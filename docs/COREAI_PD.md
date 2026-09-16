# CoreAI 本机 Prefill / Decode

这条路径为完整 48 层模型提供独立的 CoreAI 函数：`prefill` 处理导出的主块，`main` 一次处理 1 个 token。它们共用同一个 `AIModel` 和显式状态，不加载两个完整模型。初版主块为S4；当前实验支持S4至S2048及较小尾块函数，按剩余长度和system前缀快照边界选择，不补假token。大块kernel与1000 token/s目标见[Prefill优化](COREAI_PREFILL.md)。

服务内部将 prefill 与 decode 分成独立操作，以一次性 handoff 交接模型 offset、最终 logits、n-gram history 和阶段统计。handoff 绑定会话与请求，取消或 reset 后不可使用。当前调度仍是**单请求串行**：`pd_scheduling=serial`，没有多请求交错、独立部署或跨进程状态传输。

## 函数与权重

每个 decoder layer 把 HC read、attention、HC write、MoE 和对应写回合在一个 CoreAI 调用中。第 1 层同时包含 PLE 投影和卷积。完整模型每个 S1/S4 块调用 embedding、48 个层和最终 head，共 50 次；之前逐组件路径每 token 为 291 次。调用数减少不能直接当作性能提升。

GDN 在块内顺序更新 recurrent state；QSA 为每个位置保持正确 mask、池化边界与计数。两个入口采用相同的状态布局，包含 GDN、QSA 以及 PLE 历史。输出 head 只投影块的最后一个位置。CPU 继续从 SSD n-gram 表按需读出每个 token 的行，不把整张 51.2 GB 表加入 CoreAI 资产。

路由为每个 token 独立选择 top-10，保留相同的并列选择规则与 FP16 归一化边界。原 affine group-64 Q4 权重保留压缩形式，未将 512 专家常驻展开成浮点权重。`--q4-kernel metal` 使用自定义 Metal kernel 直接解包、反量化并累加，避免生成选中专家的完整浮点矩阵；`reference` 保留原先的展开路径供对照。它仍由 CoreAI 加载与调用，不链接 MLX。

## 导出与比较

从仓库根目录执行，Python 环境需安装本工作区 CoreAI authoring 依赖。资产约 78 GB，不随 Git 提供。导出是 CPU 工作，设备运行单独验证：

```sh
../../.venv/bin/python scripts/export_coreai_pd.py \
  --output results/coreai-pd/fused-s4-metal --chunk 4 --capacity 4096 --q4-kernel metal

.build/release/coreai-runner generate \
  --model-dir ../qwen38-ssd/models/Qwen3.8-Flash-Next-MLX-SSD-Stream \
  --attention-manifest results/coreai-service/attention4096/manifest.json \
  --dense-manifest results/coreai-native/dense/manifest.json \
  --moe-manifest results/coreai-native/moe/manifest.json \
  --pd-manifest results/coreai-pd/fused-s4-metal/manifest.json \
  --prompt-file results/prompt.txt --max-tokens 8 \
  --compare-prefill true --output results/coreai-pd/compare.json
```

指定 PD manifest 时只加载该 manifest 的资产，旧三份 manifest 不额外加载。CLI仍保留其参数位置。运行时校验完整48层以及全部主块、尾块和S1函数。S32至S2048需用实验性`--prefill-kernels tensor --q4-kernel metal`导出；支持加载不等于完成质量、性能或服务验收。

`--compare-prefill true` 在同一模型实例按 S1、S4、S4、S1 跑四轮，每轮重置所有状态与 n-gram history。报告包含最终 prefill logits 的相对 L2/最大绝对误差、输出 token 是否一致，以及独立的 prefill/decode 耗时。首轮调用可能包含系统编译开销，不能混入热态吞吐比较。

`prefill_group_milliseconds` 与 `decode_group_milliseconds` 按业务阶段区分；其内部 `decode.*`/`prefill.*` 标签表示实际选用 S1/S4 函数。因此提示词尾部的 `decode.*` 仍属于 prefill。分组时间是等待函数返回的墙钟时间，不是 GPU kernel profiler。SSD 行读取另行计时。

## 验证边界

CPU 检查覆盖多 token HC/embedding、PLE 非零历史、逐 token MoE 路由、GDN/QSA 跨块状态和 QSA 稀疏切换。真实 layer-0 的 S4 与 4 次 S1 设备结果相近，随后第 5 个 token 的输出及状态一致。这不代替完整模型质量或性能验收。

运行时检查完整 48 层、512 专家元数据、源配置哈希及所有入口和状态形状；不进行完整资产字节身份校验。源 BF16 与当前 FP16 主干的质量等价尚未验收。4096 为资产容量，262K、SSD KV offload 和并发 PD 调度尚未迁移到此 CoreAI 路径。

## 2026-09-17 实测与负结果

M5 Max / macOS 27，完整 48 层、512 专家、capacity4096。原始文件在工作区 `results/coreai-pd/`，不提交模型或大体积临时资产。

- 直接 Q4 kernel：真实完整 layer-0 MoE 的 S1/S4、实际输入/零输入四组均通过预设数值阈值，路由 ID 完全一致。13 项诊断最大相对误差约 0.00060，最大绝对误差 0.0009766；这些诊断也包含普通 router/shared 路径，不能把全部误差归因于自定义 kernel。热态 S1 约 2.39 ms，S4 约 3.39 ms。证据 `q4-real-device.json`。
- 完整层：原展开路径 S4 热态约 20.15 ms，直接 Q4 路径约 4.91 ms；块后继续解码的状态和输出检查通过。证据 `layer0-device.json`、`layer0-metal-device.json`。这是单层结果，不能乘层数当作整模型速度。
- 同进程、同 120-token prompt，按 S1/S4/S4/S1 执行。预填分别 16.09/5.15/5.28/14.91 s，最后两轮约 22.71/8.05 token/s；四轮均输出47。解码单步热态约 0.12 s，首次从 S4 切回 S1 可出现 0.86–0.99 s 的额外开销。证据 `metal-short.json`。最终 S4/S1 logits 相对 L2 差异约 0.0733，最大绝对误差 0.798；不能据答案相同声称数值等价或完成质量验收。
- 17 项真实 HTTP/缓存/隔离/取消/队列检查通过，另有 79 份 health 样本的阶段统计检查；`http-acceptance.json`、`health-audit.json`。18 项 CPU fake-backend 网络检查通过，`transport-cpu.json`，不计作设备性能。
- 同一 2064-token 请求，首次 82.18 s，完整前缀命中 0.281 s，均输出47，5项检查通过；`long-acceptance.json`。这是端到端时间，当时同时进行下一候选资产的 CPU 导出；旧 S1 服务同份请求的历史结果是804.79 s，不能将两次结果当作严格隔离的同期A/B。
- 展开权重的整模型 S4 参考路径在测试中出现交换空间持续增长（系统 SSD swap 约0.98→10 GiB），主动停止，未得到完整报告。单层通过不意味着完整模型可用。`fused-short-stopped.json` 和采样保留这一负结果；不要把它启用为默认服务路径。

为定位 S1/S4 差异，CPU 同输入检查分别固定 GDN/MoE 的 linear 形状、QSA 的 linear 与 SDPA 查询形状，输出与状态可以完全一致。GPU 全模型逐层 trace 则显示小误差随深度和序列长度放大；尚未逐一证明路由翻转的贡献。可选 `--stable-projections` 使用固定 SIMD 归约的 dense Metal kernel 做进一步对照，默认关闭。它并不承诺复现普通 CoreAI 或 CPU GEMM 的求和顺序，结果与性能必须重新验收。

固定归约对照已完成整模型测试：同一120-token提示词，S1/S4/S4/S1预填分别13.22/4.79/4.82/12.24秒，答案仍为47，但S4/S1最终logits相对L2为0.1171，最大绝对误差1.2753，反而高于普通投影路径。证据 `results/coreai-pd/stable-short.json`。因此保留为关闭的诊断选项，不作为数值修复或吞吐优化路线。

2026-09-17优先级调整：先达到 **10K以上提示词、前缀缓存未命中的完整CoreAI prefill至少1000 token/s**，decode单独统计。S4约23 token/s尚未满足目标。当前推进大块矩阵投影、按专家分组的Q4矩阵乘法、单次调用内完成的GDN递推与QSA增量更新；单kernel通过不等于整模型达到目标。其它性能方向暂缓，最新结果统一记录在[Prefill优化](COREAI_PREFILL.md)。
