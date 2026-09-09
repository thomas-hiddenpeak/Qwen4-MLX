# 从 vllm-metal 吸收的分页读取能力

2026-09-09。继续维护独立 Swift/MLX runner。上游固定到
[`023e544fec59f872f65f66e23232706e4e17ff2e`](https://github.com/vllm-project/vllm-metal/tree/023e544fec59f872f65f66e23232706e4e17ff2e)，不把其服务层、Python 调度器或原生库加入运行依赖。

## 采用顺序

| 上游机制 | 本项目的处理 |
| --- | --- |
| [物理页表与尾页 COW](https://github.com/vllm-project/vllm-metal/blob/023e544fec59f872f65f66e23232706e4e17ff2e/vllm_metal/attention/caches/kv_cache.py) | 先实现 Metal 直接读取页表；共享页池、跨请求引用计数和尾页 COW 是后续增量。 |
| [Hybrid 状态边界对齐](https://github.com/vllm-project/vllm-metal/blob/023e544fec59f872f65f66e23232706e4e17ff2e/vllm_metal/attention/state/align.py) | 页共享仍必须绑定同一位置的 GDN、QSA、PLE 状态。32-token 页可以整除现有 416-token 检查点；不把可裁切 KV 当成整个模型都可任意裁切。 |
| [MLX 读写依赖](https://github.com/vllm-project/vllm-metal/blob/023e544fec59f872f65f66e23232706e4e17ff2e/vllm_metal/attention/impls/sdpa.py) | 新 reader 将 Q/K/V、页表和掩码全部作为 primitive 输入，临时归约数组由 command encoder 保留到完成。以后写页也必须返回并传递真实依赖。 |
| Prefill/decode 专用 kernel | 保持现有阶段分离。NAX prefill 后续独立评估；上游 NAX 不直接覆盖本模型学习型 QSA 布尔掩码，不能直接替换。 |

上游 GDN 适配并未直接覆盖本项目的 `qwen4_exp_text`、外置 SSD n-gram 表和 PLE 状态。其页缓存设计值得借鉴，但不是本模型的即插即用实现，也不提供本项目完整混合状态 SSD 恢复的验收证据。MTP 的优先级保持靠后。

## 第一增量

`native/paged-attention` 增加固定 BF16、B1/S1、Q24/KV2、D256、32-token 页的 reader。乱序物理页直接寻址；逻辑布尔掩码仍按原 token 顺序解释。K/V 的 page/head/token strides 分别传递，支持不同容量和非整页尾部。

算术来自本机固定的 MLX 0.32.2 `sdpa_vector`，保留浮点运算、BF16 中间结果和第二阶段归约；仅替换 K/V 地址计算。保留 MIT 许可。vllm-metal 提供页表与依赖管理的设计参考，未复制其 kernel 算术，也未使用 NAX 或放宽精度。

Swift 的 `GPUPagedSDPAReader` 是显式 per-generator 实验配置，默认 nil。原生 ABI 成功加载后固定 DSO 寿命，identity 元数据构造一次。当前读页边界上限 131072 tokens，不代表该长度已经实测。只允许显式单 token AR decode，MTP 在请求准入阶段拒绝。

完整模型首版读取现有 capacity256 逻辑 view，通过 identity 页表寻址；没有逐 token pack/gather。两种模式仍使用同一追加、预算、私有状态、前缀缓存和归档实现。**这不是共享物理页池**，也未完成 K07/K08：普通 capacity256+stock SDPA 本来就能直接读 view，因此不能重复计算“省去整段拼接”的收益。

页表最多 16 KiB 逻辑 payload；当前 11k 的两阶段归约暂存为 3194880 B，与固定 stock dispatch 一致。它们不是 RSS 或请求峰值。编码计数代表成功提交至 encoder，不能单独证明 GPU 完成；probe 在完成等待后检查输出。

## 验证与复跑

独立原生机制已完成同步及 async-eval 各 260 项逐位对照和 lazy owner 生命周期测试：15 个长度、4 种布局，覆盖页边界、容量边界、全掩码、仅尾 token、代表性 QSA 掩码、乱序带空洞页表及独立 K/V stride。输入物理内容和 buffer 身份保持不变。该测试未加载模型或执行学习型 QSA indexer。

原生机制的短 ABBA 计时第一组存在明显首轮波动，不作为模型提速证据。公开源代码再次构建后的 260 项机制复跑通过。Swift release 编译以及请求、阶段、容量许可共 22 项 CPU 回归通过。

完整模型 P11057/O16、chunk416/eval4、MTP0 的 ABBA 状态诊断通过：16 个参考状态锚，48 组配对、5808 个原生 BF16 张量记录，以及对应的 GDN/attention offset、PLE history 和 capture 标志一致。比较的是全部逻辑状态，刻意不要求私有 storage extent 相同。四次请求的完整输出 IDs 一致，candidate 每次 15 个 decode 步骤真实编码 180 次 reader，stock 为零；请求和 workspace 额度全部归还。

两组 P11057/O128 完整生成器 benchmark 使用相同模型实例、独立私有状态、无 observer；每组各模式先预热 O16，再执行四轮。所有 8 个测量请求、4 个 warmup、4 个状态诊断请求共 **1152 个输出 IDs** 均与改动前 O512 参考的对应前缀一致。独立分析复用既有 `validate_result`/`budget` 校验全部 phase 数值、实际 decode 数、容量许可及最终额度；每个 candidate 的编码增量均为 `12 × 实际 decode tokens`。没有丢弃样本。

| 顺序 | stock + capacity256 decode tok/s | 页表 + capacity256 decode tok/s | 本块观察变化 |
| --- | ---: | ---: | ---: |
| ABBA | 24.32 | 22.49 | −7.50% |
| BAAB | 29.90 | 29.63 | −0.88% |

速率为 `sum(actual decoded tokens) / sum(decode round seconds)`，首输出不计入 decode。第一组 stock 首末 decode 耗时变化 −17.41%，存在明显漂移；第二组 stock/paged 的首末耗时变化分别 −1.29%/−0.37%。decode service 口径变化为 −7.49%/−0.88%，结论一致。两组各只有两样本/模式：**没有证明吞吐收益，保留显式实验配置，不更改默认**。

prefill target 口径的 stock/paged 速率分别为 ABBA 494.01/427.54、BAAB 636.32/611.22 tok/s。本增量未修改 prefill 算法或接入 reader；这些计时差异原样记录，不当作 prefill 优化效果，也不能由本轮确定其原因。后续共享页池的主要验收应是多会话的实际物理字节、复制字节及吞吐/延迟综合收益，而非把 identity reader 的正确运行直接升级为性能结论。

原生机制、完整模型两轮 postflight 分别核对 460 个固定文件和 102 个模型 payload stat，均无变化；最终参考服务 PID 4046 按原参数恢复，idle、MTP/drafter 关闭。新 reader 尚未接受带缓存的 HTTP 长时负载、真实系统压力、共享页 COW 或增量 SSD 验收；既有缓存验证不能代替这些后续检查。

本地结果：`results/vllm-metal-paged-mechanism-v1/`、`results/vllm-metal-paged-model-v1/`。完整模型 runner SHA 为 `daa189d99f500b51907608ebaa33e522988977eb11963131d6e05cdd2090c5ad`；原始状态报告 SHA `dcda900757eecc3a459f79bb267a1b13d5abf7c8a50c0c26e5250a795b3ab3d0`，ABBA `821dade418a4d31da62fd4b3a8775f5b61ba2030c6181b0ce10a3adf0d3c0cb3`，BAAB `71c31fe4318b7722582e1d72da1965ee6507b3c31f41346ab44ef3a608994241`。独立分析 SHA `a73086fec0e96050e5cebe95bad4fa40eeb999000063866888d52e9b500824f6`。

```sh
python3 -B scripts/build_mlx_paged_attention.py \
  --bridge-dir native/paged-attention --output NEW_BUILD_DIRECTORY
```

通过[唯一 GPU 控制器](../EXPERIMENT_CONTROLLER.md)运行新目录下的 `bin/paged-reader-probe --run`，以及 `--run --async-eval --benchmark-iterations 40`。完整模型入口：

```sh
.build/release/ane-runner benchmark-gpu-paged-attention \
  --library ABSOLUTE_NEW_BUILD_DIRECTORY/lib/paged_reader.dylib \
  --model-dir ../qwen38-ssd/models/Qwen3.8-Flash-Next-MLX-SSD-Stream \
  --tokens-file fixtures/gpu-agent-11k/prompt-token-ids.json \
  --max-tokens 16 --context 16384 --order abba \
  --warmup false --state-check true --output NEW_STATE_REPORT.json
```

状态检查读取全部逻辑 BF16 张量和语义 host 状态，会干扰计时。测性能改为 `--state-check false --warmup true --max-tokens 128`，分别跑 ABBA/BAAB；两边均为 capacity256，prefix cache 关闭，首输出属于 prefill，decode 分母使用实际前向 token 数。不得用状态诊断时长计算性能收益。
