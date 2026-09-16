# 独立 CoreAI 文本 runner

`coreai-runner` 将本模型的全部神经网络计算交给 macOS 27 系统 CoreAI。独立 Swift 可执行文件只依赖 `ANERunnerCore`，不依赖 `ANERunnerGPU` 或 `CMLX`；没有 MLX fallback。它与此前保留 MLX MoE/HC 的 [hybrid 路径](COREAI_HYBRID.md)是两个入口。

本页保留首次 **256-token 容量、逐 token prefill** 的完整迁移证据。后续已新增[4096容量的 HTTP 试用服务](COREAI_SERVICE.md)，支持JSON/SSE与RAM前缀缓存。导出、子图数值检查和完整生成验收分别记录；这些结果尚不代表262K服务、模型质量或吞吐性能验收。

## 执行边界

| 模块 | 执行位置与表示 |
| --- | --- |
| Token embedding、HC read/write、final mixer/head | CoreAI；中间激活 FP16，最终 logits FP32 |
| 36 个 GDN、12 个 QSA | CoreAI；显式 NDArray 状态，GDN recurrence 为 FP32 |
| 每层 512 专家、top-10 路由、共享专家及门控 | CoreAI；保留 affine Q4/group64 专家库，选择专家后再解包 |
| PLE 投影、归一化、门控、膨胀卷积与残差注入 | CoreAI；第 2 层（索引 1），卷积状态 `[1,9,10240]` FP16 |
| Tokenizer、chat template、n-gram hash | CPU Swift |
| SSD n-gram 行读取、FP8 解码 | CPU；按需提供 `[1,1,2560]` PLE embedding，不把 51.2 GB 表作为 CoreAI 常量 |
| 最终 greedy token 选择、EOS 判断 | CPU；读取最终 logits |

每个 token 按原模型顺序通过全部 48 层：embedding → PLE/HC/attention/HC/MoE/HC → final mixer/head。当前每步完成 **291 次 CoreAI 函数调用**。激活和 GDN/QSA/PLE 张量状态在这些调用之间保留为 NDArray，不经 JSON 或 CPU 数值重建；除 token、SSD 行及最终 logits 外，每个 QSA 的两个整数计数器也会读回 CPU 校验 offset 与 pooled count。

`CoreAINativeModel` 校验三个 manifest 的模型目录、config SHA256、完整层集合与完整专家库；部分层或 12-expert smoke 资产不能作为完整模型加载。失败会使会话失效，必须 reset；这一入口不提供并发调度。

CLI 当前固定请求 **GPU preference**。这只是调度偏好，系统可选择其他设备；没有 Instruments 执行时间线时，不能把调用成功称为 GPU 独占或 ANE 驻留证据。原生报告将 `hardware_placement_verified`、`quality_acceptance`、`performance_acceptance` 分别保留为 `false`。

## 构建和导出

以下命令从本仓库根目录执行。需要 macOS 27、SDK 27，以及可运行导出脚本的 Python 环境。本工作区使用 `../../.venv/bin/python`；导出依赖包括 `torch`、`numpy`、`coreai-core`、`coreai-torch`、`safetensors`，复用 `export_moe.py` 的权重读取也会导入 `coremltools`，但不执行 Core ML 推理。

```sh
xcrun swift build -c release --product coreai-runner

../../.venv/bin/python scripts/export_coreai_hybrid.py \
  --capacity 256 --output results/coreai-native/attention

../../.venv/bin/python scripts/export_coreai_dense.py \
  --output results/coreai-native/dense

../../.venv/bin/python scripts/export_coreai_q4_moe.py \
  --layers all --output results/coreai-native/moe
```

已用不存在的 `ANERUNNER_MLX_ROOT` 和新的独立构建目录完成上述 product 构建，并执行其 help；原生二进制的链接列表无 MLX/CMLX。完整 package release 构建也通过。标准 XCTest 未执行，当前 CLT 缺少 XCTest；本轮导出器 CPU 回归共 16 项通过，未修改全局工具链或 Xcode 许可状态。

`export_coreai_hybrid.py` 保留历史名称，但只导出 attention 子层；这些纯 CoreAI 资产也被原生 runner 复用。不要为完整模型加入 `--layers` 子集或 `--experts 12`。导出器要求新的输出目录，不覆盖完成的 manifest。

三个导出器均读取本机已验证的原始权重与 source manifest，记录源 tensor slice/hash、配置身份和资产哈希。源模型、source verification 文件、捕获 fixture、`.aimodel` 与 `results/` 不随 Git 克隆提供；新机器需要先按本仓库的本地模型准备流程建立这些输入。导出本身只运行 CPU authoring/reference，不调用设备 runtime。

本机已完成的资产如下；磁盘大小不是 runtime RAM 或 GPU 驻留测量。

| 资产 | 完整范围 | 资产字节数 |
| --- | --- | ---: |
| Attention | 48 层，capacity 256 | 5,430,037,148 |
| Dense | 96 个 HC read，加共享 write、PLE、embedding、head，共 100 份 | 3,898,275,627 |
| MoE | 48 层，每层完整 512 专家 | 68,547,023,642 |
| 合计 | 全部神经模块 | 77,875,336,417 |

导出需要约 78 GB 的额外资产空间，并保留原模型目录和 SSD 表。当前本机 attention 资产路径是 `results/coreai-hybrid-full/manifest.json`；上面的复跑命令使用新的 `results/coreai-native/attention/manifest.json`，两者是同类 manifest，选择实际存在的一份即可。

## 生成命令

```sh
.build/release/coreai-runner generate \
  --model-dir ../qwen38-ssd/models/Qwen3.8-Flash-Next-MLX-SSD-Stream \
  --attention-manifest results/coreai-native/attention/manifest.json \
  --dense-manifest results/coreai-native/dense/manifest.json \
  --moe-manifest results/coreai-native/moe/manifest.json \
  --prompt '请用一句简短的中文介绍你自己。' \
  --max-tokens 48 --repeat 2 \
  --output results/coreai-native/generation.json
```

默认先应用本模型 chat template；`--raw-prompt true` 才直接编码原始文本。`--max-tokens` 为 1...256，`--repeat` 为 1...3。**编码后的 prompt 长度加 max-tokens 必须不超过导出容量 256**。repeat 在同一次加载后 reset，并检查重复生成的 token IDs 是否一致。

Prefill 与 decode 分开记录：prefill 当前也是 S1 逐 token 执行，不能与既有 MLX 的分块 prefill 吞吐混为一谈。首个生成 token 来自最终 prefill logits，因此报告中的 `decode_forward_steps` 通常比输出 token 数少 1；EOS 可提前结束。`load_seconds` 与这两个阶段分开，SSD 读取时间及逻辑字节也单独记录。

成功报告检查进程动态库中是否存在名称含 `mlx` 的镜像，并提供 `mlx_runtime_images`。这项检查与 package 的依赖边界共同说明原生入口没有调用 MLX；它不等同于证明所有 CoreAI 算子都实际落在指定硬件上。

## 已完成的数值检查

### Dense 子图

系统 CoreAI GPU preference 对 5 类函数执行完成。参考是相同 FP16 权重/边界下的 CPU 方程；初始子图门槛为 relative L2 ≤ 0.005 且 max absolute ≤ 0.02。

| 输出 | Relative L2 | Max absolute |
| --- | ---: | ---: |
| Embedding stream | 0，逐位相同 | 0 |
| HC read mixed | 0.000386 | 0.00390625 |
| HC read injection | 0.000700 | 0.00012207 |
| HC write stream | 0.000309 | 0.00048828 |
| PLE stream | 0.000573 | 0.00024414 |
| PLE 下一卷积状态 | 0.000386 | 0.001953125 |
| Final mixer/head logits | 0.000340 | 0.00403553 |

Embedding 使用真实 token `248046`；HC 和 head 使用其真实 embedding stream。HC write 的 block output 使用真实 HC mixed 作为测试输入，不冒充真实 attention 输出。PLE 使用真实权重、该 token stream，以及确定性低幅合成 PLE embedding 和非零历史，以覆盖门控和膨胀卷积状态；这项测试不证明实际 SSD/hash 流程已端到端通过。

本机证据：`results/coreai-native/dense-fixtures/manifest.json`、`runtime-summary.json`。另有 3 项 CPU 单元测试通过，覆盖 HC 的独立 NumPy 方程、PLE 与 grouped convolution 的状态/膨胀对照，以及 embedding/head 的接口。

### Q4 解包的 I16 兼容实现

初始 I32 packed gather/unpack 图在本机 macOS 27 beta 的 GPU preference 下出现明显错误：12-expert smoke 的 selected IDs 正确，但 MoE output relative L2 为 **0.504784**。独立解包诊断也发现 codes/dense 不同，而 CPU-only 诊断能匹配。这些观测尚不能唯一定位系统内部哪一个算子或转换导致低位丢失。

当前把原始 U32 packed 字节解释为两个 **I16 lane**，保持字节数及原始 Q4 编码不变；先 gather 选中的专家，再用整数 mask/floor-divide 解出 nibble。不是将全部专家展开成 FP16，也不是新的 16-bit 权重量化。scale/bias 与选中专家的计算采用记录的 FP16/FP32 边界。

| 当前检查 | 结果 |
| --- | --- |
| I16 独立 gather、mask、divide、codes、dense、projection | 本机 GPU preference 与 CPU 参考逐位相同 |
| 12-expert layer 0 smoke | IDs 精确相同；output relative L2 0.000512，max absolute 0.00024414 |
| 完整 512-expert layer 0，真实捕获输入 | IDs 精确相同；output relative L2 0.000461，max absolute 0.00012207 |
| 完整 512-expert layer 0，零输入 | 输出逐位相同，误差 0 |

本机证据：`results/coreai-q4-unpack-diagnostic/`、`results/coreai-q4-unpack-i16/`、`results/coreai-q4-moe-smoke-v2/`、`results/coreai-q4-moe-i16/`、`results/coreai-q4-moe-layer0/`。

全 48 层完整专家资产已导出，但这些数值检查只覆盖上述指定层和输入。gather/unpack/FP32 GEMM 目前是功能实现，尚不是融合 Q4 性能 kernel；资产保留压缩编码也不自动证明整个 runtime 的内存分配行为。

## 首次端到端运行

`results/coreai-native/generation.json` 已记录一次完整原生加载后连续两遍生成，第二遍执行 reset。输入经 chat template 编码为 20 token；两遍输出的 27 个 token IDs（含 EOS）逐位相同，文本为：

> 我是通义千问，由阿里巴巴通义实验室独立开发的大语言模型，致力于成为您真诚、有用的思考伙伴。

| 观测 | 第一遍 | reset 后第二遍 |
| --- | ---: | ---: |
| Prefill token 数 | 20 | 20 |
| Prefill 时间 | 11.582 s | 8.198 s |
| Decode forward 次数 | 26 | 26 |
| Decode forward 时间 | 9.708 s | 9.572 s |
| Decode forward 吞吐 | 2.68 token/s | 2.72 token/s |
| 全部 forward 次数 / 最终状态 offset | 46 / 46 | 46 / 46 |
| CoreAI 调用数 | 13,386 | 13,386 |
| SSD 逻辑读取字节 | 117,760 | 117,760 |

本次加载用时 97.545 s，独立记录，不计入 prefill/decode。吞吐按 decode forward 次数除以该阶段时间计算；本轮是每 token 291 次调用的未优化原型，不是性能验收。第二遍还会受系统与 SSD 缓存状态影响，不能把这两遍时间差解释成 kernel 优化收益。

成功报告的 `mlx_runtime_images` 为空，结合独立 target 无 CMLX 依赖，确认这次生成没有调用 MLX runtime。目录/config 身份检查，以及缺层、非完整专家库、重复 MoE 资产三项负向拒绝检查也通过，证据在 `results/coreai-native/negative/report.json`。

相同 prompt 下，原生输出与此前 hybrid/源 BF16 参考在第 **17 个生成 token** 开始分歧：原生选择逗号后继续扩展介绍，hybrid 选择句号并结束。两遍 reset 一致只证明当前路径可重复，不能声称原生与 BF16 模型质量等价；`quality_acceptance` 仍为 `false`。

另一次独立加载运行“只回答数字：17乘以3等于多少？”，prompt 为23 token。两遍都生成 `[20,16,248046]`，文本 `51`、正常 EOS，与此前 hybrid/源参考 token 一致；每遍25次 forward、7,275次 CoreAI 调用、最终 offset25，reset 重放精确一致。原始结果在 `results/coreai-native/arithmetic.json`；该请求期间另有独立 CPU 构建，不作为性能比较样本。

**两个请求、四次完整生成共执行 41,322 次 CoreAI 调用。** `results/coreai-native/acceptance-summary.json` 独立复核调用数、offset、reset、EOS、链接库及二进制 SHA256。此处“通过”限定于这些集成检查，仍不是模型质量或服务发布验收。

## 尚未迁移的业务能力

本页首次生成容量只有256，不会触发超过2051 token的QSA稀疏筛选。后续4096资产及实际HTTP验证见[CoreAI服务](COREAI_SERVICE.md)；尚未证明262144-token上下文。较长QSA边界子图的独立证据见[CoreAI连续状态验证](COREAI_STATEFUL.md)，不能替代完整模型长上下文验收。

后续已实现有界的单worker HTTP API、取消/reset，以及独立深拷贝的完整状态RAM前缀缓存。分块或业务独立部署的prefill/decode、前缀树、SSD KV offload、持久化导入导出及内存压力策略仍未迁入。原有MLX缓存、prefix或archive文件不能导入CoreAI会话；神经状态表示和版本不同，需要独立迁移与验收。

原 BF16 路径被 FP16 权重/激活及 FP32 GDN state/logits 替代，路由、词表排名和长程累积误差都需要完整模型质量验证。子图低误差、可生成文本或 reset 重复一致，均不足以单独判定质量合格。先完成全链路，再做质量、状态管理和性能；MTP 优化仍放在后段。
