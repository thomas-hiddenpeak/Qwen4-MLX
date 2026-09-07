# 独立 Swift GPU runner

本工程现在可以从原模型目录执行完整的 **48 层文本生成**：原生分词 → embedding → Hyper Connection → GatedDeltaNet / QSA → PLE 的 SSD 按需输入 → 动态 top-10 MoE 与共享专家 → 输出 head → 贪心采样。计算使用 Metal GPU，Swift 管理权重、状态和 SSD 读取；推理过程不启动 Python，也不调用作者的服务。

Prefill 与 decode 按独立业务阶段建设，接口与 kernel 参数见 [Prefill / Decode 分离](docs/PREFILL_DECODE_SEPARATION.md)。完整的 `prefill` / `decode` / `generate` 继续可用，新增 begin/step session 可分别恢复一个 prompt chunk 或完整 AR/MTP round。共享一份权重的 `QwenLocalScheduler` 默认仍为 `wholeStages`，显式选择 `cooperative` 才在这些边界切换作业；GPU 始终串行。调度器本身由调用者推进，不启动后台 worker 或 HTTP listener；上层已有独立的 [HTTP/SSE 适配器](docs/HTTP_SERVER_EXPERIMENT.md)，跨进程 PD 尚未实现。此前 20 项 CPU / 27 项实模检查属于完整阶段历史证据；增量接口的历史验证包括 26 项 CPU 测试和 55 项实模检查，包含 11k 长短请求实际交错。MTP 仍按独立 decode 有效吞吐与 TPOT 验收。

MTP 默认关闭。当前兼容候选可显式指定 `--mtp-depth 2 --mtp-verification batchedScalarLinear`，另有 `--mtp-draft-history 1024` 的草稿头初始历史实验（默认完整历史，主干上下文始终完整）。输出分叉已在固定短/长基准修复；更广回归、适用范围及实际收益见 [MTP 与会话接口](MTP_AND_SESSIONS.md) 和 [发布标准](docs/MTP_RELEASE_CRITERIA.md)。`generate-gpu` 不使用 ANE；已有 Core ML / ANE 子图探针仍保留为独立实验，不参与这一整模型路径。底层依赖固定版本的 MLX / MLX C 原生库，部分计算内核参考作者实现并保留来源及许可，见 [源码目录](Sources/ANERunnerGPU/) 与 [上游许可](UPSTREAM-LICENSE)。

**当前生成默认：chunk416、跨块 SSD 预取 `nextChunk`、1 个 SSD worker，prefill attention 策略为 `reference`。** `reference` 保留已有路径，包括符合条件时的融合 causal attention；设置 `ANERUNNER_FUSED_PREFILL=0` 只关闭无 QSA 的 causal 融合。GDN blocked 仍关闭，每 4 层同步，MTP 关闭；源权重格式和 `reference` BF16 累加保持不变。最初将分块从 64 改为 416 的输出一致性检查仅覆盖下述 1,217-token 文档提示；后续长上下文检查单独记录，不能推广为任意提示等价。

显式传入 `--prefill-attention fusedQSA` 可在 prefill 中已有 QSA 可见性 mask、当前块长度大于 8 时使用融合 SDPA，沿用相同的布尔 mask 和 causal 尾部。它不改变 chunk 边界，不替换单 token decode 或 MTP verification 路径；末尾保留的单 prompt token 仍走原路径。这个请求级策略与上述 causal 融合环境变量独立，省略参数或指定 `reference` 均保留原选择。融合会改变归约舍入，不能由 mask 相同推断完整输出等价；实现与验证边界见 [QSA prefill 融合](docs/QSA_PREFILL_FUSION.md)。

用户确认纳入默认后，Release 构建及[默认配置验证](results/prefill-defaults-smoke/summary.json)已通过：未传分块、预取、worker 参数，并清除相关实验环境变量，实际运行报告确认以上默认值，三轮生成的 64-token 输出全部匹配此前重测参考。该检查用于确认默认路径生效；没有据此追加性能提升结论。

[本轮完整构建汇总](results/gpu-full-model-milestone.json)保存验证范围、执行参数、文件哈希、最终计时和服务恢复记录。

可选的进程／系统采样、Instruments 时间轴与离线 phase 对齐见 [采样使用与指标口径](TELEMETRY.md)。已保存同请求有无采样的完整模型配对结果；GPU trace 的本模型实测状态在该文档单独列出，不由逻辑权重规模推算硬件带宽。

针对固定模型结构的单 token 优化、同进程交错对照与最新结果见 [专用解码优化](GPU_SPECIALIZATION.md)。单 token decode 默认保留 `reference` 路径，候选不改量化或启用 MTP。

最新的 [主要瓶颈审计与 GPU 命令计时](GPU_BOTTLENECK.md)进一步拆分实际权重格式和命令缓冲执行时间，用于定位下一步的大矩阵优化。

最新 routed MoE 候选 `--mtp-verification batchedTokenMoE` 将 S2…5 验证的逐 token 专家调用合并；正确性通过，11k/128 本轮约+3.6%、256复测基本持平，稳定性能收益未建立，默认不变，见 [token 轴实验](docs/MTP_TOKEN_AXIS.md)。

## 本地阶段调度器

`QwenLocalScheduler(generator:limits:)` 复用同一个 generator / 模型；`submit` 返回作业 UUID，调用者在同一 inference executor 上反复调用 `runNext()`。默认 `wholeStages` 推进完整阶段；可选 `cooperative` 推进一个 prompt chunk、首次 token 发布或完整 decode round，也可领取一个待处理事件。`cancel(id)` 和 `discardAll()` 在 yield 后清理等待状态，`snapshot()` 返回队列、活动作业、状态数量与额度；调用者停止推进，作业就停止在当前边界。

默认 limits 为 `executionMode: .wholeStages`、prefill 队列 8、ready decode 2、token 额度 32768、连续 prefill 1。显式开启合作模式：

```swift
let scheduler = try QwenLocalScheduler(generator: generator, limits: .init(
    executionMode: .cooperative, decodeBurst: 4, maxResidentSequences: 2))
```

同进程 PD 可逐请求选择 prefill attention，完整阶段和增量接口都使用同一不可变请求字段：

```swift
let request = QwenGenerationRequest(tokens: tokens, prefillAttention: .fusedQSA)
let jobID = try scheduler.submit(request)
```

省略 `prefillAttention` 默认为 `.reference`。该策略随请求经过 prefill chunk 恢复与 ready handoff，作用范围始终是 prefill；decode 和 MTP verification 使用各自策略。同一模型与 inference executor 的生命周期约束保持不变，不提供跨进程状态传输。CLI 在 `prefill_attention_mode` 中记录选择，阶段 API 在 `QwenPrefillStatistics.attentionMode` 中记录选择；这些字段表示请求策略，不证明每个块均满足融合条件。

合作模式保留原 chunk416 和最后一个单独 prompt token，不额外切小数值分块。decode 首步只发布首 token，后续每步完成一轮 AR/MTP；MTP 本轮全部输出发布后才 yield。未完成作业放回对应队尾，最多连续 4 个 decode 步骤后给可准入的 prefill 机会，ready 满时先继续 decode。这是软件的继续执行边界，不是硬件 kernel 抢占。

额度从 submit 开始按 `prompt.count + maxTokens` 预留，包含未开始、partial prefill、ready、活动/暂停 decode，终止时释放一次；它不是物理内存字节限制。合作模式 `maxResidentSequences` 默认 2，统计已开始的所有请求状态，排除尚未开始的请求；prefill 队列上限同时统计 partial 和新请求，并为活动 producer 保留回队槽。

调度器、ready 句柄和增量 session 均不可 Sendable。低层接口为 `beginPrefill` / `stepPrefill` 与 `beginDecode` / `stepDecode`；步骤未完成返回 nil，最后分别返回 ready 或生成结果。cursor 提供 `isFinished`、`isActive`、进度计数与 `discard() throws`。每次增量 prefill yield 排空已安排 SSD 读取，同时保留已完成的下一块预取数据；没有 GPU/SSD 工作跨过切换边界。

跨线程取消使用 `QwenCancellation`；callback 内可提交新作业或查询 snapshot，不能嵌套推进/清空队列或丢弃活动 cursor。准入前拒绝保留 cursor；步骤开始后发生取消/错误则废弃它，MTP 半轮不重试。详细合同见 [阶段设计](docs/PREFILL_DECODE_SEPARATION.md#可恢复的同进程步骤)。

确认完整模型资源已交接后，短场景验收入口为：

```bash
.build/release/ane-runner probe-gpu-local-scheduler \
  --model-dir ../qwen38-ssd/models/Qwen3.8-Flash-Next-MLX-SSD-Stream \
  --suite all --max-tokens 64 \
  --output results/manual-local-scheduler-short.json
```

真实 11k 输入只执行混排检查，避免用长提示重复每个取消分支：

```bash
.build/release/ane-runner probe-gpu-local-scheduler \
  --model-dir ../qwen38-ssd/models/Qwen3.8-Flash-Next-MLX-SSD-Stream \
  --tokens-file fixtures/gpu-agent-11k/prompt-token-ids.json \
  --suite mixed --max-tokens 128 \
  --golden-report results/prefill-agent-11k-default-check/default.json \
  --output results/manual-local-scheduler-11k.json
```

`--output` 必须是新文件；`--tokens-file`、`--max-tokens`、`--suite`、`--golden-report` 均可选。默认 suite 为 `all`、输出预算 64，允许预算 3…256；golden 比较完整 IDs，不自动裁剪预算。探针包含显式的 MTP depth2 / `batchedScalarLinear` / tail1024 作业，不改变普通 `generate-gpu` 默认关闭 MTP 的设置。

此前 17 项 handoff 检查证明阶段 API 与单次消费。历史 `wholeStages` 版本的 20 项 CPU / 27 项实模检查全部通过，包括 FIFO、两个 ready 状态、AR/MTP exact、取消清理和额度释放，见[历史调度器汇总](results/local-pd-scheduler-v1/summary.json)。该次真实 11k/128 的 AR/MTP 输出匹配原 golden，纯 decode 观察值为 29.67 / 35.30 token/s；这些结果没有覆盖新增的 chunk/round 交错，也不是 kernel 加速或全部 MTP 发布门槛通过的证据。

本轮合作调度的 Release、[26 项 CPU 测试](results/cooperative-pd-v1/cpu-tests.log)及 55 项实模检查均通过，见[验收汇总](results/cooperative-pd-v1/summary.json)。固定实模探针在一个模型上比较长/短作业的完整阶段和合作模式，并检查长 MTP 恢复、取消、状态释放和输出一致性：

```bash
.build/release/ane-runner probe-gpu-cooperative-scheduler \
  --model-dir ../qwen38-ssd/models/Qwen3.8-Flash-Next-MLX-SSD-Stream \
  --tokens-file fixtures/gpu-agent-11k/prompt-token-ids.json \
  --golden-report results/prefill-agent-11k-default-check/default.json \
  --output results/manual-cooperative-scheduler.json
```

`--output` 必须是新文件；此固定探针要求 11,057-token 输入，长/短输出预算分别为 128/64。[验收计划](results/cooperative-pd-v1/plan.json)已完成，参考服务已恢复。一次长 AR / 短 MTP 对照中，合作模式把短请求首 callback 从 20.322 s 降到 1.679 s、完成时间从 22.047 s 降到 7.827 s，整组耗时由 22.048 s 变为 22.305 s。长短请求输出均匹配独立参考，长 MTP 恢复也通过；这是调度延迟观察，不是稳定 kernel 加速结论。默认模式仍为 `wholeStages`。

增量指标将活动计算与暂停分别记录：`prefill.suspensionSeconds` / `phases.decodeSuspensionSeconds` 为可选字段；调度 `initialPrefillQueueWaitSeconds` 与 `prefillResumeWaitSeconds` 区分首次和恢复排队，`prefillQueueWaitSeconds` 是两者合计，ready wait 包含 decode 恢复等待。**提交后的实际 TTFT 使用 `submission_to_first_callback_seconds` 和原始 callback 时间。** 生成结果的 TTFT 已含 session 内暂停，再加聚合 prefill wait 会重复计数，不能用这种合成值当作精确首字延迟。

## Agent 长系统提示基准（11k）

后续 agent 首次请求的 prefill 优化使用长系统提示基准，1,217-token 文档继续用于快速回归。[新输入](fixtures/gpu-agent-11k/provenance.json)包含独立编写的 agent 规则和工具契约，以及未重复填充的真实项目文档：system 角色 **10,993 token**，加短 user 问题及 no-thinking 聊天边界后总输入 **11,057 token**。工具契约只是本地模拟文本，不执行工具，也不是宿主助手的系统指令。

使用默认 chunk416、跨块 SSD 预取、1 worker、scalar GDN 和 MTP 关闭，显式将 `--context` 设为 16384，生成 128 token。每轮重新创建全部模型状态，因此这是**没有前缀缓存命中**的长提示初始化；预热仅复用权重、已编译内核及 OS 文件缓存。

初始六轮预取开关交错测试出现未解释的后期减速，开启预取的首字时间由约 15.7 s 到 28.6 s，解码也明显下降；[记录](results/prefill-agent-11k/summary.json)保留，但不用于归因预取收益。运行前和间隙检查未发现外部 Python 训练进程，后续检查未报告供电或系统热告警；这些检查没有定位减速原因。

随后只用默认配置、加入现有 200 ms 轻量采样复跑三轮。[复跑汇总](results/prefill-agent-11k-default-check/summary.json)与[采样分析](results/prefill-agent-11k-default-check/telemetry-analysis.json)：

| 请求 | 首字时间，不含模型加载 | Prefill | 后续 decode |
|---|---:|---:|---:|
| 新进程首次请求 | 20.817 s | 531 token/s | 32.62 token/s |
| 热请求 1，无前缀缓存 | 13.298 s | 832 token/s | 32.26 token/s |
| 热请求 2，无前缀缓存 | 13.812 s | 801 token/s | 30.93 token/s |

模型加载另需 9.392 s；MLX allocator 峰值约 79.831 GB。此表带采样且热样本只有两轮，不代表不带采样的性能上限。两批共九轮的 128-token 输出逐项一致，全部十二个 QSA 层进入选择路径；这是预取开关和重复运行的一致性检查，尚未与作者在 11k 输入下逐项对照。

第 5 个 416-token 块累计到 2080 token 后触发 QSA。该历史批次的多 token prefill 使用完整 KV 的稠密 attention 与可见性 mask，绕过普通 causal 融合路径；当时开启 causal 融合开关不代表这些 QSA 块也执行融合 attention。这一描述不适用于单 token decode 的向量 SDPA 读取行为。复跑中，后段完整 416-token 块约需 0.54–0.56 s，前段约 0.41–0.47 s；预取后多数后续块的剩余 SSD 等待接近零。新增显式 `fusedQSA` 策略的检查另见 [QSA prefill 融合](docs/QSA_PREFILL_FUSION.md)，上述历史计时不是新策略的结果。跨请求系统提示缓存复用仍需单独评估，不把首次 11k 初始化与 agent 后续轮次混在一个指标里。

```bash
.build/release/ane-runner generate-gpu \
  --model-dir ../qwen38-ssd/models/Qwen3.8-Flash-Next-MLX-SSD-Stream \
  --tokens-file fixtures/gpu-agent-11k/prompt-token-ids.json \
  --context 16384 --max-tokens 128 --repeat 3 \
  --output results/manual-agent-11k.json
```

## Prefill 快测（2026-09-06）

同一份 1,217-token 项目文档、生成 32 token、关闭 MTP，排除每个进程第一轮：

| 配置 | 热 prefill token/s | 热首字时间 |
|---|---:|---:|
| 原配置，chunk64 | 351 | 3.464 s |
| chunk256 | 695 | 1.753 s |
| chunk512 | 690 | 1.763 s |
| 后续 chunk512 参考，两次热运行 | 聚合 756 | 1.769 / 1.449 s |
| chunk512 + 融合 causal attention，两次热运行 | 聚合 835 | 1.752 / 1.164 s |

分块快测见 [原始结果目录](results/prefill-chunk-quick/)，融合 attention 比参考聚合约快 10.35%，见 [汇总](results/prefill-fused-quick/summary.json)。不同批次、缓存和预热状态会影响绝对速度，不能把最好的一次当作稳定吞吐。本轮首次 prefill 约 11–13 秒，不含模型加载。

chunk512 和融合 attention 的所有本轮 32-token 输出均匹配原配置。chunk256 在第 30 个 token 起出现差异；分块会改变 GDN 的 BF16 状态写回边界，不能视为纯调度改动。这里没有覆盖 QSA 启用后的长上下文。

作者服务随后处理相同 raw prompt，确认输入 1,217 token、输出 32 token、无前缀缓存；热请求报告纯 prefill 1,491.2 token/s，HTTP 首个文本片段延迟 0.870 s，见 [请求记录](results/prefill-chunk-quick/author-stream.json) 和 [服务日志](results/prefill-chunk-quick/reference-restored.log)。作者纯 prefill 与 Swift 包含最终提示 token/首次采样的口径不同，不能直接当作相同内核计时。

当时用实验开关 `ANERUNNER_FUSED_PREFILL=1` 配合 `--prefill-chunk 512` 验证。仅在无 QSA、序列长度大于 8、head dimension 256 时使用隐式 causal mask 和 `forceFused`；已有 KV 的 causal 对齐保持一致。该阶段曾保留 chunk64、关闭融合 attention 的旧默认；当前生成默认已改为 chunk416，并开启符合条件的融合 attention。

该阶段发现 MoE 合批路径从本模型单块 205 token 起符合分派条件；这批短提示快测没有覆盖 QSA 激活后的多 token prefill。现有 `reference` 策略保留原 QSA prefill 路径，新增的显式 `fusedQSA` 选择及验证边界见 [QSA prefill 融合](docs/QSA_PREFILL_FUSION.md)。

### GDN blocked-seq 实验

已移植作者的 TB32 分块 GDN 递推，使用 `ANERUNNER_BLOCKED_GDN=1` 开启；只影响至少 64 token 的 fused GDN 前向。输入、输出及跨调用状态仍为 BF16，调用内部状态保持 FP32。来源和许可见 [内核](Sources/ANERunnerGPU/GPUGatedDeltaNetBlocked.swift) 与 `UPSTREAM-NOTICE`。

同一份 1,217-token 输入、chunk512、融合 attention 开启、生成 64 token，每种配置运行四轮。[完整结果](results/prefill-blocked-quick/summary.json)：

| 配置 | 后两轮首字时间，不含加载 | 后两轮聚合 prefill |
|---|---:|---:|
| scalar GDN | 1.459 / 1.497 s | 823.6 token/s |
| blocked GDN | 1.474 / 1.471 s | 826.4 token/s |

缓存逐渐热起来后仅相差约 0.34%，本轮没有明确整模型收益。排除首次请求、纳入其余三轮会得到约 13.7% 的表观吞吐提升，但较早轮次的 SSD 剩余等待明显不同，不据此接受该收益。解码仍约 36–38 token/s。

[局部复用输入检查和计时](results/prefill-blocked-quick/recurrence-timing.json)覆盖 64、65、512 token 的输出、最终状态及后续一步 decode。相对 L2 最大约 0.00365%；512-token 递推同步调用中位数由 1.214 ms 降至 0.804 ms。这只是递推部分，不包含 GDN 的大矩阵投影，也不等于真实 GPU 内核或整层耗时。

整模型两种配置各自四轮输出一致，但彼此在第 36 个 token 起分叉：例如原版的“真实开销”变为“真实表现”。blocked 改变了 FP32 求和顺序，不能宣称逐 token 等价。该功能保留为默认关闭的实验，不改量化、不启用 MTP。

### Prefill 同步间隔实验

`ANERUNNER_PREFILL_EVAL_LAYERS=1...48` 控制 `generate-gpu` 在 prefill 中每隔多少层同步求值，默认仍为 4。它不改变单 token 解码，也不移除最终 prompt 分块的同步；PLE 处已有的异步提交仍保留。

[快速对照](results/prefill-sync-quick/summary.json)沿用上述输入、chunk512 和融合 attention，关闭 blocked GDN，各跑五轮。每 4 层同步的四轮热请求聚合为 **730.6 token/s**；每 48 层为 **690.3 token/s**，没有观察到收益。两组全部 64-token 输出逐项一致。MLX 峰值从 79.506 GB 到 79.550 GB；后者并未显著增加内存，但仍不足以改善性能。

两组 SSD 剩余等待都有明显波动，因此不把这次约 5.5% 的吞吐下降当作稳定退化比例。默认同步间隔继续保留 4。该阶段提出的后续方向包括已知 prompt 的跨块 SSD 行预取，以及 GDN 投影和 MoE 的矩阵批处理；递推局部的加速不能替代这些整链路实验。

### 跨块 SSD 预取及专家矩阵分块实验

`--ssd-prefetch nextChunk` 已接入完整推理。**用户随后确认，下列旧批次测试期间本机还有训练任务运行。旧性能表、等待时间及约 +5%～6% 的表面变化均受该干扰，不能作为预取或分块优化的收益依据。**

旧批次使用同一份 1,217-token 文档、64-token 生成、融合 attention 开启、scalar GDN、每 4 层同步及 1 个 SSD worker；每个分块配置在同一进程按 `off,nextChunk,nextChunk,off,off,nextChunk,nextChunk,off,off,nextChunk` 跑十轮，排除最初两轮，各保留四次对照。以下仅保留[受干扰的历史读数](results/prefill-lookahead-quick/summary.json)：

| 分块 | 关闭预取：平均首字 / 聚合 prefill | 开启预取：平均首字 / 聚合 prefill | 历史读数变化（非收益证据） |
|---|---:|---:|---:|
| 512 | 2.095 s / 580.9 token/s | 1.971 s / 617.4 token/s | +6.3% |
| 416 | 2.443 s / 498.2 token/s | 2.317 s / 525.1 token/s | +5.4% |

旧 512 分块记录的平均 prefill 剩余读取/解码等待为关闭时 0.583 s、开启时 0.181 s，预取模式后续块的等待接近零；这些数值同样受训练任务干扰。预取允许等待与部分 GPU 工作重叠，等待差值本身也不能直接等同于 TTFT 收益。

416 分块将前缀从 `512+512+192` 改为 `416+416+384`，最后一个提示 token 均独立处理。固定 MLX 的本模型 sorted MoE 矩阵路径门槛为至少 205 token，因此三个大块都满足条件；旧批次不能判断这种分块的性能收益。功能与数值验证仍保留：两种分块的二十轮及随后 [4-worker 补测](results/prefill-lookahead-workers4/summary.json)的十轮，共三十轮，全部 64-token 输出均通过严格逐项对照，与此前 chunk512 参考一致；逻辑 SSD 请求量也保持一致。

旧批次绝对性能及解码速度有明显随时间漂移，不能把不同进程或前后批次直接当作严格配对，也不再据其宣称小幅收益。当时的默认仍为 chunk64、关闭预取。预取的四项 CPU 小表测试覆盖分块/EOS 历史一致性、消费校验及读失败排空，全部通过，见 [测试日志](results/prefill-lookahead-quick/ple-tests.log)。

已按用户要求原样重测，保留相同二进制、输入、量化与交错顺序。开始前及运行间隙的进程检查未见外部 Python 训练进程；这不等于证明系统完全无后台负载。[新结果及进程快照](results/prefill-lookahead-retest/summary.json)：

| 分块 / SSD workers | 关闭预取：平均首字 / 聚合 prefill | 开启预取：平均首字 / 聚合 prefill | 本批次吞吐变化 |
|---|---:|---:|---:|
| 512 / 1 | 1.679 s / 724.7 token/s | 1.527 s / 796.9 token/s | +10.0% |
| 416 / 1 | 1.545 s / 787.6 token/s | 1.407 s / 864.8 token/s | +9.8% |
| 512 / 4 | 2.130 s / 571.2 token/s | 1.966 s / 619.0 token/s | +8.4% |

每个单元格来自四轮热请求。512/1 的剩余 prefill 读取等待由平均 0.365 s 降至 0.068 s；416/1 由 0.317 s 降至 0.129 s。重测三十轮全部 64-token 输出均匹配旧 chunk512 参考，逻辑 SSD 请求量相同。解码各组热均值约 35.3–36.0 token/s，预取未改变 decode 算法。

本轮均值支持采用预取，但尚不是稳定的普遍 10% 提升：512/1 的首字中位数实际由 1.421 s 到 1.499 s，均值改善受到慢请求尾部影响；416/1 中位数由 1.516 s 到 1.449 s。416/1 加预取是本批次最快配置，4-worker 未显示额外优势；块大小和 worker 数之间是不同进程对照，不能据此确定所有提示长度的最优配置。用户确认后，`generate-gpu` 已将 `--prefill-chunk 416 --ssd-prefetch nextChunk --ssd-workers 1` 纳入默认，并默认开启符合条件的融合 causal attention；无需额外设置 `ANERUNNER_FUSED_PREFILL=1`。

## 当前验证状态：源权重与 reference BF16 累加保留

默认 `--prefill-accumulation reference` 保留作者 prefill 的 BF16 乘积与归约边界，源 Q4/BF16/FP8 权重格式不变，最后一个 prompt token 仍独立处理。当前生成默认 chunk416 改变了 GDN 的跨块状态舍入位置，因此不再笼统宣称默认保持作者全部数值语义。新配置已验证上述单一文档提示的 64-token 输出；尚不能宣称任意任务或长上下文等价。

2026-09-05 使用旧配置保存的短中文结果如下，作为历史参考；这些计时与张量对照不是本轮新默认的重测结果。

| 已保存证据 | 结果 | 范围与限制 |
|---|---|---|
| [最终 Release 整模型结果](results/gpu-full-model-release.json) 与[作者对照](results/gpu-full-model-author-comparison.json) | 三轮均生成相同的 27 个 token，包含 EOS；作者返回的全部 26 个正文 token ID 和完整文本逐项一致 | 作者 API 没有暴露末尾 EOS，不能补猜它的 ID；该可执行文件 SHA 与下述长文档运行相同 |
| 同一份 Release 记录 | 短请求暖 decode 为 **38.515 / 38.301 token/s**，暖首 token 为 **124.2 / 123.3 ms** | 每轮只有 26 个后续 decode 步；首次首 token 为 6.262 秒，另有 9.555 秒加载；不是长任务稳定吞吐或已超过作者引擎的结论 |
| [最终四层捕获](results/gpu-model-4layers-final.safetensors) 与 [比较结果](results/gpu-model-4layers-final-validation.json) | 默认 reference 模式实际执行四层；layer 0 MoE / HC 的五个 prefill 与五个 decode 边界全部逐位一致，L2 为 0 | 已验证完整 token 历史；只比较原参考实际拥有的十个边界，不能扩展为四层所有内部张量一致 |
| [分词验证](results/gpu-tokenizer-validation.json)、[权重加载验证](results/gpu-weight-loader-validation.json)、[最终测试日志](results/gpu-full-build-tests-final.log) | 原生分词、聊天模板和原始 BF16/U32 读取有独立证据；最终构建记录 54 项 XCTest、0 失败 | 单测通过不代表下述所有数值门槛都通过 |

保留作者语义与接近独立 FP32 参考是两项不同检查。[原始 prefill 局部检查](results/gpu-moe-prefill-validation.json)对 FP32 的相对 L2 为 **0.5014146%**，略高于原定 **0.5%** 门槛，报告仍为失败；同一输出对作者 BF16 capture 为 0.0011585%。不能将源实现的一致性改写成 FP32 门槛通过，也不调整旧报告掩盖这项差异。

可选 `--prefill-accumulation float32` 保留每个专家的 BF16 加权乘积，改为 FP32 累加后舍入回 BF16。[局部结果](results/gpu-moe-prefill-fused-validation.json)对 FP32 降至 0.459661%，但它改变了作者的归约语义；[早期整模型实验](results/gpu-full-model-v2.json)中生成的 token 路径也随之改变。这个选项用于独立实验，默认仍为 `reference`。

其他验证的范围与待补结果如下：

- **作者对照已完成。** [原始采集](fixtures/gpu-full-model-reference-64/reference.json)与[配对报告](results/gpu-full-model-author-comparison.json)验证相同 prompt、贪心采样、关闭投机和模型元数据。Swift 两次暖 decode 中位数 **38.408 token/s**，作者三次为 **38.700 token/s**；当前没有速度优势。两者计时边界及 EOS 计数方式不同，约 0.75% 的原生报告速率差不能解释为纯 GPU 内核差异；单个短请求也不能代表广泛质量或持续吞吐。
- **真实长上下文已完成一次执行检查，作者数值对照待补。** [项目文档 fixture](fixtures/gpu-long-context/source-provenance.json)包含 2185 个 no-thinking 输入 token；[整模型记录](results/gpu-long-context-final.json)已生成 8 个 token，12 个 QSA 层进入稀疏选择路径，最终状态 offset 为 2192。该次进程首次请求加载 9.731 秒、首 token 15.692 秒（不含加载），7 个后续 decode 步为 35.161 token/s，SSD 剩余等待 2.389 秒，MLX allocator 峰值约 78.776 GB。它包含首次编译和读取开销，输出被 8-token 预算截断；不是长任务稳态吞吐、完整摘要质量或作者等价性验证。当前短请求报告的 `qsa_active_layers` 为 0。

## 环境与构建

当前验证机器为 M5 Max、128 GiB 统一内存、macOS 26.6.2、Swift 6.3.3。Swift package 的最低部署版本为 **macOS 26.2**，配置见 [Package.swift](Package.swift)。构建和 XCTest 使用本机已有完整 Xcode；无需修改系统 `xcode-select` 设置，也无需升级系统。

下文命令均从本目录执行：

```bash
cd /Users/tom/Documents/test/coreai-models/experiments/ane-runner

env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  xcrun swift build -c release

env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  xcrun swift test
```

本机默认选择 Command Line Tools；完整 Xcode 的 XCTest 环境由上述单次命令的 `DEVELOPER_DIR` 提供。测试包含临时小模型与 GPU 数值探针，运行测试时不要同时采集性能数据。

原生库默认来自已构建的 [固定 MLX 目录](../qwen38-ssd/runtime/mlx-serve/lib/mlx/)：

- `include/mlx/c/`：C ABI 头文件。
- `lib/libmlxc.dylib`、`lib/libmlx.dylib`、`lib/libjaccl.dylib`：原生动态库及依赖。
- `lib/mlx.metallib`：配套 Metal 库。

当前源码版本分别为：作者 runtime `7dbcba04c98e4fd3bcc533c63e645547f13cc3b1`、MLX `1f8e74e3f12f31365464a6867c6579f0e9b29d85`、MLX C `56b2d39fc831f2c0eb5bb94d82ef7191f7b31fa6`。这里复用的是该本地固定构建，不会在 Swift 构建时下载另一个 Python MLX wheel。

如已在另一目录准备相同 ABI 的原生构建，可以在**构建时**覆盖路径；该目录应同时含 `include/` 和完整 `lib/`：

```bash
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  ANERUNNER_MLX_ROOT=/absolute/path/to/pinned/mlx \
  xcrun swift build -c release
```

`ANERUNNER_MLX_ROOT` 由 package 读取，用于头文件、链接与运行库搜索路径。更换路径后需要重新构建；仅在已生成的可执行文件旁设置这个变量，不会改变它的链接配置。保留配套动态库和 Metal 库，避免混用版本；当前验证不要求修改系统库或全局动态库路径。

## 内存与模型资源

直接使用已有的 [模型目录](../qwen38-ssd/models/Qwen3.8-Flash-Next-MLX-SSD-Stream/)。源 affine Q4 专家权重、BF16 权重和 FP8 n-gram 表不转换、不重写，也不再复制完整模型目录。

已记录的整模型加载包含 **77,843,121,920 字节，约77.84 GB** 源权重；MLX active memory 约 **78.5 GB**，均为十进制 GB。active memory 不包括所有系统占用和文件缓存。51.2 GB 的 n-gram 表通过 `pread` 按请求行读取，经过 FP8 解码、scale 和 BF16 舍入，不整体展开成 Float32 数组。

**完整 Swift 模型与原作者完整模型服务必须错开运行。** 两者同时常驻会让这台 128 GiB 机器承受超过可用空间的权重压力，并使性能对照失真。[已有资源交接记录](results/gpu-full-model-resource-handoff.json) 保存了当时的参考服务命令；不要依据旧 PID 操作进程，应在运行前确认当前占用并按现有服务管理方式完成交接。本工程的命令不会自动停止、启动或恢复原服务。

`tokenize` 只加载分词文件；`probe-gpu-model` 加载 embedding 和指定的少量层；`generate-gpu`、阶段交接与本地调度器实模探针均加载完整文本模型。当前没有独立 Swift HTTP 服务入口。各 CLI 命令不自行切换作者服务；已有[实验计划控制器](scripts/run_specialization_experiment.py)需显式运行，并使用最新恢复 ledger 核对当前 PID / 命令后管理资源交接。

此前阶段交接验收结束后已恢复作者服务 `http://127.0.0.1:11235`，并检查 MTP / drafter 关闭，见[该轮恢复记录](results/prefill-decode-handoff-v1/run-ledger.json)。当前 PID 与状态以[实时状态文件](../qwen38-ssd/results/experiment-status.json)及其对应最新恢复 ledger 为准，不能复用历史 PID。这是作者服务，独立 Swift runner 的调度器仍是本地库接口。

## 分词与文本生成

先检查原生分词和 no-thinking 聊天模板，不加载整模型：

```bash
.build/release/ane-runner tokenize \
  --model-dir ../qwen38-ssd/models/Qwen3.8-Flash-Next-MLX-SSD-Stream \
  --prompt '请用一句话解释太阳为什么发光。' \
  --chat true \
  --output results/manual-tokenize.json
```

`--chat true` 使用文本聊天模板；省略或设为 `false` 则直接对原文分词。输出含渲染后的文本、token IDs 和解码结果。默认匹配固定作者实现的 Unicode 处理方式；源码另有显式 NFC 模式。输出屏蔽 token 从模板及分词器的 special 标记派生，EOS 保留，不会用屏蔽来替代数值验证。

确认完整模型资源已交接后运行：

```bash
.build/release/ane-runner generate-gpu \
  --model-dir ../qwen38-ssd/models/Qwen3.8-Flash-Next-MLX-SSD-Stream \
  --prompt '请用一句话解释太阳为什么发光。' \
  --max-tokens 32 --context 4096 \
  --output results/manual-gpu-generation.json
```

此命令使用当前生成默认：chunk416、`nextChunk` 预取、1 个 SSD worker 和 `reference` prefill attention（包含符合条件的融合 causal attention）。可显式添加 `--prefill-attention fusedQSA` 选择 QSA prefill 融合。加载和生成进度写入标准错误；最终文本、token IDs、计时与内存写入 JSON。省略 `--output` 时，JSON 写到标准输出。首次运行会包含模型加载与 Metal 编译开销；用 Ctrl+C 结束当前前台进程不会自动恢复原作者服务，且可能尚未产生最终 JSON。

要对固定输入重跑，使用已有的 [19 个参考 prompt token](fixtures/gpu-full-model-reference/prompt-token-ids.json)，避免把聊天模板差异带入数值比较：

```bash
.build/release/ane-runner generate-gpu \
  --model-dir ../qwen38-ssd/models/Qwen3.8-Flash-Next-MLX-SSD-Stream \
  --tokens-file fixtures/gpu-full-model-reference/prompt-token-ids.json \
  --max-tokens 32 --context 4096 --repeat 2 \
  --output results/manual-gpu-repeat.json
```

`--repeat 2` 在同一进程保留权重，每轮重新创建会话状态；它不是前缀缓存命中。`--tokens-file` 必须是扁平 JSON 整数数组，不能直接传入 `tokenize` 返回的整个 JSON 对象，也不能同时传 `--prompt`。

| 参数 | 默认值 | 当前约束 |
|---|---:|---|
| `--max-tokens` | 32 | 1–4096；提前遇到 EOS 即停止 |
| `--prefill-chunk` | 416 | 1–512；跨块状态舍入边界随 chunk 改变，不保证不同 chunk 的完整数值等价 |
| `--prefill-accumulation` | `reference` | `reference` 保留作者 BF16 归约；`float32` 改变 prefill 累加精度，可能改变生成 token |
| `--prefill-attention` | `reference` | `reference` 保留现有路径；`fusedQSA` 仅对已有 QSA mask、长度大于 8 的 prefill 块强制融合，不影响 decode / MTP verification |
| `--context` | 4096 | 1–262144，输入与输出预算之和不可超过它；上限来自配置，不代表256K已经实测通过 |
| `--repeat` | 1 | 1–10；每轮独立状态，权重不重复加载 |
| `--decode-mode` | `reference` | `scalar` 优化取回 token；`elementwise` / `projections` 分别加共享专家融合或投影合并；`all` 同时启用。所有非 reference 模式均包含 scalar |
| `--decode-order` | 不启用 | 逗号分隔逐轮模式，与 `--decode-mode` 互斥；省略 `--repeat` 时按列表长度运行，最多 10 轮 |
| `--wired-policy` | `disabled` | `fit` 按已加载 MLX 活跃内存加 256 MiB 请求进程内驻留额度；不是物理驻留字节测量 |
| `--wired-order` | 不启用 | 逗号分隔逐轮驻留策略，与 `--wired-policy` 互斥；逐轮设置耗时独立报告，不进入首 token 或 decode 计时 |
| `--ssd-workers` | 1 | 1–4；只控制 CPU 行读取并发，本轮 4-worker 补测未显示额外优势 |
| `--ssd-prefetch` | `nextChunk` | 提前读取下一块已知 prompt 的嵌入；`off` 关闭，保留末尾单 token，decode 仍按需读取 |
| `--ssd-prefetch-order` | 不启用 | `off,nextChunk,...` 逐轮交错对照；与 `--ssd-prefetch` 互斥，最多 10 轮 |
| `--raw-prompt` | `false` | 仅配合 `--prompt`；`true` 跳过聊天模板 |
| `--profile-stages` | `disabled` | `hostBodyOnly` 或 `synchronizedStages` 用于诊断 |
| `--telemetry-dir` | 不启用 | 新目录；启动当前 runner 拥有的可选硬件采样器，另存绝对 phase 时间 |
| `--telemetry-interval-ms` | 200 | 50–10000；必须与 `--telemetry-dir` 一起使用，具体口径见 [TELEMETRY.md](TELEMETRY.md) |
| `--gpu-command-timing-output` | 不启用 | 仅配合独立诊断 MLX 库；保存真实 command-buffer 起止时间，输出必须为新路径。见 [GPU_BOTTLENECK.md](GPU_BOTTLENECK.md) |

| 环境变量 | 默认值 | 当前约束 |
|---|---:|---|
| `ANERUNNER_FUSED_PREFILL` | `1` | `0` 关闭融合 causal attention；仅无 QSA、序列长度大于 8、head dimension 256 时生效，不控制请求级 `fusedQSA` 选择 |
| `ANERUNNER_BLOCKED_GDN` | 关闭 | `1` 开启实验性 blocked 递推；已观察到生成 token 变化，未纳入默认 |
| `ANERUNNER_PREFILL_EVAL_LAYERS` | `4` | 1–48；控制多 token prefill 的层间同步间隔 |

当前固定贪心采样、文本/no-thinking；单个 GPU 活动阶段保持串行，本地调度器可排队多个请求。MTP 可显式选择，尚无工具调用协议、视觉/音频输入、流式 HTTP 或跨进程会话持久化。扩大上下文前应先通过分块连续性、QSA 阈值和内存检查；不要把 CLI 可接受的最大数值当作已通过的容量结论。

## 局部 GPU 探针

只测一层完整 MoE，使用已保存的真实输入：

```bash
.build/release/ane-runner probe-gpu-moe \
  --model-dir ../qwen38-ssd/models/Qwen3.8-Flash-Next-MLX-SSD-Stream \
  --fixture fixtures/moe-real/converted/decode.json \
  --layer 0 --warmups 3 --runs 10 \
  --output results/manual-gpu-moe.json
```

该命令加载整层512个专家的原 Q4 bank，动态选择全部top-10；它不是只装载参考文件中命中的专家。结果包含路由、分支输出、误差和计时，不能直接换算为48层生成速度。

捕获从原始 token 开始的前四层边界：

```bash
.build/release/ane-runner probe-gpu-model \
  --model-dir ../qwen38-ssd/models/Qwen3.8-Flash-Next-MLX-SSD-Stream \
  --tokens-file fixtures/moe-real/prefill-token-ids.json \
  --decode-tokens-file fixtures/moe-real/decode-token-ids.json \
  --layers 4 --prefill-chunk 64 \
  --capture-output results/manual-gpu-boundaries.safetensors \
  --output results/manual-gpu-boundaries.json
```

`--layers` 默认1、最多4；捕获同时受4096个总输入 token 和512 MiB逻辑张量预算限制，实际可能先触发后者。为避免覆盖证据，`--capture-output` 必须是不存在的新文件。

参考 prefill 是26个token；所谓 decode fixture 是 offset 26 的 `[271]`，即作者延后处理的最后一个 prompt 换行，**不是第一个生成内容 token**。捕获键为 `prefill.N.*` / `decode.N.*`，每步都保存 token IDs 和处理前后的位置。随后可从现有环境执行纯 CPU 文件比较：

```bash
../../.venv/bin/python scripts/compare_gpu_model_capture.py \
  --capture results/manual-gpu-boundaries.safetensors \
  --reference-dir fixtures/moe-real \
  --report results/manual-gpu-boundaries-validation.json
```

这里的 Python 只读取两份已保存的张量，不参与推理。比较器先验证完整 token 历史，再比较原始参考实际拥有的五个 layer 0 MoE / HC 边界；其他层和内部张量明确列为未比较。

GDN / QSA 模块还可复用已有 [序列参考](fixtures/gpu-sequence-reference/manifest.json)：

```bash
.build/release/ane-runner probe-gpu-sequence \
  --model-dir ../qwen38-ssd/models/Qwen3.8-Flash-Next-MLX-SSD-Stream \
  --fixture fixtures/gpu-sequence-reference/manifest.json \
  --max-relative-l2 0.005 --fused true \
  --output results/manual-gpu-sequence.json
```

这一序列参考将真实 MoE 激活用于另一个模块边界，并包含为跨越QSA阈值而重复的输入；它属于模块数值测试，不是真实长上下文对话证据。不要将离线参考导出与原生 runner 推理混为一谈。

## Prefill 热点探针

`probe-gpu-hotspots` 在一份已加载模型上按 `baseline,profiled,profiled,baseline` 执行四轮。每轮都实际通过两个 generator 完成 `producer.prefill(request)` → `consumer.decode(ready)`；固定 chunk416、MTP 关闭、reference attention / decode。确认完整模型资源已交接后运行：

```bash
.build/release/ane-runner probe-gpu-hotspots \
  --model-dir ../qwen38-ssd/models/Qwen3.8-Flash-Next-MLX-SSD-Stream \
  --tokens-file fixtures/gpu-agent-11k/prompt-token-ids.json \
  --golden-report results/prefill-agent-11k-default-check/default.json \
  --max-tokens 128 \
  --output results/manual-prefill-hotspots.json
```

输入须为真实 10k…12k token 提示；`--max-tokens` 默认 128，允许 1…256，golden 必须具有相同完整输入、预算及明确的 AR 配置。`--output` 必须是新文件。每轮完整 IDs、结束原因、AR 最终状态位置、callback 与 handoff 都要通过检查；错误会保存已有数据并以非零状态结束。

仅两轮 `profiled` 的 prefill 启用同步阶段记录；`--detail attention` 为默认 attention 细分，`--detail moe` 将多 token MoE 拆为路由、排序搬运、gate/up/激活、down、还原归约及共享专家合并六段。细分阶段替代对应父阶段，避免重复计算占比。baseline prefill 和全部 decode 均关闭记录。每轮重置记录，原始结果保存在 `trials[].prefill_profile`，包含 layer、tokenCount、`phase` 与块起始 `position`，可区分前、中、后段。prefill 只为最后保留的 prompt token 求值输出 head，不为诊断额外计算前面各块的 logits。

阶段时钟用于定位开销，包含主机建图、求值与同步等待；逐阶段同步会改变正常重叠，不能称为 GPU-only 时间或物理带宽测量。首轮编译与 SSD 缓存也可能影响对照。本轮 11k 诊断中 MoE 为最大模块，其内部 gate/up/激活与 down 合计约 73.6%；八轮完整输出及状态对照通过，默认计算路径不变。各自分母、原始数据与边界见 [Prefill 热点分析](docs/PREFILL_HOTSPOTS.md)。

随后使用真实 MoE 输入测试了 M5 NAX 的 BM16 路由矩阵分块：九组真实输入与一组尾部输入的局部张量全部逐位一致，但九组完整 MoE 微测聚合耗时增加 12.32%，因此不纳入默认，也未继续完整模型候选 A/B。共享专家保持常驻 BF16、每 token 执行且独立门控，不参与路由 top10。具体分布、静态 Metal 实例及测试边界见 [路由 MoE prefill 分块实验](docs/MOE_PREFILL_TILING.md)。

后续新增 prefill 逆排序/路由加权/归约融合，并对5种原生矩阵配置×4种归约选择做 [MoE 自动调参](docs/MOE_PREFILL_AUTOTUNE.md)。19个候选在11份输入上全部逐位一致，自动保存的 native4/归约512候选微测吞吐+2.59%。六轮完整11k/128 PD输出与状态检查通过；warm ABBA prefill均值观察到+2.61%，但基准自身漂移27.51%，因此保持实验状态、默认不变。归约表是请求局部策略，原生矩阵配置当前仍限于串行实验，不能用于交错请求。

进一步实现独立 NAX [prefill gate/up/SwiGLU 融合](docs/MOE_PREFILL_GATEUP.md)，使用原始Q4权重和BF16舍入，不叠加上一轮调参库。两个候选共220项真实输入比较全部逐位一致；BN32版本九组完整MoE微测吞吐+6.31%。两轮相反顺序的完整11k/128 PD对照共1536个输出 token及状态/计数检查通过，每个候选prefill实际调用1296次，decode为0。两窗口prefill均值分别观察到+3.81%/+1.34%，但运行随时间漂移，相邻对照方向相反，因此保留请求局部可选实验、默认不推广。32项CPU检查与release编译通过。

随后实现 [按专家边界分块的 MoE](docs/MOE_PREFILL_EXPERT.md)：GPU生成计划，gate/up与可选down复用，每块只处理一个专家。550项真实输入比较全都逐位一致；BM32同时用于gate/up和down的完整MoE微测吞吐+37.67%。两组完整11k/128 PD对照共1536个token及状态/计数检查通过，prefill热均值分别观察到+22.62%/+3.10%，仍有较大运行漂移。33项CPU检查通过；保持可选，默认不变。普通`generate-gpu --prefill-moe-config PATH`已接入保存配置及插件/基础库身份检查，只在prefill选择新路径，短于205token的块回退参考。

普通生成入口另经11k/128-token golden验证，prefill gate/up、plan、down各1296次且decode为0；错误库哈希/旧全局调参会在权重加载前被拒绝。配置与使用示例见上文链接，默认不传参数时保持原实现。

## 如何解读速度和带宽

性能对照使用 `--profile-stages disabled`。`hostBodyOnly` 主要记录主机提交开销；`synchronizedStages` 在每个阶段前后同步，会改变调度、融合和SSD计算重叠，仅用于找等待位置。[现有阶段诊断](results/gpu-full-model-v2-stage-profile.json) 与正常吞吐记录应分开解读。

- `load_seconds` 是模型加载阶段；`time_to_first_token_seconds_excluding_load` 不含它。
- `prefill_tokens_per_second` 包含最后一个保留的 prompt token 的 decode 形状前向与首次采样，不能直接比较作者的纯 prefill 吞吐。
- `decode_tokens_per_second` 使用后续decode步数除以decode时间，不含首token、prefill和加载。生成 ID 包含 EOS，当前短请求的26个decode步不能代表长任务的稳定吞吐。
- `ssd_wait_seconds` 是整次请求在收取SSD任务结果时尚未被计算掩盖的等待，包含prefill与decode；它不是整个文件读取耗时，不能再与计算时间相加。
- `ssd_requested_row_bytes` 及阶段 SSD 字节字段记录实际消费的逻辑行字节；重复行仍计入，OS 文件缓存可能命中。它不等于物理磁盘流量，也不包含取消时已发起但尚未消费的下一块预取。
- `logical_decode_weight_footprint_rate_gbps` 是逻辑权重规模除以耗时，**不是实测DRAM带宽**。报告中物理DRAM字节/带宽仍为不可用。
- `memory.peak_bytes` 是进程累计 MLX allocator 峰值，各轮之间没有重置；它不是单轮独立峰值或全系统物理内存。
- `qsa_active_layers` 记录最终状态已有 pooled indexer 缓存。传给 SDPA 的 KV 张量保持完整逻辑形状，不等于每次读取全部 K/V：固定版本的单 token decode [`sdpa_vector_2pass`](../qwen38-ssd/runtime/mlx-serve/lib/mlx-src/mlx/backend/metal/kernels/sdpa_vector.h) 在 `if (use_key)` 内才加载该可见 key 的 K/V。多 token prefill 的路径需另按请求策略与形状判断；不能由可见比例推算相同比例的物理 DRAM 流量下降。

当前记录没有可归属的物理 DRAM 流量计数，也没有设备计算单元利用率证据；不能从 token/s、权重大小、MLX active memory 或阶段等待推导“带宽已打满”或 GPU/NAX 利用率。

新增 `--telemetry-dir` 可采集目标进程 CPU／磁盘计数与系统 IOReport 分档；`scripts/analyze_gpu_telemetry.py` 将其按 load、prefill、decode 对齐并报告完整／部分覆盖。[采样文档](TELEMETRY.md)列出实际开销配对、诊断命令、GPU trace 待验证项和所有 `null` 字段的含义。采样器不会把这些系统信号改写成物理 DRAM 流量。

[有界并发SSD模块](Sources/ANERunnerGPU/GPUSSDReader.swift) 已接入 `generate-gpu --ssd-workers 1...4`，保留顺序、重复行及原始错误。默认仍使用原单worker路径，并发度应在相同输入、相同输出和独占资源条件下测量后选择。

`--ssd-prefetch nextChunk` 使用每请求独立的串行队列，按输入顺序准备当前及下一块。队列内部仍遵守 `--ssd-workers`，不并发启动多个 chunk 的读操作。预取只复制 token/hash 历史并读取 CPU 数据；真正消费时校验 token、历史及表实例，然后推进 PLE 历史，模型卷积和递推状态照常在 GPU 前向中更新。成功或失败结束时等待已提交读取完成。增量 session 还在每个 yield 前排空读取，保留同一预取对象和已完成数据供恢复使用。

预取在请求和首块计时开始后才创建，因此准备及未被覆盖的读取时间仍算入 TTFT。新增 `prefill_ssd_wait_seconds` 与 `prefill_chunk_ssd_wait_seconds` 将 prefill 剩余等待单独列出；这些等待包含 CPU 读取与解码，不代表物理 SSD 流量。无跨请求缓存，不加载整张 n-gram 表，也不预测未知的生成 token。
