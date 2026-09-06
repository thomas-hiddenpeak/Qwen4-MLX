# DwarfStar：Redis 作者 runner 的可吸收设计

调研日期：2026-09-07（Asia/Shanghai）。这是方案筛选和源码核对，没有运行上游模型，也没有把上游性能数字当作本项目的收益。

用户提到的项目可确定为 **[antirez/ds4，现名 DwarfStar](https://github.com/antirez/ds4)**。Redis 官方确认 antirez 即 Redis 创建者 Salvatore Sanfilippo；该仓库 README 也以 Salvatore 的身份介绍项目。[作者身份](https://redis.io/press/redis-creator-salvatore-sanfilippo-antirez-joins-redis-labs/)。

固定源码为 [`9ab705347c1775e7599ede7eb81a6255ec7dccb5`](https://github.com/antirez/ds4/commit/9ab705347c1775e7599ede7eb81a6255ec7dccb5)，提交时间 `2026-09-05T20:55:01+02:00`，标题 `Record long GLM quality controls and Spark decode gains`。浅克隆位于被 Git 忽略的 `results/research/upstream/ds4`，只作阅读；未编译、安装依赖或下载模型。

上游根 [LICENSE](https://github.com/antirez/ds4/blob/9ab705347c1775e7599ede7eb81a6255ec7dccb5/LICENSE) 为 MIT，保留 ds4.c authors 与 ggml authors 两项版权声明；仓库也包含独立许可的第三方内容。本文未复制实现。将来若移植任何代码，应在具体文件保留其实际版权、许可和固定来源，不能仅凭根目录 MIT 推断所有附带数据的许可。

## 判断

DwarfStar 值得深入参考，尤其是它面向 Metal、少数模型和真实 agent 使用的组合设计。它已经扩展到 DeepSeek 与 GLM 等模型及 CUDA/ROCm，但与我们的 Swift/MLX runner 一样，明确不追求任意模型兼容。[项目范围](https://github.com/antirez/ds4/blob/9ab705347c1775e7599ede7eb81a6255ec7dccb5/README.md)。

最直接的内核候选是 **在保持归约顺序时增加 BF16 读取并行度**；随后可实验 shared/routed 分支的 Metal 调度重叠。最有长期业务价值的是 **完整状态的前缀复用**。这三项分别减少单 token 等待、利用 GPU 空隙、避免重复 prefill；不能用其中一项的微测代替另外两项的实际效果。

以下 Adopt / Adapt / Defer 为本项目建议，不是已实现能力。小门槛是停止无效实验的筛选标准，不是对收益的承诺。

## 1. Adapt：BF16 GEMV 提前读取，保持原有加法顺序

上游 `metal/glm53_bf16.metal` 的每行计算把 8 组权重和输入 load 提前列出，再按既定次序执行 FMA，最后 `simd_sum`。这增加一个 SIMD 内可同时等待的内存读取，不依赖另一个模型或投机 token。其 QKV 版本只把独立矩阵投影放到 dispatch 的第三个轴。[固定源码，29–112 行](https://github.com/antirez/ds4/blob/9ab705347c1775e7599ede7eb81a6255ec7dccb5/metal/glm53_bf16.metal#L29)。

我们的 [GDN](../../Sources/ANERunnerGPU/GPUGatedDeltaNet.swift) 已有 QKV+Z 和 A+B 的同类拼接实验；[验证投影](../../Sources/ANERunnerGPU/GPUVerificationLinear.swift) 已复刻固定 MLX 的 lane 分配和归约。**不能直接用上游一行一个 SIMD 的 FMA 方式替换我们的 MLX 算术**，否则会重现 MTP 曾经遇到的数值分叉。

建议在当前 MLX 派生 GEMV 里仅改变 load 的排列或展开量，保留每个 accumulator 的加法、dot 分组和 BF16 回存位置。先固定 S1、K=2560/N=10240 与 N=6144、K=6144/N=2560 的真实 GDN 权重；测试 2/4/8 组预载。寄存器增加可能降低占用率，因此应直接测完整投影，不推断更大的展开必然更快。

小门槛：所有输出有限并与既有 S1 逐位一致；微测聚合至少改善 5%，没有主要矩阵超过 3% 的回退，才进入短输入实模；短输入通过后做 11k/128、MTP 关闭的 ABBA 与反序复测，decode 至少改善 3% 且所有输出 IDs 相同才考虑普通候选。若仅改善小矩阵而主导矩阵不变，停止扩大框架。

## 2. Adapt：shared/routed 分支并行调度

上游 `ds4_metal.m` 使用一个 `MTLDispatchTypeConcurrent` encoder，先安排 routed gate/up 与 shared gate/up，再用显式资源 barrier 进入两个 down 分支，最后结束 encoder 才进入依赖最终输出的阶段。中途退出会清除 pending 状态，避免后续普通 dispatch 意外继承 concurrent 状态。[固定源码，9481–10035 行](https://github.com/antirez/ds4/blob/9ab705347c1775e7599ede7eb81a6255ec7dccb5/ds4_metal.m#L9481)。

这正对应我们的常驻 shared expert 和 routed experts：它们都依赖当前层输入，各自投影后才合并。我们的 shared 分支还具有 sigmoid gate，采用 BF16 权重；上游示例是 Q8 shared / IQ2+Q2 routed，量化和公式不通用。[本地合并位置](../../Sources/ANERunnerGPU/GPUMoE.swift)。

建议先做单层原生 Metal 可选探针，不同时改量化或归约。应由同一 MLX primitive 声明完整输入、输出、资源生命期与 stream 依赖；两个独立 Swift 队列不等于正确的 GPU 重叠。两个分支都可能受 DRAM 限制，并行只在当前执行未充分利用资源时有用。

小门槛：真实 top-k 输入下完整 MoE 逐位一致；单层热 ABBA 至少改善 5%；取消和失败后不会残留未提交操作或引用；完整模型 decode 改善至少 3% 才保留。微测平或更慢就保持现有串行图。先于这一实验完成 GDN 主导矩阵的筛选，避免同时改两个瓶颈。

上游另有把 router 与 shared gate/up 放进同一 dispatch、不同 threadgroup 区域的实现，但这主要省 launch，不直接减少权重字节。[dense.metal，930 行起](https://github.com/antirez/ds4/blob/9ab705347c1775e7599ede7eb81a6255ec7dccb5/metal/dense.metal#L930)。只有 timeline 证明这些小 dispatch 留出显著空隙时才提升其优先级。

## 3. Adopt：先保留完整前缀状态，再做磁盘复用

DwarfStar 的 cache payload 保存精确 tokens 与图状态；外层文件版本和内部图状态 ABI 分开。磁盘查找检查模型、上下文与兼容条件，写入先使用临时文件、完成后 rename。源代码使用 `fflush`/`fclose`，不能据此称为断电持久化保证。[header](https://github.com/antirez/ds4/blob/9ab705347c1775e7599ede7eb81a6255ec7dccb5/ds4_kvstore.h#L36)、[ABI 和边界](https://github.com/antirez/ds4/blob/9ab705347c1775e7599ede7eb81a6255ec7dccb5/ds4_kvstore.c#L25)、[写入](https://github.com/antirez/ds4/blob/9ab705347c1775e7599ede7eb81a6255ec7dccb5/ds4_kvstore.c#L1054)。

我们的 [PD 交接文档](../PREFILL_DECODE_SEPARATION.md) 已明确完整状态，但现有 `QwenPrefillResult` 是同实例、单次消费的 live handle，不能直接拿它作为可复用 cache entry。Qwen 的 GDN recurrent/conv、PLE conv/hash 历史、Attention KV/QSA 和位置，以及启用 MTP 时的 head 历史，都必须恢复到同一提交边界。

建议顺序：

1. 内存里只缓存一个真实 10k 系统提示词的完整、不可变 checkpoint；多个后缀分别获得独立状态。先 AR，之后补 MTP 专属状态合同。
2. 以精确 token IDs 加模型权重、tokenizer、dtype、kernel 数值配置、chunk/对齐策略和 payload ABI 为身份。不能只对文本或模型目录名称做 hash。
3. 验证通过后才加 token radix 索引和字节预算；最后实现磁盘 envelope、完整性校验、临时文件发布、损坏回退及有界 restore。不同量化的状态复用在本项目默认拒绝，即使上游允许配置接受。

小门槛：同一 10k 前缀接 3 个不同后缀，与冷 prefill 的 logits/完整输出一致；A/B/A 重用不污染状态；2051/2053 QSA 附近和当前 416-token chunk 边界通过；MTP snapshot 单独验证 accept/reject 后恢复。缓存命中 TTFT 至少下降 50%，decode 无可重复超过 3% 的回退，并记录缓存真实字节、保存/恢复耗时、后缀 prefill 时间。磁盘损坏或身份不符必须回退冷 prefill，不发布半份状态。

上游缓存有两个不宜直接继承的设计：它的自动磁盘 key 基于 rendered byte prefix，并可保留旧 token 序列后仅重编码 suffix；我们的首版应坚持全提示词 token 前缀匹配。它还按长度扫描 entries 查找磁盘前缀，`rax` 主要用于 tool replay 字典；因此不能把该实现称为 RadixAttention。[前缀查找](https://github.com/antirez/ds4/blob/9ab705347c1775e7599ede7eb81a6255ec7dccb5/ds4_kvstore.c#L1190)、[tool 字典](https://github.com/antirez/ds4/blob/9ab705347c1775e7599ede7eb81a6255ec7dccb5/ds4_server.c#L9205)。

## 4. Adapt：按 decode 压力选择 prefill 让出间隔

上游 server 的模型 coordinator 负责图状态；网络线程只提交作业。`server_prefill_quantum_for` 在没有生成任务时选择较大 quantum，有 active decode 时缩小；GLM 5.3 又有模型专属最小值。部分后端仍是逐请求执行 fallback，文档明确不把调度公平性当原生 batch 吞吐。[执行所有权](https://github.com/antirez/ds4/blob/9ab705347c1775e7599ede7eb81a6255ec7dccb5/ds4_server.c#L9)、[quantum](https://github.com/antirez/ds4/blob/9ab705347c1775e7599ede7eb81a6255ec7dccb5/ds4_server.c#L11135)、[能力边界](https://github.com/antirez/ds4/blob/9ab705347c1775e7599ede7eb81a6255ec7dccb5/docs/SERVER.md#multiple-sessions)。

本项目已经有 cooperative scheduler、prefill chunk/AR-MTP round 边界与 decodeBurst，第一步可吸收的是测量两请求的服务质量。**暂不在请求中途改变算术 chunk**：GDN 舍入和 QSA 边界与数值配置绑定。先在请求准入时固定 208/416 两种配置，比较等待时间；改变中的 chunk 只有专门数值回归通过才考虑。

小门槛：一个长 11k prefill 与一个已开始 decode 的请求交错，保存实际 callback 时间，报告 decode 间隔 p50/p95/max、总吞吐和新请求 TTFT；两边输出与各自固定 chunk 的独立运行一致。希望 p95 等待至少下降 20%，聚合有效吞吐回退不超过 5%；否则保持固定 416。decodeBurst 计数是步骤，不把一次 MTP round 当一个输出 token。

## 5. Adapt/Defer：SSD 缓存计量先做，专家 offload 后置

DwarfStar 的 SSD streaming 是 **routed expert 权重缓存**；cache miss 通过显式 `pread` 和有界 worker pool 填充 buffer，并累计 hits/misses、实际读取字节、等待和 buffer reuse。cache 预算之外还预留非路由权重、context、临时图和 prefill 空间。[内存范围](https://github.com/antirez/ds4/blob/9ab705347c1775e7599ede7eb81a6255ec7dccb5/docs/SSD_STREAMING.md)、[Metal 计量](https://github.com/antirez/ds4/blob/9ab705347c1775e7599ede7eb81a6255ec7dccb5/ds4_metal.m#L4374)、[读取和 pool](https://github.com/antirez/ds4/blob/9ab705347c1775e7599ede7eb81a6255ec7dccb5/ds4_metal.m#L13258)。

我们的 [GPUSSDReader](../../Sources/ANERunnerGPU/GPUSSDReader.swift) 读 n-gram 表行，已有最多 4 个 worker；[GPUPLE](../../Sources/ANERunnerGPU/GPUPLE.swift) 的已知 prompt 预取也已存在。应先记录行访问重用率、unique rows、实际请求 pread 字节、读取耗时和消费等待。若重复率可观，再测试固定 64/256 MiB 的 PLE 行缓存。`pread` 返回字节仍可能来自 OS page cache，不能标作硬件 SSD 物理流量，更不能标作 DRAM 带宽占用率。

小门槛：重复行、乱序行和相同 n-gram 的返回结果、错误顺序完全一致，内存严格有界；命中缓存使 11k prefill 的 SSD 等待至少减少 20%，并带来总 prefill 至少 3% 的可重复收益才保留。完全常驻的 routed weights 不为模仿上游而主动 offload；只有需要降低内存/增加驻留会话数量时才开专家缓存容量实验。当前收益目标是 M5 Max 的推理性能，SSD 容量能力单独评估。

## 6. Adopt later：API 与 token 历史保持同一事实来源

上游把 chat 模板、tool-call 输出、精确 token 历史与 live state 一起测试；对已生成的原生 tool block 保存有界 replay 映射，避免客户端再次提交等价但格式不同的 JSON 时丢失前缀。HTTP 支持 SSE，网络线程与推理状态所有权分离。[server 功能](https://github.com/antirez/ds4/blob/9ab705347c1775e7599ede7eb81a6255ec7dccb5/docs/SERVER.md#tool-history-and-debugging)、[测试入口](https://github.com/antirez/ds4/blob/9ab705347c1775e7599ede7eb81a6255ec7dccb5/docs/TESTING.md)。

本项目仍是 Swift 库/CLI，不能把参考 xlm-server 的 HTTP 能力算作自己的。将来做最小服务时，先一种 OpenAI chat 流式接口、严格有界排队、断连取消和 slow-client 背压；推理 executor 继续单一所有权。不要一次纳入上游所有协议、原生 coding agent、视觉和联网功能。

小门槛：现有 tokenizer 的中文、代码、tool 参数 JSON、EOS 边界进行 render → tokens → stream → replay；无重复或丢 token，客户端断连后额度和状态释放；慢客户端不阻塞另一个请求推进。首次 API 上线先保持 AR，之后分别报告 MTP 的接受率、有效输出吞吐和 callback 间隔。

## 暂缓事项与实施顺序

| 决策 | 项目 | 原因 |
| --- | --- | --- |
| 立即筛选 | BF16 load staging | 直接对应当前 GDN decode 主导矩阵，可在原算术内做小实验。 |
| 下一内核候选 | Shared/routed overlap | Apple Metal 贴合度高，但先证明资源存在重叠空间。 |
| 计划内优先能力 | 精确完整状态 checkpoint | 可以直接消除 agent 重复 10k 系统提示词的 prefill。 |
| 小型调度实验 | 固定请求配置的 quantum / decodeBurst | 我们已有本机 PD 分离，不必先造多进程服务。 |
| 条件实验 | PLE 行缓存 | 以真实行重用率和暴露等待为依据；命中率本身不是吞吐。 |
| 后置 | SSD routed expert offload、跨 Mac TP/RDMA、压缩 KV | 当前权重与设备资源目标不同；需要容量或状态精度的独立门槛。 |
| 不吸收为默认 | 上游激进专家再量化、不同 quant 的 cache 互用、近似 speculation | 会改变我们固定 checkpoint / greedy 的对照合同。 |
| 不纳入当前 runner | 内置 coding agent、方向 steering、多个模型/后端 | 分散当前专用模型推理与稳健 API 的任务。 |

每个实验单独保存候选选择、正确性结果和阶段吞吐；正确但更慢的实现保持诊断用途或回退。阶段提交应描述实际完成的能力，不把本计划中的 Adopt 条目写成已支持。
