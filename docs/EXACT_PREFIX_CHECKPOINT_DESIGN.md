# 精确 system-prefix checkpoint：原始设计与实现入口

2026-09-08 更新：AR 完整状态缓存、请求私有恢复、radix/LRU 及容量限制已实现，HTTP 默认512 MiB/8条；[实际接口和实模验收](AR_PREFIX_CACHE.md)是当前状态依据。MTP 请求仍冷 prefill，SSD 状态缓存尚未实现，MTP 性能继续排在计划后期。

**以下保留实现前的设计推导。** “单条首版”“建议”“待验收”描述当时的拆分计划；实际版本已在同一轮完成多条前缀索引，不能再据下文认定没有缓存。MTP 补充设计仍未落地。

首版建议只保存一个、同模型实例内的不可变完整 trunk checkpoint，在原有 416-token prefill 边界复用精确系统前缀。命中后为请求恢复私有状态，继续现有 prefill → 单次 handoff → decode。先验收 AR；MTP 仍是显式选项，首版 MTP 请求整体冷 miss，保持原来的完整 prefill 与 MTP decode，不能悄悄降级成 AR，也不能把 AR-only 缓存称为 MTP 缓存支持。后文保留 MTP 的最小补充设计，不作为首版 AR 缓存的实施依赖。

## 当前实施顺序

1. 先推进 AR 的完整混合状态精确前缀 checkpoint 与请求私有恢复，同时补齐对应的取消、失败清理、内存预算和服务调度验证。数值正确、状态隔离与生命周期安全是这一能力自己的验收条件。
2. 单条 checkpoint 通过后，再扩展前缀索引、容量预算与淘汰，随后开发 SSD 状态卸载和恢复。磁盘缓存必须保存完整混合状态，不能只落盘 Attention KV。上述能力都不等待 MTP 加速比。
3. MTP 缓存适配按其功能需求另行验收；原有 MTP 配置继续显式启用，并持续保留已有数值正确性、接受/回滚状态隔离及 AR 回归检查。AR 缓存通过不等于 MTP 缓存通过。
4. 在基础推理、服务与缓存能力稳定后，再集中做 MTP 性能调优及默认启用评估。[MTP 发布条件](MTP_RELEASE_CRITERIA.md) 仍约束 MTP 本身的性能结论和默认发布，不再阻塞前面的 AR 工作。

## 当前落点与合法边界

源码里的状态是 `QwenModel.State`，prefill job 的实际执行游标是 `QwenPrefillSession` / 私有 `QwenPrefillProgress`，并没有独立名为 `QwenPrefillJob` 的类型。scheduler 的 `QwenLocalWork.producer` 持有该游标。[QwenGeneration.swift:282、308、540](../Sources/ANERunnerGPU/QwenGeneration.swift#L282)、[QwenLocalScheduler.swift:95](../Sources/ANERunnerGPU/QwenLocalScheduler.swift#L95)

设完整提示词 token 数为 P，可信系统前缀与完整编码相符的 token 数为 S，选择 `K = 416 × floor(S / 416)`，要求 `0 < K < P`。完整 token IDs 必须从现有完整 chat render 后编码获得；候选系统片段的独立编码只用于确定保守候选，须再与完整编码逐 ID 核对，不能把 UTF-8 字节长度当 token 边界，也不能拼接“旧前缀 tokens + 独立重编码 suffix”绕过核对。当前 `renderChat` 要求有 user，不能直接拿仅 system 的消息调用它生成缓存请求。[QwenTokenizer.swift:195](../Sources/ANERunnerGPU/QwenTokenizer.swift#L195)

冷路径分块是先处理 `[0..<P-1]`，每次最多 416，再单独处理 `[P-1..<P]`。因此只缓存真正完成过的 416 整块边界；系统末尾不足 416 的部分留给本次请求重算。`K=P-1` 且 K 整除 416 时仍合法，恢复后只剩最后一个输入 token；`K=P` 不合法。不能把系统前缀作为一个独立完整请求跑完再保存：它会提前走“尾块 + 最后一个 token”，GDN 的 BF16 回存边界与冷完整请求不同。也不能从更晚的 GDN 状态倒切出 K。[QwenGeneration.swift:553–570](../Sources/ANERunnerGPU/QwenGeneration.swift#L553)

例如稳定系统前缀为 10000 token，K=9984，剩下 16 个系统 token 与新 user 内容照常 prefill。416 同时是 4 的倍数，但这只保证 QSA 分组对齐，不代表能改成任意 4-token 边界。2051/2052 附近的 QSA 状态仍按实际执行保存，不能自行推导成固定存在或不存在。

最小接入位置：

1. `QwenGenerator` 在固定执行器上持有可选的单条 `QwenExactPrefixStore`，默认未启用。可信调用方登记一个系统候选 token 前缀；首版不做最长前缀树、自动学习用户消息或 HTTP 新参数。模型/生成器的原准入检查仍先执行。
2. 冷 `prefillStep` 完成 K 对应整块、所有状态评估成功、PLE lookahead 已 join、取消检查通过后，尝试发布 checkpoint。原请求继续运行；其下一块 `PreparedInput` 留在原游标中，不能进入 cache entry。完整阶段包装器也必须在这个保存点 join，不能依赖只有 cooperative 才执行的现有 join 分支。[QwenGeneration.swift:574–588](../Sources/ANERunnerGPU/QwenGeneration.swift#L574)
3. `makePrefillSession` 在命中时构造新的私有 State 与新的 progress：`state.offset=K`、`p.offset=K`、`processedTokenCount=K`；本请求的计时、实际执行 chunks/bytes、取消对象全新。其余状态机、最终 `QwenPrefillResult` 的单次消费、decode 路径保持原合同。
4. **预取必须只接收 `Array(request.tokens[K...])`，state 已在 K。** `PrefillPrefetch` 的内部 cursor 从 0 起，另存 `initialOffset=state.offset`。若恢复 K 后仍把完整 prompt 传给当前第 551 行，会预取旧前缀而不是后缀，后续 token/hash 检查失败。生成器的 `p.offset` 仍是全 prompt 的绝对游标，不能一起改成 0。[QwenModel.swift:207–234](../Sources/ANERunnerGPU/QwenModel.swift#L207)、[QwenGeneration.swift:550](../Sources/ANERunnerGPU/QwenGeneration.swift#L550)

## checkpoint 完整字段与所有权

entry 只在模型所属 OS 线程创建、恢复和释放，不是 Sendable、Codable 或磁盘格式。模型权重、tokenizer、PLE 表句柄仍由原 owner 持有；entry 不额外拥有网络连接、回调、计时器或 SSD 任务。

| 数据 | 必须保存 / 恢复的内容 | 归属与约束 |
| --- | --- | --- |
| 身份 | 精确 `[Int32]` 前缀、K、payload schema、模型实例 UUID、数值配置 | hash 只加速匹配，最终比较 tokens；模型目录 basename 不是权重身份。首版只同一加载实例，不跨重载、量化、tokenizer/template 改动或不同构建恢复。 |
| 数值配置 | chunk=416、prefill evaluation interval、attention mode、完整 prefill MoE 配置及模型 accumulation、初始化时冻结的 fused-prefill/kernel 策略 | 首版严格同配置命中；请求 context/output 预算仍独立验证，不因跳过前缀减少逻辑上下文。 |
| 主干元数据 | `State.offset=K`、owner、valid、48 层结构 | checkpoint 必须 valid；恢复通过 `makeState` 获得**新的 sessionIdentity**，再填入复制的数据。inactive 层对应分支保持原空状态。 |
| 36 层 GDN | 各层 offset=K；BF16 recurrent `[1,48,128,128]`、convHistory `[1,3,10240]` | 无法从仅 KV 或最终 recurrent 恢复历史边界。`verificationCapture` 必须为 nil，不能把尚未提交的验证输入当成前缀。 |
| 12 层 full attention | offset=K、K/V、rawIndexerKeys、pooledIndexerKeys 的值/shape/dtype 与 nil 状态 | 同时保留 retainedRowCount / retainedPooledBlockCount 的正确语义；紧凑复制后用现有 State initializer 按真实复制范围重建，不能沿用含额外尾部的原 allocation 范围。trunk 的 positionBase 固定为 0。 |
| PLE | layer 1 的 UInt32 n-gram `history` 与 BF16 `convolution` | history 不在 `State.tensors` / `namedTensors` 中，必须显式带上。长度按当前配置 `(pleConvKernel-1) × ngramSize`；verificationCapture=nil。已准备下一块的 historyAfter 属于 lookahead，不能覆盖 K 的 history。 |
| AR hidden / logits / token | 首版不保存 | K<P，尚未到生成首 token 的最终输入边界；后缀会重新产生最终 hidden/logits。不能拿缓存块末尾的预测当成请求首输出。 |
| 请求运行状态 | 不保存 | QwenPrefillResult、可变 decoder、prefetch、UUID/job、取消、stats、deadline、输出缓冲与 handoff 都归新请求。 |
| MTP 补充材料 | 后续可选：带绝对位置与原 chunk 分段的 raw trunk HC stream tail | 是重建材料，不是可直接共享的 head state；首版不提供。详细范围见下一节。 |

字段依据：[QwenModel.State:9–35](../Sources/ANERunnerGPU/QwenModel.swift#L9)、[GDN.State:47](../Sources/ANERunnerGPU/GPUGatedDeltaNet.swift#L47)、[Attention.State:57](../Sources/ANERunnerGPU/GPUAttention.swift#L57)、[PLE.State:86](../Sources/ANERunnerGPU/GPUPLE.swift#L86)。`namedTensors` 是诊断列表，不是完整序列化合同。

建议在 `QwenModel.swift` 内新增仅模块可见的封装 snapshot 与 `freezePrefix` / `forkPrefixState` 方法，利用现有 fileprivate 字段完成校验/复制，不把 State 内部扩大为 public。现有 `checkpoint(state:)` 只先 evaluate 再返回 State 值，`restore` 直接赋值，同时沿用旧 sessionIdentity；它适合同请求 MTP rollback，**不直接作为跨请求 fork**。不改变这两个既有方法的含义。[QwenModel.swift:150–165](../Sources/ANERunnerGPU/QwenModel.swift#L150)

## MLX lazy、alias 与私有恢复

当前 forward 以新 Tensor 结果替换 State 字段；attention 追加使用 concat，GDN fused kernel 分别声明 state_in/state_out，PLE 也生成新卷积状态。这为保留快照提供基础，但 Swift `let snapshot` 或 struct 拷贝不等于 device buffer 私有：Tensor 是持有 MLX C handle 的引用类型。

本机固定 MLX 的 `Copy::eval` 实际执行 `copy_shared_buffer`；contiguous 也不能保证独占或剔除所有 backing storage。当前 GDN 的 3 行 convHistory 可能保留整个 419 行输入 allocation。普通 `MX.copy` 不满足本设计要求。已有 `GPUVerificationCopy.tensor` 通过 batch-axis gather 在 Metal backend 分配 `out.nbytes()` 并逐值复制，可复用于所有 batch=1 的持久 tensor。[GPUAttention.swift:396–408](../Sources/ANERunnerGPU/GPUAttention.swift#L396)

本次另外只读核对了本机固定源码 `../qwen38-ssd/runtime/mlx-serve/lib/mlx-src/mlx/backend/common/common.cpp:56`、`backend/metal/indexing.cpp:57–68`、`array.h:294` 与 `array.cpp:117`：copy 共享 buffer，gather 分配输出，donation 检查 ArrayDesc/data 引用数，detach 清理图依赖。不能只凭引用计数就承诺未来任意自定义 in-place kernel 安全。

首版采用明确的两次所有权转换：发布时把所有持久 tensor 紧凑复制到 entry，**联合 evaluate 全部复制结果**后才原子替换 cache 条目；恢复时再从 entry 复制到新请求的私有 State，并联合 evaluate 后才返回 session。源 entry 在恢复评估完成前保持强引用。这样将缓存寿命与当前图分离，也不依赖后续 kernel 正确实现共享 buffer 的写时复制。紧凑布局也可能影响后续算子选路，仍以完整 logits/输出验证等价性。先测这份复制成本，之后才考虑只读 KV 共享优化。

恢复失败不能发布半份 State。身份/预算不符是正常 miss，走冷路径；设备评估失败须先走现有 recover / model health 合同，不能把失败吞掉后在不明设备状态下重算。请求取消只释放其私有副本；模型 poison、重载或关闭则释放整个 store。所有释放仍在原执行器。

store 首版限一条，建议起始 entry 额度 512 MiB；超出就不发布，不扩容。这是 payload 上限，不是物理峰值或服务总内存上限。scheduler 继续按完整 `prompt.count + maxTokens` 预留 token；缓存常驻字节、恢复临时峰值另行预算。compact payload 的张量 nbytes 可作为逻辑计数，但仍要单独观测 MLX active/peak/cache：旧图、allocator cache、复制中的源/目标以及多个 resident 请求都可能使峰值大于 entry 大小。

## MTP：为什么不能克隆完整 head，怎样最小补齐

**否定直接把已有完整 decoder/head clone 当作跨任意后缀等价缓存。** 源码有两项请求依赖：

- `consumePrompt` 用 trunk `h[p]` 配对已知 `token[p+1]`。在 chunk 结束于 K 且 K<P 时，若该行位于保留历史中，最后一对已经是 `(h[K-1], token[K])`。A/B 即使前 K 个 token 全同，只要第一个后缀 `A[K] != B[K]`，这份 head state 就可能不同；把最后一对省掉后另跑 1 行，又改变了原 head forward 的输入形状与舍入。[QwenMTPDecoder.swift:86–92](../Sources/ANERunnerGPU/QwenMTPDecoder.swift#L86)
- tail H 的起点是 `s(P,H)=4 × floor(max(0,P-1-H)/4)`，head positionBase=`s+1`。即使后缀首 token 相同，P 改变也可能改变从哪里构建历史；较晚构建的 head 不能从另一份更早历史的结果直接裁剪成等价结果。H=1024 是初始历史截断，生成后仍继续增长，不是滑窗。[QwenMTPDecoder.swift:76–84](../Sources/ANERunnerGPU/QwenMTPDecoder.swift#L76)

可行的下一小步是缓存 **trunk 的原始输出 HC stream**，按新请求重建 head，不保存从 A 得到的可变 decoder。固定 H=1024、chunk416、K 整块时，保留最后三个完整 chunk 的 `[1,S,10240]` BF16 stream，及每块的绝对 offset/length，范围 `R=max(0,K-1248)` 到 K。P 至少 K+1，因此最早需要的 pair row 是 `max(0,K-1024)`；三个完整 chunk 能覆盖它并保留原分段。材料最多 1248 行，即 25,559,040 字节（24.375 MiB）逻辑 tensor 数据，不含恢复临时值。H 或 chunk 改变不沿用这个固定容量方案。

重建的私有路径应放在 `QwenMTPDecoder` 内：

1. 已知新请求完整 P 后，建立同一 head owner 下的全新 `headState(positionBase:s+1)`，显式设置 `promptHistoryStart=s`、`promptRows=R`、valid=true，统计计数/时间从零开始、`statistics.historyStartPosition=s+1`。现有公开 `consumePrompt` 要求 offset==promptRows，且仅 promptRows==0 时初始化 s；所以不能从外部直接以 offset=R 调用一个普通新 decoder。这里只需要一个受校验的内部恢复入口，不放宽公开顺序校验。
2. 按保存的**原 chunk**逐块喂入，仍由相同 pairStart/pairEnd 截取本次请求所需行，使用本次完整 prompt 的 token[p+1]。早于 s 的块可以跳过 head 运算，但顺序游标与 previousStream 必须正确。不要把三块拼成一个大 forward，也不要拆开其边界上的最后一行。
3. 缓存材料来自主干 block 后、mixer 前的 `Output.stream`，不是 MTP 的输出 stream。恢复到 K 后继续原后缀 trunk prefill，并照旧 `consumePrompt`。最终在 P 验证 `promptRows=P`、`headState.positionBase + headState.offset=P`；`previousStream` 是真实 `h[P-1]`，与首次待生成 token 配对。[QwenMTP.swift:128–132](../Sources/ANERunnerGPU/QwenMTP.swift#L128)、[QwenMTPDecoder.swift:95–116](../Sources/ANERunnerGPU/QwenMTPDecoder.swift#L95)
4. 保存 tail 时同样紧凑复制并评估；它必须在原 chunk 完成时收集，不能从最后一行或 final State 推回整个 tail。恢复的 head 所有 KV/raw/pooled、offset、positionBase、valid/owner、previousStream、promptRows/promptHistoryStart 和统计都为请求私有。verification、evaluation interval、depth、H 仍用该请求已验证配置；不带入原请求的 rounds/acceptance/timing。

上述可以保留原 head 数值分段，同时跳过前缀 trunk 重算；其等价性与收益仍是**待验证的设计判断**。只有一份 raw tail 也不代表支持 full-history、任意 H、MTP 生成后任意位置分叉，或 head QSA 阈值之外的输出预算。材料不全或配置不支持时继续冷 prefill，不能悄悄把请求从 MTP 改成 AR。

## 分阶段验收，实施时再跑

1. **AR 完整状态与私有性**：冻结一个约 10k 的真实系统前缀，按同一 chunk/profile 跑冷 A、冷 B，再 cache A/B/A；A/B 后缀内容不同。K 处比较全部持久 tensor 位值、PLE UInt32 history、每层 offset/nil 与最终 logits/完整输出 IDs。中间让一个恢复副本推进后取消，确认 entry 内容和另一副本不变；包含一个系统边界跨 416 的变体，并在同组小输入中用 K=1664、P=2051/2053 验证后缀跨 QSA 启用边界。不能只比较生成文本。
2. **后续 MTP 补充材料**：只在开发 MTP 缓存适配时执行，不是 AR 缓存或前缀索引/淘汰/SSD 的前置条件。在 AR 状态验收通过后，H=1024、depth2、已准入的 verification/profile，分别令 A/B 的 token[K] 不同、P 改变到 s 跨 4-token 边界；与各自冷路径对照完整 head state、最终 previousStream、draft/accept/reject/输出 IDs。复用已有会产生接受与拒绝的冻结任务，并做取消后再次命中；证据不能只覆盖高接受的一个任务。没有材料或不支持的 profile 必须观察到冷 miss。

每一阶段在实施时分别报告保存/恢复耗时、命中 token 数、实际执行的后缀 prefill token/秒数、TTFT、decode 有效吞吐、逻辑 payload 与 MLX 内存峰值；MTP 缓存阶段另报 head 重建时间。AR 缓存接入共享生成器或调度器时，仍须复跑受影响的现有 MTP 数值与状态生命周期回归，确认冷 miss 后继续执行所请求的 MTP 模式；这不是要求先取得 MTP 性能提升。`promptTokenCount` / API usage 保留完整 P；prefill compute 吞吐分子只能用实际计算的 P-K，不能把跳过的 token 算进带宽或 kernel 吞吐。本请求 chunkCount 与 SSD 字节从零累计，另记 cached tokens；禁止复制原请求的计时或“省下的时间”塞入本次 compute。

AR 缓存继续沿用研究计划的性能目标：真实 10k 命中 TTFT 至少下降 50%，decode 无可重复超过 3% 的回退；复制或额外内存若抵消收益，就保留冷路径，不因功能正确而默认启用。后续 MTP 缓存适配再单独计入历史重建成本并验收收益；MTP 投机解码的加速比调优仍属计划后期。不为这两阶段另造通用缓存测试框架。

## 上游借鉴的边界

DwarfStar 固定 `9ab705347c1775e7599ede7eb81a6255ec7dccb5`（2026-09-05、MIT）把 graph payload ABI 与文件 envelope 分开，并要求保存的 token 前缀确实对应 live state；其默认对齐明确服务于冷 prefill 的分块一致性。本设计吸收这些检查，但暂不做磁盘文件，也不继承其 byte-prefix 重编码或跨量化复用。[ds4_kvstore.c:25–41](https://github.com/antirez/ds4/blob/9ab705347c1775e7599ede7eb81a6255ec7dccb5/ds4_kvstore.c#L25)、[live 边界检查](https://github.com/antirez/ds4/blob/9ab705347c1775e7599ede7eb81a6255ec7dccb5/ds4_kvstore.c#L950)

SGLang 固定 `2c05ed4e7776c876478f4b2db61acb12b9a27d01`（本次 2026-09-07 核对、Apache-2.0）的 Mamba component 独立验证 checkpoint 是否存在，并为命中请求分配/登记私有恢复槽；Full KV 命中可能长于可复用 Mamba 前沿。借鉴的是完整组件边界与私有恢复，不照搬 CUDA 内存池，也不把 KV 截断等同于 GDN/PLE/MTP 完整恢复。[Mamba validator 与恢复](https://github.com/sgl-project/sglang/blob/2c05ed4e7776c876478f4b2db61acb12b9a27d01/python/sglang/srt/mem_cache/unified_cache/components/mamba_component.py#L131)

本文没有复制上游实现。来源与许可的更完整记录见 [DwarfStar 研究](research/REDIS_AUTHOR_RUNNER.md) 和 [vLLM/SGLang 研究](research/VLLM_SGLANG.md)。
