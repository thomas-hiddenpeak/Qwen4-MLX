# Prefill / Decode 业务分离

2026-09-06。用户明确要求两个业务阶段分别统计、可以独立执行，并根据各自需求调整 kernel。MTP 的主要性能门槛属于 decode；整请求耗时作为业务参考，不用 prefill 占比否定 decode 的独立优化。

**范围已确认：先在本机拆分，后续再考虑独立部署。** 当前阶段采用同进程、共享一份模型权重的 prefill / decode 作业与调度边界；独立进程服务、跨机器传输和状态序列化属于后续部署工作，不作为本阶段或 MTP 上线的前置条件。

`QwenLocalScheduler` 提供有界队列、token 额度和阶段调度。默认 `wholeStages` 保留完整阶段切换；新增显式选择的 `cooperative`，可在 prompt chunk 或 decode round 完成后切换作业。此前 20 项 CPU / 27 项实模检查属于完整阶段模式的历史证据；本轮增量接口已构建并通过 26 项 CPU 测试及 55 项实模检查，实际交错与 callback 延迟见文末。GPU 执行始终串行。

## 阶段与指标

| 阶段 | 主要工作 | 独立指标 |
| --- | --- | --- |
| 模型 / head 初始化 | 加载权重、首次构建 | load / preparation，冷暖分开 |
| Prefill 主干 | 已知 prompt 的批量前向，最终 prompt 行与首 token 选择 | target seconds、prompt token/s、分块配置、SSD 等待与逻辑字节 |
| MTP 提示历史准备 | 对齐真实 hidden 与下一 prompt token，建立草稿头历史 | draft history seconds、实际历史行数；当前在 prefill 内执行 |
| 阶段交接 | ready 结果等待接收、句柄消费；未来可扩展传输 | wait、consume；序列化 / 传输尚未实现 |
| Decode | AR 单步或 MTP draft / verify / commit / history | 有效输出 token/s、平均 TPOT、round 数、接受长度及各阶段时间、SSD 等待 |
| 输出回调 | 交付已提交 token | callback seconds、含回调的 decode service seconds |

`prefill.targetSeconds` 不包含草稿头历史；`prefill.totalSeconds` 包含 producer 的活动操作及历史准备，排除增量 session 的暂停时间，两者不能相加。prefill.targetTokensPerSecond 与 readyTokensPerSecond 分别体现主干计算与完整 producer 的活动吞吐。可选字段 `prefill.suspensionSeconds` 单独记录 prefill 步骤间暂停；旧报告缺字段表示未记录。

首 token 已在 prefill 选择；decode 首先发布它一次。decode 的性能分子为首 token 之后实际输出的 token 数，包含输出 EOS，不能用草稿数或 round 数替代。`decodeSecondsPerToken` 是平均 TPOT；MTP 多 token 一批发布，因此它不是逐回调间隔的 p95。纯生成时间与包含 callback 的 service 时间分别报告。

`timeToFirstTokenSeconds` 包含 session 开始之后的暂停、handoff 等待及到首 token 发布准备的时间，不能当成纯 prefill；它不包含模型加载、首次 head 构建及 session 开始之前的首次排队。`totalSeconds` 同样包含 session 内暂停；`phases.decodeServiceSeconds` 只累加 consumer 的活动步骤，包含 callback，排除暂停。可选字段 `phases.decodeSuspensionSeconds` 单独记录 decode 恢复等待。组合 `generate` 仍持有整个请求的 admission，并保持原整次调用计时口径。

## 当前同进程接口

```swift
let job = QwenGenerationRequest(
    tokens: promptIDs, maxTokens: 128, contextLimit: 16_384,
    prefillChunk: 416, mtpDepth: 2, verification: .batchedScalarLinear,
    draftHistoryTokens: 1024,
    prefillEvaluateEveryLayers: 4, verificationEvaluateEveryLayers: 4)
let ready = try prefillWorker.prefill(job)
// Admission has been released; another job may run on the same executor.
let output = try decodeWorker.decode(ready) { token in /* publish token */ }
```

两个 worker 可以是共享同一 `QwenModel` 的两个 `QwenGenerator`。它们仍必须运行在同一个串行 MLX inference executor；下面的本地调度器负责在这个执行器上选择作业，不启动后台 worker、独立 HTTP 服务或跨进程传输。`generate` 作为便捷接口组合两个阶段，并继续在整个请求期间持有 admission。

业务分离不要求在本机复制两套完整权重。本地调度器复用一个 generator 及其模型，在其上维护 prefill 与 ready decode 两个队列。并行执行或跨进程部署需要另行设计共享资源与传输，不能从两个 API 或两个队列推导为已支持 GPU 并发。

`QwenPrefillResult` 是不透明、不可 Sendable、不可 Codable 的单次消费句柄：

- producer 返回前排空 SSD 预读，求值主干与 MTP 状态，校验 head/trunk 位置一致；首 token 尚未消费到 trunk，也尚未触发回调。
- 私有持有模型强引用、完整 trunk state、请求级 MTP decoder、pending token 和固定请求配置。这里只移动句柄，不复制或重算整套状态。
- consumer 在任何输出前检查模型实例身份。busy、错误模型或 admission 前取消不消费句柄；claim 后无论成功、取消或 callback 抛错，都不能再次消费。
- `discard()` 可释放不使用的 ready 状态；句柄与 discard 均限同一 inference executor。单次消费后清空句柄 payload，避免调用者继续保留整套请求 tensors。
- MTP depth / verification / 初始 history cap 在 producer 之前声明，因为建立历史需要完整 prompt hidden。仅凭最终 KV 与最后 hidden 不能在 decode 临时补建之前全部历史。

## 可恢复的同进程步骤

[QwenGeneration](../Sources/ANERunnerGPU/QwenGeneration.swift) 提供以下增量 API；它们与原 `prefill` / `decode` / `generate` 复用内部计算步骤，原完整调用不会自动开始交错执行。

| API | 返回与步长 |
| --- | --- |
| `beginPrefill(_:cancellation:) throws -> QwenPrefillSession` | 校验并冻结请求/MTP 配置，创建请求级状态；不执行 prompt chunk。 |
| `stepPrefill(_:cancellation:) throws -> QwenPrefillResult?` | 恰好处理一个原 prompt chunk；未完成返回 nil，最后一块返回单次消费 ready 句柄。 |
| `beginDecode(_:cancellation:) throws -> QwenDecodeSession` | 校验同一模型身份后，只 claim 一次 ready 句柄；此时尚未发布首 token。 |
| `stepDecode(_:cancellation:onToken:) throws -> QwenGenerationResult?` | 首次调用只发布首 token；以后每次最多执行一轮完整 AR/MTP。请求结束才返回结果。 |

两个 session 均不具备 Sendable，必须留在同一模型的 inference executor。它们提供 `isFinished`、`isActive`、`discard() throws`，以及 `processedTokenCount` 或 `generatedTokenCount`。外部可在 yield 后主动 discard；活动步骤或 callback 内 discard 会报 busy。错误模型、busy 或步骤准入前取消不会消费 cursor；进入步骤后的取消、模型错误或 callback 异常会废弃整个 cursor，不能从半轮输出重试。

默认 prompt 分块仍为 416，最后一个 prompt token 单独执行；切换作业不会额外切小 chunk 或改变 BF16 状态的分块舍入边界。一次 MTP round 可能提交多个 token，必须按顺序发布本轮全部输出后再 yield，不能将已经推进的 trunk/head 状态与半轮输出分开恢复。EOS 或预算结束后 cursor 不再可用。

每次增量 prefill yield 前排空本请求已安排的 SSD 读取，保留同一个预取对象和已完成的下一块数据，下一步不重建或重复读取。主干状态完成求值，MTP 中间保留的末行可能是已物化 stream 的惰性切片视图；没有额外 GPU/SSD 工作跨越 yield。完整阶段兼容接口在内部循环继续保留原预取重叠，最终交接前统一排空。这里增加的是显式继续执行的边界，不是 Metal kernel 的硬件抢占。

## 本地有界调度器

[QwenLocalScheduler](../Sources/ANERunnerGPU/QwenLocalScheduler.swift) 是由调用者主动推进的同步库对象。`submit` 只做校验、准入与排队，返回 UUID；`runNext()` 按 `executionMode` 推进完整阶段或一个可恢复步骤，也可能只返回一个待领取的终态事件。没有后台执行循环；调用者停止推进时，队列不会自行运行。

```swift
let scheduler = try QwenLocalScheduler(generator: generator, limits: .init(
    maxQueuedPrefills: 8, maxReadyDecodes: 2,
    maxResidentTokens: 32_768, maxConsecutivePrefills: 1,
    executionMode: .cooperative, decodeBurst: 4, maxResidentSequences: 2))
let cancellation = QwenCancellation()
let jobID = try scheduler.submit(job, cancellation: cancellation) { token in
    // Publish a committed token on this same inference executor.
}
while let event = try scheduler.runNext() {
    // Inspect event.kind, event.jobID, event.result and event.timing.
}
let status = scheduler.snapshot()
```

| API | 行为 |
| --- | --- |
| `init(generator:limits:) throws` | 绑定一个 generator / 模型并校验额度，不复制整套权重。 |
| `submit(_:cancellation:onToken:) throws -> UUID` | 校验请求后立即预留 `prompt.count + maxTokens`；队列满报 `queueFull`，额度不足报 `overBudget`。 |
| `runNext() throws -> Event?` | `wholeStages` 推进完整阶段；`cooperative` 推进一个 chunk、首次 token 发布或完整 AR/MTP round；也可返回待领取事件。无剩余工作/事件时返回 nil。失败不自动重试。 |
| `cancel(_:) throws -> Event?` | 在阶段边界删除 queued/ready 作业、释放状态与额度；未知或已结束 UUID 返回 nil。 |
| `discardAll() throws -> [Event]` | 在阶段边界取消全部等待作业并返回终态事件，释放 ready 状态；不会重开因模型不可用而关闭的调度器。 |
| `snapshot() -> Snapshot` | 返回两个队列的数量/UUID、running job/stage、预留 token、`residentSequences`、待领取事件、空闲与准入状态。 |

`Limits` 默认 `executionMode: .wholeStages`，queued prefill 8、ready decode 2、预留 token 32768、连续 prefill 1。示例显式开启 `.cooperative`；其 `decodeBurst` 默认 4、`maxResidentSequences` 默认 2。完整阶段模式的公平性仍发生在阶段结束时，长阶段会使其他作业等待。

合作模式在步骤完成后将未完成作业放回对应 FIFO 队尾；有 ready 作业时，连续 prefill 达上限后进入 decode，最多连续执行 `decodeBurst` 个 decode 步骤，再给符合容量条件的 prefill 机会。这个数是步骤数，包含首次 token 发布，不是输出 token 配额。ready 已满或没有可推进的 prefill 时继续 decode；resident 额度已满时跳过尚未开始的新请求，让已拥有状态的 partial prefill 继续完成。

合作模式的 `maxResidentSequences` 统计已经开始的 partial prefill、ready 与活动/暂停 decode，排除未开始的队列请求；它在首次 prefill 前预留。`maxQueuedPrefills` 统计新的和部分完成的 producer，活动 prefill 也保留一个返回队列的槽，避免 callback 提交新作业后回队超额。`maxReadyDecodes` 包含首次 ready 和暂停后待恢复的 decode。

预留从 **submit 时**开始，覆盖 queued、ready 和 running 的整个生命周期，完成、失败或取消时释放一次。阶段转移既不提前归还，也不再次计费。`maxResidentTokens` 是保守的逻辑额度，**不是物理内存字节硬上限**：GDN 固定状态、MTP 历史、验证临时张量、共享权重与 allocator 缓存仍占内存，队列/ready 上限需要共同控制状态数量。

调度器和 handoff 都不具备 Sendable；所有调度器操作、句柄释放和 GPU 工作须留在同一个 inference executor。跨线程发起取消只使用 `QwenCancellation.cancel()`，由生成过程在合作式检查点观察；它不会打断正在运行的 Metal kernel 或 SSD 读取。Callback 内允许 `submit` / `snapshot`，嵌套 `runNext`、`cancel` 或 `discardAll` 报 busy；活动请求应通过其取消对象结束。Callback 抛出的 busy 也是作业失败，不能误判为“尚未消费，可重试”。

事件的 `kind` 包含 `.prefillProgress`、`.prefillReady`、`.decodeProgress`、`.completed`、`.failed` 和 `.cancelled`，`stage` 为 `.prefill` 或 `.decode`。progress 事件不是终态，部分完成 prefill 的取消仍标为 prefill；可选 `processedPromptTokens` / `generatedTokenCount` 记录当前进度。每个已准入作业应最终收到一次终态；设备恢复失败使模型不可用时，调度器关闭后续准入并为等待作业产生终态事件。此处描述源码行为，真实故障恢复能力仍需独立证据。

队列等待与计算分别保留：`Timing.prefillStageSeconds` / `decodeStageSeconds` 累加活动步骤，不包含恢复排队；前者包含首次 head 构建，后者包含 callback。`initialPrefillQueueWaitSeconds` 和 `prefillResumeWaitSeconds` 是新增可选字段，分别记录首次执行前和分块间的排队；`prefillQueueWaitSeconds` 为两者之和。`readyQueueWaitSeconds` 累加首次 handoff 等待及后续 decode 恢复等待。更细的计算与历史指标从 `event.prefill` / `event.result.phases` 读取。

**从提交开始的 TTFT 以实际首 callback 的单调时钟为准。** 新探针保存 `submission_to_first_callback_seconds` 及原始 callback 时间。`result.timeToFirstTokenSeconds` 已包含 session 内的 prefill/decode 暂停，不能再加聚合的 `prefillQueueWaitSeconds`，否则会重复计入 prefill 恢复等待。初始排队和 head 构建虽分开可查，也不将近似字段组合当成精确的首 callback 时刻。MTP 的平均计算 TPOT 与交错作业后的实际 callback 间隔是不同指标。

SSD 字节字段记录模型实际消费的逻辑行字节，重复行仍计数，不代表物理流量或所有已发起读取；取消时已经预取但尚未消费的下一块不计入这些字段。

## 完整交接状态

| 内容 | 原因 |
| --- | --- |
| Attention KV、QSA raw/pooled keys、有效长度与保留尺寸 | 可见历史与池化位置必须一致 |
| GDN recurrent 与 convolution history | 模型包含递推层，不能只传 KV |
| PLE convolution、n-gram/hash 历史 | 下一 token 的 SSD 行查找依赖连续历史 |
| Trunk offset、pending 首 token、EOS / budget | 避免首 token 重复消费、少算或多算一步 |
| MTP head KV/QSA、绝对位置、previousStream / promptRows | 继续起草必须从真实主干 hidden 与相同位置出发 |
| 权重 / tokenizer / dtype / layout / kernel 数值配置身份 | 将来跨实例或跨进程需校验，不能只按模型名字接收 |

当前句柄由原对象私有持有上述状态，模型实例检查比目录字符串匹配更严格。将来跨进程必须新增可传输的数据布局、版本与身份校验、执行完成边界及所有权协议，不能发送 MLX handles 或用 `@unchecked Sendable` 替代传输。

## Kernel 独立优化

`QwenModel.forward` 新增显式 `QwenExecutionPhase`：prefill、decode、verification。高层调用全部显式路由，最后一个 S1 prompt 仍是 prefill，S2/S3 MTP 验证仍属于 decode 业务。低层旧探针省略 phase 时暂保留旧 shape 推断，不代表业务边界。

当前可分别设置 prefill 与 verification 的 evaluateEveryLayers，默认都为 4；decodeMode 只传给 decode / target verification，不传给 prompt。模型未准备相应投影 / elementwise 实现，或 `batchedScalarLinear` 搭配非 reference decode mode，会在高层请求进入 GPU 前拒绝。

benchmark CLI 同样显式路由并新增 `phase_metrics`、`--prefill-eval-layers`、`--verify-eval-layers`；旧 `prefill_tokens_per_second` 保留其包含 MTP prompt-history 的历史口径，新的 `phase_metrics.prefill_target_tokens_per_second` 才是单独主干指标。旧历史报告不反向改写。

目前改变的是接口、指标与调度策略隔离，默认算术和 kernel 保持不变。后续 prefill 面向大块矩阵 / 吞吐，decode 面向低延迟 / 权重读取复用，verification 面向少量 token 的共享读取及正确状态提交。两阶段可以使用不同实现，但必须通过交接后继续 decode 的数值与输出验证。

评估 decode / MTP 时固定 prefill 配置及输入状态，避免将 prefill kernel 数值改变引起的输出变化误判成 decode 回归。优化 prefill 时另做其输出状态与下游继续生成的验收；不要求两个阶段使用同一种 kernel，也不把算子级逐位一致当作所有 prefill 优化的唯一标准。当前 MTP 的固定 greedy 兼容合同仍在各自固定的 prefill 配置内执行。

## 此前阶段 API 验证

[阶段交接探针](../results/prefill-decode-handoff-v1/handoff.json)在一个完整模型、两个 generator 上完成 17 项检查，全部通过。覆盖拆分前后完整 IDs、ready 期间执行其他请求、busy 与预先取消保留句柄、消费后拒绝复用、首 callback 取消/抛错与恢复、discard、预算 1 零 decode round，以及原生 MTP depth2 的拆分生成。12 项 CPU 契约测试通过，包括阶段路由、独立求值间隔和单次消费释放。

这是同进程、同执行器的交接验证；不包含跨进程迁移、独立队列背压、错误模型实例的实模注入，或设备故障恢复。错误模型在输出前有显式实例检查，当前只是源码审查证据。探针故意插入等待，阶段时间反映不同冷暖状态，不作为性能加速依据。

[11k 回归](../results/prefill-decode-handoff-v1/long.json)使用 AR/MTP/MTP/AR，四轮 128 输出全部匹配旧 golden。独立 decode 分别为 31.26、34.26、34.74、30.17 token/s；后续三轮主干 prefill 为 14.34、15.21、15.57 秒，首次 AR 为 22.96 秒，首次请求单列而不将差异归因于新接口。MTP prompt-history 为 0.228 / 0.056 秒，单独记录。这是新路由与计时的回归及观察，未据一组结果宣布稳定收益或新的 kernel 加速。

原始运行、二进制和服务恢复见 [运行记录](../results/prefill-decode-handoff-v1/run-ledger.json)，本轮验收汇总见 [summary.json](../results/prefill-decode-handoff-v1/summary.json)。

## 历史完整阶段调度器验收（2026-09-06，wholeStages）

新入口为 `probe-gpu-local-scheduler --model-dir PATH --output NEW.json`；输出文件必须不存在。默认 `--suite all --max-tokens 64` 使用两个不同短提示；`--tokens-file` 可指定扁平 token ID JSON 数组，`--suite mixed` 只执行混排/状态/输出检查，适合真实 11k 输入，避免把全部取消分支重复成长请求。`--max-tokens` 范围 3…256，预算 2 不会触发草稿分支。`--golden-report` 可核对既有 `generate-gpu` 报告中第一轮的完整 prompt / 输出 IDs 与输出预算，并保存 golden 文件 SHA256。

探针使用一个模型，检查两个 ready 状态、各阶段 FIFO、AR/MTP 输出与独立参考、额度释放以及阶段计时；`all` 另覆盖队列/额度背压、取消、callback 失败与重入、ready 数量上限、容量复用和 discard。混排测试会显式设连续 prefill 为 2，以建立两个同时保留的 ready 状态，这不是将生产默认 1 改为 2。Release 构建与 20 项 CPU 测试通过，其中 8 项直接测试生产队列引擎的公平性、额度、时钟、取消、回调重入与恢复失败关闭。新增实模检查全部通过；MTP 默认和发布状态保持不变。


[短提示实模](../results/local-pd-scheduler-v1/short.json)完成 20 项检查：两个不同的 26 / 27-token 提示分别生成 64 token，AR / MTP 与各自独立 AR 参考全 ID 一致；P(A) → P(B) → D(A) → D(B) 顺序正确，同时 ready 数为 2。队列满、token 超额、无效输入、queued / ready 取消、第二个输出 callback 取消、callback 抛 busy、重入拒绝、callback 安全追加作业、后续生成与 discard 均通过。终态后队列与额度归零，已消费作业没有自动重试。

[长提示实模](../results/local-pd-scheduler-v1/long.json)完成 7 项检查：两份 11,057-token 输入分别使用 AR 和 MTP depth2 / batchedScalarLinear / tail1024，均生成 128 token，与原 golden 的 prompt / 输出 IDs 和预算逐项一致。先完成两个 prefill，再依次 decode，逻辑额度由 22,370 → 11,185 → 0；结束时队列、ready、pending 事件均归零。

| 11k 作业 | 主干 prefill | 主干 prefill token/s | MTP 历史准备 | 纯 decode token/s | 平均 TPOT |
| --- | ---: | ---: | ---: | ---: | ---: |
| AR | 14.420 s | 766.79 | 0 | 29.67 | 33.71 ms |
| MTP depth2 | 14.963 s | 738.94 | 0.050 s | 35.30 | 28.33 ms |

这些是一次混排功能验收中的观察值，prefill / decode 均排除队列等待；没有据此宣布稳定加速、kernel 收益或默认 MTP 已达到发布门槛。探针有意先执行两个 prefill，因此 A 的 ready 等待包含 B 的完整 prefill；其他请求的等待时间不会因逻辑拆分而自动消失。该历史版本需完整阶段结束才可切换；后续新增的 chunk / round 继续执行边界不由这些旧结果验证。

两份长状态同时 ready 时，MLX 活跃分配约 81.379 GB；完成后为 80.628 GB，下降约 0.750 GB。首次 MTP head 已驻留并供后续请求复用；共享权重没有复制。这里记录的是 MLX allocator 的活跃字节，不能当作进程物理 footprint、真实 DRAM 流量或 token 额度的通用换算。

设备失效关闭分支使用 CPU 注入，未人为制造 GPU 故障；跨线程请求提交、后台 worker、HTTP、独立部署和细粒度抢占均不在此次验收范围。原始命令、二进制哈希及参考服务恢复见 [运行记录](../results/local-pd-scheduler-v1/run-ledger.json)，测试与计时汇总见 [summary.json](../results/local-pd-scheduler-v1/summary.json)。


## 合作式分块调度验收（2026-09-06）

增量 session 与可选 `cooperative` 已完成 Release 构建；[本轮 CPU 日志](../results/cooperative-pd-v1/cpu-tests.log)记录 26 项检查、0 失败，包含新生产队列引擎的分块公平性、容量、计时和取消清理。CPU 测试不证明实模输出一致或首字延迟已经改善。

实模控制器按[本轮计划](../results/cooperative-pd-v1/plan.json)依次执行旧 handoff 回归、`wholeStages` 回归，以及固定 11,057-token 长请求与短请求的合作调度。新入口为 `probe-gpu-cooperative-scheduler --model-dir PATH --tokens-file AGENT_11K.json --output NEW.json [--golden-report PATH]`。固定探针检查 AR/MTP 输出、实际 chunk/round 交错、状态/额度归零、yield 后取消、后续新请求和从提交到实际首 callback 的时间；不运行第二份模型。

**本轮全部通过：26 项 CPU 测试、55 项实模检查。** 实模包含旧 handoff 的 17 项、原完整阶段队列的 20 项，以及新合作调度的 18 项。新探针先建立独立 AR 参考并预热短 MTP，再测试完整阶段的长 AR / 短 MTP、合作模式的长 AR / 短 MTP，以及合作模式的长 MTP / 短 MTP。长输入 11,057 token、输出 128；短输入 26 token、输出 64。所有完整输出与 AR 参考一致，长 prompt / 输出逐项匹配原 golden。

两组相同长 AR / 短 MTP 输入的实际 callback 计时如下；两请求连续入队，长请求先入队：

| 从提交开始计时 | wholeStages | cooperative |
| --- | ---: | ---: |
| 短请求首 callback | 20.322 s | 1.679 s |
| 短请求最后 callback | 22.047 s | 7.827 s |
| 长请求首 callback | 15.802 s | 17.923 s |
| 长请求最后 callback | 20.122 s | 22.304 s |
| 整组完成耗时 | 22.048 s | 22.305 s |

本次短请求首 token 少等约 18.64 秒，整组多用约 0.257 秒；长请求因穿插服务短请求而稍晚完成。原始事件确认短请求的 prefillReady 和第二个输出都早于长请求的 prefillReady，不是仅发布 progress 标记。每组只测一次且顺序固定，这里展示观察值与调度取舍，不作为稳定的吞吐提升或所有混合负载保证。

活动吞吐按阶段单列：长 AR 的主干 prefill 为 699.77 / 693.68 token/s，纯 decode 为 29.40 / 29.00 token/s；短 MTP 纯 decode 为 36.52 / 36.01 token/s（依次为完整阶段 / 合作模式）。暂停恢复时间没有计入这些分母。第三组长短均启用 MTP，也通过全部输出与交错检查；长 MTP 的主干 prefill 为 695.27 token/s、纯 decode 为 33.54 token/s，是额外正确性场景中的一次观察。

yield 后取消覆盖 partial prefill 和已输出至少两个 token 的 decode；取消后不再推进该作业，状态与额度清零，后续新请求仍匹配参考。每组 resident 数不超过 2，结束时 queued / ready / pending / resident 以及 token 预留全部归零。参考服务已恢复并确认 MTP / drafter 关闭。

本轮未改 kernel 算术；`executionMode` 默认仍为 `wholeStages`，`cooperative` 需要显式启用，MTP 默认仍关闭。原始 callback 时钟、全部 token IDs、每一步事件与快照见 [cooperative.json](../results/cooperative-pd-v1/cooperative.json)；二进制、测试汇总与服务恢复见 [summary.json](../results/cooperative-pd-v1/summary.json) 和 [run-ledger.json](../results/cooperative-pd-v1/run-ledger.json)。
