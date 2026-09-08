# 本机 HTTP/SSE 的最小接入

2026-09-07：本文保存 HTTP 接入的设计约束与首批 CPU 输出边界。随后已实现实验入口并完成29项相关CPU、19项真实loopback检查，当前支持范围与未覆盖项见 [服务验收](HTTP_SERVER_EXPERIMENT.md)。模型、调度策略和默认 kernel 没有因接入服务而改变。

2026-09-08：后续已加入 [function tools 与工具结果回传](HTTP_TOOL_CALLING.md)、[完整前缀缓存](AR_PREFIX_CACHE.md)。下文“先交付纯文本”的段落保留最初实施顺序；当前范围以这两份验收及服务合同为准。

## 已有接口与接入点

| 已有代码 | 可以直接复用 | 仍需服务层处理 |
| --- | --- | --- |
| `QwenLocalScheduler.submit/runNext/cancel` | 有界 prefill/ready 队列、逻辑 token 预留、分片推进与终态清理 | 网络线程不能直接调用；所有操作进入同一个推理执行器 |
| `QwenGenerator.beginPrefill/stepPrefill/beginDecode/stepDecode` | 每个 prompt chunk / AR 或 MTP round 的暂停边界 | 网络忙时不能暂停在半轮，也不能重复提交已消费的 handoff |
| `QwenCancellation` | 从其他线程请求取消，幂等且受锁保护 | 当前 GPU/SSD 操作结束后才能观察到；断连不代表立即停止设备 |
| `onToken(Int32)` | 只发布已提交 token，首 token 单独发布，MTP 一轮可连续多次调用 | 回调同步执行；不能在这里等 socket、排入无界 async 闭包，或逐 token 直接转 String |
| `Event.completed/failed/cancelled` | 当前 scheduler 每个 admitted job 释放一次，故障关闭会排出全部待终结事件 | 服务还要处理入队前失败、断连及 send callback 与推理终态的竞争 |

具体源位置：`QwenLocalScheduler.swift` 的 `runNext/terminate`；`QwenGeneration.swift` 的 `publish` 先追加 generated，再调用 onToken。输出入队失败后应取消并抛出 cancellation，不能重试那个 token。失败请求未必有完整 result；服务记录的已入队/发送 token 数不能冒充已生成总数。

推理对象保持原有 non-Sendable 边界。建议一个长期存活的固定 OS 线程创建、使用并销毁 model/generator/scheduler，线程每次取一批有界命令，再调用一次 `runNext`，有工作时继续，没有工作时等待通知。普通 Swift actor/串行 DispatchQueue 不保证固定 OS 线程；当前 `MX.stream` 是静态默认 stream，而 pinned MLX 的 `stream.cpp` 默认 stream 与 `device.cpp:933` encoder 表是 `thread_local`。先保持固定线程，避免为 HTTP 接入改 MLX stream 所有权。

网络收发使用独立队列的 Apple `NWListener/NWConnection`。它们提供连接与字节收发，仍需最小 HTTP/1.1 parser/framing；首版可每连接只处理一个请求并明确关闭连接。接收 header/body 与连接数、待 tokenization 请求、待提交命令都要有限额，否则 scheduler 的额度只约束最后一段队列。tokenization 放在独立 CPU 所有者中，跨边界只传请求 DTO、token IDs、Data、UUID、取消句柄和统计快照。

## 本轮 CPU 模块

[`QwenSSEOutputBuffer`](../Sources/ANERunnerCore/QwenSSEOutputBuffer.swift) 是每请求的线程安全缓冲，默认 64 KiB / 256 个入队条目，内部预留 4 KiB / 1 个条目给终态。byte 统计使用实际编码后的 SSE body Data；event 统计是入队条目数量，不解析 Data 内的 SSE 内容。一个终态条目可以由适配器封装结束信息和协议结束标记。HTTP framing、socket 内部缓冲和调用者自行保留的 Data 不计入这个对象的额度。

- `enqueue` 只短暂持锁，不等网络；超限拒绝本条，保留此前输出顺序，追加预先提供的小型 overflow 终态，并返回一次 `cancelProducer`。调用者必须在锁外执行取消。
- `beginSend` 最多发出一个 lease。取走条目不会提前释放额度，只有对应的 `acknowledgeSend(id:succeeded:)` 才释放；过期/重复回调不会影响新的 lease。
- `finish` 由 scheduler 终态驱动。完成、失败、取消、overflow、断连的第一个 outcome 胜出；之后 finish 只确认推理已结束，不重发终态。终态超出预留额度会返回 `invalidFrame`，适配器须使用固定小型 fallback。
- `disconnect` 清理未发送条目并禁止继续发送，保留 in-flight 额度直到其回调。适配器同时取消连接；不能把 `disconnect()` 当作已经回收了网络层 Data。
- `Actions.scheduleSend` 合并空队列到有数据、以及 ack 后续发的唤醒。网络队列每次只发一个 lease，随后由 send callback 驱动下一次；不能为每个 token 无条件堆积独立发送任务。

Apple 的 `contentProcessed` 表示连接已处理内容，不是客户端已经收到/呈现；本模块的 `isDrained` 也没有端到端送达保证。逻辑终态只选一次，断开的连接无法保证收到最后一帧。[Apple send 文档](https://developer.apple.com/documentation/network/nwconnection/send%28content%3Acontentcontext%3Aiscomplete%3Acompletion%3A%29-5ecuz?language=objc)

缓冲超限处理之外，适配器还需要有限的发送无进展期限和请求期限：模型已经生成完但客户端不再读时，不能无限保留连接。超时走现有取消/断连入口；不让慢请求阻塞另一个请求。正常完成必须等末帧 send callback 再关闭连接。

[`IncrementalUTF8Decoder`](../Sources/ANERunnerCore/IncrementalUTF8Decoder.swift) 由并行任务补齐，供文字适配层复用。`QwenTokenizer` 原有 `decodeBytes` 可以提供 token 的原始字节；每请求持有自己的增量 decoder，正常结束时 finish，special/EOS 根据既有 tokenizer 策略过滤。SSE 始终是 UTF-8，字节拆分不能变成额外替换字符。[SSE 标准](https://html.spec.whatwg.org/dev/server-sent-events.html)

## 最小服务范围与检查

先交付本机 text-only、greedy、单 choice 的 HTTP/SSE 流程以及 health/ready。若提供 `/v1/chat/completions`，明确这是支持 system/user/assistant 的子集：当前 `renderChat` 不支持 tools/tool role、多模态，生成器也没有随机采样或 stop-string 合同。收到不支持的字段必须明确拒绝，不能为了兼容 agent 静默丢弃工具参数。通用 agent API 需要后续独立补齐这些语义。

发送 SSE 响应头前完成结构验证和 scheduler admission：无效请求返回 4xx，队列/额度不足返回 429，模型尚未就绪/已关闭返回 503；开始流式响应后只能发约定的流内错误与终态，不能再改 HTTP 状态码。首次实现不支持自动续传或 `Last-Event-ID` 重放，客户端不得把断线自动重试当成同一次推理。

本模块对应 [`QwenSSEOutputBufferTests`](../Tests/ANERunnerCoreTests/QwenSSEOutputBufferTests.swift)：byte/event 上限、in-flight 额度、overflow 输出前缀、终态竞争、迟到 send failure、断连和并发 enqueue。现有 local/cooperative scheduler 测试继续负责 job 与 GPU handoff 的释放；它们不等同于 HTTP 验收。

首批缓冲10项、UTF-8 8项CPU测试通过（`results/service-core-v1/tests.log`）。随后协议层再补11项，直接使用真实模型完成首轮网络回归；未构建额外假backend。服务已分别记录prefill/decode与scheduler总耗时，传输时刻仍用于内部期限管理，尚未完整导出接收、admission、首帧与传输结束的分段时间；不能把内部send callback称为客户端实际TTFT。后续网络边界以[服务验收](HTTP_SERVER_EXPERIMENT.md)中的实际覆盖为准。
