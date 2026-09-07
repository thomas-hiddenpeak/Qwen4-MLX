# HTTP 输出限额与发送期限：未覆盖边界

2026-09-07 的 CPU 只读检查确认：已有 19 项 live、15 项网络边界、46 项短 soak 检查通过，但**真实应用输出 overflow、15 秒发送期限、300 秒连接期限仍未覆盖**。CPU 缓冲测试证明的是状态机合同，不能替代实际 HTTP 触发证据。本轮未运行模型、修改服务或新增猜测性的长输出 harness；MTP 窗口 A/B 期间源码和二进制保持冻结。

## 当前真实路径

| 边界 | 实现及当前可见结果 | 缺失的证据 |
| --- | --- | --- |
| SSE 应用缓冲 overflow | [服务配置](../Sources/ANERunnerCLI/GPUHTTPServer.swift#L244)在最低 8192 字节总限额内预留 4096 字节终态；[enqueue](../Sources/ANERunnerCore/QwenSSEOutputBuffer.swift#L95)在未释放的普通帧加新帧超过剩余普通额度，或普通事件达到 255 条时，选择 `slowConsumer` 并请求取消。若连接仍可发送，已接受前缀后跟 `code=slow_consumer` 错误和一次 `[DONE]`。 | 尚未从真实 HTTP 收到该错误或取得服务侧明确的 overflow 原因记录。 |
| 单次发送期限 15 秒 | [sendNext](../Sources/ANERunnerCLI/GPUHTTPServer.swift#L317)保存当前 lease ID 和起始时间；[周期检查](../Sources/ANERunnerCLI/GPUHTTPServer.swift#L345)在发送仍未完成且持续至少 15 秒时关闭连接并取消请求。header/simple 响应发送也使用该时间字段。 | 尚未证明实际有一次发送保持未完成达到期限；关闭路径不记录原因和对应 lease。 |
| 整个连接期限 300 秒 | 同一[周期检查](../Sources/ANERunnerCLI/GPUHTTPServer.swift#L345)从 accept 时刻计算连接年龄，在没有先命中发送期限时，于年龄达到 300 秒后关闭。 | 尚未观察一份已解析、仍在计算或排队的请求命中该分支。未完成 header/body 会先命中 15 秒接收期限，不能用于代测。 |

周期观察每秒执行一次，15/300 秒是检查阈值，不是承诺精确到点关闭。两个期限当前都调用相同的 `close`；只有“约在该时刻断开”不能排除发送错误、另一期限或先前取消。

## SSE 与非流式限额不是同一条错误路径

SSE `slow_consumer` 由缓冲的入队失败选择。[publish](../Sources/ANERunnerCLI/GPUHTTPServer.swift#L479)随后抛取消，scheduler 的迟到 cancelled 终态保留已选中的 overflow 结果。取得 role/content 中的请求 ID 后，收到同连接的 `slow_consumer` 和 `[DONE]`，才可作为可见的 SSE 溢出证据；它不应被要求返回 `output_limit`。

非流式没有中间帧队列：[publish 的累计文本检查](../Sources/ANERunnerCLI/GPUHTTPServer.swift#L488)保留 2048 字节 JSON/header 余量，最低配置下文本最多 6144 字节。超出时抛 `GPUHTTPOutputError.tooLarge`，但 [scheduler](../Sources/ANERunnerGPU/QwenLocalScheduler.swift#L350)将它转换为 failed，随后 [complete](../Sources/ANERunnerCLI/GPUHTTPServer.swift#L515)仅返回 `generation_failed`，丢失了明确的限额原因。这是已确认的错误归因缺口。

当前 `output_limit` 出现在 [complete 的 fallback](../Sources/ANERunnerCLI/GPUHTTPServer.swift#L521)：例如最终编码后的完整响应超过终态额度，或最后 UTF-8 flush/编码失败。成功生成日志在最终响应检查之前写出，因此日志中的 `finish=eos/length` 也不能单独证明客户端收到成功终态。构造大量需 JSON 转义的文本可能使最终编码先超限，但现无已验证、稳定且足够短的模型输出 fixture，不能据提示词意图保证触发。

## 为什么暂停读取不够

[发送额度](../Sources/ANERunnerCore/QwenSSEOutputBuffer.swift#L163)在 `contentProcessed` 回调后释放；这表示传输处理了内容，不表示远端应用已经读取。客户端暂停读取或缩小接收缓冲，仍可能由 Network/TCP/内核容纳全部短输出，并持续释放应用 lease。必须区分“另一个请求仍能推进”和“本请求应用额度确实超限”。

目前最小配置仍有 4096 字节普通 SSE 额度；MTP 输出预算最高 256 token，模型还可能提前 EOS。当前二进制没有可直接固定传输确认进度的参数，也没有已知必然超大的单 token 帧。因此没有确定的短触发条件，增加长生成或等待时间不能自动补成有效覆盖。慢读一旦产生积压，也可能先触发 overflow，再等待终态发送而命中 15 秒期限；两次原因应分别保留。

## 解冻后的最小补齐

1. 在 HTTP adapter 的每请求本地状态中保留累计文本超限原因，令 scheduler 返回 failed 后仍能发出 `output_limit`；不改变 GPU 数值路径或把普通推理故障都改写成限额错误。最终编码超限也应保留明确原因。
2. 补少量可关联记录：请求 ID、选中的输出 outcome/原因、bytes/events/in-flight 额度、关闭原因、当前 lease ID/发送经过时间和连接年龄。输出限额与网络关闭可能是先后两个事件，应避免混成一个成功终态。**不记录提示词、输出正文或 token 内容。**
3. 先选择真实冻结输出已知的短 fixture，使原始文本或最终编码能确定跨过配置限额，再写有界 HTTP 测试；SSE 应验证 `slow_consumer`，非流式应验证 `output_limit`。若当前参数仍不能确定触发，明确保留未覆盖，不加伪造推理或服务延迟开关来凑通过。
4. 对发送/连接期限，需先证明命中条件并取得服务侧对应原因，不能只以客户端停止读取、耗时或 EOF 判定通过。每次测试还须唯一关联请求 ID，检查 idle 与所有任务/预留计数归零，再运行固定短 AR 和 MTP 请求核对文本、实际 usage 与终止原因；已有 edge/soak helper 可复用。

本轮结论只收紧验证边界，不撤销已有正常请求、取消恢复与短 soak 的通过记录，也不将这些未覆盖分支算作已通过。
