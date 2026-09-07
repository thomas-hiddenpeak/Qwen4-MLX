# HTTP 输出限额、终态与未覆盖边界

2026-09-07。MTP双窗口结束后，HTTP adapter已修复非流式累计文本超限原因丢失，并增加可关联的模型、输出和连接终态记录。服务二进制`c93011f804dd287a7758568d69373089959a0b92408379390ea71d7294b0568a`完成36项Swift CPU、6项Python日志解析控制、19项live、15项edges、固定12轮46项soak及6项真实终态检查。**真实非流式生成过程中的`text_limit → HTTP 500 / output_limit`已触发并验证后续恢复；SSE应用缓冲overflow、15秒发送期限、300秒连接期限仍未覆盖。**

完整服务合同与历史结果见[HTTP实验服务](HTTP_SERVER_EXPERIMENT.md)。本页区分此次实测、源码合同与尚未触发的分支，不把正常请求或CPU状态机通过当作全部网络边界通过。

## 当前路径与覆盖

| 边界 | 实现与此次证据 | 仍不能声称的范围 |
| --- | --- | --- |
| 非流式累计文本限额 | 每请求[QwenHTTPTextBudget](../Sources/ANERunnerCore/QwenHTTPTextBudget.swift)先检查UTF-8字节，再接受文本；保留首个本地失败原因。最低8192字节配置减去2048字节JSON/header余量，允许6144字节文本。真实AR生成在decode中命中`text_limit`，scheduler failed之后仍返回`output_limit`；随后新AR/MTP成功。 | 只覆盖生成时累计文本检查，不覆盖最后UTF-8 flush才跨限或最终JSON/header编码超限；也不是模型数值故障恢复。 |
| SSE应用缓冲overflow | [QwenSSEOutputBuffer](../Sources/ANERunnerCore/QwenSSEOutputBuffer.swift)在普通帧字节或事件额度不足时选择`slowConsumer`并请求取消；若还能发送，保留已接受前缀，再发`slow_consumer`错误和一次`[DONE]`。scheduler迟到终态不能覆盖这个选择。 | 尚无真实HTTP/SSE `slow_consumer`或对应overflow记录。非流式`output_limit`不补成此项通过。 |
| 单次发送期限15秒 | [sendNext](../Sources/ANERunnerCLI/GPUHTTPServer.swift#L349)记录lease和发送起点；[checkDeadlines](../Sources/ANERunnerCLI/GPUHTTPServer.swift#L394)检查未完成发送并以`send_deadline`关闭，记录lease/发送经过时间。header/simple发送也使用此计时。 | 尚未观察真实发送保持未完成达到期限；CPU缓冲检查和暂停读取均不能代替。 |
| 整个连接期限300秒 | 从accept时刻计算年龄；未先命中发送期限时，以`connection_deadline`关闭，记录连接年龄。 | 尚未观察已解析、计算中或排队请求命中；未完成header/body会先遇到接收期限。 |
| 晚到的send确认 | 关闭保留真实in-flight lease，确认回调释放它并可记录`closed_send_released`。 | 此次终态日志没有该事件实例，不能称为真实晚确认分支已覆盖。 |

周期检查每秒执行，15/300秒是阈值，不保证精确到点关闭。当前已有有限的关闭原因枚举，可以区分发送错误、接收失败、期限、正常终态和shutdown；这些字段存在不等于每个分支都已经实测。

## 真实非流式超限证据

[`terminal-logs.json`](../results/http-output-fix-regression-v2/terminal-logs.json)及[原始服务日志](../results/http-output-fix-regression-v2/terminal-logs.server.log)来自同一自有服务PID24196，4连接、8192字节输出额度。测试为真实模型AR、非流式、4096-token预算；请求意图抄写7200字节文本，但**是否触发以实际HTTP和日志为准，不能由提示词意图推定**。

此次于43.664788375秒收到HTTP500，body为`code=output_limit`、`type=server_error`。唯一请求`chatcmpl-5f320b77-880c-4347-9f12-4e144ea9548e`对应同一连接，实际顺序为：

1. `model_terminal`：`model_kind=failed`、`stage=decode`、`reason=text_limit`。没有先记录模型成功，属于生成时文本限额。
2. `output_terminal`：`output_outcome=failed`、`reason=text_limit`、`error_code=output_limit`，`text_bytes=6144`、`text_limit_bytes=6144`。计数是已接受字节；跨限的下一段未接受，不将计数解释为越界后的总输出。
3. `connection_close`：`reason=terminal_sent`，已选`failed`保持不变，buffered/in-flight字节和事件为零、output_drained为true。

超限后所有队列、resident、reserved tokens归零；新AR非流式和MTP2 SSE均再次返回`1,2,`、prompt35/output4/total39、`finish_reason=length`。这些是文本、usage与HTTP终态回归，HTTP没有输出完整token IDs。本次不是SSE溢出、编码错误、最终响应超限或GPU运行故障的覆盖。

## 三种终态不能混成一次“成功送达”

[complete](../Sources/ANERunnerCLI/GPUHTTPServer.swift#L548)先记录模型终态，再执行最后UTF-8 flush及最终响应编码，所以`model_terminal completed`或旧`HTTP request ... finish=eos/length`只证明模型完成。adapter最后一步仍可能失败。

[logOutput](../Sources/ANERunnerCLI/GPUHTTPServer.swift#L622)只由首次overflow或`finish.accepted`记录`output_terminal`；它表示输出缓冲选定结果，**不表示客户端已经读取**。本次live/edges/soak/terminal的所有output_terminal记录当时均`output_drained=false`。后续`connection_close reason=terminal_sent`来自Network的contentProcessed确认；客户端完整收到响应的证据仍来自测试端解析，不能只凭发送回调推定。

[close](../Sources/ANERunnerCLI/GPUHTTPServer.swift#L372)记录网络终态和关闭前/后outcome。RST经常先选择`disconnected`，随后scheduler报告cancelled；迟到`finish`返回alreadyTerminal，不再产生output_terminal。未进入scheduler的拒绝请求也可能只有connection_close。不能承诺每份请求都有三个事件，或把缺少output_terminal直接当作丢日志。

日志schema为`qwen-http-lifecycle-v1`，用PID、request ID和connection ID关联，只含有限原因、计数与时间；不写提示词、正文或token内容。SSE的累计`text_bytes/text_limit_bytes`不适用，记录null而非伪造0。旧`HTTP request id=...`仍保留给已有取消gate，但它与新的model_terminal是同一模型事件的两种表示。

| 本次c930报告 | 旧model日志 | 新model_terminal | 新output_terminal | 新connection_close |
| --- | ---: | ---: | ---: | ---: |
| v1 live | 14 | 14：9完成、5 prefill取消 | 9 | 171 |
| v2 edges | 8 | 8：7完成、1 decode取消 | 7 | 135 |
| v2 soak | 43 | 43：31完成、12 decode取消 | 31 | 197 |
| v2 terminal | 8 | 8：6完成、1 decode取消、1 text_limit失败 | 7 | 135 |

connection_close还包含health、参数拒绝等非生成连接，不能当作生成请求数。旧43-request日志的CPU格式兼容检查只证明旧格式可读，不是新schema实测；此表soak的43则来自c930新的一次真实运行。同一行的新旧model计数不能相加。

## 为什么暂停读取仍不够

发送额度在实际contentProcessed回调后释放；这表示传输处理了内容，不表示远端应用已读取。OS/TCP可能容纳全部短输出并持续释放应用lease，因此“暂停读取后另一请求仍能推进”与“应用缓冲确实溢出”必须分开。MTP最高256-token预算还可能提前EOS；此次live的`live_overflow_observed=false`，不能依赖长输出意图、等待或缩小接收缓冲补成SSE通过。

后续SSE溢出测试需要取得请求ID、明确的slow_consumer选择及同连接错误/[DONE]；期限测试需要服务侧send_deadline或connection_deadline、对应lease/年龄与恢复证据。若触发条件不成立，继续记未覆盖，不伪造推理或延迟凑通过。

## 同步stderr仍有活性缺口

c930的[log](../Sources/ANERunnerCLI/GPUHTTPServer.swift#L670)仍在共享mutex内同步写stderr。若stderr接到无人读取且已满的pipe，写入能阻塞调用它的网络或推理线程；独立网络队列不消除共享日志阻塞。此次服务日志写普通文件，以上通过结果没有覆盖满pipe行为。

最小有界日志写入候选及pipe活性验证正在准备，**尚未应用或验证，不能声称该缺口已经修复**。保持c930历史证据与后续候选分开。真实SSE overflow、发送/连接期限、晚到send确认、长期稳定及精确MTP verify/replay取消也仍有各自门槛。
