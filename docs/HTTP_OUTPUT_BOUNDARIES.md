# HTTP 输出限额、终态与未覆盖边界

2026-09-07。MTP双窗口结束后，HTTP adapter已修复非流式累计文本超限原因丢失，并增加可关联的模型、输出和连接终态记录。服务二进制`c93011f804dd287a7758568d69373089959a0b92408379390ea71d7294b0568a`完成36项Swift CPU、6项Python日志解析控制、19项live、15项edges、固定12轮46项soak及6项真实终态检查。**c930这批已触发真实非流式生成过程中的`text_limit → HTTP 500 / output_limit`并验证后续恢复；当时SSE应用缓冲overflow、15秒发送期限、300秒连接期限未覆盖。**

当前b039已加入有界异步诊断日志及显式服务进程SIGPIPE策略，release构建、43项Swift CPU、11项Python控制及五组真实服务回归19/15/46/6/3项全部通过；包含实际非流式text_limit再次触发，以及真实未读满stderr管道下的服务活性。随后同一b039二进制的独立窗口真实触发AR SSE slow_consumer，4项检查通过；15/300秒期限和晚到send确认仍未覆盖。下面c930旧实测独立保留，不计入新版本验收。

完整服务合同与历史结果见[HTTP实验服务](HTTP_SERVER_EXPERIMENT.md)。本页区分此次实测、源码合同与尚未触发的分支，不把正常请求或CPU状态机通过当作全部网络边界通过。

## 输出路径与c930实测覆盖

| 边界 | 实现与此次证据 | 仍不能声称的范围 |
| --- | --- | --- |
| 非流式累计文本限额 | 每请求[QwenHTTPTextBudget](../Sources/ANERunnerCore/QwenHTTPTextBudget.swift)先检查UTF-8字节，再接受文本；保留首个本地失败原因。最低8192字节配置减去2048字节JSON/header余量，允许6144字节文本。真实AR生成在decode中命中`text_limit`，scheduler failed之后仍返回`output_limit`；随后新AR/MTP成功。 | 只覆盖生成时累计文本检查，不覆盖最后UTF-8 flush才跨限或最终JSON/header编码超限；也不是模型数值故障恢复。 |
| SSE应用缓冲overflow | [QwenSSEOutputBuffer](../Sources/ANERunnerCore/QwenSSEOutputBuffer.swift)在普通帧字节或事件额度不足时选择`slowConsumer`并请求取消；若还能发送，保留已接受前缀，再发`slow_consumer`错误和一次`[DONE]`。scheduler迟到终态不能覆盖这个选择。 | c930时没有实际记录；后续b039独立AR窗口已观察slow_consumer、错误/[DONE]和恢复，见文末。该证据不覆盖MTP SSE或每条成功enqueue的完整交付。 |
| 单次发送期限15秒 | [sendNext](../Sources/ANERunnerCLI/GPUHTTPServer.swift)记录lease和发送起点；[checkDeadlines](../Sources/ANERunnerCLI/GPUHTTPServer.swift)检查未完成发送并以`send_deadline`关闭，记录lease/发送经过时间。header/simple发送也使用此计时。 | 尚未观察真实发送保持未完成达到期限；CPU缓冲检查和暂停读取均不能代替。 |
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

[complete](../Sources/ANERunnerCLI/GPUHTTPServer.swift)先记录模型终态，再执行最后UTF-8 flush及最终响应编码，所以`model_terminal completed`或旧`HTTP request ... finish=eos/length`只证明模型完成。adapter最后一步仍可能失败。

[logOutput](../Sources/ANERunnerCLI/GPUHTTPServer.swift)只由首次overflow或`finish.accepted`记录`output_terminal`；它表示输出缓冲选定结果，**不表示客户端已经读取**。本次live/edges/soak/terminal的所有output_terminal记录当时均`output_drained=false`。后续`connection_close reason=terminal_sent`来自Network的contentProcessed确认；客户端完整收到响应的证据仍来自测试端解析，不能只凭发送回调推定。

[close](../Sources/ANERunnerCLI/GPUHTTPServer.swift)记录网络终态和关闭前/后outcome。RST经常先选择`disconnected`，随后scheduler报告cancelled；迟到`finish`返回alreadyTerminal，不再产生output_terminal。未进入scheduler的拒绝请求也可能只有connection_close。不能承诺每份请求都有三个事件，或把缺少output_terminal直接当作丢日志。

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

后续独立AR窗口取得了同请求ID的slow_consumer选择、实际错误/[DONE]和EOF，证据在文末；旧live未命中的历史保持不变。期限测试仍需服务侧send_deadline或connection_deadline、对应lease/年龄与恢复证据。若触发条件不成立，继续记未覆盖，不伪造推理或延迟凑通过。

## 诊断日志的背压与信号策略

c930的log在共享mutex内同步写stderr。若stderr接到无人读取且已满的pipe，写入能阻塞网络或推理线程；c930服务日志写普通文件，以上通过结果没有覆盖这种活性问题。**这项历史缺口不能被旧19/15/46/6检查或早期19+15+46结果补成通过。**

当前b039已将写入移到[QwenHTTPLogger](../Sources/ANERunnerCore/QwenHTTPLogger.swift)的单独固定线程。producer只在短锁内入队，不等待sink IO；固定额度为65536字节、128条、单条4096字节，正在写的完整记录继续占额度。满额或过大记录整条丢弃，失败写入记errno并停止接受；health的`logging`提供保留/in-flight计数、累计drops和write_failures。它独立于SSE/非流式响应缓冲，日志丢弃不会选择slow_consumer/output_limit或取消模型。

关闭时stop丢弃排队记录，不join可能仍卡在IO的writer；该writer最多保留一条记录至返回或进程退出。因此这是可丢弃的诊断日志，不能承诺最后一条终态、刷盘或完整审计。正文、提示词、token内容和原始异常正文仍不进入日志。

第一版writer线程mask方案实际失败：[xctest日志](../results/http-async-logger-regression-v1/cpu-tests.log)显示真实closed-pipe测试被SIGPIPE13终止。固定Apple XNU源码中，非socket写入的EPIPE路径调用进程级`psignal`；`F_SETNOSIGPIPE`设置共享fileglob的FG_NOSIGPIPE，dup仍指向同一个fileglob。所以只屏蔽writer线程不足，在私有dup上设置该flag也不是隔离方案。[Apple sys_generic.c](https://github.com/apple-oss-distributions/xnu/blob/f6217f891ac0bb64f3d375211650a4c1ff8ca1ea/bsd/kern/sys_generic.c#L601)，[F_SETNOSIGPIPE](https://github.com/apple-oss-distributions/xnu/blob/f6217f891ac0bb64f3d375211650a4c1ff8ca1ea/bsd/kern/kern_descrip.c#L2944)，[finishdup](https://github.com/apple-oss-distributions/xnu/blob/f6217f891ac0bb64f3d375211650a4c1ff8ca1ea/bsd/kern/kern_descrip.c#L551)。公开commit早于本机kernel，源码解释与此次真实失败一致，不声称取得本机内核完整对应版本。

最终选定的最小策略是在独立`serve-gpu`进程入口显式设置SIGPIPE=SIG_IGN，持续至该进程退出；不改变父进程、其他应用或共享stderr FD标志。Core仅只读核验，未配置就拒绝默认stderr sink。忽略信号让断管以EPIPE返回，满pipe写入仍可能阻塞，所以固定writer及非等待退出同样必要。

v3第一次编译又因本机Swift SDK将`Darwin.sigaction(...)`解析为struct构造器而失败；只在3文件7处改成未限定函数调用。保留[首编译错误](../results/http-async-logger-regression-v2/build.log)、[导入修正记录](../results/http-async-logger-regression-v2/import-fix.json)及[后续成功构建](../results/http-async-logger-regression-v2/build-fixed-import.log)，不写成一次通过。当前`b0391391b4af4dbdcd31bb16cffbf268e790112129e4887c3ac84ef547be1bc1`的[43项Swift CPU](../results/http-async-logger-regression-v2/cpu-tests.log)与[11项Python控制](../results/http-async-logger-regression-v2/python-production-tests.log)通过，含实际断管EPIPE、调用线程mask/FD标志不变和未配置策略被拒绝。它们不是实际满pipe服务验收。

五项服务postflight后，包装器已改为加载生产`scripts/test_http_terminal_log_validator.py`，与旧ignored副本内容SHA一致。仅含Package.swift和5个生产脚本、无results或模型的临时目录中，[相同11项CPU再次通过](../results/http-async-logger-regression-v2/python-clean-layout-tests.log)，见[修正记录](../results/http-async-logger-regression-v2/logger-wrapper-path-fix.json)；这是同组复跑，不累计为22项。Swift二进制和历史冻结SHA未改。

## b039实测诊断归因与满管道活性

日志可晚于响应和health出现，需按唯一请求/连接ID有界等待；旧兼容行仍与model_terminal对应同一个事件。edges/soak保留原取消前提、请求和耗时口径，等待最多10秒并检查零丢弃/写入失败，最后原有idle快照再检查累计计数；terminal脚本也如此。计数缺失或任意drops/failures均不能判日志归因完整。

[`本轮计划`](../results/http-async-logger-regression-v2/plan.json)的live/edges/12轮soak/terminal-logs/pipe最终检查数为19/15/46/6/3，全部complete、passed、graceful_shutdown且退出码0；[postflight](../results/http-async-logger-regression-v2/postflight-and-release.json)核对133文件、102模型payload及b039二进制，参考服务按原argv恢复PID31650、11235监听与idle均已核对。完整请求与RSS/FD范围见[HTTP服务实测](HTTP_SERVER_EXPERIMENT.md#b039有界日志cpu与五组实模回归通过)。

普通文件日志的实际计数如下；旧model行与新model_terminal逐ID一致，不能相加。

| b039报告 | 旧model / model_terminal | output_terminal | connection_close |
| --- | ---: | ---: | ---: |
| live | 14 / 14：9完成、5 prefill取消 | 9 | 174 |
| edges | 8 / 8：7完成、1 decode取消 | 7 | 154 |
| soak | 43 / 43：31完成、12 decode取消 | 31 | 213 |
| terminal | 8 / 8：6完成、1 decode取消、1 text_limit失败 | 7 | 140 |

edges/soak/terminal的最后idle快照分别为written/enqueued 184/184、337/337、170/170，队列和in-flight为零，drops/dropped_bytes/write_failures均为0。live最后保存的是SIGTERM前active快照（215/215、零loss），没有退出后的health快照；其raw取消终态已核对，不据此承诺退出日志可靠投递。soak的43个模型终态对应31份完成HTTP响应与12个不同ID的decode取消，每轮取消后及轮末资源归零。850.750708709秒窗口内5次外部idle RSS采样末次比基线增加320 KiB，FD始终11；这是有限观察，不证明无泄漏或长期稳定。

[`本轮terminal报告`](../results/http-async-logger-regression-v2/terminal-logs.json)与[raw日志](../results/http-async-logger-regression-v2/terminal-logs.server.log)再次命中真实非流式生成超限：PID31366，请求`chatcmpl-da642831-c778-4dbf-94dd-a12c9c8576fe`，51.953498875秒收到HTTP500/output_limit。模型记录failed/decode、reason=text_limit；输出记录failed/output_limit、已接受text_bytes与limit均6144；同连接terminal_sent后额度归零，后续AR/MTP均返回`1,2,`、prompt35/output4/total39、length。仍是generation_text_limit，不能补成最终UTF-8 flush或JSON/header超限覆盖。

[`pipe报告`](../results/http-async-logger-regression-v2/pipe.json)使用自有stderr管道，读端保持打开且退出前从未读取，stdout独立写普通文件。161次填充health后应用日志达到128条事件上限；3个额外health样本均有65536字节未读、written固定165、in-flight419字节、总保留52926字节，drops为2/3/4。这同时观察到实际管道积压、writer不再前进和应用限额丢弃，不能仅以暂停读取替代。

日志堵塞期间，真实AR非流式与MTP2 SSE均完成`1,2,`、prompt35/output4/total39、length；随后idle的全部请求/模型资源归零，但日志仍为128条、52926字节、written165，累计丢弃13条/6581字节、write_failures为0。SIGTERM在0.7258455秒内退出0，父Python原SIGPIPE策略保持不变；退出后才捕获65536字节stderr，stdout为0字节。实际[stderr记录](../results/http-async-logger-regression-v2/pipe.stderr.log)共165个完整行（157条无请求ID的connection_close及8条启动信息），没有任何模型请求终态，符合此项明确的`lifecycle_attribution_complete=false`。

这证明当前自有未读满pipe条件下的服务活性，不证明完整日志、退出刷盘或所有sink行为。断管EPIPE由独立Core真实pipe CPU测试覆盖。这五项日志回归没有触发SSE overflow，后续由独立AR窗口补测如下；15秒发送/300秒连接期限、晚到send确认、长期稳定及MTP verify/replay内部取消仍未覆盖。

## b039独立AR SSE溢出实测

[`attempt.json`](../results/http-sse-overflow-public-v1/attempt.json)、[服务raw日志](../results/http-sse-overflow-public-v1/attempt.server.log)和[原始SSE](../results/http-sse-overflow-public-v1/attempt.sse)来自独立自有服务PID35012，同一b039二进制，8192字节输出额度、4连接；冻结prompt4039 token，真实AR、stream=true、4096-token输出预算。客户端实际接收缓冲1024字节，先收到两条非空content后暂停读取，观察到明确的应用slow_consumer才恢复，实际暂停60.267929166秒。暂停本身不是判据，实际同ID错误选择和客户端终态才构成覆盖。

唯一请求`chatcmpl-7b88d73e-3f24-48d6-bbea-4186dd34ac64`的记录与收到的流相互对应：

1. `output_terminal`唯一，reason/error_code为slow_consumer，outcome为slowConsumer，cancellation_requested=true。记录时4117字节/18事件保留，其中234字节in-flight；SSE text_bytes/text_limit_bytes为null。此为错误终态选定后的快照，不据此反推超限瞬间的精确额度。
2. `model_terminal`唯一，cancelled/decode，旧兼容行也是同ID的decode取消；迟到模型终态没有覆盖slowConsumer。
3. 原始body共480541字节、2055帧：role 1 + 非空content 2052 + slow_consumer错误1 + DONE 1，随后正常EOF。严格UTF-8内容合计12312字节，wire/text SHA均与报告一致；暂停前702字节（含两条content）的wire/text前缀保持。错误流没有正常finish/usage，2052条内容帧不能当作生成token计数。
4. `connection_close`唯一且为terminal_sent，outcome仍slowConsumer，transport_closed/output_drained为true，buffered/in-flight字节与事件归零。没有closed_send_released或deadline记录。

4项检查全部通过，报告complete/passed为true，服务graceful_shutdown且退出0。fresh AR非流式和MTP2 SSE均返回`1,2,`、prompt35/output4/total39、length，idle各请求/模型额度归零；最终logging141/141、零drops/dropped_bytes/write_failures、无保留记录。[postflight](../results/http-sse-overflow-public-v1/postflight-and-release.json)核对141文件、102模型payload及二进制；[ledger](../results/http-sse-overflow-public-v1/run-ledger.json)记录自有组清空，参考服务按原argv恢复PID35281、11235监听与idle已核对。

本次只证明这一AR配置下的真实应用overflow、已收到前缀保持、错误尾帧和取消后恢复。没有逐条记录成功enqueue的内容，因此不宣称每个已入队输出都完整交付；不是MTP SSE溢出、长时间慢读或性能门槛。非流式text_limit、日志pipe满、SSE slow_consumer是三个分别实测的边界；15秒发送期限、300秒连接期限、晚到send确认仍未覆盖。
