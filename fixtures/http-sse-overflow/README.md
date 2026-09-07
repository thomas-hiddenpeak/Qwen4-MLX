# 单次真实 SSE 慢消费者回归

本目录保存固定输入及其身份清单；回归入口是 `scripts/test_http_sse_overflow.py`。脚本验证固定的一次 SSE 输出 overflow、收到的前缀和生命周期，不测性能。输入与 11 项 CPU 控制已验证；本清单不表示真实服务或 GPU 回归已经通过，实测结论以独立运行报告为准。

输入于2026-09-07T02:30:22.087165Z、任何生成之前冻结：要求原样抄写`山川日月`重复2000次，共8000汉字/24000 UTF-8字节。使用现有`ane-runner tokenize --chat true`的CPU入口实际得到**4039输入token**；AR、stream=true、max_tokens4096，合计**8135≤16384**。完整messages、分词报告、IDs及模型tokenizer元数据SHA见`manifest.json`。CPU分词二进制SHA为`836678a69f7e41d727e574d977ece6b084add32f85e57d45e2d4c3d14e8c34e7`；正式运行另记录实际二进制SHA，不把这个CPU分词版本当作logger实测版本。

唯一长请求采用现有合法`--output-buffer-bytes 8192`：常规SSE预算4096字节、保留终态4096字节，服务策略/发送15秒deadline/连接300秒deadline均不改变。客户端在连接前请求`SO_RCVBUF=1024`并记录OS实际返回值。获得HTTP200、assistant role、同一request ID及至少两个非空content帧后暂停；保留同一`HTTPResponse`对象及其已有缓冲，避免重建reader丢失读前缓存。

暂停期间只从普通文件读取有界、完整的日志行，每50ms检查一次同ID。只有实际`output_terminal`的`reason=slow_consumer`与`output_outcome=slowConsumer`才算触发；一旦观察到立即恢复读取，不等待模型cancel日志或做health请求。若先出现正常/其他终态、连接deadline，或者150秒固定暂停上限到达，则记录`not_triggered`。没有第二个候选、不延长budget、不强制禁EOS；OS网络缓冲可能吸收全部输出，模型也可能提前结束，因此不保证触发。

恢复后保留所有完整收到的UTF-8 SSE帧，核对原先两帧前缀不变、所有普通帧request ID/created/model一致、顺序为role→content→恰好一个`slow_consumer` error→恰好一个DONE→EOF。error帧本身没有ID，按同一socket及普通帧ID关联。必须有同ID唯一overflow、model cancelled/decode、旧格式cancelled/decode日志与`terminal_sent`关闭，终态仍为slowConsumer、没有保留额度，健康状态资源清空，logger无drop/write failure。随后只运行fresh AR4和MTP4两个短请求，按已有冻结的数字任务验证`1,2,`、length和完整usage 35/4/39，并核对各自生命周期日志。合计最多3个模型请求。

**范围**：没有事件序号/token IDs或全部enqueue计数，本次不能独立证明服务端曾接受的每个事件均无缺失；重复文本也不能提供此oracle。通过只说明收到的前缀、SSE终态与资源恢复符合以上合同。不将buffered_events跨线程快照当成应收到帧数，不将`contentProcessed`当成客户端交付。没有日志overflow证据时，暂停本身绝不构成通过。若已触发overflow但发送deadline/断连抢先导致终态不能送达，应失败并保留证据。

## CPU控制

```sh
python3 -B scripts/test_http_sse_overflow_parser.py -v
python3 -B scripts/test_http_sse_overflow_exit.py -v
```

11项控制已通过。原9项覆盖正常收到前缀+终态、重复终态/后续内容、身份/顺序、UTF-8/重复JSON键/非有限值、完整帧/截断、同ID真实日志判定、deadline/自然结束非触发、坏日志/额度/旧日志，以及冻结HTTP模板和实际token数。新增2项实际调用脚本的`main()`，以mock替代全部进程、socket、reader和已另测的日志判定：干净未触发返回2；嵌套SSE协议错误仍先完成两次fresh恢复及自有child清理，再返回1且`complete=false`。没有启动真实子进程或网络，不能算服务/GPU证据。

读取超时或协议错误不改变“未观察到 overflow”的事实，但会使运行未完成并失败；嵌套 `attempt.read_error` 不能被归类为正常未触发。CPU 测试的临时产物放在系统临时目录，结束后清理。

## 单一 controller 执行

以下命令从仓库根目录运行，作为外层 `OwnedProcessGroup` 的一个 case，指定新输出路径和空闲端口。运行前冻结脚本、fixture 和两个复用 helper 的 SHA；`manifest.json` 中的路径均为本目录输入或仓库公开路径，没有 ignored 候选依赖。脚本核对输入、tokenizer 元数据和 helper 身份，另在报告中记录运行二进制、脚本和 manifest 的 SHA。外层须先按既有流程暂停 reference，确保没有另一个模型，完成后确认整个自有组已清空再恢复 reference。

```sh
python3 -B scripts/test_http_sse_overflow.py \
  --runner .build/release/ane-runner \
  --model-dir ../qwen38-ssd/models/Qwen3.8-Flash-Next-MLX-SSD-Stream \
  --port 11240 \
  --output results/http-sse-overflow-v1/attempt.json
```

产物为`attempt.json`、`attempt.server.log`（普通文件）和`attempt.sse`（完整收到帧的原始字节）。输入不通过命令行向服务器传递，只从冻结messages构造HTTP请求。脚本继承外层自有process group，绝不setsid/daemonize/killpg；只终止自己的Popen child和自己的socket。TERM/INT/ALRM在Popen未赋值窗口只记flag，拿到句柄后再退出。工作上限360秒，finally屏蔽重复中断并TERM等待30秒、必要时KILL等待10秒，总上限约400秒（含少量文件收尾）；外层timeout应留至少415秒，强制清理grace仍至少45+10秒，不能用原先5秒wrapper抢先杀harness。

退出0表示真实overflow与所有gate通过；退出2表示没有`attempt.read_error`且正常收尾的`not_triggered`（`complete=true, passed=false`）；协议/读取/日志/恢复失败或未完成退出1。所有情况下外层controller均须完成组归属与恢复检查，不因退出2跳过恢复。若先达到固定工作上限，记录未完成，不在该次进程追加重试。
