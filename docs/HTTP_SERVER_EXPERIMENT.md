# 本机文字 HTTP/SSE 实验服务

2026-09-08 新增并完成 [AR 前缀缓存验收](AR_PREFIX_CACHE.md)及 [function tools 闭环](HTTP_TOOL_CALLING.md)：同一 `2941a0dd…5a9a54` 二进制通过新功能40项检查、9次推理及另一个服务进程的19项旧HTTP回归。HTTP默认512 MiB/8条快照；MTP保持冷prefill。下文旧版soak/背压数据仍绑定各自版本，不表示本轮已重跑这些较长套件。

此入口复用已有 Swift/MLX generator 和 cooperative PD scheduler，已通过真实本机 HTTP/SSE、补充网络边界及固定12轮短测。2026-09-07的c930版本还修复了非流式文本超限归因，并实测到HTTP500 / output_limit及后续恢复；各版本证据分列如下。当前b039版本已接入有界异步诊断日志，43项Swift CPU、11项Python控制及五组真实服务回归（19/15/46/6/3项）全部通过，包括真实未读满stderr管道下的健康检查、AR/MTP及退出。同一b039二进制随后单独完成一次真实AR SSE缓冲溢出与恢复，4项检查通过；各版本和运行窗口分列，不累计成同一批。它仍是实验功能，不表示已经达到上线标准，也不提供完整 coding-agent API。

```bash
.build/release/ane-runner serve-gpu --model-dir /absolute/path/to/model
```

默认只绑定 `127.0.0.1:11236`，没有外网 host 选项。模型 ID 是规范化模型目录的 basename，可从 `GET /v1/models` 读取。只有一个固定 OS 推理线程创建/使用/释放 tokenizer、model、generator 和 scheduler；Network 的连接、收发、超时在独立串行队列处理。每个 GPU slice 之间最多处理一个待 tokenization/admission 请求。

| 参数 | 默认值 | 范围 |
| --- | --- | --- |
| `--model-dir` | 必需 | 模型目录绝对或相对路径 |
| `--port` | 11236 | 1024…65535 |
| `--max-connections` | 8 | 1…32 |
| `--max-body-bytes` | 262144 | 1024…1048576 |
| `--output-buffer-bytes` | 65536 | 8192…1048576 |
| `--prefix-cache-bytes` | 536870912 | 0…8589934592；0关闭 |
| `--prefix-cache-entries` | 8 | 1…256 |

固定 header 上限 16 KiB、待提交邮箱 8 请求、SSE 输出 256 条目（含一个最多 4 KiB 终态条目）。推理调度沿用 8 prefill / 2 ready、32768 逻辑预留 token、2 resident sequences、decodeBurst 4。请求接收期限 15 秒，单次发送无进展期限 15 秒，整个连接期限 300 秒。连接数满时直接关闭新连接，邮箱/推理队列满时返回 429；模型尚未 ready 或不可用返回 503。

每连接只处理一个 HTTP/1.1 请求，`Connection: close`。POST 必须有唯一 Content-Length；不接收 chunked、Transfer-Encoding、重复关键 header、折叠 header、重复 JSON key 或额外 pipelined 请求。SSE body 使用关闭连接作为边界，没有 chunked 编码。

## 接口合同

`GET /health` 返回顶层 `status=loading/ready/stopping/failed`，ready 时 HTTP 200，其余 503。还包含 `idle`、`active`、`active_jobs`、`pending_requests`、`queued_prefills`、`ready_decodes`、`resident_sequences`、`reserved_tokens`。这是 worker 的阶段边界缓存，网络线程不访问模型。`running_job` 仅在能观察实际活动 ID 时填写；prefill 中无法从边界采样得知时为 null，`running_job_known=false`，不猜测。判断取消清理完成时应同时检查 idle 和所有队列/资源计数，而非只看 running_job。

当前还返回独立的`logging`快照：`buffered_bytes/buffered_events/queued_events/in_flight_bytes`及固定额度，累计`enqueued_events/written_events/dropped_events/dropped_bytes/write_failures`、`last_write_errno`、`accepting/writer_exited`。这些是诊断日志状态，不是模型响应字节或显存计数。日志器固定最多65536字节、128条记录，单条最多4096字节；正在写入的记录仍计入额度。`--output-buffer-bytes`不调整这个日志额度。

`GET /v1/models` 返回一个本机模型。`POST /v1/chat/completions` 支持：

```json
{
  "model": "EXACT_MODEL_DIRECTORY_BASENAME",
  "messages": [{"role": "user", "content": "用两句话介绍太阳。"}],
  "max_tokens": 128,
  "stream": true,
  "temperature": 0,
  "mtp_depth": 0
}
```

model 必须精确匹配。基本 messages 使用字符串 role/content；角色为 system/user/assistant/tool，system 只能在第一条，至少有一条 user。携带 tool_calls 的 assistant 可用 null content，tool 需按顺序提供匹配的 tool_call_id。使用 no-thinking 模板；function tools 与调用历史的详细合同见[工具协议](HTTP_TOOL_CALLING.md)。max_tokens 默认128，AR范围1…4096；stream默认false；temperature只能省略或为数字0；mtp_depth只能为0（默认AR）或显式实验2（batchedScalarLinear、draft history1024）。**工具请求必须mtp_depth=0；纯文本mtp_depth=2的max_tokens仅允许1…256**，更长输出尚未验证。context16384、prefill chunk416，prompt加max_tokens仍按完整逻辑长度验证，不因缓存命中缩小预算。

tools支持function定义，tool_choice支持auto/none。required、指定函数及strict=true明确拒绝。stop、top_p、多choice、随机采样、图片/数组content、reasoning等其他字段仍返回400，不静默丢弃；布尔值不能充当整数。已支持工具闭环不等于完整agent协议兼容。

流式成功响应依次发出 assistant role、零到多个 content、带 finish_reason 和 usage 的最终 chunk、`data: [DONE]`。EOS 映射为 stop，输出预算停止映射为 length。usage 来自 result 的实际 prompt/output token 计数，output 包括生成的 EOS，与本项目原始报告一致。字节经过每请求独立的 IncrementalUTF8Decoder，token 边界不会额外引入 Unicode 替换字符。非流式返回一个 JSON completion；文本与最终编码仍受 output-byte 限制，超限返回明确错误。

工具请求还可产生结构化tool_calls；每个调用完整解析验证后才发送，成功终态为tool_calls。缓存命中时usage新增prompt_tokens_details.cached_tokens，完整prompt_tokens不变。health新增prefix_cache统计，缓存payload额度不包含请求私有副本及MLX allocator；详情见[缓存合同](AR_PREFIX_CACHE.md)。

## 慢读、取消与关闭

onToken常规路径做有限的CPU解码/编码与入队，不等待网络发送确认。当前推理和网络线程只向独立日志器有界入队，stderr写入由一个固定线程承担；本轮已实测满且未读的自有pipe下健康检查、短AR/MTP和SIGTERM仍能完成。发送端一次只有一个 in-flight lease，送出后仍计入额度，实际 send callback 才释放；终态由 QwenSSEOutputBuffer 只选择一次。SSE缓冲overflow请求取消，并在连接可发送时保留已接受前缀，再发error/[DONE]；非流式文本超限终止生成，返回HTTP500/output_limit，不发送正文前缀。发送失败或超时会断开并取消，不能保证死连接收到终态。非流式只缓存有界文字与一个最终响应，不建立额外发送队列。

HTTP 请求后 TCP write half-close 仍可能是合法的读响应客户端，所以 EOF 本身不当作“对方已死”。真正断连通过 NW failed、发送错误或期限处理；非流式计算途中若无法及时区分 half-close/full-close，取消可能延后。暂停客户端读取的一次测试也不保证触发应用缓冲 overflow：OS 可能容纳全部短输出，必须把“其他请求继续运行”与“应用额度超限”分别记录。

SIGINT/SIGTERM 停止监听，关闭连接并请求取消，唤醒邮箱，等待当前 tokenization / GPU / SSD 操作返回后清理 scheduler 与 MLX 状态。模型加载中也需要等待当前同步加载结束；没有从另一线程释放 GPU handle 或强行中断 kernel。日志器停止接收并丢弃排队记录，退出不join可能阻塞的写线程；最多一条in-flight记录继续保留至写入返回或进程退出。因此退出时的诊断日志可能不完整，也不承诺刷盘或可靠投递。已运行服务的失败保持非零退出，避免再进入通用CLI的同步stderr错误输出；服务初始化之前的通用错误路径仍不在满pipe运行期活性合同内。

c930增加`qwen-http-lifecycle-v1`结构化日志，将model_terminal、output_terminal和connection_close分开；按PID、请求ID、连接ID关联，不记录提示词、正文或token内容。旧`HTTP request id=...`仍保留为同一模型终态的兼容表示，不能与新model_terminal相加计数。output_terminal只代表缓冲首次选定结果；断连或准入前拒绝不保证出现这个事件，模型完成也不保证最终编码/发送成功。prefill、decode与scheduler elapsed分别记录，不把网络确认当客户端实际读取或TTFT。

## 验证状态

已完成的c930版本证据为36项Swift CPU、6项Python日志解析控制、19项live、15项edges、46项soak及6项终态检查；真实生成text_limit已覆盖，当时同步日志满pipe、SSE overflow和发送/连接期限未覆盖。当前b039有界日志版本已完成release构建、43项Swift CPU、11项Python控制及19/15/46/6/3项五组实模回归；之后独立AR SSE溢出窗口另通过4项检查，两次冻结身份和参考服务恢复分别核对。下方分别保留f955、c930历史及b039当前状态，不混合各版本计数。

### 早期f955服务基线

2026-09-07（本机时间）release构建通过；29项CPU测试全部通过：HTTP parser / chat DTO / framing 11 项、有界输出与生命周期 10 项、增量 UTF-8 8 项。真实 loopback 回归的 19 项检查也全部通过，使用 `Qwen3.8-Flash-Next-MLX-SSD-Stream`、8 连接上限和 8192 字节输出额度。

| 实测范围 | 结果与边界 |
| --- | --- |
| ready、模型身份与 5 类参数拒绝 | 模型名一致；错误 model、非零 temperature、tools、布尔 max_tokens、字符串 stream 均返回 400。 |
| 长提示词 AR SSE / MTP2 非流式 | 同一 11216-token prompt、128-token 输出预算，实际生成 85 token 后 EOS；文本、实际 usage、stop 原因与冻结的直接生成结果一致。HTTP 不暴露 token IDs，因此这里验证的是文本与计数，不能称为 HTTP token-ID 逐项对照。 |
| 中文与 emoji | AR 非流式与 MTP2 SSE 文本一致；实际 prompt 31 / output 9 token，未产生额外 Unicode 替换字符。 |
| 两请求并发与清理 | 长、短请求均正确结束，随后所有队列与资源计数归零。该轮不单独证明短请求抢占已在执行的长请求。 |
| 逻辑 token 预留过载 | 前两份长请求共预留 22688 token；第三份会达到 34032，超过 32768，收到 429 / queue_full。取消后恢复 idle，预留与 resident 均归零。此项不是连接数或邮箱上限测试。 |
| half-close、RST 与后续请求 | 合法 write half-close 仍收到完整响应；prefill 中 RST 后队列与资源清空，新请求文本正确。该次观测约 0.42 秒完成取消清理，不构成延迟上限保证。 |
| 暂停读取 | 一份已活动的长 SSE 请求暂停读取后，另一短请求仍正确完成；没有观察到应用输出缓冲 overflow，不能据此声称已覆盖慢读超限。 |
| 活动请求期间 SIGTERM | 确认存在活动 prefill 与资源预留后退出；服务在 30 秒等待窗口内以 0 退出，无强制 kill。 |

复现入口为 `scripts/test_http_server_live.py`；本机原始记录在 `results/http-service-v1/live.json`、`live.server.log` 与 `run-ledger.json`，这些运行产物不随源码提交。冻结对照为 `results/mtp-agent-expansion-v1/tools-128.json` 的首轮结果。服务二进制 SHA-256：`f95565cb2bcb32b494c2fe9d3a6397f69c01e4433a761c7d910887ff874dd4be`。这轮用于正确性和生命周期验收，运行顺序与负载没有为性能比较设计，不据其中耗时宣布 AR/MTP 加速比例。

### 早期f955补充网络边界回归

同日补充的 15 项检查全部通过，使用与首轮相同 SHA-256 的服务二进制；这一轮设置 `--max-connections 4`，仍为 8192 字节输出额度。独立脚本 `scripts/test_http_server_edges.py` 只启动、关闭自己的一份服务并管理自己的连接，SIGTERM/KeyboardInterrupt 进入 finally 清理；外层 controller 统一安排 GPU 窗口和参考服务恢复。

| 补充实测范围 | 结果与边界 |
| --- | --- |
| 连接数量硬上限与槽位恢复 | 4 条未完成 header 占位期间，第 5 条连接收到 RST；其后完成原 4 条请求，全部得到 200，观测连接数依次为 4、3、2、1。饱和期间没有另开 health 探测干扰计数；最终新 health 成功且所有任务/资源计数为零。这项不验证邮箱容量。 |
| header/body 接收期限 | 未完成 header 与未完成 body 分别约 15.185 秒收到 408 / request_timeout，随后仍为 idle，无任务或资源预留。这是接收期限测试，不是发送期限测试。 |
| Content-Length 截断与 EOF | 声明 body 为 2 字节，只发送 1 字节后 write half-close，收到 400 / Incomplete HTTP request；未进入推理调度。与上一轮完整请求 half-close 可正常读响应的结果互补。 |
| 真实输出预算停止 | AR 非流式和 MTP2 SSE 的输出预算分别为 1、2、4；文本对应 `1`、`1,`、`1,2,`，均为 length。每份 prompt 为 35 token，实际 completion 计数等于预算、total 等于两者之和，AR/MTP 文本和 usage 一致。预算 1 只验证首 token 的结束处理，不能作为执行了 MTP round 的证据。 |
| 未验证 MTP 范围拒绝 | `mtp_depth=2, max_tokens=257` 在真实 HTTP 返回 400，维持当前最高 256-token 输出预算的候选范围。 |
| MTP decode 期间 RST | 同一长工具目录 fixture、MTP2、256-token 预算，观察到两条非空 content（`{` 和换行）后 RST。唯一响应 ID 对应日志 `terminal=cancelled stage=decode`；约 0.109 秒后 health 观察到所有队列与资源归零，后续新 AR 请求的预算 4 文本、usage 和 length 与此前一致。首 token 可来自 prefill，因此等待两条；本项没有定位到 verify/replay 内部的取消点，也不构成取消延迟上限保证。 |
| 独占执行与退出 | 自有服务以 0 退出，SIGTERM 清理完成、未强制 kill；外层 ledger 确认参考服务恢复 ready。 |

原始记录在 `results/http-service-edges-v1/edges.json`、`edges.server.log` 与 `run-ledger.json`。报告保留了取消请求原文、观察到的完整 SSE 帧、唯一请求 ID、匹配日志及清理后的计数；脚本与 runner 的 SHA-256 均已独立核对。该轮没有交错重复或冷热控制，不从短请求的耗时推断性能收益。

### 早期f955固定12轮短时持续回归

同日 `scripts/test_http_server_soak.py` 完成预先固定的 **12 轮、46 项检查，全部通过**；总脚本时间 808.770 秒（约 13 分 29 秒，含加载、基线和退出），未重试或因中途结果调整轮数。仍使用同一服务二进制、4 连接上限和 8192 字节输出额度。脚本设置 1150 秒工作期限，为自有服务的有界清理预留时间，总预算 1200 秒。

每轮用冻结的 11216-token tools prompt、MTP2、128-token 输出预算，读到至少两条非空 SSE content 后 RST。随后同 prompt、同预算的新 MTP SSE 请求必须与初始基线的文本、实际 usage、stop 原因全部一致，并穿插短中文请求；第 3、6、9、12 轮再运行完整 AR 对照。初始 MTP、AR 基线均独立匹配冻结结果：自然输出 85 token 后 EOS。HTTP 不暴露 token IDs，以上一致性仍限于文本、计数和结束原因。

| 短测证据 | 实际结果 |
| --- | --- |
| 生成请求及终态 | 日志逐条对应 **31 个自然 EOS 完成、12 个主动取消**的生成请求，合计 43 个；health 请求另计。12 个取消 ID 互不重复，每个 ID 都唯一匹配 `terminal=cancelled stage=decode`。 |
| 取消后的恢复 | 12 份新 MTP、12 份短请求和 4 份追加完整 AR 对照均与各自基线一致；每轮取消后及轮末的 idle 记录中，active、jobs、prefill/ready 队列、pending、resident 和 reserved token 计数全部为零。 |
| 取消到首次 idle 观测 | 最小 / 中位 / 最大值为 **0.104675 / 0.108646 / 0.114811 秒**；这是 RST 到 health 确认的观察值，受约 100 ms 轮询粒度影响，不是 kernel 内部中断耗时或取消延迟上限。 |
| 服务退出与参考恢复 | 自有服务 SIGTERM 后以 0 退出，未强制 kill；外层 controller 同样以 0 结束，ledger 确认参考服务恢复 ready。 |

仅在空闲基线和第 3、6、9、12 轮末，读取**同一个自有服务进程**的外部 RSS 与数字文件描述符计数：

| 已完成轮数 | RSS（KiB，`ps rss`） | 数字 FD 数（`lsof`） |
| --- | ---: | ---: |
| 0（基线） | 30056336 | 11 |
| 3 | 30057408 | 11 |
| 6 | 30057536 | 11 |
| 9 | 30057744 | 11 |
| 12 | 30057888 | 11 |

RSS 末值比基线增加 **1552 KiB，约 1.516 MiB**；五次数字 FD 计数均为 11。`lsof` 不把 cwd、txt 或内存映射条目算成数字 FD。这些是五次外部进程采样，**不是 MLX active memory 或 macOS physical footprint，也不能证明没有泄漏或已稳定运行 8 小时**。

原始记录在 `results/http-service-soak-v1/soak.json`、`soak.server.log` 与 `run-ledger.json`，包含每轮完整结果、取消前的 SSE 帧、唯一请求 ID、匹配日志、恢复观察时间及空闲计数。脚本、复用的 edge helper、runner、冻结对照的 SHA-256，以及实际 fixture 文本均已独立核对。这一轮的结论限于固定 12 轮短时持续回归。

### c930输出原因与终态回归

MTP双窗口结束并解除旧冻结后，加入每请求`QwenHTTPTextBudget`、明确关闭原因和结构化终态记录；不改变模型数值路径。新服务二进制SHA-256为`c93011f804dd287a7758568d69373089959a0b92408379390ea71d7294b0568a`。release构建、36项相关Swift CPU（原29项加7项文本限额测试）与6项Python日志解析控制通过。CPU解析还确认旧43-request日志可读；这不是新schema或新的模型请求证据。

| c930实测 | 结果与原始记录 |
| --- | --- |
| live 19项 | 全部通过；[`v1/live.json`](../results/http-output-fix-regression-v1/live.json)保留长AR SSE/MTP非流式、Unicode、并发、配额、half-close、RST、后续恢复与活动shutdown。paused-reader仍未触发应用overflow。 |
| edges 15项 | 全部通过；[`v2/edges.json`](../results/http-output-fix-regression-v2/edges.json)保留4连接硬上限、header/body接收期限、截断请求、AR/MTP的1/2/4输出预算、MTP257拒绝、两条非空content后的唯一ID decode取消和fresh恢复。实际header/body接收期限约15.343/15.344秒，不是发送期限。 |
| 12轮soak 46项 | 全部通过；[`v2/soak.json`](../results/http-output-fix-regression-v2/soak.json)实际797.792412416秒，未追加替换轮。31个正常完成、12个decode RST取消，每轮fresh MTP/短请求与第3/6/9/12轮AR均匹配文本、usage、stop；每次恢复和轮末所有任务/资源计数归零。 |
| 终态6项、8请求 | 全部通过；[`v2/terminal-logs.json`](../results/http-output-fix-regression-v2/terminal-logs.json)逐ID关联6个正常length完成、1个decode RST取消、1个真实非流式text_limit失败。超限收到HTTP500/output_limit后，再次AR/MTP均恢复。 |

首次v1的edges在服务加载前因`Address already in use`停止，失败记录仍保留于`v1/edges.json`和ledger；续批使用各自独立端口执行edges、soak、terminal三项，不把首次失败抹掉或算作通过。三份自有服务均以0退出，控制器清理自有进程组并恢复参考PID24434，UTC01:46:09.163439（北京时间09:46:09.163439）确认ready；90文件postflight、精确argv、listener/meta/idle后续核对通过，见[`恢复核对`](../results/http-output-fix-regression-v2/restoration-verification.json)与[`身份核对`](../results/http-output-fix-regression-v2/postflight-identity.json)。该版本冻结已结束；历史结果仍绑定c930。

此次12轮soak的五次空闲进程采样如下，必须与早期f955表分开：

| 已完成轮数 | RSS（KiB） | 数字FD数 |
| --- | ---: | ---: |
| 0 | 21411328 | 11 |
| 3 | 21412048 | 11 |
| 6 | 21412336 | 11 |
| 9 | 21412720 | 11 |
| 12 | 21412672 | 11 |

末值增加1344 KiB（1.3125 MiB），采样峰值比基线增加1392 KiB；FD五次均11。12次取消到首次idle观测的最小/中位/最大为0.004207/0.108743/0.114620秒，只是受轮询时点影响的观察量，不是kernel中断时间或延迟保证。五个外部RSS样本不证明没有泄漏或稳定运行八小时。

#### 新日志实际记录的边界

| c930记录 | 旧兼容model日志 | 新model_terminal | 新output_terminal | 新connection_close |
| --- | ---: | ---: | ---: | ---: |
| live | 14 | 14 | 9 | 171 |
| edges | 8 | 8 | 7 | 135 |
| soak | 43 | 43 | 31 | 197 |
| terminal | 8 | 8 | 7 | 135 |

soak旧43条与新model43条对应同一组31完成+12取消，不能算86个请求；connection_close还包含health和参数拒绝连接。此次每个正常请求只选一次output终态，记录时尚有待发送数据；断连先选disconnected，迟到scheduler cancelled保留它，通常没有output_terminal。此次没有closed_send_released实例，真实晚到lease确认仍不算已覆盖。旧43-request格式兼容控制也与上述新43-request实测分开。

真实超限请求使用AR、非流式、4096-token预算、8192字节总输出额度。43.664788375秒后客户端收到HTTP500/output_limit；同一请求记录`model_terminal failed/decode reason=text_limit`，随后`output_terminal failed error_code=output_limit text_bytes=6144 text_limit_bytes=6144`，最后`connection_close terminal_sent`且额度归零。6144是已接受字节，跨限的下一段被拒绝。本次命中的是**生成时累计文本限制**，不是final JSON/header编码超限、最后UTF-8 flush跨限、SSE slow_consumer或模型数值错误。后续AR非流式与MTP2 SSE均返回`1,2,`、prompt35/output4/total39、length。更完整的来源、first-terminal-wins与未覆盖分支见[输出边界](HTTP_OUTPUT_BOUNDARIES.md)。

仍需分开补齐以下边界，不能由上述通过结果外推：

- c930在共享锁内同步写stderr，满且无人读取的pipe可阻塞网络或推理日志路径；当时的普通文件回归没有覆盖它。后续b039已替换此实现并通过下述真实满pipe活性回归；这不改变c930当时的覆盖范围。
- c930这批尚未触发真实SSE应用缓冲overflow；后续b039独立AR窗口已补齐下述一次溢出与恢复。15秒发送期限、300秒连接期限、晚到send确认仍未触发；邮箱满与槽位回收也未独立验证。非流式text_limit与SSE slow_consumer分开登记。
- MTP verify/replay内部的精确取消位置尚未覆盖；已有证据限于decode内容后的网络断连、取消及后续恢复。
- 长期内存/FD增长、模型运行故障后的服务状态，以及SSE已经发送之后的生成错误终态，仍需后续回归。

### b039有界日志：CPU与五组实模回归通过

当前[QwenHTTPLogger](../Sources/ANERunnerCore/QwenHTTPLogger.swift)用一个固定OS线程写stderr；NSCondition只保护有界FIFO和计数，IO期间不持该锁。完整或部分写入完成之前，整条in-flight记录都占用字节和事件额度。超限记录整条丢弃，不截成损坏JSON；写入失败记有限errno并关闭日志器、丢弃余下排队记录。计数饱和而不回绕。日志丢弃不会取消推理，也不等于SSE的`slow_consumer`或非流式的`output_limit`。

`serve-gpu`在自己的一次性CLI进程入口显式将SIGPIPE设为SIG_IGN，并保持到进程退出；它不修改调用父进程、其他应用或共享stderr描述符标志。Core默认stderr sink仅用`sigaction`只读核验该前提，不隐式设置进程信号策略；拥有者必须在logger生命周期内保持它。忽略SIGPIPE让断管返回可统计的EPIPE，**不会让满管道写入变成非阻塞**，所以仍需要固定writer与不join的退出路径。

这次并非一次通过，原始失败均保留：

1. 首个异步logger使用writer线程级SIGPIPE屏蔽，release构建后，真实closed-pipe CPU测试使xctest被signal13终止，见[`v1/cpu-tests.log`](../results/http-async-logger-regression-v1/cpu-tests.log)。Apple公开XNU的EPIPE路径向进程发`psignal`，线程mask不足；`F_SETNOSIGPIPE`又作用于dup共享的fileglob，不能当成私有描述符隔离修复，见[输出边界中的源码依据](HTTP_OUTPUT_BOUNDARIES.md#诊断日志的背压与信号策略)。
2. 改成显式服务进程策略后，本机Swift SDK将限定名`Darwin.sigaction(...)`解析为struct构造器，首次编译失败，见[`v2/build.log`](../results/http-async-logger-regression-v2/build.log)。只将3个文件7处调用改为未限定的`sigaction(...)`，保留候选及首失败日志，见[`import-fix.json`](../results/http-async-logger-regression-v2/import-fix.json)。
3. 导入修正后的release构建51.66秒通过，当前二进制SHA-256为`b0391391b4af4dbdcd31bb16cffbf268e790112129e4887c3ac84ef547be1bc1`。见[`build-fixed-import.log`](../results/http-async-logger-regression-v2/build-fixed-import.log)。[43项Swift CPU](../results/http-async-logger-regression-v2/cpu-tests.log)及[针对生产解析函数的11项Python控制](../results/http-async-logger-regression-v2/python-production-tests.log)全部通过。

五项服务回归postflight后，仅将Python包装器的测试集入口从ignored历史副本改成生产`scripts/test_http_terminal_log_validator.py`；两个测试集内容SHA一致。在只含Package.swift和5个生产脚本、没有results或模型的临时目录中，[同一组11项控制再次通过](../results/http-async-logger-regression-v2/python-clean-layout-tests.log)，见[路径修正记录](../results/http-async-logger-regression-v2/logger-wrapper-path-fix.json)。这解决了该CPU入口的复跑依赖，不能把两次执行计为22项；Swift二进制不变，原冻结清单保留修正前SHA。

43项Swift包含原36项与7项logger测试：真实断管使用同一descriptor write路径，验证EPIPE、调用线程mask和原FD标志不变，并在writer退出后恢复测试进程原信号策略；另测默认sink拒绝未配置策略。其余logger测试用受控CPU sink检查in-flight额度、固定writer、失败计数与producer/stop不等待IO。它们不代替真实服务满pipe验收。

异步日志可能晚于HTTP响应或health快照出现。edges/soak对原有请求ID最多等待10秒，且每次等待与原有最后idle快照均要求累计drops/write_failures为零；终态脚本同样检查。未知计数、任何丢弃或写入失败都不能宣称日志归因完整。正常写入也只是诊断记录写给stderr，不是客户端收到模型响应。关闭可能只留下connection_close，仍不能因没有output_terminal就认定丢日志。

本轮[`冻结计划`](../results/http-async-logger-regression-v2/plan.json)依次运行以下五个自有服务进程；每份最终报告均complete/passed/graceful_shutdown为true、退出码0。普通文件日志与未读pipe分别验收。

| 本轮报告 | 检查数 | 实际范围 |
| --- | ---: | --- |
| [live](../results/http-async-logger-regression-v2/live.json) | 19/19 | 长提示词AR/MTP、Unicode、拒绝、并发、断连/half-close及活动请求SIGTERM；实际SSE overflow仍未命中。 |
| [edges](../results/http-async-logger-regression-v2/edges.json) | 15/15 | 1/2/4-token预算、接收期限、连接上限、MTP decode RST取消与新请求恢复。 |
| [soak](../results/http-async-logger-regression-v2/soak.json) | 46/46 | 固定12轮，每轮MTP decode取消、资源归零及新请求；每3轮完整AR对照。 |
| [terminal-logs](../results/http-async-logger-regression-v2/terminal-logs.json) | 6/6 | 8个HTTP请求ID逐一对应结构化/旧终态；实际非流式text_limit及AR/MTP恢复。 |
| [pipe](../results/http-async-logger-regression-v2/pipe.json) | 3/3 | 未读stderr满管道下的health、真实AR/MTP、idle和SIGTERM活性。 |

普通文件原始日志中，live/edges/soak/terminal分别有14/8/43/8条model_terminal，且与旧兼容行ID一一对应。soak的43条是31完成和12个不同ID的decode取消；31份完成响应文本、usage及stop与对应基线一致，长提示词基线仍为prompt11216/output85/total11301。每轮取消后及轮末队列、resident和reserved tokens归零。edges/soak/terminal最后idle快照分别为written/enqueued 184/184、337/337、170/170，保留字节/事件/in-flight均为0，累计drops/write_failures为0。live保存的最后日志快照是SIGTERM前的活动请求状态（215/215、零loss），不冒称它是退出后的最终idle快照。

此次soak总观察窗口850.750708709秒，在第0/3/6/9/12轮的ps RSS分别为26037200/26037152/26037296/26037344/26037520 KiB，末次比基线多320 KiB（0.3125 MiB）；FD均为11。取消到首次观察idle的min/median/max为0.005001/0.107982/0.110143秒，是轮询观测，不能当作kernel内部取消延迟。五个外部idle样本不证明无泄漏或长期稳定，RSS也不等于MLX分配或physical footprint。

本轮真实AR非流式超限在51.953498875秒返回HTTP500/output_limit，同ID日志为failed/decode/text_limit，已接受6144字节，终态发送后额度归零；新AR非流式和MTP2 SSE均恢复为`1,2,`、prompt35/output4/total39、length。命中阶段仍是生成时累计文本检查，不是最后UTF-8 flush或最终响应编码。

pipe项的自有stderr读端直到进程退出前保持打开且不读取，stdout独立普通文件。161次填充health后日志达到128条事件额度；连续3个额外health样本均显示管道未读65536字节、written固定165、in-flight419字节、总保留52926字节，drops从2升到3再到4。真实AR/MTP分别完成后队列/模型额度归零，日志仍堵塞，累计丢弃13条/6581字节、write_failures为0。两份响应均为`1,2,`、prompt35/output4/total39、length；随后SIGTERM在0.7258455秒内正常退出，父Python的SIGPIPE策略不变。退出后才捕获65536字节stderr，原始165行没有任何请求终态；这与报告明确的`lifecycle_attribution_complete=false`一致，不从缺失诊断记录推断完整归因。详见[输出边界](HTTP_OUTPUT_BOUNDARIES.md#b039实测诊断归因与满管道活性)。

[postflight](../results/http-async-logger-regression-v2/postflight-and-release.json)核对133份冻结文件、102份模型payload及二进制身份，控制器五项均退出0；[ledger](../results/http-async-logger-regression-v2/run-ledger.json)记录参考服务按原完整argv恢复为PID31650，127.0.0.1:11235监听归属、meta及running/waiting为0均已核对。冻结解除后才执行上述Python包装路径修正。这五项当时没有触发SSE溢出，之后由下述独立窗口补测；15秒发送期限、300秒连接期限和晚到send确认仍未覆盖。

### b039独立SSE溢出：一次真实AR窗口通过

2026-09-07，沿用同一b039二进制，独立[`计划`](../results/http-sse-overflow-public-v1/plan.json)只做一次真实AR SSE尝试：冻结prompt4039 token、输出预算4096 token、输出额度8192字节、4连接，客户端实际SO_RCVBUF为1024字节。先收到两条非空content帧后暂停读取，观察到服务选择slow_consumer才恢复读取；未加入推理延迟或伪造GPU输出。[报告](../results/http-sse-overflow-public-v1/attempt.json)的4项检查全部通过，服务正常退出0。

[原始SSE](../results/http-sse-overflow-public-v1/attempt.sse)共480541字节：一条role、2052条非空content、一条slow_consumer错误和唯一`[DONE]`，随后正常EOF。暂停60.267929166秒；暂停前已收到的702字节（role及两条content）在最终记录中逐字节一致，完整wire/text SHA与报告匹配。2052是内容帧数，不是token数；错误流没有正常finish/usage，不能推导实际生成token总数，也未将每条成功enqueue与收到的内容逐项对照。

唯一请求`chatcmpl-7b88d73e-3f24-48d6-bbea-4186dd34ac64`在[原始服务日志](../results/http-sse-overflow-public-v1/attempt.server.log)中先选择`output_outcome=slowConsumer`、reason/error_code=slow_consumer并请求取消；记录时保留4117字节/18事件，其中234字节in-flight。随后model_terminal为cancelled/decode，最终connection_close为terminal_sent，保持slowConsumer，buffered/in-flight/事件全部归零。旧取消日志与同一模型事件一致。超限日志快照是在选定错误终态之后读取，不能用4117字节反推越界瞬间或每个enqueue的交付情况。

恢复后fresh AR非流式和MTP2 SSE均返回`1,2,`、prompt35/output4/total39、length；最后idle的请求与模型额度归零，logging written/enqueued为141/141、零丢弃/写入失败、无保留记录。此次覆盖的是**AR SSE应用输出限额、已收到前缀的保持、错误尾帧、decode取消和后续恢复**，不是MTP SSE溢出、完整已入队前缀的交付证明、长期背压或性能测量。

[postflight](../results/http-sse-overflow-public-v1/postflight-and-release.json)核对141份冻结文件、102模型payload及同一二进制；[ledger](../results/http-sse-overflow-public-v1/run-ledger.json)确认自有进程组清空，参考服务按原完整argv恢复为PID35281，11235监听归属和idle已核对。本次没有send_deadline、connection_deadline或closed_send_released记录；15/300秒期限及晚到确认仍未覆盖。更细口径见[输出边界](HTTP_OUTPUT_BOUNDARIES.md#b039独立ar-sse溢出实测)。

当前可作为本机文字客户端的实验入口；缓存、前缀复用、认证与远程部署尚未纳入此服务。这些HTTP结果不替代MTP双窗口性能门槛，也不是性能对比或生产发布通过。
