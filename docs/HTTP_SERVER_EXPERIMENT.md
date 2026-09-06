# 本机文字 HTTP/SSE 实验服务

此入口复用已有 Swift/MLX generator 和 cooperative PD scheduler，已通过一轮真实本机 HTTP/SSE 与模型回归。它仍是实验功能，不表示已经达到上线标准，也不提供完整 coding-agent API。

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

固定 header 上限 16 KiB、待提交邮箱 8 请求、SSE 输出 256 条目（含一个最多 4 KiB 终态条目）。推理调度沿用 8 prefill / 2 ready、32768 逻辑预留 token、2 resident sequences、decodeBurst 4。请求接收期限 15 秒，单次发送无进展期限 15 秒，整个连接期限 300 秒。连接数满时直接关闭新连接，邮箱/推理队列满时返回 429；模型尚未 ready 或不可用返回 503。

每连接只处理一个 HTTP/1.1 请求，`Connection: close`。POST 必须有唯一 Content-Length；不接收 chunked、Transfer-Encoding、重复关键 header、折叠 header、重复 JSON key 或额外 pipelined 请求。SSE body 使用关闭连接作为边界，没有 chunked 编码。

## 接口合同

`GET /health` 返回顶层 `status=loading/ready/stopping/failed`，ready 时 HTTP 200，其余 503。还包含 `idle`、`active`、`active_jobs`、`pending_requests`、`queued_prefills`、`ready_decodes`、`resident_sequences`、`reserved_tokens`。这是 worker 的阶段边界缓存，网络线程不访问模型。`running_job` 仅在能观察实际活动 ID 时填写；prefill 中无法从边界采样得知时为 null，`running_job_known=false`，不猜测。判断取消清理完成时应同时检查 idle 和所有队列/资源计数，而非只看 running_job。

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

model 必须精确匹配。messages 仅允许 role/content，两者是字符串；角色为 system/user/assistant，system 只能在第一条，至少有一条 user。使用当前 no-thinking、无工具的原始 chat template。max_tokens 默认 128，AR 范围 1…4096；stream 默认 false；temperature 只能省略或为数字 0；mtp_depth 只能为 0（默认 AR）或显式实验 2（batchedScalarLinear、draft history 1024）。**mtp_depth=2 时 max_tokens 仅允许 1…256**，超过即返回 400：当前候选只回归过此输出预算范围，更长输出可能越过 MTP head 的 QSA 阈值，尚未验证。context 16384、prefill chunk 416，prompt 加 max_tokens 超预算仍由 worker 校验并拒绝。

任何其他字段都返回 400，包括 tools、tool_choice、stop、top_p、多 choice、随机采样、图片/数组 content、reasoning 字段；不静默丢弃。布尔值不能充当整数参数。因而可供文字客户端试用，但不能宣称兼容需要 tool calling 的完整 agent 流程。

流式成功响应依次发出 assistant role、零到多个 content、带 finish_reason 和 usage 的最终 chunk、`data: [DONE]`。EOS 映射为 stop，输出预算停止映射为 length。usage 来自 result 的实际 prompt/output token 计数，output 包括生成的 EOS，与本项目原始报告一致。字节经过每请求独立的 IncrementalUTF8Decoder，token 边界不会额外引入 Unicode 替换字符。非流式返回一个 JSON completion；文本与最终编码仍受 output-byte 限制，超限返回明确错误。

## 慢读、取消与关闭

onToken 只做有限的 CPU 解码/编码与入队，不等待网络。发送端一次只有一个 in-flight lease，送出后仍计入额度，实际 send callback 才释放；终态由 QwenSSEOutputBuffer 只选择一次。输出超限取消该请求，并尽可能发送已接受前缀之后的 error/[DONE]。发送失败或超时会断开并取消，不能保证死连接收到终态。非流式只缓存有界文字与一个最终响应，不建立额外发送队列。

HTTP 请求后 TCP write half-close 仍可能是合法的读响应客户端，所以 EOF 本身不当作“对方已死”。真正断连通过 NW failed、发送错误或期限处理；非流式计算途中若无法及时区分 half-close/full-close，取消可能延后。暂停客户端读取的一次测试也不保证触发应用缓冲 overflow：OS 可能容纳全部短输出，必须把“其他请求继续运行”与“应用额度超限”分别记录。

SIGINT/SIGTERM 停止监听，关闭连接并请求取消，唤醒邮箱，等待当前 tokenization / GPU / SSD 操作返回后清理 scheduler 与 MLX 状态。模型加载中也需要等待当前同步加载结束；没有从另一线程释放 GPU handle 或强行中断 kernel。

每个请求完成/取消会写不含提示词内容的日志；成功记录分别含 prefill 秒数、decode 秒数与 scheduler elapsed，不把网络发送回调称作客户端实际送达或 TTFT。

## 验证状态

2026-09-07（本机时间）release 构建通过；29 项 CPU 测试全部通过：HTTP parser / chat DTO / framing 11 项、有界输出与生命周期 10 项、增量 UTF-8 8 项。真实 loopback 回归的 19 项检查也全部通过，使用 `Qwen3.8-Flash-Next-MLX-SSD-Stream`、8 连接上限和 8192 字节输出额度。

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

仍需分开补齐以下边界，不能由上述通过结果外推：

- 当前真实请求均由 EOS 停止；真实 HTTP 输出预算停止的 length、MTP decode / verify / replay 中取消尚未覆盖。RST 与活动退出测试发生在 prefill。
- 真实应用输出 overflow、15 秒发送/接收期限、300 秒连接期限尚未触发；连接数量硬上限、邮箱满与槽位回收也未独立验证。CPU 缓冲测试不能代替真实 TCP 期限测试。
- 长期运行的内存/文件描述符增长、模型运行故障后的服务状态，以及故障发生在 SSE 已发送之后的错误终态，仍需后续回归。

当前可作为本机文字客户端的实验入口；缓存、前缀复用、认证与远程部署尚未纳入此服务。
