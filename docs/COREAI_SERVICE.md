# CoreAI 文本服务试用

`coreai-runner serve` 为独立 CoreAI runner 提供有限的 OpenAI 兼容 HTTP 接口。全部神经网络计算使用系统 CoreAI；CPU 负责 tokenizer、SSD n-gram 行读取和解码、greedy 选词及网络。可执行目标仅依赖 `ANERunnerCore`，没有 MLX fallback。模型目录名中的 `MLX` 是源资产名称，不表示服务调用 MLX。

当前面向本机和局域网试用：macOS 27、完整模型资产、**4096-token 导出容量、单请求串行推理**。原组件路径使用 S1 逐 token prefill；可选的[本机 PD 路径](COREAI_PD.md)提供共享权重的 S4 prefill/S1 decode 独立函数。容量来自选用的 manifest，不是启动参数；4096 资产完整不等于完整模型已通过 4096-token 质量与性能验收。`quality_acceptance` 和 `hardware_placement_verified` 仍为 `false`。原生计算与量化边界见 [CoreAI 原生 runner](COREAI_NATIVE.md)。

## 启动

以下命令从仓库根目录执行，需要已构建的 `.build/release/coreai-runner`、macOS 27，以及来自同一模型的完整 attention、dense、MoE 资产。原始模型目录还须包含配置、tokenizer 和 SSD n-gram 表。模型与 `results/` 资产不随 Git 克隆提供。

本工作区 4096-token attention 资产为 `results/coreai-service/attention4096/manifest.json`，复用完整的 dense 和 MoE 资产：

```sh
.build/release/coreai-runner serve \
  --model-dir ../qwen38-ssd/models/Qwen3.8-Flash-Next-MLX-SSD-Stream \
  --attention-manifest results/coreai-service/attention4096/manifest.json \
  --dense-manifest results/coreai-native/dense/manifest.json \
  --moe-manifest results/coreai-native/moe/manifest.json \
  --host 0.0.0.0 --port 11236 \
  --prefix-cache-bytes 536870912 --prefix-cache-entries 2 \
  --request-timeout-seconds 1800 --max-pending-requests 2
```

默认监听 `127.0.0.1:11236`；上例显式使用 `0.0.0.0`，其他设备应连接这台 Mac 的局域网 IP，例如 `http://<Mac局域网IP>:11236/v1`。`0.0.0.0` 是监听地址；其他设备的 `127.0.0.1` 指向设备自身。服务当前不提供 TLS 或 API key 校验。

使用本机 PD 资产时在上述命令中追加 `--pd-manifest results/coreai-pd/fused-s4-metal/manifest.json --prefill-chunk 4`。此时只加载 PD 资产，旧三份资产不会同时加载。`--prefill-chunk 1` 可强制使用该资产的 S1 入口做对照。

加载期间 HTTP listener 可以响应健康查询，但推理请求返回 503。必须等待 `/health` 的 `ready` 为 `true`。加载时会核对源目录、配置 SHA256、完整 48 层、512 专家库和资产接口；smoke 或部分层资产不能代替完整模型。

CLI 只接受唯一的 `--option value` 参数对；未知或重复选项报错。主要服务参数如下：

| 参数 | 默认值 | 范围或含义 |
| --- | ---: | --- |
| `--host` | `127.0.0.1` | 仅 `127.0.0.1` 或 `0.0.0.0` |
| `--port` | 11236 | 1024…65535 |
| `--pd-manifest` | 无 | 完整共享权重 PD 资产；省略时使用原组件路径 |
| `--prefill-chunk` | 0 | 自动选择；支持 1，或配合 PD manifest 使用 4 |
| `--prefix-cache-bytes` | 536870912 | 0…2147483648；默认 512 MiB |
| `--prefix-cache-entries` | 2 | 0…8；任一缓存额度为 0 即禁用 |
| `--max-pending-requests` | 2 | 1…8；包含当前执行和等待的请求 |
| `--max-connections` | 8 | 1…32；含健康查询等所有连接 |
| `--max-body-bytes` | 262144 | 1024…1048576；默认请求体 256 KiB |
| `--max-output-bytes` | 65536 | 4096…1048576；默认响应体 64 KiB |
| `--request-timeout-seconds` | 1800 | 1…86400；包括接收、排队和生成 |

当前一个消费任务独占完整模型；默认总共最多接纳两个推理请求，通常为一个执行、一个等待。它不是两个请求同时推理，也没有 continuous batching。达到推理准入上限返回 429；达到连接上限会直接拒绝新连接。

## 健康查询与模型名

```sh
curl -sS http://127.0.0.1:11236/health
curl -sS http://127.0.0.1:11236/v1/models
```

`GET /health` 返回 HTTP 200 的状态快照，加载失败或尚未就绪时也须检查 JSON 中的 `ready`，不能只看 HTTP 状态。主要字段：

| 字段 | 含义 |
| --- | --- |
| `ready`、`stopping`、`phase` | 就绪、停止中，以及 loading/idle/prefill/decode/stopped 阶段 |
| `capacity` | 已加载模型的上下文容量；加载完成前为 0 |
| `progressCompleted`、`progressTotal` | 当前阶段的 token 进度；idle 时归零 |
| `requests_in_flight`、`active_requests`、`queued_requests` | 已准入总数、执行中数量、排队数量 |
| `max_pending_requests` | 执行中加排队请求的总限额 |
| `completed`、`cancelled`、`failed`、`lastError` | 累计请求结果及最近错误 |
| `prefix_cache_enabled`、`prefix_cache_limit_bytes` | 缓存配置 |
| `cacheEntries`、`cacheBytes` | 最近一次请求清理后记录的缓存条目和逻辑字节数 |
| `cacheHits`、`cachedTokens` | 成功请求的累计命中次数和复用 token 数 |
| `prefillTokensProcessed`、`prefillSeconds`、`decodeSeconds` | 成功请求的累计阶段统计 |
| `backend`、`prefill_policy` | `native-coreai`，以及 `tokenwise` 或 `chunked` |
| `independent_pd_functions`、`prefill_chunk_size` | 是否加载独立 PD 函数，以及当前预填块大小 |
| `pd_scheduling` | 当前为 `serial`，没有多请求交错执行 |
| `prefill_group_milliseconds`、`decode_group_milliseconds` | 成功请求的独立阶段函数墙钟累计值，非 GPU kernel 时间 |
| `compute_preference` | 当前为 `gpu`；不证明算子实际硬件归属 |
| `quality_acceptance`、`hardware_placement_verified` | 当前均为 `false` |
| `mlx_runtime_loaded` | 当前为 `false`；worker 初始化时检查动态库 |

`GET /v1/models` 返回一个模型，ID 是实际源目录的最后一段加 `-CoreAI`。上例为 `Qwen3.8-Flash-Next-MLX-SSD-Stream-CoreAI`。它可在加载阶段返回模型名，不能代替 readiness 检查。

## 非流式对话

```sh
curl -sS http://127.0.0.1:11236/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary '{
    "model": "Qwen3.8-Flash-Next-MLX-SSD-Stream-CoreAI",
    "messages": [
      {"role": "system", "content": "请用简短中文回答。"},
      {"role": "user", "content": "17乘以3等于多少？"}
    ],
    "max_tokens": 16,
    "temperature": 0,
    "stream": false
  }'
```

成功时返回 `object: "chat.completion"`、`choices[0].message.content`、`finish_reason` 和 `usage`。正常 EOS 的结束原因为 `stop`，耗尽输出预算为 `length`。`usage.prompt_tokens` 是包含 chat template 的完整 prompt 长度；`completion_tokens` 包含生成的 EOS。缓存命中时额外返回 `usage.prompt_tokens_details.cached_tokens`，未命中时该字段省略。缓存不会缩减报告的完整 prompt token 数。

请求合同按当前实现执行：

| 字段 | 接受范围 |
| --- | --- |
| `model` | 必填，精确匹配 `/v1/models` 的 ID |
| `messages` | 非空数组，普通文本 system/user/assistant，至少一条有效 user；system 只能在第一条 |
| `content` | 纯字符串；不接受图片、音频或其他多模态内容数组，特定非文本 token 也会拒绝 |
| `max_tokens` | 默认 128，整数 1…4096；完整编码后的 prompt 加此预算必须不超过导出容量 |
| `temperature` | 省略或数字 0；当前仅 greedy，不采样 |
| `stream` | 默认 false，必须为 JSON 布尔值 |
| `mtp_depth` | 省略或 0；CoreAI worker 拒绝非零 MTP |
| `tools`、`tool_choice` | 仅空 `tools` 和省略/`none` 的 `tool_choice` 可用；非空工具、tool 消息或调用历史均拒绝 |

采用当前文本 chat template 的 no-thinking 分支。接口不接受 `reasoning_content` 等额外消息字段。顶层只识别上表字段；`top_p`、`stop`、`seed`、`n`、`stream_options`、`max_completion_tokens` 等未知字段返回 400，不能假设 SDK 默认附加参数会被忽略。重复 JSON key、无效类型及非 UTF-8 JSON 也拒绝。

HTTP transport 使用 HTTP/1.1，每条连接只处理一个请求，响应后关闭。请求需要 Host，POST 需要 Content-Length；不接受 chunked 请求、流水线请求或 Expect 协商。上述 curl 命令自动提供所需长度和 Host。

## SSE 流式对话

```sh
curl -N -sS http://127.0.0.1:11236/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary '{
    "model": "Qwen3.8-Flash-Next-MLX-SSD-Stream-CoreAI",
    "messages": [{"role": "user", "content": "用一句话介绍你自己。"}],
    "max_tokens": 32,
    "temperature": 0,
    "stream": true
  }'
```

请求校验完成后先发送 assistant role 帧，再发送 `choices[0].delta.content`。未命中的 prefill 每执行 8 个 token 发送一次 `: keep-alive` SSE 注释；这不是输出 token，也不是固定时间间隔的心跳。结束帧含 `finish_reason` 和 `usage`，最后为 `data: [DONE]`。客户端应忽略 SSE 注释，并从完整内容增量重建文本。

校验或准入失败时，尚未发送 SSE 成功响应头，返回普通 JSON 错误和相应 HTTP 状态。流已经开始后的失败以 SSE `error` 对象和 `[DONE]` 结束，HTTP 状态仍为 200；客户端必须检查事件内容。连接失效或发送超时可能直接断开，不能保证仍有终结帧。

`--max-output-bytes` 同时约束文本和编码后的整个响应体。SSE 帧、JSON 转义、usage、心跳都会占用响应预算；它不是允许生成的纯文本字节数承诺。transport 为终结帧保留 2048 字节，超限会终止请求。

## RAM 前缀缓存

默认 **512 MiB、最多 2 条、LRU 淘汰**。先对完整对话应用模板并编码，然后选择已保存条目中最长的精确 token 前缀；文本相似或语义相同不算命中。缓存命中只省去重复 prefill，不改善新 token 的 decode kernel。

保存候选是已完整计算的 system token 前缀和完整 prompt。system 边界通过独立系统块与完整编码 token 的共同前缀确定，避免把独立编码的字符串直接拼接。只有相应 token 边界执行成功才可生成快照；没有 system 时只保存 prompt。系统前缀和 prompt 各占一条，因此默认两个条目通常容纳一个系统前缀与一个最近 prompt。生成的回答后缀不另行发布为缓存条目。

每条缓存包含全部 48 层 GDN/QSA 状态与 offset、PLE 卷积状态，以及 CPU 侧的 n-gram history、对应 token IDs 和最终 logits。张量深拷贝到独立存储，restore 再复制，运行中的写入不会修改已发布快照；它不只是传统 attention K/V。快照绑定当前模型实例，不能在进程或模型实例之间交换。

创建快照前按逻辑字节预估判断预算，超预算则跳过；需要时先淘汰旧条目。`cacheBytes` 包含状态逻辑字节、token/history 和 logits，不包含模型权重、活跃会话、CoreAI 临时分配、对象开销或复制时的峰值。因此 512 MiB 是缓存记账额度，不是进程 RAM 上限。

请求取消或失败后，当前模型状态 reset；此前已经完整发布的前缀可以保留，未完整的 token 状态不能发布。服务重启会丢失全部 RAM 缓存。当前**没有 SSD KV cache/offload、持久化导入导出、前缀树或分页缓存**。SSD 上的 n-gram 权重表读取与 KV 缓存是不同功能。

## 取消、超时和错误

网络关闭、发送失败或超时会设置该请求的取消标志。推理在 token/预填块边界协作检查；已经提交的 CoreAI 调用不承诺立即中断。TCP 请求写端的正常半关闭允许服务器继续返回结果；连接真正不可读的情形依靠发送失败、连接状态或 deadline 识别。

总请求 deadline 默认 1800 秒，从连接建立开始计算，包含排队和 prefill；慢请求体接收的 deadline 固定为 15 秒，单次发送等待固定为 15 秒。超时尚能发送时返回 408，流式响应已经开始则发送错误事件；已有发送阻塞时可能直接关闭连接。网络确认取消后，尚未执行的排队请求立即从有界 FIFO 移除并释放名额；已开始执行的请求仍需等待 token 边界和 reset 才释放。唤醒流不保存请求内容，因此反复取消不会遗留占位。正常 TCP 写端半关闭不等于取消。

SIGINT/SIGTERM 停止接收新请求、关闭连接、取消请求，并等待 worker 清理。模型归单一推理任务持有，HTTP 的健康查询不进入模型执行器。

| 状态 | 典型原因 |
| --- | --- |
| 400 | 参数、模型名、文本格式、上下文预算、tools/MTP 不支持 |
| 404 / 405 | 端点不存在 / 方法不支持 |
| 408 | 请求或接收 deadline 到期 |
| 413 / 431 | 请求体 / 请求头超限；请求头固定最多 16 KiB |
| 429 | 已准入推理请求达到上限 |
| 500 | 模型执行、非有限数值或输出预算错误 |
| 503 | 模型加载中、不可用或服务正在停止 |

## 分阶段统计与当前限制

Prefill 和 decode 单独统计。health 的 `prefillSeconds` 包含缓存恢复、未命中 token 执行、SSD 读取和快照发布等 prefill 阶段开销；tokenization 在计时开始前完成。`prefillTokensProcessed` 只累计成功请求实际重算的 prompt token。完整缓存命中仍可能花费状态复制时间，不能把完整 prompt 长度除以该时间当成 prefill kernel 吞吐。

`decodeSeconds` 累计后续生成 token 的 forward 与对应 SSD 读取时间，不含全部选词、网络等待或端到端延迟。第一个输出来自最后一次 prefill logits，最后一个输出 token 不再 forward；因此 decode forward 次数通常比生成 token 数少 1。需要逐步 profile 时使用 `coreai-runner generate` 的 `prefill_seconds`、`prefill_coreai_calls`、`decode_forward_steps`、`decode_forward_seconds` 和 `decode_step_seconds`，不要把两个阶段合并成单一吞吐。

指定 PD manifest 后，prefill/decode 拥有独立函数与一次性状态交接，S4 预填可批量执行投影和路由；S1 尾部仍按业务归入 prefill 统计。当前请求调度仍串行，没有两个阶段独立部署。服务不支持 262K 上下文、工具调用、MTP 或多请求批量推理。GPU preference 不是 ANE 使用证明；更大容量、数值质量、缓存生命周期和性能均需各自验收，不能由 HTTP 可连接或能生成文本推定完成。

## 运行验证

2026-09-17，M5 Max / macOS27，以4096 attention、既有完整dense和Q4 MoE资产测试。首版验证二进制SHA256为 `5d3498b3b84f771a8b1e5703a4aab4f1866f55aec0512a50ab34ee6822adeb9a`；后续排队取消修复版另行记录于下文。

- `results/coreai-service/http-final.json`：16项通过。实际模型覆盖JSON/SSE、37-token完整缓存命中、system前缀复用、47/83不同系统消息隔离、多轮53、参数拒绝、预填中断及后续请求恢复。最终ready、在途请求0。
- `results/coreai-service/transport-cpu/repository-tool-report.json`：18项CPU fake-backend网络检查通过，覆盖限制、合法TCP半关闭、断连、慢读、deadline、连接上限和SIGTERM。这些网络时延不是实际模型性能。
- 发现并修复232,034,400字节、145张量的状态清零开销。独立相同形状测试中，原序列填充7.094/7.068s，typed pointer批量填充0.0122/0.0114s，四轮逐字节全零；实际新helper两轮0.0293/0.0125s也全零。原始证据为 `zero-benchmark.json` 与 `helper-zero-benchmark.json`。
- 完整服务复测，同一缓存命中算术请求在旧/新版本分别约9.23/0.77s；取消测试整体17.31/1.86s，越界请求8.01/0.01s。改善来自消除请求间的状态清空等待，不能称为decode kernel提速。两轮原始报告均保留。

复跑短请求及网络边界：

```sh
python3 scripts/check_coreai_service.py \
  --base-url http://127.0.0.1:11236 --output results/local-coreai-http.json

python3 scripts/check_coreai_http_transport.py \
  --port 11239 --output results/local-coreai-transport.json

python3 scripts/check_coreai_request_queue.py \
  --output results/local-coreai-queue.json
```

上述 `5d3498b3…2adeb9a` 二进制还完成了真实2064-token冷请求和完整前缀恢复：`results/coreai-service/long-acceptance.json` 的5项检查通过，两次均输出47、completion计数3（含EOS），热请求命中2064/2064。冷请求804.79s，重复请求0.857s，都是端到端HTTP耗时；没有把缓存命中折算成kernel吞吐。这个prompt跨过QSA稀疏切换边界，但不等于4096全容量或普遍长文本质量验收。

长请求已经直观显示S1首次prefill的性能限制；服务可用于功能试用，尚不适合agent长系统提示词业务。


排队取消修复版二进制为 `7edd1c57ef5d676be5f5e5934915c5cd73767ea40a56637a32c1e8b3062c0726`。修复将请求载荷移出不可撤回的stream缓冲，使用独立有界FIFO和合并唤醒；网络确认取消时，queued请求立即移除，active请求仍需reset。模型计算、attention资产和checkpoint格式未改变；上面的2064-token测试对应前一二进制，没有把它伪记成修复版重新执行。

- `results/coreai-service/http-queue-final.json`：17/17通过。新增实模A持续prefill、B尚未收到响应头就RST取消、C在满队列时429的三轮验证；每轮取消后inflight回到1、queued回到0，后续短请求输出51。最终cancelled=5（1个原active检查，加新检查的3个queued和1个active），没有遗留请求。`failed=1`来自故意越界的400检查，并非模型执行异常。
- `results/coreai-service/request-queue-cpu/report.json`：7/7通过，含5000次取消后复用、900个并发请求核账、FIFO与stop竞态。这是队列检查，不是模型并发吞吐。
- `results/coreai-service/transport-cpu/queue-version-report.json`：当前源文件的18/18 CPU网络检查复过。
- 独立CoreAI产品及原有 `ane-runner` release产品均构建成功；本轮没有运行XCTest。短请求回归后服务ready、active/queued均为0、缓存开启。

修复版另做两条普通文本人工核对（`results/coreai-service/language-examples.json`）：中文一句话解释缓存，31个prompt tokens、37个completion tokens、25.23s；Python加法函数，41个prompt tokens、18个completion tokens、17.54s，复用11个system tokens。两者均正常EOS，中文可读且回答问题，代码AST确认为 `def add(a,b): return a+b`。这是两个功能示例，不是全面代码能力或源模型质量等价评测。
