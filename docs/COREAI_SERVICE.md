# CoreAI 文本服务试用

`coreai-runner serve` 为独立 CoreAI runner 提供有限的 OpenAI 兼容 HTTP 接口。全部神经网络计算使用系统 CoreAI；CPU 负责 tokenizer、SSD n-gram 行读取和解码、greedy 选词及网络。可执行目标仅依赖 `ANERunnerCore`，没有 MLX fallback。模型目录名中的 `MLX` 是源资产名称，不表示服务调用 MLX。

当前面向本机和局域网试用：macOS 27、完整模型资产、**4096-token 导出容量、S1 逐 token prefill、单请求串行推理**。容量来自 attention manifest，不是启动参数；4096 资产完整不等于完整模型已通过 4096-token 质量与性能验收。`quality_acceptance` 和 `hardware_placement_verified` 仍为 `false`。原生计算与量化边界见 [CoreAI 原生 runner](COREAI_NATIVE.md)。

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

加载期间 HTTP listener 可以响应健康查询，但推理请求返回 503。必须等待 `/health` 的 `ready` 为 `true`。加载时会核对源目录、配置 SHA256、完整 48 层、512 专家库和资产接口；smoke 或部分层资产不能代替完整模型。

CLI 只接受唯一的 `--option value` 参数对；未知或重复选项报错。主要服务参数如下：

| 参数 | 默认值 | 范围或含义 |
| --- | ---: | --- |
| `--host` | `127.0.0.1` | 仅 `127.0.0.1` 或 `0.0.0.0` |
| `--port` | 11236 | 1024…65535 |
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
| `requests_in_flight` | 已准入且尚未释放的推理请求数 |
| `completed`、`cancelled`、`failed`、`lastError` | 累计请求结果及最近错误 |
| `prefix_cache_enabled`、`prefix_cache_limit_bytes` | 缓存配置 |
| `cacheEntries`、`cacheBytes` | 最近一次请求清理后记录的缓存条目和逻辑字节数 |
| `cacheHits`、`cachedTokens` | 成功请求的累计命中次数和复用 token 数 |
| `prefillTokensProcessed`、`prefillSeconds`、`decodeSeconds` | 成功请求的累计阶段统计 |
| `backend`、`prefill_policy` | 当前分别为 `native-coreai`、`tokenwise` |
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

网络关闭、发送失败或超时会设置该请求的取消标志。推理在 token 边界协作检查；已经提交的 CoreAI 调用不承诺立即中断。TCP 请求写端的正常半关闭允许服务器继续返回结果；连接真正不可读的情形依靠发送失败、连接状态或 deadline 识别。

总请求 deadline 默认 1800 秒，从连接建立开始计算，包含排队和 prefill；慢请求体接收的 deadline 固定为 15 秒，单次发送等待固定为 15 秒。超时尚能发送时返回 408，流式响应已经开始则发送错误事件；已有发送阻塞时可能直接关闭连接。取消后队列名额在 worker 清理完成时释放，不保证客户端断开即刻释放。

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

当前 prefill 与 decode 只有阶段区分，使用同一套 S1 函数逐 token 执行；没有批量/分块 prefill kernel，也没有两个阶段独立部署。服务不支持 262K 上下文、工具调用、MTP 或多请求批量推理。GPU preference 不是 ANE 使用证明；更大容量、数值质量、缓存生命周期和性能均需各自验收，不能由 HTTP 可连接或能生成文本推定完成。

## 运行验证

2026-09-17，M5 Max / macOS27，以4096 attention、既有完整dense和Q4 MoE资产测试。最终服务二进制SHA256为 `5d3498b3b84f771a8b1e5703a4aab4f1866f55aec0512a50ab34ee6822adeb9a`。

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
```

完整2064-token稀疏边界与热恢复请求正在另行验证，结果完成后补入；当前上述通过结论仅对应已记录的短请求和CPU网络检查。
