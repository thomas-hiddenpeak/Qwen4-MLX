# HTTP 工具调用与结果回传

`serve-gpu` 已支持本机 AR 的 function-tool 闭环：客户端声明工具，模型返回结构化调用，客户端执行函数并以 `role: "tool"` 回传结果，模型继续回答。服务监听 `127.0.0.1`；默认端口为 `11236`。工具函数由客户端执行，runner 不执行请求中的命令或外部操作。

## 已实现范围

| 请求或输出 | 当前行为 |
| --- | --- |
| `tools` | 最多 64 个名称唯一的 `type: "function"` 定义；`function.parameters.type` 必须为 `object` |
| `tool_choice: "auto"` | 向模型提供工具定义，由模型决定是否调用；需要非空工具数组 |
| `tool_choice: "none"` | 本轮不向模型提供可调用定义；工具请求中意外出现的工具控制输出会失败，不返回调用 |
| 省略 `tool_choice` | 非空工具数组默认 `auto`，否则默认 `none` |
| `assistant.tool_calls` | 支持历史调用，`function.arguments` 是包含 JSON 对象的字符串；有调用时 `content` 可为字符串、`null` 或省略 |
| `role: "tool"` | 必须携带字符串 `content` 和匹配的 `tool_call_id`；按原调用顺序逐个返回，每个 ID 恰好一次 |
| 非流式响应 | `message.tool_calls` 包含完整调用；有调用时 `finish_reason` 为 `tool_calls`，无正文时 `content` 为 `null` |
| SSE 响应 | 先发 assistant role；普通正文仍增量输出，每个完整且通过检查的调用以一个结构化 `delta.tool_calls` 事件发送；末尾为终态和 `[DONE]` |

调用 ID 在整段历史中须唯一。一次 assistant 消息支持 1–16 个调用；返回完该轮全部工具结果后，才能追加下一条普通消息。模型模板将连续工具结果合并为一个 user turn，且不在模板中写入调用 ID，因此当前接口要求结果顺序与调用顺序一致。

函数名允许 ASCII 字母、数字、`_`、`-`、`.`，最长 64 字节；顶层参数名使用相同字符范围，最长 128 字节。历史调用 ID 最长 128 字节。字符串参数中不能含有会与此 XML 方言冲突的工具控制分隔符。

工具请求使用 greedy AR：`temperature` 只能为 `0`，`mtp_depth` 必须为 `0`。当前不支持 `tool_choice: "required"`、指定函数的 named choice、`strict: true`、多模态内容或随机采样，这些请求明确返回 `400`。`strict: false` 可以省略或显式传入。输出检查覆盖声明的 JSON 类型、必填键、`enum`、数组元素和 `additionalProperties: false` 等已实现规则；它不是完整 JSON Schema 约束解码。

HTTP 本轮仍不接受 `reasoning_content` 等 reasoning 字段。没有工具定义或工具历史的请求继续使用既有文本模板与文本输出路径。

## 请求与回传示例

先通过 `GET /v1/models` 确认 `model` 标识。以下示例使用本地测试数据，不是实时天气查询。向 `POST /v1/chat/completions` 发送：

```json
{
  "model": "Qwen3.8-Flash-Next-MLX-SSD-Stream",
  "messages": [
    {
      "role": "system",
      "content": "需要测试天气时调用工具。拿到结果后，用一行报告城市、温度和观测编号。"
    },
    {
      "role": "user",
      "content": "请查询北京的测试天气，并报告观测编号。"
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "lookup_test_weather",
        "description": "查询本地测试天气观测，观测编号只能通过此函数获得。",
        "parameters": {
          "type": "object",
          "properties": {"city": {"type": "string", "enum": ["北京"]}},
          "required": ["city"],
          "additionalProperties": false
        }
      }
    }
  ],
  "tool_choice": "auto",
  "temperature": 0,
  "mtp_depth": 0,
  "max_tokens": 192,
  "stream": false
}
```

收到 `finish_reason: "tool_calls"` 后，读取实际返回的 `message.tool_calls`。例如 `function.name` 为 `lookup_test_weather`，`function.arguments` 为 `"{\"city\":\"北京\"}"`。客户端解析这个 JSON 字符串并执行自己的函数；不能用示例 ID 代替模型响应里的实际 ID。

执行完成后保留上一个请求的 `model`、`tools`、`tool_choice` 等字段，将原 `messages` 数组追加下列两条消息，再发送请求。第一条应直接使用上一次响应的完整 assistant `message`；下面的 `call_example` 仅表示同一个实际返回 ID：

```json
[
  {
    "role": "assistant",
    "content": null,
    "tool_calls": [
      {
        "id": "call_example",
        "type": "function",
        "function": {
          "name": "lookup_test_weather",
          "arguments": "{\"city\":\"北京\"}"
        }
      }
    ]
  },
  {
    "role": "tool",
    "tool_call_id": "call_example",
    "content": "{\"city\":\"北京\",\"temperature_c\":23.5,\"condition\":\"晴\",\"observation_id\":\"LOCAL-7319\"}"
  }
]
```

这里的天气结果必须来自客户端实际执行的测试函数。将 `stream` 改为 `true` 即可使用 SSE；按 `tool_calls[].index` 重组调用并等待成功的 `tool_calls` 终态后，再执行函数和提交下一轮。SSE 中的 `function.arguments` 同样是 JSON 字符串。

## 模板与序列化边界

模板沿用当前模型的 `<tool_call>` / `<function=...>` / `<parameter=...>` 方言，以及连续 `<tool_response>` 的合并规则。thinking-disabled 的生成前缀和历史 assistant reasoning 的默认保留规则也延续现有模板实现；HTTP 没有因此新增 reasoning 输入字段。

工具定义的 canonical JSON 键排序、ASCII/HTML 安全转义和历史参数排序，是本 runner 选择的稳定序列化方式。它们与 Hugging Face 或作者 runner 的合法 JSON 内容具有相同语义，**不保证渲染文本或输入 token 逐一相同**。与其他 runner 对比时应保存实际渲染文本和 token，不能把序列化差异误判为推理数值差异。HTTP 的无工具文本路径保持原有行为。

系统前缀缓存包含本轮实际渲染的工具定义。是否能命中由完整请求 token 的精确前缀决定；`auto` 与 `none` 形成不同模板时，不假定二者共享同一缓存条目。命中量由 `usage.prompt_tokens_details.cached_tokens` 报告，`usage.prompt_tokens` 仍为完整提示长度。

## 失败、限额与取消

解析器保留跨 token / UTF-8 边界的控制片段。完整调用通过函数名、参数 JSON 和已实现 schema 规则检查前，不发送该调用；原始 XML 不降级为普通正文。未知函数、重复参数、缺少必填参数、错误类型、截断或未闭合调用会得到 `invalid_tool_call`。非流式为 `500` JSON 错误；已开始的 SSE 使用错误事件和 `[DONE]` 结束。此前已经发送的普通正文或完整调用不能撤回，客户端应以终态判断本次响应是否成功。

每个 pending 调用片段及累计保留的调用数据均有字节上限，一次响应最多 16 个调用。非流式正文和调用数据受累计输出预算约束，最终 JSON/HTTP 编码仍需通过实际字节检查。SSE 调用事件复用原有有界队列与事件额度，不绕过背压；输出过大或客户端积压沿用 `output_limit` / `slow_consumer` 路径。默认 `--output-buffer-bytes` 为 65,536，单个完整调用必须能够装入现有额度。

输入错误在推理前拒绝。生成失败、断连、取消和服务停止继续经过同一调度器与输出终态清理流程；工具适配器不拥有额外模型执行线程。

## 本机验证

`results/ar-prefix-cache-v1/http/tools-prefix.json` 记录本轮真实服务测试，`complete=true`、`passed=true`：**40 项检查、9 次推理请求通过**。该本地产物保留完整请求、原始响应字节、SSE 重组结果、客户端测试函数执行记录和 health 快照；`results/` 不作为仓库随附测试数据发布。

其中非流式和 SSE 都由模型实际生成 `lookup_test_weather({"city":"北京"})`，客户端执行固定测试数据函数后回传结果；两种模式的模型续答均为：

> 北京，23.5°C，观测编号 LOCAL-7319。

这里的 23.5 和 `LOCAL-7319` 是明确标注的测试数据，未查询真实天气。其余检查包括 `auto` / `none`、不支持组合的 `400`、真实 10k+ 系统提示的首次未命中 / 重复命中、相同系统更换用户问题后的复用，以及更换系统内容后的隔离。这是当前本机协议流程验证，不代表完整 agent 协议或持续运行验收。

在已经 ready 且 idle 的服务上复现；探针只连接服务，不启动或停止进程，输出路径必须不存在：

```sh
python3 scripts/probe_http_tools_prefix.py \
  --base-url http://127.0.0.1:11236 \
  --suite all \
  --output results/http-tools-prefix-new.json
```

也可以选择 `--suite tools` 或 `--suite prefix`。探针记录 HTTP 总耗时；prefill/decode 仍从服务阶段统计读取，不以 HTTP 总耗时替代 kernel 性能测量。
