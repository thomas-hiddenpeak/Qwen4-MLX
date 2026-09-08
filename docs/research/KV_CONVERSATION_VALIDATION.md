# K02 完整会话缓存：最小实模验收设计

2026-09-09。探针已实现，等待根执行器统一构建和实模验收；本页不表示已通过。与 [K02 接线设计](KV_CONVERSATION_IMPLEMENTATION.md) 配套，首轮只验证专用模型、AR、chunk416、单进程 cooperative prefill/decode；MTP 和跨机部署不进入此次范围。

## 实施边界

新增 [GPUConversationCacheProbe.swift](../../Sources/ANERunnerCLI/GPUConversationCacheProbe.swift)，复用 `CacheReliabilityAnchor`、现有 `QwenTokenizer.encodeConversation`、`QwenGenerator`、`QwenLocalScheduler`、`QwenPrefixDiskStore`。模型只加载一次；一个不启用 cache 的 generator 建立参考，启用 RAM/SSD 的 generator 执行对照。C6 排空后换成默认 RAM 容量的 generator；所有实例共享同一个 `QwenModel`，模型调用在同一 executor 串行推进。不另起推理服务、不实现新的状态序列化或测试调度框架。

入口：`probe-gpu-conversation-cache --model-dir ... --system-file fixtures/gpu-agent-11k/system-prompt.txt --cache-directory NEW_EMPTY --output NEW_JSON`。四项均必需，由根执行器接入 CLI。报告的单份 `fixtures` 数组保存 snake_case 的 OpenAI messages/tools、完整 token IDs、rendered/token 摘要和 plan，可供 HTTP 原样回放；`cold_oracles` 保存实际生成 IDs、native stop 和 `text`。终端仅打印 counts/LCP。诊断读回会干扰耗时，首轮不作为性能默认验收。

## 先用 CPU 固定六个输入

使用已有 `fixtures/gpu-agent-11k/system-prompt.txt`。其旧独立 system tokenization 为 10993 tokens；加入真实 tools 后的数字必须重新完整渲染/分词，不能复用旧数字当新系统锚点。工具使用纯合成的 `measure(city: string)`，历史 assistant 带实际 `QwenToolCall` 对象，tool 消息带相应 call ID。工具结果是固定 fixture，没有真实网络工具调用。

| 输入 | 消息结构 | 用途 |
| --- | --- | --- |
| A | 长 system + user 历史文档 + assistant `measure` call + tool result A | 根历史、重放、分层恢复和取消后的输出参考 |
| B | 与 A 相同，只有 tool result 开头和其后正文改为 B | 相同 10k+ system、不同真实尾部生产者；旧 A tail 必须不能冒充 B 命中 |
| N1 | A 的完整历史 + assistant 固定已完成答复 + 新 user 问题 alpha | canonical 多轮继续，证明命中超过 system 的历史 checkpoint |
| N2 | 与 N1 相同，最后 user 改为 beta | 相同父历史的分叉；N1 新 tail 不应因文字会话 ID 相同而被借用 |
| E | A 的较早 user 文档开头被修正，其他结构不变 | 回退到实际共同 system；旧历史 tail 失效 |
| T | A 的 messages 不变，只改变 `measure` schema 的 description 或增加可选属性，旧 call 仍合法 | 工具定义改变导致 token 身份改变；不得把无效 tool history 混入缓存测试 |

历史文档、工具结果和两个问题用短的固定英文行填充，初始各重复64次；CPU 预检最多12轮按实际 LCP/grid 调整次数，目标总 prompt 约 12k–15k，`P + maxTokens <= 16384`。每轮真实长度、重复数和条件先写入报告。超长时缩减历史文档/工具结果/末轮问题，原始长 system 保持不变。不截断渲染后的 token 列表来凑长度，不独立编码各消息再拼接。

模型加载前必须验证并写入报告：

- 每个输入实际 `P`、system 候选 `S`、tail 候选 `K`、完整渲染文本/token 摘要及可复现实例参数。普通日志只显示摘要与长度；不输出整份 10k token 数组。
- `S >= 10000`、各输入候选为原 416 网格、末 token 不缓存。若加 tools 后 `S` 没有达到 10k，停止并报告实际长度；禁止改动原 fixture 文件或把追加内容当作原始 system。
- A/B 的前 `S` 个 token 完全相同，且 `LCP(A,B) < min(K_A,K_B)`。这保证两个尾部 key 真正不同。
- `LCP(A,N1)` 和 `LCP(A,N2) >= K_A`，且 `K_A > S + 416`；`LCP(N1,N2) < min(K_N1,K_N2)`。必要时让最后 user 问题足够长，避免新 tail 仍落在相同的历史里。
- `S <= LCP(A,E) < K_A`；工具定义位于当前模板 system 的最前部，T 的改动应保证 `LCP(A,T) < S`。如果实际 LCP 不满足，停止并报告实际 token 边界，不把预期写死成错误的“必须零命中”。
- 对每次将要恢复的输入，依据**真实存在过且未清除的完整候选 token key**算期望最长命中。不同长度、同一树节点、文本会话 ID 都不能代替完整前缀相等。

## 独立 oracle 与状态证明

六个输入各做一次 cache-disabled AR 冷算，`maxTokens=16`。保存完整实际 output IDs、finish reason、统计、每个 system/tail 候选处的 `CacheReliabilityAnchor`。缓存关闭时保留原网格 `coldBoundary` 诊断回调，可让一次 prefill 同时取得两个独立锚点，避免再做一套手工 forward。

每条后续缓存路径的 `coldBoundary/publish/restore` 都与对应输入、对应准确 token 边界的冷 oracle 比较原生 BF16 tensor hash 和全部 host offset/PLE 历史。共享 system 的 A/B 两份独立 oracle 也应相同。建立 oracle 本身不计为“交叉状态对照通过”。

恢复的较早 `K_A` 未必是 N1/N2 的本次发布候选。可以使用 A 在 `K_A` 的独立锚点作为参考，因为 CPU 已证明 N1/N2 的前 `K_A` tokens 完全相同且计算网格/数值配置相同；报告必须写明引用的是哪个独立请求及边界，不能与恢复结果自身比较。

保留原生 EOS：`maxTokens=16` 是上限，短输出并非错误。检查 finish 与 EOS/长度一致、没有中途 EOS 后仍继续生成；不为了凑 16 tokens 禁用 EOS。取消时优先停在 prefill/system 边界或 decode 首 token 发布之前，避免把“至少生成很多 token”偷偷变成 fixture 前提。诊断说明明确：state hashes 覆盖 checkpoint，不宣称所有最终 decode 张量已被读回。

## 十二个必要成功请求，兼顾所有权故障

前六组使用 RAM 1 GiB/8 entries、SSD 4 GiB/16 entries、state budget 4 GiB；SSD pending 为2 jobs/1 GiB，明确高于默认队列字节容量，使两份约350 MiB的候选可同时入队。加载后先记录模型实际估计的 S/K、最大 request lease 和去重候选总字节，再检查共存/联合额度条件；不满足即停止，不将容量问题当作数值错误。C6 清空后用同一模型/同一 SSD 换成默认 RAM 512 MiB generator 执行 C7/C8，pending 仍保持受控的1 GiB。

| 顺序 | 成功请求数 | 操作与必须证明的结果 |
| --- | ---: | --- |
| C1：冷并发与 clear epoch | 2 | 空缓存开始 old-A producer、old-B waiter；B 一次 slice 后 processed 必须仍为 0。clear 两层后先创建 new-A，再取消/丢弃 old 两者，最后创建 new-B；new-B 必须等待 new-A，不能被旧 owner deinit 误释放。这个顺序最多保留三个 request lease。交错完成 new-A/new-B：A 冷算；B 复用共同 S；只生成一次该 epoch 的共同 S；两者 tail 均可继续发布，各自 output/state 精确。 |
| C2：原历史重放 | 1 | 重放 A；恢复 `K_A > S`，确认 system 发布后上下文仍活到 tail 发布。记录有效 cached/computed 和来源，不把查找次数当命中。 |
| C3：完整多轮与分叉 | 2 | 先 N1，再 N2。两者复用 A 的完整历史 `K_A`；N2 不能借 N1 的不同尾部；共享页尚未实现时各自仍恢复私有完整 state。各自输出对自己的冷 oracle。 |
| C4：早期编辑与 schema 变化 | 2 | E 只恢复实际共有的 S；T 在已证明 LCP<S 的前提下冷算，绝不读回旧 system/tail。每个变体对独立冷 oracle，不拿 A 的输出当正确答案。 |
| C5：浅 RAM、深 SSD | 2 | 不清 SSD，仅清 RAM。用 A 的旧单边界 hint=S 完成一条受限查找请求，确认它从 SSD 只恢复 S 并提升为唯一 RAM entry，深 `K_A` 仍在 SSD。再用 A 的完整 plan 请求：必须选择更深 SSD `K_A`，不能见浅 RAM 就返回。同深度才优先 RAM。两次完整 output 都与 A oracle 一致。 |
| C6：活跃私有恢复与 clear/cancel | 1 | 同时从已保留 A checkpoint 建立 left/right；right 完成 prefill 并建立 decode cursor但尚未发首 token。clear 两层，再取消 left。right 继续生成完整 A oracle；清 cache 不改活跃 request lease、取消 left 只归还自己的份额。排空后 request/workspace/flight 全归零，缓存为空。 |
| C7：有效 producer 在 S 前取消，默认 RAM 容量 | 1 | 空缓存 A producer 只执行一块416，B waiter processed=0；取消 A 后 B 取得生产权、冷算完成，对 B oracle。B tail 发布后，再建立不同尾部 A 的只恢复游标，确认立即从 RAM 恢复 S、`retainedSystemAnchorSkips > 0`，随即取消。验证真实 producer 接管和默认容量的共享 system 保留，不增加完整生成数。 |
| C8：S 已发布后 producer 取消 | 1 | 清两层，A/B 重建；A 算到 S、完成发布与 lookahead join 后取消，未生成 A tail。B 采用已完成 S，继续自己的 tail并对 B oracle；A 已完成的 S 不能随请求取消消失，A 未完成 tail 不得出现在发布事件里。 |

合计：6 个独立冷参考 + 12 个成功对照请求 = 18 次完整生成，O16 上限，另有6个被主动取消的短/边界游标。C1 的 clear/新 owner 次序必须在同一 executor 实际交错，不能仅给事件改标签。不能把永久等待算成功。

实际执行次序为 C1 → C2 → C5 → C3 → C4 → C6 → C7 → C8。C5 紧接 C2，开始前检查已排空且 SSD 至少有 S/A-tail/B-tail 三份候选，保持请求集不增加。不需要诊断性的直接 index 修改或修改 SSD 文件来布置场景。

## 每个场景的最少断言

1. 完整输出 IDs、finish、最终 token offset 与独立 oracle 一致；取消请求有明确终态，不计为成功生成。
2. 采用来源和深度正确，`computed + RAM restored + SSD restored = prompt`；先浅候选再选深候选不能计两次。记录期望 token LCP、实际 adopted boundary 和 state hash reference。
3. 状态 observer 的事件/offset 列表证明 system 与 tail 分别完成。C1/C7/C8 明确记录 producer/waiter 的 processed count、waiting 状态与取消点；`liveFlights=0`、scheduler 可继续接收任务、无反向等锁。若 root 提供 read/publication fence 数量，排空也须归零；没有字段不能写成已验证。
4. 任一阶段 clear/cancel 前后记录 request/cache/workspace、pending jobs/bytes、累计发布和读写字节；尚未结束的实际 I/O 仍占自己的预算。使用已有 flush/close，而不是假定 callback 触发就已经释放其 payload。
5. 每组结束后排空采样；保留有效缓存时 `request/workspace=0`，最终 clear 后 cache/总状态额度为 0；记录 MLX memory 作物理观察，不能把逻辑 lease 归零换算成进程 RSS=0。

每条诊断应先写 actual 再 require，尤其是输入长度、EOS、候选存在、来源和请求存活数。失败报告继续保留独立 oracle、已完成 trials 和阶段位置，不能只留下一个 false 总标签。

## 最小 probe 结构与接线依赖

单文件采用现有 probe 的局部 helper：`object/write/require`、`finishPrefill`、`decode/record`、`clean`，以及一个 `(inputID, offset) -> CacheReliabilityAnchor` 字典。fixture 生成、CPU precheck、oracle、上述 C1–C8、最终排空顺序执行。每个轮询有600秒上限；只有所有等待者等待 CPU I/O 且无 GPU 工作时短暂退避，禁止持完整模型 gate 等被外部暂停的 producer。

必需运行时接口只有 `QwenGenerationRequest(prefixCachePlan:)`、原来的旧 hint 兼容、`isWaitingForPrefixCache/processedTokenCount`、原观察者和 cache/disk/budget statistics。不得为了 probe 添加可篡改 index 或跨线程取 Tensor 的公开入口。统计能证明多少就报告多少；不要求加通用 replay framework。

还需一条独立 HTTP 冒烟以证明 worker 确实调用 `encodeConversation` 并传 plan：同一真实工具历史原请求 + 带完整 assistant/user 续轮 + 早期编辑，至少覆盖非流及 SSE 的缓存 usage。此 HTTP 验证用已有 HTTP 工具回归脚本扩充，由根执行器安排；CLI 成功不能代替 API 接线或工具调用协议验收。

本设计首轮刻意不追加小时级 churn、新页 kernel、成本 autotune 或联网工具执行。K02 实际实现与上述故障/正确性通过后，再用既有固定 trace 单独看 TTFT/TPOT 与默认 512 MiB 的容量收益。
