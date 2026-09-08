# K02 完整会话复用：接线设计与 CPU 策略

2026-09-09。基于 `QwenGeneration.swift`、`QwenPrefixCache.swift`、`GPUHTTPServer.swift` 当前实际流程。根执行器已完成 release build（61.38s，`results/kv-night-a1/build-final.log`）及 108 项选定 CPU 测试，其中本文新增的 7 项全部通过（`results/kv-night-a1/cpu-tests.log`）。本文新增的 tokenizer 接口尚未接入运行入口，不代表 HTTP 已获得完整会话复用，也没有 K02 运行时实模证据。

## 已准备的小接口

`QwenTokenizer.swift` 增加：

```swift
public struct QwenConversationPrefixPlan: Equatable, Sendable {
    public let promptTokenCount: Int
    public let prefillChunk: Int
    public let lookupMaxTokens: Int
    public let publicationTokenCounts: [Int]
    public let systemProducerTokenCount: Int?
    public init(promptTokenCount: Int, exactSystemPrefixTokenCount: Int,
                prefillChunk: Int = 416) throws
}

public struct QwenTokenizedConversation: Sendable {
    public let tokens: [Int32]
    public let prefixPlan: QwenConversationPrefixPlan
}

public func encodeConversation(messages: [ChatMessage],
    tools: [QwenToolDefinition] = [], prefillChunk: Int = 416)
    throws -> QwenTokenizedConversation
```

完整请求先按现有模板渲染，然后完整分词一次。独立 system/tool 文本分词只用于与完整 tokens 比较精确公共前缀，绝不拼接到最终输入。不会按消息断点添加额外计算块。

- 查找上限：`P - 1`，允许命中已有的任一完整检查点。
- 发布候选：系统/工具精确前缀的原 chunk 网格位置，以及 `floor((P - 1) / chunk) * chunk`；去除 0、去重、升序，最多两个。
- 共享 system 生产者边界：系统/工具网格位置，没有完整 chunk 时为 nil。真正的 key 必须再结合实际前缀 token IDs 和完整运行 namespace，**长度本身不是身份**。
- 系统锚点与尾部重合时只有一个发布候选；没有系统/工具时仍可保存完整文档的尾部候选。低于一个 chunk 的请求不造检查点。
- 候选只说明允许保存的位置；不保证存在、不等于 hit、也不自动拿容量。未改变 AR-only、profiler bypass、字节 admission 和 private restore 的边界。

例：P=11057、精确系统前缀=10000、chunk=416，查找 cap=11056、发布候选=[9984,10816]、共同 system producer=9984。不同 user/tool 尾部改变 cap 与尾部候选，前 9984 个真实 token 不变时仍能合并系统生产者。

## 当前耦合点与必要改造

现有 `QwenPrefixCache.begin(request, maximum:)` 将 `tokens.prefix(maximum)` 同时用作：

1. radix/SSD 查找输入；
2. single-flight key；
3. 最终发布 token key。

`QwenPrefillProgress` 也只有一个 `cacheBoundary`、一个 `cacheFlight`；`prefillStep` 仅在 `end == cacheBoundary` 保存一次。HTTP 目前把 systemPrefixTokenCount 作为 maximum。仅把 maximum 改成全 prompt 会使不同尾部生成不同 flight key，失去共同系统前缀的冷合并。

建议分离三个实体，而不是把 plan.publicationTokenCounts 最后一个值直接传给旧 begin：

- **Request lookup context**：完整 tokens、lookup cap、namespace、clear epoch、restore read ticket；只负责寻找及恢复已有完整状态。
- **Checkpoint producer ticket**：namespace + 完整 checkpoint tokens + epoch、唯一 owner、发布中状态；只负责这个准确边界的同前缀计算合并。
- **Publication opportunities**：最多两个原网格边界以及系统/尾部用途；只在前向计算原本抵达边界、状态已 evaluate 且 PLE lookahead 已 join 时触发可选保存。

请求参数可添加 `prefixCachePlan: QwenConversationPrefixPlan?`，旧 `prefixCacheMaxTokens` 保持兼容现有 probes。两者同时出现时明确拒绝或有明确迁移规则，不能静默产生不同边界。校验 plan.promptTokenCount 与请求 tokens.count、plan.prefillChunk 与实际 chunk 一致。HTTP AR/cache enabled 时接 plan；MTP 或 cache disabled 不接 plan。

## 有界 single-flight 的实施顺序

先查找整个 cap 的现存状态。若更深的可用 RAM/SSD 检查点已经存在，不应为了等待较浅 system producer 而放弃该命中。查找候选与真正恢复、GPU 导入仍使用现有租约与 K04 截止期。

没有更深可用检查点时，对共同 system 锚点取得/等待 producer ticket；若没有系统锚点，可对尾部的准确 key 合并相同整前缀。生产者在原网格计算到 checkpoint 后保存，释放该边界的 producer；等待者在正常模型执行器恢复。发布失败时释放 owner，等待者可接管或者走受控冷算；缓存可选性不能把请求永久挂起。

扩展到多个 producer ticket 时只按 token 深度递增取得，且同一请求最多持有一个尚未完成的计算生产者。不得先占尾部 ticket 再等待系统 ticket，避免两个请求反向持有形成等待环。SSD read dedup 的 key 是实际读取的 checkpoint，与整条请求 lookup cap 同样分离。

**第一小步可以只对共同 system 锚点合并，保留现有行为，再增加尾部合并。** 若请求已经计算了部分前缀，再加入其他尾部生产者并替换自己的已算状态，会产生丢弃工作和统计复杂性；初版应避免这种状态切换。已经开始计算的游标若发现尾部 ticket 被别人持有，可继续当前私有计算、到边界再去重发布。尚未计算的请求则可以等待准确的完整尾部。此策略会保留一些尾部重复工作，但不会破坏 10k 共同系统前缀的核心收益和现有正确性。

## 多轮、分叉与存储约束

- 一个请求的两个候选不等于一条长期会话只能/一定保留两个状态。新增尾部前仍须执行全局/路径条目数与字节准入；现有 512 MiB RAM 下两个长快照可能放不下。保存失败安全跳过，不能为了同时保留两个候选突破硬预算。
- 第一版保留系统锚点加有界的近期尾部；分叉点只能复用真实保存的网格检查点。会话软偏好不能变成硬 pin，不能仅因出现较深状态就删除仍有分叉价值的较浅 GDN/PLE 状态。
- 如果从超过 system 锚点的快照恢复，不能向后裁 GDN/PLE 来补保存缺失的 system 锚点；本请求直接跳过已越过的候选。
- 模板 history 中的 assistant 内容按下一轮 canonical prefill 再计算。禁止把上轮 decode state 直接登记成 canonical prefill checkpoint。最后一个 prompt token 按现有 logits 协议执行。
- checkpoint 键使用真实完整 token 前缀，因此历史编辑、tool schema 变化和 BPE 边界变化由匹配自然产生更短命中/冷算，不用文本 session ID 代替 token 证据。
- 有效 token 守恒统计使用最终采用状态的来源。不得把“先恢复 system K1，再恢复 tail K2”计成 K1+K2；执行过又被丢弃的计算另记 actual/recomputed tokens。

## 已新增的测试与未验证部分

新文件：`Tests/ANERunnerGPUTests/QwenConversationPrefixPolicyTests.swift`，7 个 CPU 测试：

1. lookup cap / 双发布边界 / system producer 分离，两个不同尾部保留共同锚点。
2. 416、417、单 token、512 备用网格与尾 token 边界。
3. 无效输入拒绝及 Int.max 算术边界。
4. 真实 tokenizer 的不同尾部共享完全相同 system token key。
5. 真实工具调用/结果、多轮分叉通过 radix 检索已有历史检查点；早期编辑退回 system，工具定义改变拒绝旧锚点。
6. 无 system 的长文档仍有有界尾部候选。
7. 真实 BPE 的完整分词与独立片段分词不等价，最终输入保留完整渲染 token 序列。

前三项无外部 fixture；后四项只读取现有 tokenizer/template JSON，无权重或 MLX tensor。缺 fixture 时沿用已有测试 skip 机制。根执行器已运行这 7 项，全部通过、没有 skip；这些 CPU 测试不验证实际 state、single-flight owner、取消接管或 TTFT。

接线后的实模/HTTP 必测：共同 10k+ system + 不同 user/tool 尾部冷并发；完全相同整前缀并发；原网格与 QSA 结构临界点；双候选容量不足；leader 在 system 发布前/后取消；历史编辑与工具 schema 变更；完整 state/输出对照；少算 token、复制字节、保存成本及两阶段延迟。构建/GPU/Git 仍由根执行器唯一负责。

## 冻结期间的函数级接线审阅

以下为 2026-09-09 对当前代码的只读审阅建议，**尚未实现**。当前 Sources、Tests、scripts 均处于根执行器实模验证冻结期。这里的函数新名与状态字段是建议接口，不是已存在的能力。

### 1. 请求策略只归一化一次

`QwenGenerationRequest.init/validate` 新增可选 `prefixCachePlan`，保留旧 hint；两者同时非 nil 时拒绝，避免 probes 和 HTTP 获得不同的隐式优先级。校验 plan 的 prompt 长度、chunk 与实际请求相同。增加内部 `prefixCachePolicy`，返回：

```swift
struct PrefixCachePolicy {
    let lookupMaximum: Int
    let publicationBoundaries: [Int] // 升序、去重、最多 2 个
    let systemProducerBoundary: Int?
}
```

新 plan 使用 `lookupMaxTokens` 和两个原网格边界；旧 hint 映射成原 `prefixCacheBoundary` 的单候选策略。MTP、profiler、cache disabled 仍禁用运行时缓存。选择策略不改变 `prefixCacheNamespace` 的数值配置身份；相同数值计算所得的现存 system 快照应继续兼容。

`Sources/ANERunnerCLI/GPUHTTPServer.swift` 的 worker 入队段，在 AR/cache enabled 时调用 `tokenizer.encodeConversation(messages:tools:prefillChunk:416)`，将返回的完整 `tokens` 与 `prefixPlan` 一起传入 request。去掉这个分支里旧的独立 `renderChat/encode/systemPrefixTokenCount` 组合，不把 plan 的尾部值传给旧 hint。其他请求继续完整模板分词，不创建缓存 plan。仍在 tokenization 后、scheduler.submit 前检查取消。

### 2. 保留一个请求上下文，拆开三个 checkpoint key

可以保留 `QwenPrefixCacheFlight` 类名以减少调用改动，但不能继续让它的单个 `key/tokens` 同时代表查找与发布。建议明确保存：

| 状态 | 生成方式与拥有者 | 有效期 |
| --- | --- | --- |
| `lookupTokens` | 请求完整 tokens 的 `prefix(lookupMaximum)`；请求上下文持有 | 初次查找及恢复结束，保留作发布取前缀即可 |
| `ownedProducer` | `CheckpointTicket(key, boundary, identity, epoch)`；至多一个未完成计算 producer | 到准确边界发布、跳过、取消或清理 |
| `publicationBoundaries` | 请求策略的两个候选；上下文持有 | 整个 prefill，**system 发布后仍保留** |
| `read` / `readCheckpoint` | 当前实际 SSD 候选的 ticket 与准确 key；请求消费者和 I/O callback 分别持有 | 实际 I/O、Data 使用及 import/recovery 均结束 |
| `waitingCheckpoint` | 等待的准确计算/发布 key；只记录身份与截止期，不占 owner | 下一次 executor slice 重查 |

新增私有 `checkpointKey(namespace:tokens:boundary:)` 统一使用真实 `tokens.prefix(boundary)`；不能使用请求 cap、尾部文本、session ID 或长度代替实际 token key。复用现有 SHA256 构造及 numeric namespace。`flights` 只记录计算 producer，`publications` 只记录已接收的 CPU 写回；二者用 checkpoint key 关联。

建议最小接口：

```swift
begin(request, policy, model, allowWaitingForLeader) -> LookupContext
resolve(context, model, checkCancellation) -> Resolution?
publish(context, at boundary, state, model, checkCancellation, observer)
tryClaimProducer(context, boundary, mayWait) -> ProducerDecision
releaseProducer(context, boundary) // 比较 identity + epoch，不能删除别人的 owner
```

`resolved` 仅说明“请求初次查找已完成”，不能表示“之后不用发布”。`publish(context, at:)` 校验 `state.offset == boundary`、边界在计划集合中、边界为原 chunk 网格且 `< promptTokenCount`，再从真实 tokens 构造自己的 key。不要用 `ownedProducer.key` 保存，因为当前请求可以没有这个边界的 producer，仍允许做去重后的可选保存。

### 3. `resolve` 先比较两层完整深度，再决定等待谁

当前 `resolve` 是 RAM 命中即返回；当 RAM 只有 system=9984、SSD 已有 tail=10816 时会丢掉尾部复用。接线时需要同时看 `index.peek` 与 `disk.peek`，比较完整可恢复边界，再做私有 state 复制/导入。相同深度优先 RAM；较深 SSD 在有 workspace 且处于截止期内时优先，否则明确回退 RAM。

建议把候选选择分成不复制 Tensor 的 `selectCandidate(context)`，之后才进入 `restoreCandidate`：

1. TTL/epoch 检查后，在整个 lookup cap 内选最深的已完成候选 D。RAM peek 不应提前调用统计/LRU lookup，把“选中过”误记为有效命中；最终采用时才记 hit。
2. 找出计划中第一个 `B > D` 的待生产边界。冷请求优先共同 system；system 已完成时可以对准确 tail 合并；已有最深 tail 时没有 producer。尚未执行任何 forward 时，可以取得/等待 B。不要为了浅 system 的旧 owner 阻塞已经存在的更深快照。
3. 如果 B 已有别的计算 owner，协作式请求返回 nil，保留 `cacheResolved=false`；不提交 GPU forward。下一 slice 必须从查找重新判断，因为新发布的完整状态可能已经满足请求。完整阶段 `runPrefill(...allowPrefixWait:false)` 持有模型 gate，必须保留“不等待被外部暂停的计算 producer”的冷算/已有快照回退行为，不能覆盖别人的 owner。
4. 取得 owner 后恢复候选。SSD read 必须绑定所选的 D，调用现有 `disk.lookupAsync(...maxPrefixTokens:D, ...)`。不能仍传完整 cap 而不设上限，否则 peek 之后新出现的更深快照可能超过本次按 D 预留的 workspace。
5. SSD 结果可能因 TTL/损坏清理退回更短边界。最小实现可只接受 `actualOffset == D`；其他结果释放请求引用后回退 RAM/冷算，本请求不再启动第二次 SSD read。重新规划时先释放旧的未来 producer，再根据真实 fallback 深度选边界，不能持有 tail owner 再去等待 system owner。预算 lease 仍由真正的 I/O/Data 生命周期释放。
6. RAM 候选可能在 workspace reserve 的 LRU 淘汰中消失。进入 fallback 前重查 index；不要将“peek 时看到过 RAM”当作无租约可恢复状态。若用强引用暂存候选，则必须连同它的 cache lease 保留，并接受这会减少可驱逐空间；初版建议重查。
7. 初次恢复最终确定后才设置请求 `p.offset` 与 `cachedTokens`。恢复边界验证为合法原网格、`<= lookupMaximum`。第一版进入 forward 后不再切换成后来出现的别人快照；从而 `computed = P - adoptedCached` 仍明确成立。

对尚未完成的 SSD publication 同样按准确 checkpoint key 等待，优先采用已经完成且可用的更深状态。现有写回截止期和超时冷算应保留；超时只终止等待，不能移除仍在写的 publication，也不能重复导出/排队同一 key 的第二份大 archive。

### 4. SSD read 合并不能随尾部 producer 拆分而丢失

当前一个 flight 同时合并读取和计算。拆开后，A/B 不同尾部可以各自拥有 tail producer，却共同读取同一个 system SSD 快照；如果只把 `flights` 改为 tail key，会重复占用 workspace 和 SSD 队列。

建议添加独立、准确 checkpoint key 的 read fence；它只负责“一个读作业完成之前，其他请求等待后重查”，不共享可变 GPU state。所有 Tensor/index 操作仍只在模型 executor。`QwenPrefixDiskRead.take()` 是破坏性取出，不能给多个消费者轮流调用来模拟共享读取结果。

最小 read fence 记录 `ioComplete` 与 `consumerComplete` 两个线程安全状态。CPU callback 只完成 Data/read ticket 和 `ioComplete`；请求在导入并可选 RAM promotion 后，或取消/超时放弃后，置 `consumerComplete`。只有二者都完成，executor 才可以清理 fence/允许下一次读取。这样 callback 标记 ready 到 RAM promotion 之间不会突然放进第二个相同读请求。

- 请求开始 SSD read 时最多持有一个未来计算 producer；不同请求共享读取时，各自的未来 producer 可以不同。不存在 read fence 依赖其他计算 producer 的反向锁顺序。
- 任何 fallback 需要改等更浅计算 producer 时，先放弃旧 read 消费者并释放旧计算 producer，再重查；不得带着 tail owner 转而等待 system owner。
- 取消/超时只标记消费者放弃。callback 继续持有 read ticket/lease 直到实际 I/O 和回调使用 Data 结束，不手工提前释放；保留现有 `withExtendedLifetime(read)` 保护 import/recovery。
- fence 自己不必永久持有 payload，避免 registry 的清理频率延长 archive 保留。CPU job 的 completion 闭包才是取消后的 Data/lease owner。
- RAM 太小导致 promotion 失败时，后续请求在前一个 read 真正结束后可串行再读。这是容量导致的受控成本；不能为了“合并”把一个可变请求 state 直接借给别人。
- `clear` 增加 epoch 后，旧请求不能导入/发布到新 epoch；清 registry 不等于停止 I/O。已有 CPU callback 仍持有 lease，不允许 clear 或 context deinit 提前归还预算。

若第一阶段暂缓 read fence，应明确标为拆分后的性能回归风险并单独测试，不应继续宣称不同尾部仍合并共同 SSD 恢复。

### 5. Generation 的四处必要改动

`QwenPrefillProgress` 将单个 `cacheBoundary` 改为候选集合/升序数组，保留一个完整 prefill 生命周期的 cache context。不要 system 发布后把 `cacheFlight=nil`：这样尾部将永远无法发布。

`makePrefillSession`：仍在第一次 state 分配前检查 K03 pressure、保留完整 request lease；归一化策略后以完整 lookup cap 创建 context，首次 `resolvePrefix`。不要为了两个发布候选预先申请两个长期 cache lease；沿用每次保存时按字节准入。

`resolvePrefix`：等待返回 nil 时不标 resolved，不更改请求 state；收到确定的恢复结果后只采用一次 offset，并累计 lookup/restore/wait。不要将两次查到的前缀长度相加。`isWaitingForPrefixCache` 仍从未完成的初次 resolve 推导，现有 scheduler 的等待/退避逻辑可以继续工作。

`prefillStep`：保留当前 `end = min(P - 1, offset + chunk)` 和最后单 token 的原调度；仅将 `end == cacheBoundary` 改为候选集合包含 end。在既有 evaluate、取消检查、PLE lookahead join 后调用 `publish(context, at:end, ...)`。跳过小于/等于恢复 offset 的候选；不能反向裁 GDN/PLE 补 system 快照。

system 发布结束时只释放这个边界的计算 owner；之后可对下一个 tail `tryClaimProducer(...mayWait:false)`。若已被别人占用，当前请求已经开始计算，应继续私有计算，到达 tail 时去重保存，不能中途等待再更换状态。tail 发布仍走同一 state/lease/observer 保障。

**统计开关也必须迁移**：当前结尾只有 `if request.prefixCacheMaxTokens != nil` 才拷贝 cached/computed/cache timers。改为新 plan 或旧 hint 存在时设置；否则 HTTP 即便复用了完整会话，也会继续报 0 cached tokens。`runPrefill` 重新构造统计的路径也必须保留所有字段。prefill/decode 计时继续分开，PD handoff 仍转移私有执行 state 与 request lease，不把两个可选前缀快照混为 handoff。

### 6. 所有者状态转移与取消位置

```text
请求 admitted，持有 request lease
  -> Lookup：无 forward，可等待准确 producer/read/publication
  -> Restore 或 Cold：取得本请求的私有状态，采用 cached offset 一次
  -> Compute(system)：拥有 system producer 或有明确 bypass
  -> system evaluate + lookahead join + cancellation check
  -> 可选 RAM 快照 / 接收 SSD write -> 释放 system producer
  -> Compute(tail)：请求 context 仍活着；可选非阻塞 claim tail
  -> tail evaluate + join + 可选发布 -> 释放 tail producer
  -> final prompt token -> PD handoff/private request lease -> decode
```

取消前未完成的 GPU/lookahead 仍按现有 error path join/recover，再 `session.invalidate()`；不能在网络线程刚收到 RST 时直接释放 producer，让别人覆盖仍活跃的请求所有权。context 销毁仅释放自己 identity/epoch 对应的计算 owner；它不撤销已经独立发布的 RAM 快照、不取消已接收的 SSD archive、不提前释放 CPU read/write workspace。

发布失败、pressure skip、epoch 变化也必须退出自己的 producer，等待者后续重查/接管。建议 `publish` 内保持 defer 释放，但用本次 boundary 的准确 ticket 匹配，不能每次释放任意 `ownedProducer`。写回 publication 的完成状态由 CPU completion 独立推进；GPU owner 的释放不代表磁盘可恢复。

具体顺序示例：A/B 的 system 均为真实相同的 9984 tokens，尾部分别 10816/12064。冷启动 A 占 system，B 等 system；A 发布 system 后释放，A 可以继续持有自己的 tail。B 重查完整 cap，恢复可匹配的最深状态，随后计算自己的尾部。两条尾部使用不同真实 token key；下一次 A 即使 RAM 只有 system，也应选中 SSD 上更深的 A tail。A 若在 system 发布后取消，system 仍可用；未发布的 A tail 不凭空存在。等待者应看到这些实际状态，而非只看到旧 leader 的“已结束”标志。

### 7. 根执行器解冻后的最小验收

CPU 侧优先补策略映射与状态机测试：两种参数冲突/长度及 chunk 不符、两个不同尾部共享 system owner、system 释放后尾部仍能发布、错误 owner UUID 不能释放另一请求、RAM 浅/SSD 深选择及失败回退、read consumer 取消后的独立完成、pending publication 超时不重复写回、新 plan 的统计开关。已有 tokenizer/radix 测试不覆盖这些 runtime 规则。

实模只需先做小范围、可判定的完整请求对照：同 system 不同尾部冷并发；原请求重放命中 tail；RAM system + SSD tail 强制分层；system 发布前/后取消；双发布超过 RAM/SSD admission 时仍完成且最终 lease drain。比较完整实际输出、原网格混合 state、cached/computed 守恒及实际 read/write/保存开销。尚未完成这些验证时，不把“已有策略类型/七个 tokenizer 测试”标成完整会话复用已上线。
