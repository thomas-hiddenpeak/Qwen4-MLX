# SSD 恢复与发布等待期限：最小实模回归设计

2026-09-09。本方法已在 `results/kv-night-a2/timeouts.json` 完成6次真实请求、96个生成IDs、50项检查；一次读超时及一次发布等待超时均触发，扣除自身基准后5组完整状态对照通过。实现位于 [GPUCacheTimeoutProbe.swift](../../Sources/ANERunnerCLI/GPUCacheTimeoutProbe.swift)，入口 `RunnerCLI.probeGPUCacheTimeouts`，CLI 已接线为 `probe-gpu-cache-timeouts`。复用当前真实模型、整状态 archive 和现有 probe 的原始 BF16/host-state 对照方法；不修改磁盘挂载、不占满 SSD、不暂停真实磁盘。MTP 关闭。

## 先明确可注入边界

`QwenPrefixDiskStore.lookupAsync` 每次预留完整 `maxPendingBytes`。`availableSpace` hook 位于一个已经持有正 payload 额度的写任务中，因此**用它阻塞写，再把异步读排到后面是不可行的**：后来的读会因 pending byte admission 立即失败。调大额度也无效。这个场景只能证明读准入失败或同 key publication 等待，不能标成 read timeout。

本轮不为测试增加生产 `beforeRead` hook，也不改变 quota。采用两个互补方式：

- 真正读超期：空闲 store 中已有完整大 archive，给一个专用 generator 配置极短、正数的恢复期限，实际发起读取。只以生成器超时计数确认路径；调度竞争导致未超期时如实记录。
- 发布等待超期：使用现有 `availableSpace` hook 有界阻塞一次后台写。RAM 限额故意低于快照，使 follower 必须等待该 publication。由 semaphore、总超时与 `defer` 保证释放。

CPU `QwenPrefixDiskReadTests` 负责确定性的时钟、未完成 ticket、正常 take 后 callback 仍持有 owner、迟到 callback 和最终 lease 释放边界；实模负责这条分支下的完整输出和状态正确性。

## 共用负载与证据

优先沿用现有 11,057 token agent fixture，`prefix=9,984`、`prefillChunk=416`、AR、输出 16 tokens、上下文 16,384。它产生实际数百 MiB 的完整快照，覆盖 GDN/PLE 及 QSA 已启用池化后的状态。最小回归不做吞吐结论；若采用较短 fixture，必须另记准确 tokens/hash，不能混用已有长提示词 oracle。

同一进程只装载一次完整模型；推理操作仍在单一模型执行器串行进行。两个 generator 共享该模型和同一个 disk store，均设置 `RAM maxBytes=1`，明确跳过 RAM 保留。读期限探针使用 `1 µs`；正常 SSD 命中与 publication 测试使用默认 `5 s`。期限不改变模型数值或 cache namespace，但实际报告仍应保存两个配置。

记录：源码/可执行文件与相关动态库摘要、checkpoint 身份和 payload stat、完整命令、fixture 摘要、精确输入/输出 IDs、prefill 数值配置、缓存目录与额度、系统/机器信息。模型资源沿用现有控制器冻结、独占、finally 恢复流程；不引用其他版本 oracle。报告初始 `complete=false`，每个断言及时落盘，退出失败保留证据。

状态比较复用 `CacheReliabilityAnchor` 的语义：全部 named tensor 的 shape/dtype/原始字节摘要以及 offsets、PLE history、capture flags；不要求新恢复对象拥有相同 session identity。比较冷计算边界与恢复边界，不能把它们自身的摘要相等当成跨路径验证。

## 最小成功路径：六次 GPU 请求

| 请求 | 路径 | 必须观测到的结果 |
| --- | --- | --- |
| R1 | 默认期限 generator，冷计算并写入 SSD，hook 不阻塞 | 建立唯一冷 oracle、K=9,984 状态锚点；flush 后恰有有效 archive、无在途 IO、request/workspace 为零；RAM 保留为零 |
| R2 | 极短期限 generator，从该 archive 发起真实 SSD read | `beginPrefill` 返回时已接受读取；随后立即 step，`diskReadTimeouts` 恰增 1；最终 cold、cached=0、完整输出及冷边界状态等于 R1 |
| R3 | 默认期限 generator，同样请求，store 已排空 | source=disk、cached=9,984、diskHits 增长；完整输出与恢复锚点等于 R1；排空后 request/workspace/pending 均零 |
| R4 | 清空专用 SSD，默认期限 generator 冷生产同一前缀；在其写任务的第一次空间采样处阻塞 | R4 完整输出/状态等于 R1；RAM 不保留；发布尚未完成，写 job、pending bytes 与归档 workspace 都保持正数 |
| R5 | R4 已完成但后台写仍阻塞，同 key follower 以 cooperative cursor 推进 | 首先确实等待 publication；达到 5 s 期限后回退 cold；在解除 hook 前完成，输出/冷状态等于 R1；未重复提交同 key 写 |
| R6 | 释放 hook、flush 原 publication 后再次请求 | source=disk、cached=9,984、状态/输出等于 R1；只有原 publication 成功，最终 request/workspace/pending 清零 |

R2 的 read 超期和 R5 的 publication 超期必须使用不同计数/原因。一个 pending disk job、cold source 或 cache wait 大于阈值，都不能单独证明 read timeout。

### R2：非确定性的真实读竞争，有限重试

1. 先确认 archive 有效、RAM 空、store pending 为零。用极短期限 generator 的 `beginPrefill` 开始请求；此时读取由后台执行，尚未执行 prompt chunk。
2. 立即保存线程安全的 disk/模型 budget 快照。读取仍在途时，应有对应 workspace 与 pending reservation；该请求已有独立 request reservation。
3. 立即调用 `stepPrefill`，完成 cold 或 restore 路径并记录相对计数。若 callback 已 ready，生成器允许正常使用它；这是正确竞争结果，标记“本次 read timeout 未覆盖”，不可判为故障或擅自篡改 ticket 时钟。
4. 最多尝试三次。至少一次 `diskReadTimeouts +1` 且输出/状态通过，才称实模 read-timeout 分支已覆盖；三次均抢先 ready 则保留未定，不继续堆叠请求，也不以 publication 测试替代。
5. 真正 IO 不受门闩控制，可能在一次 prefill chunk 完成前就结束。因此只在实际 pending 的采样点报告正额度；“timeout 后始终持有直至 callback 退出”的确定性证明来自 CPU owner 测试。不要伪造可控 IO 阻塞，或要求实际完成的 IO 继续占用额度。

可附加一个低频 CPU 采样器，只读线程安全的 `stateBudget.statistics` 和 `disk.statistics`，不访问 MLX/tensor/生成器索引。两个账本的读值不是同一原子快照，边界差异只能作时间线，不能据单个跨账本采样报假泄漏。结束后停采样、flush 并在稳定排空点核对。

R2 回退后的 request 已事先保留完整冷算状态额度；超期不会释放在途 IO 的 workspace，也不会触发第二笔 SSD read。若预算不足以容纳实际 request 加保留中的 IO，不能绕过预算冷算，必须明确失败而非 OOM。

### R4–R5：确定性 publication 门闩

清空专用缓存必须发生在所有旧 IO 已结束之后。空间 sampler 使用目录 FD 实际查询可用空间；仅在显式 arm 后的第一调用查询前阻塞，后续正常查询。不伪报无限空间，实际存储空间保护仍生效。报告应写明这是**可控的 CPU/IO 调度延迟注入**，不是实盘满空间或真实卡死 IO。

R4 的 prefill 到达 K 时，真实 generator 导出 archive 并 enqueue；工作线程进入空间 hook 后发出 `entered`。先等 `entered`，再确认 R4 能完成输出而无须等待可选写入。生成器 publication 表仍持 owner，store 写任务仍持 Data；R4 的 request lease 结束，归档 workspace 保留。

R5 必须使用 `beginPrefill/stepPrefill` cooperative 接口；不能使用 complete-stage 的 `generate` 替代，因为后者本来就禁止等待外部 paused producer。至少观察一次 `isWaitingForPrefixCache=true` 和 processed=0。门闩保持关闭，推进 R5 至超期后的完整 cold 输出。

解除前在稳定点核对：`requestBytes=0`、`cacheBytes=0`、`workspaceBytes>0`、`pendingJobs=1`、`pendingBytes>0`，原 write 尚未 published；R5 的冷 prefill 到 K 不能再 enqueue 同 key archive。计数与 R4 返回后的基线相比不能出现第二次 write admission，具体 workspace 值以实际预留公式记录。

然后释放门闩，flush 原 write/callback；此时 publication 只完成一次、entries=1、全部在途额度归零，再执行 R6。RAM 仍为 1 byte，避免将 RAM 命中误报成 SSD 恢复。

## 退出与超时契约

- 门闩是一次性、可重复 release 的 gate，watchdog 固定 60 s。它包含 R4 剩余输出及 R5 等待/冷算时间；过短的 fixture 期限会让正常 10k 冷算误触发。watchdog 被触发就把用例标记失败，释放线程，不能继续报告正常通过。
- 定义 gate 后立即注册最外层 `defer`：**先 release gate，再取消/discard 活跃 cursor，最后 flush/close store**。不可在仍锁住 hook 时先调用同步 flush/close。
- 每个等待 `entered`、prefill loop、decode 和回调排空都有合理上限；控制器再提供进程级总期限。发生任何断言或模型异常，都保留报告、释放 gate，并走现有 finally 服务恢复。
- 已发生真实 OS I/O 无法因客户端 timeout 被强制终止。上述门闩只延迟一次 CPU hook，可由本 fixture 确定释放；不得把该成功推论为任意坏盘的关闭期限保证。

六次请求是首次 read-timeout 命中时的最小成功路径；最多两次额外 read 尝试，总上限八次。既有 CPU ticket 测试承担边界穷举，本 fixture 不再扩成长期压力工具。5 s 默认 read 延迟边界本轮只有 CPU 时钟测试，实模极短期限只确认路径与状态/所有权；真实慢盘、持续 SSD 压力、小时/天级门槛仍独立保留。
