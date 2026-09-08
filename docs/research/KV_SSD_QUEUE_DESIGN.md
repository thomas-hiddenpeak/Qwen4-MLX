# K04：SSD 前台读优先与可选写节流

2026-09-09，第一增量已集成；177 项所选 CPU 测试通过。C1 实模探针在计时断言失败，整轮未通过；C2 重跑完成，具体范围见下文。本文记录设计、当前源码合同与已完成报告证据，构建和实模执行由统一控制器完成。

当前交付为“一个前台读意图 + 既有硬额度内等待 + 分类观测”，解决忙写期间 SSD 候选直接被放弃的问题。继续使用单条物理 I/O 队列，保留原有完整读取 charge 和 callback lifetime。分块中止写、多线程读取、暂停大写状态及通用优先任务框架均未实现；后续是否增加，以竞争样本为依据。

## B1 能说明什么

读取完整的 `results/kv-night-b1/conversation.json` 和 `results/kv-night-b1/timeouts.json`。以下为现有字段实际值，毫秒取一位小数：

| 路径 | 恢复 tokens | cacheWait | cacheRestore | cacheLookup |
| --- | ---: | ---: | ---: | ---: |
| C5 浅 SSD | 11232 | 163.6 | 54.6 | 11.8 |
| C5 深 SSD | 12064 | 188.4 | 57.9 | 13.6 |
| C3 N2 SSD | 12064 | 179.3 | 57.3 | 13.0 |
| timeout probe R3 正常读 | 9984 | 135.4 | 35.0 | 0.064 |
| timeout probe R6 发布后正常读 | 9984 | 135.4 | 33.9 | 0.058 |

`cacheWait` 是整个 cache flight 经过的时间减 lookup/restore 工作，混合了异步 I/O、线程/执行器调度和等待 producer；它不是纯磁盘队列等待。`cacheRestore` 包含模型导入及可选 RAM promotion。conversation probe 的 observer 会读回/hash，不能将其 TTFT 用作性能默认。C1 的15.6秒、C8 的21.8秒 cacheWait 是故意暂停/推进 producer 所产生，不能说是 SSD 慢。

三条成功的 conversation SSD 请求之前都已 `clean/flush`，没有给出“前台读与大写竞争”的样本。它们只提供安静队列的诊断基线。本节仅使用已完成的原生报告，不外推 HTTP 端到端时延。

优先级依据来自确定的准入路径，而非把上述约0.18秒归因于写阻塞：长请求一旦失去 SSD 命中，可能重算11232–12064 tokens；B1同一 A 的冷前向12354，深恢复仅290。先避免这种功能性退化，比尝试从安静队列读的0.18秒中猜测 kernel 收益更有依据。

## 修复前的阻塞点与保留的所有权

来源：[QwenPrefixDiskStore.swift](../../Sources/ANERunnerCore/QwenPrefixDiskStore.swift)、[QwenPrefixCache.swift](../../Sources/ANERunnerGPU/QwenPrefixCache.swift)。

1. 修复前 Core `lookupAsync` 直接以 `limits.maxPendingBytes` 调用共同 `admitLocked`。只要某个写仍占任何正的 pendingBytes，异步读就不能入队；GPU `resolve` 随即释放 workspace、设 `skipDisk` 并回退。这是“忙写时读被拒绝”的确定路径，旧报告没有证明其自然发生频率。当前 cooperative 路径先取得 metadata intent 并等待旧 charge 释放，再尝试这一次完整准入；原来的硬额度没有降低。
2. 同步 `lookup` 使用 `queue.sync`，确实会排在先前整份写之后；它没有异步读的 pending 准入。不要把测试中的同步 lookup 行为外推成服务的异步路径。
3. 一份写的整个生命周期在同一个 `queue.async` 内：manifest/hash、`makeRoom`、临时文件、整份 payload SHA、`writeAll`、文件 fsync、rename、目录 fsync、索引发布。`writeAll` 每次把全部剩余区间交给系统调用，没有主动小块预算。约360–383 MiB的 checkpoint 在这个闭包结束前独占队列。
4. `decodeFile` 已每1 MiB读取、校验并 append 到预留的完整 Data，但循环一直在同一 I/O block 内。小块缓冲并不等于会让出队列。读结果本身仍是完整 archive；不能把1 MiB块尺寸称为1 MiB恢复内存。
5. 写入完成回调另外排入 callbackQueue。Core 写 pending 在 I/O结束释放，闭包最后的 `withExtendedLifetime(completion)` 和 GPU completion 所持 workspace lease 共同防止回调抢先结束而提前释放 archive 使用者。异步读的 Core pending 一直保留到 callback返回；GPU read ticket/fence 又独立覆盖 import 或已取消消费者。不能将这些所有权合并成“文件读完即可释放全部额度”。
6. GPU 导出完整 archive 本身在推理 executor 上执行，发生在 `disk.enqueue` 之前。当前新增 intent 存在时的导出前跳过；它只是尽力检查，检查后才出现 intent 的竞争仍由 enqueue 锁内拒绝兜底。其他准入拒绝仍可能已付出导出成本。B1 cacheSave 含导出、RAM副本和诊断，不够拆分这个成本。

## 已集成第一增量：cooperative 运行时的有界准入等待

采用收窄后的实现：不将还不能计费的 read job 提前交给 Core 队列。只引入至多一个 metadata-only intent；原 `lookupAsync` 的整份 `maxPendingBytes` charge、jobs上限和回调 lifetime全部保留。

- Core `acquireReadIntent` 返回 acquired/busy/closed/unavailable。handle按 UUID+epoch 标识，weak引用 store；不持Data、FD、IO job或读取workspace。它的 state 区分 ready/busy/closed/unavailable/invalidated；release和deinit幂等。
- 持有intent就拒绝新的可选 `enqueue`；此前接纳的写/读/control工作照常完成。active intent单列0/1统计，不能冒充已入队读取或已缓存tokens。
- cooperative GPU resolver在已有有效SSD候选、future checkpoint producer放行后首次建立单调时间。等待同key read fence、获取intent、store忙、最后accepted IO共用这一5秒窗口；普通长system producer的计算时间不算在这5秒内。
- store busy时返回nil让出slice，processed不动，不先预留读取workspace。ready只是提示：再次peek确认候选深度和metadata/payload大小，随后执行真实joint workspace reserve，再用intent原子提交原来的异步read。已接收的replacement写可能刚完成，所以不能仍用忙写前的旧summary大小计费。
- 只有accepted才消费intent、完整charge Core pending并安装GPU read ticket/fence；failed提交不调用completion。race busy时立刻归还未转交lease，只保留metadata intent继续原deadline；closed/unavailable/invalidated则释放intent冷退。joint workspace本身不足也保持既有冷退，不新增无法证明会腾出空间的重试。
- deadline到期且IO尚未接收：释放intent、只冷退一次，不制造迟到读取。已经接收的IO保留原来的迟到completion/fence所有权；结果已经ready时，即使执行器稍后才来消费，也不把它当过期结果丢弃。5秒约束未完成等待，不是结果有效期。
- wholeStages busy立即回退，不等另一个被暂停cursor持有的intent；否则完整generation持有model gate时可能挡住释放intent所需的cursor推进。
- 任何实际放弃路径、取消/析构、RAM-only clear和SSD clear都撤销对应metadata handle；cache登记自己的handles以便clear无需等待暂停cursor再次执行。UUID+epoch保证旧析构不能移除新owner。
- 已知存在foreground intent时，在GPU导出前尽力跳过可选SSD归档。该检查没有reservation承诺；最终enqueue锁内判断仍是准入权威。

Core写pending归零时，其completion还可能短暂持有GPU workspace lease。因此 ready不能替代joint reserve；首增量允许此时安全冷退，不改变callback计费协议。

当前五秒是在 GPU resolver 重新执行时检查，并非 Core intent 自动到期。公开 cursor API 允许调用方暂停或在 step 前取消后保留 cursor；若之后既不 step、discard，也不 clear，未提交 intent 可一直阻止该 store 的新可选写。HTTP scheduler 持续推进并在取消/关闭时 discard，因此没有同样的闲置 owner 路径。下一项最小修复建议为 metadata 绝对期限的锁内惰性过期；仅在后续 enqueue/acquire/state/statistics 时撤销已过期 UUID，不改已接收 IO、callback 或 cursor 的保留合同。此修复尚未实现，不能把当前五秒称为不依赖调用者推进的 Core 租约期限。

此方案只新增metadata状态与cooperative等待，不增加物理I/O队列、未入队的写permit或“以后才提交”的隐形IO owner。close仍只等待原来已接纳的串行IO闭包及callback marker；metadata intent在clear/close即时撤销即可。仅提高Dispatch QoS不能解决共同字节额度问题。

## 暂缓的第二增量：在固定内存下中止可选写

只有“一个大写活动中到达一个前台读”的实测竞争样本仍有明显等待，才增加分块。保留单个写 I/O block，每4 MiB左右检查一次 epoch 和前台读标记；先从固定4 MiB开始，不新增 autotune。

1. payload SHA 改为有界增量更新，每块后检查。先保留现有“先算 hash，再写 header+manifest+payload”两遍流程和文件格式，避免首轮同时改 header回填/恢复协议。
2. payload `writeAll` 每次最多4 MiB，处理短写/EINTR，每块后检查。metadata/hash等阶段也在进入前检查；不要只分块 write 而保留不可中止的整份 SHA。
3. 遇到前台读时，中止尚未发布的可选写，关闭/移除自己的临时文件，完成一次 `completion(false)`。按照真实闭包/回调 lifetime归还额度后，已入队前台读才启动。
4. 不暂停并保留整份400 MiB写 payload，再叠加400 MiB读 buffer。默认 pending512 MiB 容不下这两份；如果保留写状态，字节额度不会因“让出队列”自动减少。中止会损失一次可选缓存填充，用独立 `writePreemptedForRead` 计数，不当作介质 writeFailure。
5. 文件 fsync、rename和目录 fsync 的提交段保持当前顺序；不能停止已在内核执行的 fsync。已经 rename 的候选沿用现有 epoch/清理逻辑，不允许让新写替换同名文件后再由旧写 cleanup 删除它。

读已经有1 MiB校验缓冲，首轮不改变。将读循环也改成任意 yield，会引入打开 inode、LRU/revision pin 和多份 live read Data的问题，当前没有证据需要承担这些成本。

## 若以后需要真正的优先队列

只有要允许多个排队写时，才引入 bounded read/write deques 和一个串行 pump。job必须有单调 submission ID、epoch、kind、阶段、callback-finalized 状态和各自真实 charge；同类FIFO，前台读优先，活动读最多1、活动写最多1，写 payload 总和仍受同一硬 byte cap。read intent等待时，先丢弃尚未执行的可选写并完成其回调释放 Data，不能只挪队列位置。

此时原来的 `queue.sync {}` 不再是任务完成 barrier：一个大写分块后排出的 continuation 可能位于 marker 后面。必须按 submission ID维护 IO完成水位和callback完成水位；`drain/flush/clear/close` 等待调用时已接纳ID集合的正确阶段，不能只等一个队列空块。这个改动范围明显大于第一增量，建议暂缓。

## 必须守住的取消与关闭边界

- IO接收前的请求取消/超时释放metadata intent，不产生callback或迟到读。IO接收后的请求取消/超时仅取消消费者使用权；原callback必须最终完成，未完成IO的fence/lease不能提前释放。
- `clear` 先推进 epoch并清 summaries，再等物理清理。旧 read/write/intent均不能重新发布旧 metadata；控制任务保持提交顺序，不能被新高优先请求越过。已打开旧 inode可完成安全清理，但不对新epoch返回旧状态。
- 保留 `invalidateAsync` 的 revision语义。当前测试特意证明“先排队的同 key替换”不会被后来的旧revision失效删除。不能将 invalidation无条件提到任何写前面。
- `close(drain:true)` 立刻停止新准入，等接纳前所有 I/O及callback；`close(drain:false)` 推进epoch但仍等真实FD/owner清理。超时仅结束等待，不能释放仍使用中的FD/Data；后续 close等待同一事件。callbacks可在 I/O完成后仍保留额度。
- 原有队列闭包方案下，只有accepted IO才在admission锁内计费并排队；metadata intent没有IO闭包和未来callback，close可直接撤销。禁止把intent改成已接受但未入队的IO，否则关闭marker可能先于其continuation。
- callbackQueue继续独立；不能让callback等待flush/close，也不能持admission锁做物理I/O、调用回调或等待释放。

## 当前统计合同与后续观测

Core statistics、HTTP health 的 SSD 快照及 Prometheus 映射已经接入三项；HTTP 指标前缀沿用服务现有前缀：

| Core 字段 | HTTP 指标后缀 | 当前语义 |
| --- | --- | --- |
| `foregroundReadIntents` | `ssd_foreground_read_intents` | 0/1 metadata owner；不计作已准入 IO、pending bytes 或恢复命中 |
| `foregroundReadIntentAcquisitions` | `ssd_foreground_read_intent_acquisitions_total` | 成功取得意图次数；可能随后取消、超时或预算回退 |
| `optionalWritePriorityRejections` | `ssd_optional_write_priority_rejections_total` | 已调用 enqueue、因 intent 优先而拒绝的可选写；不含 GPU 导出前的尽力跳过 |

意图被实际读取准入消费后，第一项回到 0，原有 `pendingJobs/pendingBytes`、GPU read fence/workspace 继续覆盖 IO 与消费者。因此 intent=0 不能单独证明系统空闲。恢复命中仍只在完整 import 成功后增加 `diskHits/restoredHits`；HTTP usage 的 cached tokens 与实际恢复深度相同，不由 intent 或 Core 文件命中推算。

`diskReadTimeouts` / `prefix_read_timeouts_total` 现在明确包含尚未完成的 SSD 准入、同 checkpoint read fence 和已接收 IO 等待超时；每一项超时不一定有实际 archive 读取。`restoreWaits` 保持原来的同 checkpoint fence 等待口径，不是全部准入等待次数。正常 compute producer 等待有自己的协调合同，未进入 SSD attempt 前不消耗本次五秒期限。

HTTP lifecycle 分别保存 prefill/实际 forward tokens、decode service/decoded tokens、cache lookup/restore/save/wait。`cacheWaitSeconds` 扣除同步 lookup 与 restore，不能单独断言“等满五秒”；一般情况下 lookup+restore+wait 仍包含 producer 等待，也不是纯 SSD 读期限。C1 的受控场景没有 producer 等待或恢复，才可用三项之和核对准入总窗。`bytesRead` 是完整 archive 文件读取记账，不是物理设备 IO；不应据此直接计算设备带宽占用率。

尚未实现：intent→IO开始、文件 read/hash、callback 排队、写 hash/write/fsync、GPU export 的独立阶段耗时，以及全部导出前跳过计数。只有后续竞争实测需要解释收益时才加这些分段；本次三项指标足以区分 metadata owner、真正准入和已提交可选写受限，但不足以声称已能拆出纯队列延迟或公平性分布。

## 已完成证据与验收边界

`results/kv-night-c1/cpu-tests.log` 在 2026-09-09 04:27 记录所选 XCTest **177 项、0 失败**，其中新增 [Core read-intent 测试](../../Tests/ANERunnerCoreTests/QwenPrefixDiskReadIntentTests.swift) 10 项、[GPU 模块的 CPU deadline 测试](../../Tests/ANERunnerGPUTests/QwenPrefixReadAdmissionDeadlineTests.swift) 2 项；后两项没有创建模型或 GPU stream。其余所选缓存/预算/寿命回归也通过。该数字不是本仓库全部测试数，末尾另一测试框架的“0 tests”不覆盖这 177 项结果。

C1 使用 binary `bb2d6f8d21672402ebc5c8b540f99ba0d5c1019da0cd5d5c8de0e14c44e0f549`。`results/kv-night-c1/admission.json` 为 `complete=false, passed=false`：23 个通用 checks 中 22 真、`R3_5s_admission_timeout` 假，控制器随后退出并恢复参考服务。已完成 R0 oracle、R1 seed、R2 放行旧写后的 SSD 恢复、R3 等待后的冷回退四条 trial；每条实际输出 4 tokens 后 EOS，不能称为固定 O16 的四条完整长度输出。R1/R2/R3 的 IDs、结束原因和各自 121 tensor+host 状态边界均与独立冷 oracle 一致；这只是已完成子集，未运行到 R4–R6 的取消、clear、wholeStages 和最终关闭验收。

R2 已直接覆盖本次修复路径：不同 namespace 的小 CPU 写保留 pendingJobs=1、pendingBytes=2 时，读取仅持 1 个 intent，processed=0、read workspace=0、bytesRead 不变；新的可选写被拒绝。放行旧写后恢复 2080 tokens，只前向 1 token，产生一次实际 SSD 恢复，完整 archive 读取 118161408 bytes，结束后各 owner 清零。这个 gate 不是物理慢盘，也不代表并发大写下的性能收益。

C1 R3 的失败边界是探针把 `cacheWaitSeconds >= 5s` 当成总期限。报告实际为 wait=4.641774144s、lookup=0.359523772s、restore=0s，合计 **5.001297916s**；lookup 已被 wait 定义扣除。R3 已冷回退并保持输出/状态一致，但旧断言将多个条件合并且没有保存其后快照，因此不能由这个失败报告单独补认全部超时计数/未提交 IO/剩余 owner 断言通过。当前 probe 将它们拆为：恰好一次超时、bytesRead/命中不变、三项总时间、旧写仍持 charge、read workspace 与 intent 已释放，并先保存实际值。

另外，旧 [timeout probe](../../Sources/ANERunnerCLI/GPUCacheTimeoutProbe.swift) 的 R2 专门要求**已接收 IO 随后超时**，drain 后每次须有实际 archive bytes；仅 `diskReadTimeouts` 增加不能满足这个场景。新总期限可能让原 1µs 参数在读提交前就到期，因此当前独立短期限改为 5ms，保留最多三次竞争尝试和 bytesRead 断言。这是该探针的确定性覆盖参数调整，不改变服务默认五秒，也不声称 5ms 一定在所有机器上完成准入；仍以报告实际 IO 与超时路径为准。

C2 使用 binary `4d728eb5d3bb7efdf6d35df11799da76bd02aa5a347297b273bcf92fb02d7bf4` 完成五 case 回归：准入58项、已接收IO超时51项、完整会话71项全部通过；45个HTTP成功请求和3个取消均与48条唯一模型终态独立对账。R3总解析5.001235秒，timeout增1、hit/read字节无增，旧写仍拥有pending。已接收IO超时场景实际完成342,798,336-byte归档读取。236文件/102模型stat postflight和参考恢复已核对，详见[可靠性记录](../KV_CACHE_RELIABILITY.md)。这仍不是物理慢盘或小时级竞争性能验收。

后续C3隔离补丁已完成：库调用方合法暂停waiting cursor时，store在enqueue/acquire/state/statistics访问中按同一个绝对期限撤销未提交的优先权。无timer/额外worker，不改变已接收IO、私有request lease或producer所有权；低层Core API省略deadline仍为手动释放合同。clear/close/存储不可用先于expiry分类，旧UUID不能撤销新owner。187项CPU（新增9项Core fake-clock、1项时间转换）、70项准入、51项已接收IO超时和36成功/3取消HTTP通过。R7游标不合作5.054100375秒，其他enqueue自行过期旧优先权并成功；旧cursor随后冷退、timeout+1、无读回且保留新owner。详见[可靠性C3记录](../KV_CACHE_RELIABILITY.md)。C2历史证据不重写为C3结果。

## 最小验收清单

以下保留设计清单，完成范围以上节已完成证据为准；不把清单本身当成执行结果。

CPU验收使用已有阻塞available-space gate、临时小文件和现有read ticket CPU测试；不制造真实系统压力或大盘占用：

1. 已发布A，different namespace的一字节B写处于gate时获取A intent：intent busy、Core pending仍只计B、不增加读取job/bytes；第二intent和新的optional写受限。释放B后state.ready，再accepted读取A。
2. accepted读在callback gate停住时，Core仍完整charge，close须分别报告IO完成与callback未完成。释放原metadata handle不取消已接收IO。
3. clear/close即时撤销metadata-only intent且不等其调用方；旧handle release或foreign-store handle不得撤销新owner。RAM-only clear由GPU registry一并验证。
4. unavailable、closed、busy明确区分；ready到submit之间的clear/close等竞争不能安装fence或保留未转交workspace。已接收replacement写结束后重新peek验证大小。
5. CPU固定timestamp测试admission耗时传入accepted ticket、不重置5秒；已ready结果保留可用语义。保留全部已有timeout/迟到callback/revision测试。

实模最小场景只需一份真实A archive和原生cold oracle，无需第二份大GPU状态或真实慢盘：

1. 先缓存A并清RAM，使用store既有availableSpace注入gate暂停一个different namespace的小CPU dummy write。A cooperative恢复必须停在admission wait，processed=0、没有新增读workspace、没有read bytes增长。释放小写后A按原prefix从SSD恢复，完整IDs与原生state/host锚点一致。
2. 再次暂停dummy写，A等完整5秒后只冷退，不重复提交IO；后续释放dummy也不得产生这条已放弃请求的迟到读。
3. 分别在等待intent时取消、RAM-only clear、SSD clear；再释放dummy验证旧owner不清新intent、所有metadata/workspace归零。wholeStages遇busy立即冷退保兼容。

以上确定性注入证明“任何非零pending写使旧full-charge读取拒绝”的修复，不声称自然负载已复现。600秒churn预检的 `rejected=0` 没有覆盖这个拒绝路径。

通过条件：原生输出IDs、发布/恢复checkpoint及host状态一致；pending与joint workspace硬限额均不绕过；取消和close后的队列/fence/lease排空；busy意图接受后optional新写不能抢占；超时及wholeStages路径有界且无迟到IO。之后再做自然竞争/无竞争的TTFT、decode和较长组合验收，不把B1静态诊断值设为硬编码性能阈值。分块、真正优先队列和写入permit不属于本次已集成增量。
