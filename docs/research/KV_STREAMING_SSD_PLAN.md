# 262K SSD流式归档实施计划

2026-09-14。**状态：计划，尚未实现或验证。** 目标是在原生262144总窗口下持久化并恢复完整AR前缀状态。当前262K RAM/HTTP有界验证与约1.915GB SSD恢复结果见[长上下文记录](KV_LONG_CONTEXT_RESULTS.md)；它们不等于支持完整262K SSD。内部restore漏传1GiB上限的问题已修复，大归档仍被v1绝对2GiB上限及整份Data暂存共同阻挡。本页是[KV关键能力计划](../KV_CACHE_CAPABILITIES.md)的大归档增量，不等待MTP性能或跨检查点页去重。

**建议首版按tensor流式处理完整dense AR归档，保留原公开v1 API和默认值。** 每次只组装/导入一个模型规定的tensor，IO以8MiB分块；不先实现新Metal导入kernel或页去重。这能移除整份约7.5GB host payload，又不依赖尚未证明的零拷贝接口。它的暂存上限是一个tensor，不是仅8MiB：该区别必须写入配置和验收。

## 1. 模型推导的上限与待验证准入

- [归档描述](../../Sources/ANERunnerCore/QwenPrefixStateArchiveDescriptor.swift)的`expectedTensors(layout:offset:)`已能从本机模型配置推导超过2GiB的规范布局。B262080有121个BF16 tensor，payload为7,506,284,544B，另有8B PLE历史。最大K/V tensor为268,369,920B；模型最大offset262144时为268,435,456B（256MiB）。offset、121名称/顺序、shape、dtype、每tensor长度、总长度均取模型推导值，磁盘声明只能用于严格相等校验。
- 首版一个stream transfer，只有一个未ack的tensor缓冲；IO片段C=8MiB。预先分配准确长度的私有Data并按区间填充，禁止反复append增长、无界chunk数组或生产者提前排满整份归档。GPU消费并释放该tensor暂存后才能读下一tensor。并行双缓冲、跨请求批量传输后置。
- 初始workspace预算假设为`W = 2×H + 2×C + M`，H为该布局最大tensor，M拟设64MiB控制额度（v2 manifest另限8MiB、metadata仍64KiB、token key至多262144项），代入得到592MiB。**这不是已经证明充足的额度。** 实现时必须逐个核对host Data、contiguous设备暂存、IO块、metadata解析副本及最后owner；发现额外同时存活副本就调整准入，不能直接套用592MiB。最终恢复State另计，也不能因多个引用指向同一Data就提前归还其lease。
- 现[AR请求准入](../../Sources/ANERunnerGPU/QwenGeneration.swift)已预留`2×estimatedPrefixStateBytes(P+O)`。满窗口为15,016,206,352B；加B处RAM快照7,506,284,552B及592MiB暂存，约21.554GiB。这个模型形状与假设预算的算术结果低于24GiB profile，不是实测RSS、实际峰值或预算充分性的证明。旧2×完整payload workspace会把相同组合推至约34.96GiB。新restore的最终State使用请求已预留部分，明确转交owner，不额外无说明地再收两份；可选RAM promotion须另行准入。模型权重、一般activation、allocator仍不在此账本内。

## 2. 最小代码落点与线程合同

| 落点 | 必须改变的行为 |
|---|---|
| [Core descriptor](../../Sources/ANERunnerCore/QwenPrefixStateArchiveDescriptor.swift) | 提取共同的结构校验，v1继续守住原2GiB合同；新stream校验使用模型推导的精确总量及显式maxArchiveBytes，不把不可信磁盘shape送入MLX。 |
| [Core disk store](../../Sources/ANERunnerCore/QwenPrefixDiskStore.swift)的新增stream路径 | 新建纯CPU transfer owner，持有已打开FD、epoch/revision、规范byte offset、增量SHA、唯一buffer和lease；区分总归档/磁盘额度与实际在途buffer额度。offer/ack一次只交一块，queue不等待GPU回调，不保留Tensor。 |
| [GPU archive](../../Sources/ANERunnerGPU/QwenPrefixStateArchive.swift)与[模型](../../Sources/ANERunnerGPU/QwenModel.swift)的新增cursor | 在固定推理线程持有不可变dense源或未发布的私有目标State。export按规范tensor读取BF16原始字节；import用现有`MX.array(data:shape:dtype:)`生成私有tensor，eval完成后ack并释放本轮Data。所有contiguous/view/array/eval/State销毁只在该线程。 |
| [Prefix cache](../../Sources/ANERunnerGPU/QwenPrefixCache.swift) / [generator](../../Sources/ANERunnerGPU/QwenGeneration.swift) / [HTTP安全点](../../Sources/ANERunnerCLI/GPUHTTPServer.swift) | 把现一次性的export/lookup改为有界cursor推进；读未就绪时yield，后台写只在有空槽时取得下一tensor，空闲worker也推进。源必须是已私有化的compact RAM Snapshot，pin其现有cache lease；首版没有这个可信源或没有额度就跳过可选写，避免再复制整份状态。 |

不能在推理线程阻塞等SSD ack，否则会破坏已有PD调度/可选写回语义；也不能让IO callback调用MLX。Core callback只更新host fence/mailbox，GPU cursor在后续安全点消费。request结束后仍可由generator保留有限的export cursor，直到结束或取消；pending transfer需要可见计数，空闲判定不能忽略仍持有的源快照和workspace。

[当前Tensor包装器](../../Sources/ANERunnerGPU/Tensor.swift)的`MX.array`调用复制输入的`mlx_array_new_data`，不能据其他API的名称推断已有零拷贝导入路径。先利用这个已有私有复制合同；若之后要求严格O(8MiB) host暂存，再单独比较私有tensor sliceUpdate与native builder，验收设备复制成本和别名寿命。首版不需要此优化。

## 3. 格式、校验与原子发布

采用独立v2 magic和`qwen-prefix-v2-`文件名前缀，临时文件也用v2前缀；新store在同一个既有目录锁下识别v1/v2并保留旧v1 reader，不按格式另建互不排斥的目录锁。不可把大文件偷偷写成v1：现`decodeFile`把payload>maxPendingBytes判坏，重启scanner随后会删除它。当前旧scanner只遍历v1 entry/tmp名字，因此仅更改header版本不够，必须同时隔离entry和tmp命名；旧binary的启动、clear与临时文件清理都须忽略v2。新reader也要将不支持的格式/size-policy与实际损坏区分，不能因当前准入额度较低就删除合法v2。manifest仍绑定真实token前缀、namespace、revision/epoch、期限及完整状态metadata，不只比较offset。

写入前按整份最终文件及临时文件分配块预留磁盘额度、检查可用空间；仅buffer占用计入pending RAM。创建私有`O_EXCL/O_NOFOLLOW`临时文件，写固定header占位及有界manifest，逐tensor按顺序写入并增量SHA；最后回填总长度/manifest SHA/payload SHA，`fsync(file) → rename → fsync(directory) → epoch复核`后才公开索引。取消或失败清理临时文件/已rename但未公开的candidate，沿用现有丢失可选缓存而不破坏推理的边界。

恢复先用有界header/manifest及本机模型布局校验全部121描述和PLE history，然后从同一个FD顺序读取，逐tensor构造**不向任何请求/缓存发布**的私有目标State。总byte数、完整payload SHA、文件末尾/同inode及mtime复核、全部offset/history验证、最后eval和取消检查全部通过后，才一次性转交有效State并计成功hit。首版单pass允许在全payload SHA完成前分配私有目标；这改变构造时机，不能把部分State用于forward。若合同要求先验完整SHA后才分配，必须显式接受第二次全盘读取，不能隐瞒成本。

## 4. 取消和close必须与流式生命周期一起改

取消/超时/clear立刻撤销新chunk资格与结果发布资格，已在执行的POSIX IO继续持有FD、Data和额度至真实返回；GPU线程同步后释放自己的部分State/源引用。IO completion不捕获GPU cursor/State，防止最后引用在callback线程析构。一个请求放弃等待后不能另起同key重复stream，直到旧owner退出；设备执行失败需要保留可区分的错误身份并判定模型可用性，不能因后续同步成功就归为普通SSD miss或磁盘corruption；本地额度拒绝和可恢复的分配失败不应一律标成永久设备故障。当前包装器在这方面的缺口见[失败分类边界](KV_SSD_FAILURE_CLASSIFICATION.md)，流式路径不能继承含糊的错误合同。

**现有close仅在队列尾插marker，不能原样套给暂停的stream。** marker可能越过尚未供给的后续chunk，并提前关闭directoryFD或报告完成。应在begin stream时进入全生命周期group，最终chunk/abort清理及callback最后owner结束才leave；关闭同时停止新stream/chunk admission。HTTP退出先在推理线程取消尚需GPU供料的cursor并释放其设备状态，再让Core用同一已有deadline排空已接受的host IO和清理。不能先阻塞Core close、再等待已停止的GPU线程生成下一块。超时返回明确未完成，保留真实owner，既不提前关在用FD也不追加第二个无限等待。

## 5. 最小验收顺序

先在原2GiB以内验证新路径：121个tensor的规范原始字节及完整host状态逐位roundtrip、两分支不变、每tensor ack/慢consumer、取消/clear/close迟到完成、截断/多余字节/换文件、临时发布失败。检查最大持有buffer数与bytes随上述单tensor/IO上限变化，不随archive总长增长；逐owner验证预算假设、最终lease/FD归还，并分别记录恢复墙钟和业务等待。另有两项明确的CPU生命周期门禁：

- **旧scanner兼容：** 新版创建小型v2 entry/tmp后退出，旧版依次启动、恢复、clear和清理临时文件，再让新版重开；在旧版操作阶段v2文件数量及完整字节哈希必须不变，不能出现被误认损坏而删除的记录。旧v1继续按原合同读取，两个版本仍由同一个目录锁互斥。大文件长度的计算和size-policy拒绝先用CPU受控输入覆盖，无需为兼容测试先写7.5GB文件。
- **完整transfer关闭：** 分别暂停在首块尚未供给、IO执行中、最后一块等待ack/callback时执行cancel/clear/close。只要真实owner仍持有FD/Data，就不得报告已排空或归还额度；允许完整abort清理后正常完成。超时必须报告未完成，放行后最终归还，内部chunk/ack等待不能延长同一次close调用的期限；后续显式调用仍可按自己的期限等待同一关闭结果，但不得重复释放transfer或关闭FD。另验证HTTP退出先终止GPU供料cursor、后进入Core关闭，避免等待已停止执行器的死锁。

随后显式打开v2大归档：用262K RAM checkpoint写入，清RAM或重启新进程后只能从SSD恢复B262080，真正计算62-token suffix并decode；与独立冷参考的完整121 tensor/host/输出比较。读/写字节、总文件与pending-buffer额度、取消/关闭、重启及最终清理各有证据，才可称支持full262 SSD。不把大文件写成功、RAM命中或1.915GB现有回归当作这项通过，也不让MTP阻塞此增量。

现有生命周期与关闭合同见[缓存可靠性](../KV_CACHE_RELIABILITY.md)，发布目标见[关键能力计划](../KV_CACHE_CAPABILITIES.md)。本页所有流式接口、buffer上限及预算均为待实现并验证的方案；不据此宣布已有功能、实际内存节省或性能收益。
