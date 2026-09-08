# KV cache 管理与可靠性

当前工作集中在 AR 缓存生命周期。这个混合模型的恢复单位包括 Attention KV/QSA、GDN recurrent/conv、PLE convolution 和 n-gram 历史；只有 KV 不能恢复请求。MTP 性能优化继续延后。

2026-09-09 的[关键能力计划](KV_CACHE_CAPABILITIES.md)统一安排完整会话复用、真实压力控制、分层 I/O 调度、物理页共享与工业发布门槛。完整会话复用已接入；本页分别描述当前合同和各版本的验证，规划本身不构成交付证明。

## 使用合同

RAM 前缀缓存默认 512 MiB / 8 条。SSD 是显式开启的可选层，服务重启保留内容：

```sh
.build/release/ane-runner serve-gpu --model-dir /absolute/model/path \
  --prefix-cache-directory /Users/yourname/qwen-cache \
  --prefix-cache-disk-bytes 8589934592 --prefix-cache-disk-entries 32 \
  --prefix-cache-ttl-seconds 86400 --prefix-cache-min-free-bytes 1073741824 \
  --prefix-cache-restore-timeout-seconds 5 --prefix-cache-shutdown-timeout-seconds 30 \
  --state-budget-bytes 4294967296
```

父目录必须存在；最后一级由 store 创建为 0700，也可使用已有的专属 0700 目录。文件为 0600；同一目录只允许一个 store，通过进程锁拒绝重复持有。底层逐级打开目录且不跟随符号链接。请使用物理路径，尤其注意 macOS 的 `/tmp`、`/var` 是别名路径。缓存管理只删除符合自身命名格式的文件。禁用 RAM 缓存时不能同时配置 SSD。

RAM 与 SSD 都有条目、字节、key token 和 TTL 上限。RAM TTL 从本层发布/提升时计算；SSD TTL 从原始发布时计算并跨重启保留。SSD 文件字节取文件长度和实际分配块数中的较大者，写临时文件前也保留磁盘额度。后台读写最多 2 个 job、512 MiB 待处理数据；单个超限归档跳过缓存，继续正常计算。写入拒绝或失败不等于推理失败。

HTTP对完整canonical会话一次分词，查找至prompt倒数第二个token，并在原416-token网格上最多发布系统/工具锚点和会话尾部两个完整检查点。准确token前缀相同才复用；编辑历史、工具结果或工具定义不会跨越不一致处。共同系统锚点的producer身份与每个请求的完整查找范围分开，分支可以共同等待系统状态后独立执行。最深完整SSD状态可优先于较浅RAM状态；当前选择基于可复用深度，尚未用延迟成本模型判断哪个更快。

默认512 MiB无法同时保留约360 MiB系统和383 MiB尾部时，可选尾部RAM保存/提升不驱逐该请求的共享系统锚点；尾部仍可尝试SSD写回。硬字节/条目/key额度始终生效，压力trim、显式clear及联合额度回收仍可移除锚点。RAM-only时长会话尾部可能无法保留；这是一项容量取舍，不是每个会话的保留承诺。

`QwenGenerator.clearPrefixCache()` 只清 RAM；`includingDisk: true` 同时清 SSD。clear 不破坏已恢复的请求私有状态，并使旧生产者失去发布资格。`flushPrefixCacheWrites()` 等待已接收 I/O 与回调，旧 `closePrefixCache(drain: true)` 为同步排空接口。服务使用新的 `closePrefixCache(drain: true, timeout: 30)` / store有界close：一个单调时钟期限覆盖已接收IO、描述符关闭和callback完成，返回两个完成标志；超期只停止等待，实际工作继续保留Data/lease/FD所有权，不能提前释放。HTTP在唯一关闭路径记录实际排空结果，不删除有效归档。该期限不约束此前的GPU同步，也不能保证挂起POSIX调用立即结束。应用应在推理执行器上调用生成器接口。

## 联合额度和并发

每个模型共享 `QwenStateBudget`，分别计量 request、cache、workspace；多个 generator 不会绕过模型的总额度。请求在首次设备操作前，按最大提示词加输出长度预留完整状态，并额外预留一份旧/新状态更新空间。MTP 保守预留回滚与草稿份额，仍绕过 AR 缓存。

请求 lease 跟随 prefill 游标、单次 handoff 和 decode 游标，直至状态释放。RAM 快照在复制前申请独立 lease；主机归档、读回与序列化副本申请 workspace lease。可淘汰 RAM 后重试申请；仍不足则拒绝请求或跳过可选缓存。非流式容量拒绝为 HTTP 429 / `resource_limit`；SSE 若已发响应头，则发送结构化错误并结束。

这套额度覆盖逻辑模型状态和相应副本余量，**不是进程物理内存硬上限**。权重、一般 activation、MLX allocator 保留及初始化工作单独观察。不能拿 ledger 的峰值替代 RSS 或系统内存压力测试。

cooperative 调度中，同一 namespace 和完整 token 前缀只允许一个冷 producer；其他请求暂停 prefill，等待可用状态。producer 取消/失败后释放资格，下一个请求接管。RAM 无法保留且 SSD 尚在发布时，等待该次写入完成或失败；后台发布与读回分别受默认 5 秒的请求等待期限约束。超期后请求继续冷算，已有 I/O 和 workspace 由完成回调保留，不能提前归还额度。同 key 的未完成归档不会因超期重算而重复入队。等待期间不提交 GPU 工作；存在可运行 decode 时优先照常推进。

完整阶段的同步库调用持有模型 gate，无法等待一个由调用方暂停的外部 producer。这种混合使用场景安全回退到冷计算，可能重复 prefill；HTTP 使用 cooperative 模式。压缩 radix 只能恢复真实完整快照边界，不从中间树节点推导 GDN 状态。

服务接入 macOS Dispatch 内存压力通知。warning 暂停可选缓存保存/提升，并在推理执行器安全点每次回收一条 RAM 快照；critical 还会在 HTTP 入队和首次状态分配前拒绝新请求，已拥有私有状态的请求继续执行。收到较低级别后需稳定 5 秒再恢复准入。回调只修改小型策略状态，不访问 Tensor。首次事件前明确报告 unknown，不能视作已观察到系统正常；这一策略与逻辑字节账本共同工作，仍不构成进程物理内存硬上限。

## 持久化和故障

保存遵循临时文件、完整写入、fsync、rename、目录 fsync。启动清理自有未完成文件、校验归档并重建有界 LRU 索引。格式包含版本、token key、namespace、metadata 与 payload 校验；CPU 验证完整层布局、dtype、shape、长度、offset、PLE 历史后才分配 Tensor。归档保持 BF16 原始位，不经 Float32 转换。

namespace 绑定 checkpoint、本次运行实现及数值配置。小模型配置、tokenizer、模板、可执行程序和相关 MLX 动态库使用内容摘要；大权重和 n-gram 文件绑定物理路径、设备/inode、大小及纳秒 mtime/ctime。这是本机不可变 checkpoint 的缓存身份，不能作为可搬迁的模型内容证明。更新程序或 checkpoint 会自然 miss；旧文件仍受 TTL/LRU/总额度约束。

SSD 默认保留 1 GiB 文件系统可用空间，可通过 `--prefix-cache-min-free-bytes` 调整，0 禁用该保护。写入前按 `f_bavail × f_frsize` 检查水位和完整临时归档占用；临时文件写完并同步后再检查水位。空间不足或查询失败时跳过可选写入，已有归档仍可读，后续写入重新检查并允许恢复。该水位是尽力保护，不能防止其他进程在检查后占用磁盘。

损坏、截断、未知版本或状态描述错误回退正常 prefill。设备同步恢复失败仍沿用模型不可用机制；不能用缓存 miss 掩盖设备故障。目录不可写或删除失败时停止 SSD 新 admission，保留字节账目，避免失控累积；成功清理后可恢复。

## 观测与验收

`/health` 包含 `prefix_cache`、`prefix_cache_limits`、`prefix_disk_cache`、`prefix_disk_cache_limits`、`state_budget`、`mlx_memory` 、`memory_pressure`、`memory_pressure_monitor_running` 和 `waiting_prefix_sequences`。空闲执行器每 100 ms 刷新快照。分别看索引 hits、真正 restoredHits、diskHits、corruptions/writeFailures、pending jobs/bytes、liveFlights、diskReadTimeouts/diskPublicationTimeouts、spaceRejections/spaceQueryFailures/spaceRecoveries，以及 request/workspace 在空闲后的归零。

prefill报告`cacheSource`、cached/computed tokens、lookup/restore/save/wait，以及`actualForwardTokenCount`和`recomputedTokenCount`。成功HTTP终态JSON记录`prompt_tokens = cached_prompt_tokens + computed_prompt_tokens`，并单列`actual_prefill_tokens = computed_prompt_tokens + recomputed_prefill_tokens`。来源只记录最终实际采用的前缀，SSD读后提升RAM不重复计费；`restoreWaits`仅表示读fence等待，不证明节省了第二次SSD读取。日志分别保留prefill计算、执行器active/suspension及decode耗时。

`GET /metrics`提供[Prometheus文本格式0.0.4](https://prometheus.io/docs/instrumenting/exposition_formats/)；固定指标名，无prompt/token/request标签，包括真实有效恢复、SSD归档读写、等待超时、空间拒绝、联合额度、压力通知及MLX统计。尚未观察的值省略，压力unknown显示为-1。当前每次约41–42条series，普通请求身份仅在既有生命周期日志中出现。指标counter按进程/显式统计reset生命周期理解；不提供请求延迟直方图，分位数需从有界终态记录计算。

SSD归档是状态缓存，与PLE的n-gram读取分别管理。应用层成功archive字节计数不是物理SSD流量或DRAM带宽；诊断全状态读回会影响时间，不作为吞吐结论。

工业发布门槛要求：完整输出与混合状态正确；并发、取消、清理及预算耗尽后恢复；跨进程重启与损坏回退；长期压力下额度不越界、请求/临时 lease 不残留、资源不持续增长；有明确的长提示词尾延迟和恢复成本。短窗口验证只是这些门槛的一部分，不自动意味着可生产部署。Paged KV、跨请求 GPU batching、跨机器 PD 和 MTP 缓存尚未交付。

## 2026-09-09 批次B：完整会话、有限关闭和监控

构建70.58秒，165项相关Swift CPU检查全部通过，其中11项会话运行时和41项SSD测试。实测二进制SHA256为`8b694fcdcf656cd5fddc1b50a420c0c4f7581976f57b3c9e4afecce13a057dc2`，结果位于`results/kv-night-b1/`与`results/kv-night-b2/`。

原生会话探针71项检查通过：6次cache-disabled参考与12次对照生成，后者192个输出IDs全部一致，另有6个取消游标；完整状态共33次独立事件记录，其中21次发布/恢复、2541条BF16张量记录及对应host状态证明存储变换，余12次为重算边界对照。A重放恢复12064/12354 tokens，后续两分支恢复父历史12064/13028；旧消息/工具定义编辑、旧epoch取消、clear后私有decode继续、默认512 MiB系统保护均通过。详细计数及16-token输出边界见[独立复核](research/KV_CONVERSATION_B1_RESULTS.md)。同二进制pressure89checks和timeout50checks复测通过；pressure仍为注入策略测试，非实际OS压力。

B2 HTTP会话9次成功：多轮/分支/工具历史输出与独立CLI参考一致；新进程在空间水位拒绝新写时仍恢复12064-token SSD状态，新的冷请求跳过写入并正常完成。四次`/metrics`格式/唯一series/有限值检查及idle health对账通过，两次服务正常退出且记录IO/callback排空完成。A单窗口冷请求总耗时20.284秒、同历史重放1.125秒；计时包含响应与随后排空，不是单独TTFT，也不是重复性能验收。

补充mixed窗口47.03秒，32请求+4串行参考全部成功，288 completion tokens；17次SSD恢复、6,229,626,880归档字节读回、189 health samples。3次RST逐一与唯一`model_terminal=cancelled`对账，45次成功HTTP的usage、cached/computed/actual token守恒及非负阶段时间全部对账，合计48条唯一模型终态，无多余/重复记录。空闲request/workspace及SSD pending归零。

B1先前7个HTTP会话/两次metrics检查成功，但第二进程启动前测试socket bind报端口占用；前一服务已close完成并exit0。失败原始记录保留，B2仅修改harness使用三个独立端口，源码和二进制未变。B1的219文件与B2的222文件、各102模型payload stat均通过postflight；B2参考服务56391按原argv恢复、idle与MTP/drafter关闭已独立核对。后续操作必须读取最新ledger，不能直接使用本文历史PID。

本批仍未完成小时级持续淘汰、真实系统压力、发布配置24小时及稳定性能护栏；后续用1 GiB SSD、160 MiB RAM和多个不同10k+前缀进行持续读写/淘汰测试。

验证命令：`probe-gpu-cache-reliability --mode populate|restore|corrupt|lifecycle|pressure`；每次需新 output，restore/corrupt 需要对应 populate oracle。HTTP `scripts/probe_http_cache_reliability.py` 在已受控启动的服务上运行 100 次并发混合请求并持续采样。客户端 RST 的记录还应与服务器终态日志按请求 ID 对照。

## 2026-09-09 压力策略与 SSD 等待回归

本轮实现 macOS 压力通知接线、warning/critical 准入与恢复滞回、逐条 RAM 回收、SSD 最小可用空间及读回/后台发布等待期限。目录 rename/fsync 移出共享 admission 锁，迟到提交仍经 epoch 验证；读、写、提交线程与完成回调各保留自己的 workspace owner。CPU 先运行108项相关检查，最后修改后再运行54项压力/读owner/SSD检查（14+4+36），均通过；另9项新 churn 脚本控制测试通过。最后 release 构建61.96秒。

`results/kv-night-a2/` 使用二进制 `fc02358d194a3b8af858eef1153a61b61da2916208a57807f9da6ab0dd38502e`：

| 实模场景 | 完整请求 / 生成 IDs | 独立跨状态检查 |
| --- | --- | --- |
| warning、critical、取消资格/恢复、复制后压力变化 | 14 / 133，含2个冷基准 | 18组 / 1962个张量及host状态 |
| 11,057-token SSD读取期限、等待发布期限及恢复 | 6 / 96，含1个冷基准 | 5组 / 605个张量及host状态，剔除基准自身 |

压力模式的89项检查通过：活跃prefill/decode继续、已有私有状态不受RAM trim影响、warning下不保存/提升、critical拒绝首次分配且调度器仍可恢复。这里使用注入事件和虚拟稳定时钟，**不是实际系统内存压力**。真实HTTP监视器已运行，本窗口未收到OS事件，health明确为unknown。没有运行系统级 `memory_pressure -S`，该工具影响其他进程且不等于物理压力验证，依据见[macOS压力验收边界](research/KV_MACOS_PRESSURE_VALIDATION.md)。

SSD模式50项检查通过：1微秒测试期限确实触发一次未完成读回超时；5秒期限触发一次后台发布等待超时。请求均安全冷算；原I/O和workspace继续持有至完成，同key未重复归档，后续请求正常从SSD恢复。后台发布延迟由CPU可用空间采样门控制，并非真实挂盘；普通读取仍为实际文件读回。两套模型probe最终request/cache/workspace及leases均为零。当前 `flush/close` 仍可能等待正在运行的POSIX I/O；有限请求等待不等于有限服务关闭，后者继续开发。

同一二进制的强制SSD HTTP smoke为70.59秒，4个基准加40个并发请求，共44次成功/352生成tokens。4次客户端RST全部按ID核对唯一服务取消终态，44次成功也匹配唯一终态及prompt/completion/cached用量。实际21次SSD恢复、7,695,421,440 B归档读回、10次同前缀等待合并；282次health采样，无恢复失败、损坏、写失败或额度越界，最终request/workspace/pending jobs/bytes为零，仅保留有效RAM缓存93,523,976 B。该短窗不代替小时级churn。

215项源码/二进制/脚本冻结摘要与102项模型payload stat复核无变化。控制器恢复原参数参考，最新ledger为 `results/kv-night-a2/run-ledger.json`，本次PID52289、精确argv/idle/关闭MTP与drafter已另行复核。a1的失败报告保留：pressure测试误把maxTokens上限当成必须输出16个，P833合法EOS提前停止；修正测试后重跑通过，不将首次失败隐去。

## 2026-09-08 状态与故障回归

175 项 Swift CPU 测试通过，其中包含 21 项 SSD 存储、9 项归档结构、10 项联合账本测试；新增容量拒绝后的调度恢复、等待前缀时 decode 继续执行的检查也通过。HTTP parser 另有 7 项 CPU 回归。

| 实模报告 | 完整请求 / 生成 IDs | 扣除自身 anchor 后的跨状态检查 |
| --- | --- | --- |
| 持久化 populate | 1 / 32 | 1 组 / 121 张量 |
| 新进程 restore | 2 / 64 | 3 组 / 363 张量 |
| 损坏归档 corrupt | 1 / 32 | 2 组 / 242 张量 |
| 合并请求与预算 lifecycle | 10 / 95 | 本套无张量回读 |
| 既有缓存 lifecycle 回归 | 13 / 59 | 15 组 / 1635 张量 |
| 合计 | 27 / 282 | 21 组 / 2361 张量记录，及对应 host 状态 |

前四套使用二进制 `7de8fd305faa128a9f8e5e478e9360b6d40adb67fa5f5c08b1286ebcf1141ed0`；后续只修正游标注释并重新构建，既有 lifecycle 及 HTTP 使用 `c89d6b0598cba8e4b66b278008775a9abdaa941f1d74afe81be07a4291523418`。原始本机报告位于 `results/cache-reliability-v1/` 与 `results/cache-reliability-v2/`，不上传大型结果与模型。

11,057-token 提示词在新进程恢复 9984-token SSD 前缀，完整输出与 native BF16 状态匹配冷 oracle。两份私有游标推进后清空两层缓存，取消一份，另一份仍输出一致。损坏 payload 启动时拒绝，并通过冷路径重新发布有效条目。P833/K416 和 P2053/K1664 验证等待合并、leader 取消接管、预算拒绝后恢复。

四套新 probe 完成后 request/cache/workspace 与 lease 全部为零。populate/restore/corrupt 的联合逻辑预留峰值分别为 1,776,486,440 / 1,839,218,728 / 1,776,486,440 B；lifecycle 的 4 GiB 峰值是人为占满账本，触发两次预期拒绝，不能说实际分配了 4 GiB。旧 lifecycle 没记录该账本，不能补称它也实测了 lease 清零。唯一 MTP 请求为 depth2 / 8 IDs，仅作状态隔离回归。

首次 HTTP 运行被测试脚本的错误假设终止：冷 miss 可以省略 `prompt_tokens_details`，服务首个请求正常完成。已修复缺省为零的解析规则并补 CPU 测试；失败报告和当轮参考恢复记录保留，不将其记为服务故障，也不隐去失败。

## 1000 请求服务窗口

`results/cache-reliability-v2/` 的持续服务验证全部通过：966.47 秒（约 16.1 分钟），4 路并发，短/10k+ 系统前缀、流式/非流式交替；4 个串行输出基准加 1000 个并发请求，共 1004 次成功请求、8032 个生成 tokens。另有 84 次主动 RST：全部按 request ID 在服务日志中找到唯一 `model_kind=cancelled`，成功请求也逐项匹配唯一完成终态、输出用量和缓存计数。

3656 次健康采样及 84 个排空检查点没有额度越界或残留请求/临时 lease；最终只保留两条 RAM 缓存 lease，共 459,890,704 B。联合逻辑额度峰值 1,954,787,376 B，小于配置的 4 GiB。SSD 最终两条记录、460,001,280 B，pending jobs/bytes 均为零，没有新损坏、写失败或存储不可用。

191 次进程采样中数字 FD 为 11–18；前、后四分之一窗口的 RSS 中位数差为 +832 KiB。排空点 MLX active allocation 范围为 78,955,563,324–79,299,152,188 B；这些是不同统计口径的观察值，不能相加，也不能据此把逻辑账本称为物理 RAM 限额。没有观察到该窗口的持续资源增长，不外推为小时/天级无泄漏。

随后分别验证正常关闭后的新进程，以及空闲时 SIGKILL 后的新进程。两次均从 SSD 恢复相同长前缀，HTTP 200、完整输出一致、实际 diskHits 增长，请求/临时额度归零。SIGKILL 是进程退出试验，不是系统断电或活动写入中断试验；后两者的文件边界目前由 CPU 故障样本覆盖。

这一千次负载的主要命中层是 RAM，不能用它代替持续 SSD 读回压力证据。额外强制 SSD 轮次将 RAM 限额设为 160 MiB，使长前缀无法保留在 RAM 中。

强制 SSD 轮次已通过，结果在 `results/cache-reliability-ssd-v1/`：93.10 秒，4 个基准加 100 个并发请求，共 104 成功请求 / 832 生成 tokens；9 次 RST 全部在服务日志中确认唯一取消。实际 51 次 SSD 恢复、18,688,880,640 B 文件读回，25 次同前缀等待合并，无恢复失败或冷回退。大于 RAM 限额的长快照被跳过 52 次，最终 RAM 只保留一条 93,523,976 B 短前缀。

本轮联合逻辑峰值 2,321,514,435 B，375 次健康采样和 9 个排空检查点通过；最终 request/workspace、SSD pending jobs/bytes 均为零。它补足了真实 SSD I/O 与并发合并验证。19 次 RSS 采样的首/末四分之一中位数增加约 361 MiB，后段在反复读回时波动；本窗口不足以判断小时级内存趋势，不能把前一轮 RAM 命中的平稳 RSS 结论移植到 SSD 负载。

三个实测阶段均通过源码/二进制冻结核对及 102 项模型负载 stat，控制器在 finally 恢复原参数参考服务。最新恢复 ledger 为 `results/cache-reliability-ssd-v1/run-ledger.json`，本次 PID 17735、idle、MTP/drafter 未加载；接续时重新核对身份，不能复用陈旧 PID。核心存储阶段已推送 `1c5e3aa`，运行时/API 集成已推送 `d097657`。

当前定位是具备有界两层缓存、故障恢复和本轮验证证据的候选版本。工业发布仍保留两个明确未关闭的验证项：小时/天级持续 SSD、不同前缀与频繁淘汰的组合负载；真实系统内存压力下的 admission、延迟和恢复表现。持续运行试验应固定业务 SLO，并继续分别统计 prefill、decode 和缓存等待。SSD 保持显式配置，不因本轮短窗口自动提升为默认。
