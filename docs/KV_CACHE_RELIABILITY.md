# KV cache 管理与可靠性

当前工作集中在 AR 缓存生命周期。这个混合模型的恢复单位包括 Attention KV/QSA、GDN recurrent/conv、PLE convolution 和 n-gram 历史；只有 KV 不能恢复请求。MTP 性能优化继续延后。

2026-09-09 的[关键能力计划](KV_CACHE_CAPABILITIES.md)统一安排后续完整会话复用、真实压力控制、分层 I/O 调度、物理页共享与工业发布门槛。本页继续描述已实现合同和历史验证；计划中的能力不因列入路线而视为已交付。

## 使用合同

RAM 前缀缓存默认 512 MiB / 8 条。SSD 是显式开启的可选层，服务重启保留内容：

```sh
.build/release/ane-runner serve-gpu --model-dir /absolute/model/path \
  --prefix-cache-directory /Users/yourname/qwen-cache \
  --prefix-cache-disk-bytes 8589934592 --prefix-cache-disk-entries 32 \
  --prefix-cache-ttl-seconds 86400 --state-budget-bytes 4294967296
```

父目录必须存在；最后一级由 store 创建为 0700，也可使用已有的专属 0700 目录。文件为 0600；同一目录只允许一个 store，通过进程锁拒绝重复持有。底层逐级打开目录且不跟随符号链接。请使用物理路径，尤其注意 macOS 的 `/tmp`、`/var` 是别名路径。缓存管理只删除符合自身命名格式的文件。禁用 RAM 缓存时不能同时配置 SSD。

RAM 与 SSD 都有条目、字节、key token 和 TTL 上限。RAM TTL 从本层发布/提升时计算；SSD TTL 从原始发布时计算并跨重启保留。SSD 文件字节取文件长度和实际分配块数中的较大者，写临时文件前也保留磁盘额度。后台读写最多 2 个 job、512 MiB 待处理数据；单个超限归档跳过缓存，继续正常计算。写入拒绝或失败不等于推理失败。

`QwenGenerator.clearPrefixCache()` 只清 RAM；`includingDisk: true` 同时清 SSD。clear 不破坏已恢复的请求私有状态，并使旧生产者失去发布资格。`flushPrefixCacheWrites()` 等待已接收 I/O 与回调，`closePrefixCache(drain: true)` 排空并关闭而不删除归档。应用应在推理执行器上调用这些接口。

## 联合额度和并发

每个模型共享 `QwenStateBudget`，分别计量 request、cache、workspace；多个 generator 不会绕过模型的总额度。请求在首次设备操作前，按最大提示词加输出长度预留完整状态，并额外预留一份旧/新状态更新空间。MTP 保守预留回滚与草稿份额，仍绕过 AR 缓存。

请求 lease 跟随 prefill 游标、单次 handoff 和 decode 游标，直至状态释放。RAM 快照在复制前申请独立 lease；主机归档、读回与序列化副本申请 workspace lease。可淘汰 RAM 后重试申请；仍不足则拒绝请求或跳过可选缓存。非流式容量拒绝为 HTTP 429 / `resource_limit`；SSE 若已发响应头，则发送结构化错误并结束。

这套额度覆盖逻辑模型状态和相应副本余量，**不是进程物理内存硬上限**。权重、一般 activation、MLX allocator 保留及初始化工作单独观察。不能拿 ledger 的峰值替代 RSS 或系统内存压力测试。

cooperative 调度中，同一 namespace 和完整 token 前缀只允许一个冷 producer；其他请求暂停 prefill，等待可用状态。producer 取消/失败后释放资格，下一个请求接管。RAM 无法保留且 SSD 尚在发布时，仍等待该次写入完成或失败。等待期间不提交 GPU 工作；存在可运行 decode 时优先照常推进。

完整阶段的同步库调用持有模型 gate，无法等待一个由调用方暂停的外部 producer。这种混合使用场景安全回退到冷计算，可能重复 prefill；HTTP 使用 cooperative 模式。压缩 radix 只能恢复真实完整快照边界，不从中间树节点推导 GDN 状态。

## 持久化和故障

保存遵循临时文件、完整写入、fsync、rename、目录 fsync。启动清理自有未完成文件、校验归档并重建有界 LRU 索引。格式包含版本、token key、namespace、metadata 与 payload 校验；CPU 验证完整层布局、dtype、shape、长度、offset、PLE 历史后才分配 Tensor。归档保持 BF16 原始位，不经 Float32 转换。

namespace 绑定 checkpoint、本次运行实现及数值配置。小模型配置、tokenizer、模板、可执行程序和相关 MLX 动态库使用内容摘要；大权重和 n-gram 文件绑定物理路径、设备/inode、大小及纳秒 mtime/ctime。这是本机不可变 checkpoint 的缓存身份，不能作为可搬迁的模型内容证明。更新程序或 checkpoint 会自然 miss；旧文件仍受 TTL/LRU/总额度约束。

损坏、截断、未知版本或状态描述错误回退正常 prefill。设备同步恢复失败仍沿用模型不可用机制；不能用缓存 miss 掩盖设备故障。目录不可写或删除失败时停止 SSD 新 admission，保留字节账目，避免失控累积；成功清理后可恢复。

## 观测与验收

`/health` 包含 `prefix_cache`、`prefix_cache_limits`、`prefix_disk_cache`、`prefix_disk_cache_limits`、`state_budget`、`mlx_memory` 和 `waiting_prefix_sequences`。SSD 开启时空闲快照至少每 100 ms 刷新。分别看索引 hits、真正 restoredHits、diskHits、corruptions/writeFailures、pending jobs/bytes、liveFlights，以及 request/workspace 在空闲后的归零。

prefill 报告 `cacheSource`、cached/computed tokens、lookup/restore/save/wait 时间；prefill 与 decode 分开统计。SSD 归档是状态缓存，与 PLE 的 n-gram SSD 读取不是同一种 I/O。诊断全状态读回会影响时间，不作为吞吐结论。

工业发布门槛要求：完整输出与混合状态正确；并发、取消、清理及预算耗尽后恢复；跨进程重启与损坏回退；长期压力下额度不越界、请求/临时 lease 不残留、资源不持续增长；有明确的长提示词尾延迟和恢复成本。短窗口验证只是这些门槛的一部分，不自动意味着可生产部署。Paged KV、跨请求 GPU batching、跨机器 PD 和 MTP 缓存尚未交付。

验证命令：`probe-gpu-cache-reliability --mode populate|restore|corrupt|lifecycle`；每次需新 output，restore/corrupt 需要对应 populate oracle。HTTP `scripts/probe_http_cache_reliability.py` 在已受控启动的服务上运行 100 次并发混合请求并持续采样。客户端 RST 的记录还应与服务器终态日志按请求 ID 对照。

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
