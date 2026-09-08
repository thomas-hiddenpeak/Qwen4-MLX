# K02 会话缓存 B1 实模结果复核

2026-09-09。根执行器完成实模运行后，本页独立读取 [conversation.json](../../results/kv-night-b1/conversation.json)，重新比较输出 ID 数组、实际前向统计和原生状态锚点；没有再次运行模型。报告 `complete=true`、71/71 checks 通过。

范围是单模型、单 executor、AR、chunk416、合成但结构真实的完整多轮/工具历史。6 次独立 cache-disabled 参考、12 次成功对照、6 个取消游标。默认前半段为 RAM 1 GiB，C7/C8 换用 RAM 512 MiB；SSD 全程为4 GiB、16 entries、pending 2 jobs/1 GiB。这不是完整默认配置的容量或性能验收。

## 实际命中与前向量

公共 system/tool checkpoint 为 S=11232。A/B/E/T tail 为12064，N1/N2 自己的 tail 为12896；续轮能够恢复已有 A 的12064，虽然它不是续轮本次的发布候选。模型实际前向量与 `prompt - cached` 每条相等，`recomputedTokenCount` 全部为0。

| 场景 | 输入 | Prompt | 实际复用 | 实际前向 | 来源 |
| --- | --- | ---: | ---: | ---: | --- |
| C1 新 epoch producer | A | 12354 | 0 | 12354 | cold |
| C1 相同 system、不同 tool result waiter | B | 12354 | 11232 | 1122 | RAM |
| C2 原历史重放 | A | 12354 | 12064 | 290 | RAM |
| C5 旧 hint 限制浅恢复 | A | 12354 | 11232 | 1122 | SSD |
| C5 深 SSD 优于浅 RAM | A | 12354 | 12064 | 290 | SSD |
| C3 完整续轮 | N1 | 13028 | 12064 | 964 | RAM |
| C3 同父历史另一分叉 | N2 | 13028 | 12064 | 964 | SSD |
| C4 早期历史编辑 | E | 12355 | 11232 | 1123 | RAM |
| C4 工具 schema 编辑 | T | 12354 | 0 | 12354 | cold |
| C6 clear、同伴取消后的私有 decode | A | 12354 | 12064 | 290 | RAM |
| C7 producer 在 S 前取消，waiter 接管 | B | 12354 | 0 | 12354 | cold |
| C8 producer 发布 S 后取消 | B | 12354 | 11232 | 1122 | RAM |

12 条对照合计149597 prompt tokens、复用105248、实际前向44349；其中6条RAM命中、3条SSD命中、3条按设计冷算。这个包含清空/取消故障的特定请求集不能当作生产命中率。

从 S 恢复改为12064恢复，A 的前向量由1122降至290，减少832；N1/N2 则各只需964。诊断读回会改变时序，本次不据此声称 TTFT 或吞吐提升比例。

CPU 实际 LCP：A/B=11829、A/N1=A/N2=12354、N1/N2=12368、A/E=11256、A/T=26。因此不同工具结果、分叉和早期编辑不会误借旧 tail，工具定义改变也不能误借 S。原 system 未修改，三类重复次数均为64，首轮 fixture 检查即通过。

## 输出与完整混合状态

- 12 条对照各16个输出 IDs，合计192个，与对应冷 oracle 的数组逐项相等；结果内嵌数组和报告数组也相等。finish reason 均为 `length`，最终 offset 均为 `prompt + outputCount - 1`。6 个参考合计96个 IDs；本轮实际没有触发 EOS，不能据此新增短 EOS 覆盖声明。
- 冷参考保存12个独立 checkpoint。缓存发布/恢复的跨状态比较采用较窄口径：10次 `publish` + 11次 `restore` = 21次、2541份原生 BF16 tensor shape/dtype/byteCount/SHA/finite 记录，加相应 host 状态，全部相等。`publish` 在 `privatePrefixStateCopy` 之后观察保存副本，相对前置冷 oracle 没有同对象自比较。
- 报告另有12次后续执行的 `coldBoundary` 重算边界对照；它们引用已经完成的独立 cold generator 锚点，不能新增发布/恢复覆盖。33次事件、3993份张量记录是包括这12次在内的总数，不称3993次独立缓存转换比较。前置 `cold_oracles` 的12个参考锚本身没有列入 `state_checks`。
- host 的 offset、GDN/attention offsets、PLE history 和 capture flags 全部与相应参考匹配，capture 均为空。恢复副本的物理保留容量不作为逻辑等价条件。
- N1/N2 在12064的恢复引用 A 的冷锚点；复核实际输入 IDs 的前12064完全相等。A/B/N1/N2/E 的独立 S 锚点彼此一致。

这些比较覆盖 checkpoint 和完整生成输出；没有读回每个最终 decode tensor，也不能把33次事件当作33次独立完整请求。SSD 发布计数与 RAM `publish` observer 不是同一口径：RAM 保留 S 时尾部仍可写 SSD。

## 取消、默认 RAM 容量与排空

6 个取消游标全部记录 `cancelled=true`、`finished=true`，取消点依次为旧 A/B 的0/0、C6 left 的12353、C7 producer 的416、C7 默认容量额外恢复的11232、C8 producer 的11232。四次显式 waiter 检查均为 processed=0、waiting=true、没有生成 handoff。

C1 clear 后先建立新 owner，再取消旧 producer/waiter；快照实际同时保留3个 request leases。新 B 仍等待新 A，共同 S 只计算/发布一次，A/B 的各自 tail 都发布。C8 则在 S 已发布后取消 producer，B 从11232恢复，未出现 A 未完成 tail 的发布事件。

S 估计378187784 bytes（360.67 MiB），A tail 401829896 bytes（383.21 MiB），合计743.88 MiB，确实不能同驻512 MiB RAM。C7 B 发布 tail 后 `retainedSystemAnchorSkips=1`、RAM 恰为一个 S 条目；不同尾部 A 随即直接恢复11232后取消。该额外游标只证明恢复/所有权，不是第13条完整输出对照。

所有阶段排空快照的 request/workspace、liveFlights 和 SSD pending jobs/bytes 均归零，scheduler 仍 idle 且 accepting。C6 clear/cancel 后幸存 decode 输出一致；最终 clear 后 request/cache/workspace/total bytes、currentLeases、RAM/SSD entries 全部为0，逻辑账本峰值3225837632 bytes，rejections=0。模型权重、激活和 MLX allocator 不在这个账本中，不能解读为 RSS 归零。

## 证据标识与边界

- 报告 SHA256：`36545805400e1e02c682b39f03cfacb554ee0ba8f40dd66ea133331dbbb99091`。
- 实际执行文件 SHA256：`8b694fcdcf656cd5fddc1b50a420c0c4f7581976f57b3c9e4afecce13a057dc2`。
- 原 system SHA256：`6306b05508091996acba55e484170ae8c8b470f70695087ddff891f73466d476`；完整模型配置和 tokenizer 摘要保留于报告。

本探针明确 `http_transport_tested=false`。HTTP/nonstream/SSE、真实默认 SSD 队列容量、长时 churn 和性能对照由其各自报告验收；不得用本页替代。
