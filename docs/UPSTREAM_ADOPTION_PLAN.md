# 从 vLLM、SGLang、DwarfStar 吸收的实施计划

2026-09-07。目标仍是让当前 Qwen3.8-Flash-Next checkpoint 在 M5 Max 上稳定、快速运行，保留 Swift 独立 runner；项目名 Qwen4-MLX 不表示更换模型。本页区分已实现基础、正在验证的候选和后续工作，不把上游功能清单当成本项目的支持列表。

主要结论：先减少每个 decode 步骤的等待、把 MTP 的有效产出和完整成本算清楚，再利用现有本机 PD 分离改善多请求等待。MTP 和请求生命周期稳定后，完整前缀状态复用最贴近 agent 重复提交 10k 系统提示词的负载。单独缓存 Attention KV 不足以恢复这个混合模型。

## 来源与适配边界

| 项目及固定快照 | 最值得吸收 | 本机需要调整的部分 |
| --- | --- | --- |
| vLLM `6865e67f0be02d53694517f6f71d7fb96492792d` | 分阶段调度、请求级 speculation 指标、混合缓存组与取消生命周期 | CUDA batching/分页及跨实例传输不能直接接入单执行器 MLX；并发请求数不等于 GPU batch size |
| SGLang `2c05ed4e7776c876478f4b2db61acb12b9a27d01` | 完整混合状态的公共前缀、ReplaySSM、分层缓存和会话淘汰 | GDN、PLE、QSA 与 MTP 状态必须同边界；统一内存先做 RAM→SSD 两层 |
| Redis 作者的 DwarfStar，`antirez/ds4`，`9ab705347c1775e7599ede7eb81a6255ec7dccb5` | Metal BF16 提前载入、shared/routed 调度重叠、精确前缀 checkpoint、SSD 计量 | 模型公式、量化与 lane 归约不同；其 SSD 专家权重读取不同于我们的 n-gram 行读取 |

逐项原始链接、源码位置、许可和适配判断见 [vLLM / SGLang 调研](research/VLLM_SGLANG.md)及 [DwarfStar 调研](research/REDIS_AUTHOR_RUNNER.md)。当前仅借鉴设计，没有复制这些项目实现；之后若移植实现，应记录具体文件与版权，而非只引用根许可。上游速度和模型规模不作为本机收益预测。

## 已有基础

- 普通 AR、显式 MTP、真实 11k prompt、分阶段计时、完整 token 对照和独立参考服务。
- 显式 prefill / decode / verification 阶段；同进程、同份权重、单执行器的 cooperative 调度，有界排队、token 预留、取消和错误清理。
- GDN / PLE / Attention / QSA / MTP 的事务状态交接。当前 prefill handle 单次消费，尚不具备多请求共享 checkpoint 的生命周期。
- SSD PLE 预取、专用 MoE / GDN 探针、独立原生库构建和可恢复的实验控制器。

这些基础支持下面的局部实验；不表示已有 HTTP 服务、跨进程 PD、连续批处理或共享前缀树。此前 MoE 组合单层 +5.21%，完整 prefill 828→824 token/s，已保留为可选，并未因此替换默认。

## 实施顺序与决策

| 顺序 | 工作 | 最小落点 | 进入下一步的依据 |
| --- | --- | --- | --- |
| 1A，已实现并回归 | [MTP 请求成本摘要](MTP_COST_SUMMARY.md) | 既有计数导出实际 decode token/s、平均步骤产出、草稿浪费和计时覆盖；不增加同步 | 46项相关CPU检查、9轮11k及预算/EOS通过，原始字段可重算 |
| 1B，本版未通过性能筛选 | [GDN BF16 提前载入](GDN_PREFETCH_EXPERIMENT.md) | 仅 S1、K2560/N10240 QKV，原 lane/FMA/归约顺序；scalar 与 vector 两种 load | 四层权重逐位与命中计数通过，但QKV约慢1%–2%，未进入整模型 |
| 1C，已完成首轮 | [输出间隔与调度实验](SCHEDULER_LATENCY_EXPERIMENT.md) | 实际 callback p50/p95/max、提交到终态耗时、既有 decodeBurst 参数 | 同时提交11k与短请求后交错，各18项gate通过；burst4/8正序观察有漂移，默认未改，延迟到达场景另测 |
| 2A | MTP depth 0/1/2 成本对照 | 先静态配置，不在线切换；高接受和低接受各一个真实任务 | 单独报告 prefill、decode 与服务等待；以实际输出/总 decode 时间判断，而非接受率 |
| 2B，单层筛选已完成 | [GDN ReplaySSM](GDN_REPLAY_FEASIBILITY.md) | 复用现有小 recurrence，生产capture不变 | 720项逐位比较通过，S3加权单层约+2%–3.5%；尚不足以证明整模型收益，暂不替换生产路径 |
| 2C，单层筛选未达门槛 | [shared/routed down 调度](MOE_BRANCH_OVERLAP_FEASIBILITY.md) | 合并两个down节点的输入hazard，再调用原primitive；MLX本来已用concurrent encoder | 三行逐位与编码计数通过，reference/fused约+3.04%/+4.16%，未达5%，pair未稳定胜过串行/原recipe，不进入整模 |
| 3，依赖 MTP/生命周期稳定 | 不可变内存 checkpoint | 先有界精确系统提示词表、私有恢复，再加最长前缀索引 | 10k 公共前缀 + 不同后缀，A/B/A 无污染；节省的 prefill 大于保存/恢复成本 |
| 4，依赖内存 checkpoint | SSD 状态缓存 | 有版本、身份与完整性校验的文件，独立磁盘额度，有界读写 | 冷盘恢复快于重算，损坏/中断写入正常回退，不拖慢 PLE 读取 |
| 服务交付线 | HTTP/SSE、背压和断连 | 复用单推理执行器、现有取消/额度；网络线程处理有界输出 | 客户端慢读/断连不阻塞其他请求，终态只发一次、资源只释放一次；AR/MTP 分别验收 |

1A/1C 是测量与可用性补足，1B 是性能实验，可以并行写代码，但 GPU 实测串行。微测平或更慢就停止扩大该候选；完整模型没有可重复收益就维持现有默认。初步以局部约 5%、整模型约 3% 作为值得继续的筛选量级，最终决策结合运行漂移，不能因一次跨过阈值宣布成功。

当前插入一项有明确原因的诊断：11k深度对照出现持续降速，allocator计数稳定且已记录PLE等待不足以解释主要下降。[固定AR驻留对照](RESIDENCY_DRIFT_DIAGNOSIS.md)已完成，fit未阻止降速，进程分页和footprint未发现对应增长；下一步用原生GPU命令缓冲计时缩小发生位置，再继续发布性能结论。不能将一次较慢结果自动归因为训练或热降频。

## MTP 的统计与状态约束

`accepted / drafted` 表示草稿质量。实际 decode 产出排除 prefill 已算出的首 token，EOS 和剩余预算按真正发布/提交数量处理；不能用 `1 + accepted / rounds` 代替实际产出。剩余预算为一时的 target-only 收尾也产生 verify 成本，但不一定增加现有 speculative rounds。计时摘要必须保留这一区别。

阶段时间只在定义互斥时求和。未覆盖的 host 时间、回调/排队时间单独列出，不给未知值编造分布。后续自适应深度要以整轮成本与等待控制，并保持 head history；暂停 MTP 后重新启用需要明确同步规则。[现有 MTP 发布条件](MTP_RELEASE_CRITERIA.md)继续生效。

ReplaySSM 的机会是少写验证期间每位置的完整 recurrent state，不是跳过主干验证或更改权重投影。当前单层每位置 1.5 MiB 的逻辑 capture 估计只用来确定测量对象；不能把它当成已测 DRAM 流量或承诺 decode 加速。

## 前缀复用的首版合同

一个 cache entry 代表同一 token 前缀、同一已消费边界的完整状态：Attention KV，QSA 原始/池化键与绝对位置，GDN recurrent/conv3，PLE conv9 与 hash 历史，以及适用的 MTP head/history。待输出 token、EOS、请求预算和已发布游标由新请求重建。

身份至少包括精确 token IDs、权重/tokenizer、dtype/布局、kernel 数值配置与固定 prefill chunk 策略。初期只在已有合法 chunk 边界保存，命中后沿原边界继续。recurrent 最终状态不能任意裁剪回早期前缀，不能将可变的单次 prefill handle 交给两个请求共同修改。

先支持一个真实 10k 公共前缀和多个后缀，测试 A/B/A、QSA 阈值、chunk 边界，再加淘汰/前缀树。统一内存预算区分随 token 增长的 KV/QSA、每请求固定 GDN/PLE、MTP history/临时 capture、共享权重，并与 MLX 实际内存一起观察。session ID 只提示复用倾向，不替代 token 校验或无限 pin 内存。

## 苹果硬件取舍

当前主要 decode 时间在 GPU 内，先尝试减少 GPU 访存等待、临时状态流量与小 dispatch 空隙。CPU 负责有界调度、tokenizer、SSD I/O 和状态元数据；从真实等待决定是否增加读取 worker 或 PLE 行缓存。PLE 计量应报告 unique rows、实际请求读取字节和暴露等待，区分 OS 页缓存与物理 SSD。

ANE 仍是可选的异构实验方向。只有某个可独立执行的子图连同转换/同步成本一起降低关键路径时间，才值得接入；为使用所有计算单元而同时复制权重或拆碎逐层依赖，不保证吞吐提升。当前三项目没有提供可直接吸收的该模型整模 ANE 路径。

暂不纳入：为模仿上游而主动 offload 已能常驻的 routed weights、多 Mac/RDMA、通用模型兼容层、激进再量化、近似 speculation、内置 coding agent。容量问题、数值变化和 API 扩展各自建立实际需求后再排期。

## 提交与接续

阶段提交应包含完成的源码、必要测试、结果结论和下一步，推送后核对远程 SHA；大模型、构建和大结果保持忽略。实际进度与最新参考服务恢复 ledger 记录在 [自主接续记录](AUTONOMOUS_PROGRESS.md)，本页的计划项不自动等于已完成。
