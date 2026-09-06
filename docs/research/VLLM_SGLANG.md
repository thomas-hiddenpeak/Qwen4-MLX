# vLLM / SGLang：适合 Qwen4-MLX 吸收的能力

核对日期：2026-09-07。范围是本机 Swift / MLX / Metal 的 Qwen3.8-Flash-Next runner。本文是研究与实施建议，不表示这些能力已经在本项目落地；上游的大模型、CUDA 和多卡测试数字不能作为 M5 Max 的预期收益。

**最值得吸收的是完整状态管理、按阶段的调度与测量，以及按真实成本选择 MTP。** 现有本机 PD 边界已经可作为基础，无需改造成通用框架。先推进 MTP 与 decode 优化，达到既定门槛后，再把整套前缀状态缓存接入服务；不因研究到新特性而绕过当前 [MTP 发布条件](../MTP_RELEASE_CRITERIA.md)。

## 已有基础与实际缺口

本地依据：[PD 设计](../PREFILL_DECODE_SEPARATION.md)、[MTP 与会话](../../MTP_AND_SESSIONS.md)、`QwenGeneration.swift`、`QwenLocalScheduler.swift`、`QwenMTPDecoder.swift`、`GPUGatedDeltaNet.swift`。

| 已有能力 | 仍需补齐 |
| --- | --- |
| 单份权重、同一推理执行器、prefill / decode / verification 显式阶段 | 多进程 PD、可序列化的跨实例状态尚未实现 |
| 有界队列、逻辑 token 额度、resident sequence 数量、cooperative chunk / round 调度 | 逻辑 token 额度不等于物理内存限制；缺少完整状态类别的内存预算 |
| 每阶段时间、实际输出 token、MTP draft / verify / commit / history 与接受分布 | 更易比较的成本摘要、阶段等待与用户可见输出间隔；将来的服务级聚合 |
| GDN / PLE / QSA / MTP 的事务提交、取消与异常清理 | 可供多个请求复用的不可变前缀缓存；SSD 持久化与淘汰策略 |
| 同步库 API 与 CLI | 独立 HTTP / SSE 服务和网络断连传播尚未实现 |

研究时保留当前边界：GPU 作业串行；cooperative 是步骤间切换；MTP 默认关闭。已有结果不能扩大解释为连续批处理、硬件抢占或生产 HTTP 服务。

## 优先顺序

以下是本项目建议，优先级依据当前代码和负载判断，并非上游给出的排序。

| 顺序 | 吸收内容 | 本机落点与依赖 | 最小实测 |
| --- | --- | --- | --- |
| P0 | 请求级有效产出与 MTP 成本摘要 | 扩展已有统计，无需换 kernel；区分计算吞吐和含排队/回调的服务延迟 | 同一长输入 AR / MTP 各一次，手工核对计数和时间分母；不增加 GPU 求值或同步 |
| P0 | 以 decode 等待时间评价 cooperative 策略 | 复用 `QwenLocalSchedulerCore`；先调已有 `decodeBurst`，保留冻结的 prefill chunk 和数值配置 | 一个 11k prefill 与一个已开始 decode 交错，核对完整输出、最长输出间隔、两个阶段吞吐 |
| P1 | MTP 的成本选择，而非仅看接受率 | 先离线比较 depth 0/1/2；在线切换需要在 round 边界保持 head / trunk 同步 | 高接受和低接受两个真实任务；加入所有 draft / verify / history / commit 时间后仍有价值 |
| P1 | 评估 GDN 输入回放式状态提交 | 先量化现有 capture 的字节与时间占比；有收益空间再改 `GPUGatedDeltaNetFused` 与 prefix commit | 一层的所有接受长度输出/state 对照；通过后才跑短整模型，再到 11k |
| P2 | 不可变完整状态 checkpoint + 内存前缀复用 | MTP 门槛通过后；统一前缀身份、完整状态、私有恢复与内存预算 | 同一 10k 系统提示 + 两个不同后缀；cache on/off 输出一致，第二次 prefill/TTFT 实际下降 |
| P2 | 服务断连、缓冲上限与准入反馈 | 接在已有取消对象和有界调度器之上；HTTP 属于后续交付 | 流式客户端断连/慢读后资源释放，其他请求继续；满队列明确拒绝 |
| P3 | SSD 缓存、会话提示与分层淘汰 | 先有可靠 L1 checkpoint；再做独立文件格式、校验、磁盘额度和异步 I/O | 冷盘命中确实快于重算；中断写入/错误版本回退；PLE 读取未被缓存 I/O 拖慢 |
| 后置 | 大规模分页、跨机传输、连续批处理 | 需要 Attention / QSA 存储接口、批量 recurrent 状态和全生命周期所有权 | 先证明当前 append/copy 或多请求吞吐确为瓶颈；不为功能列表先重写 |

## 1. 分阶段调度：借鉴策略，保留本机算术边界

vLLM V1 的 chunked prefill 会优先安排 decode，再用剩余 token budget 放入 prefill。其文档明确指出 token budget 会影响 TTFT、ITL 和吞吐之间的取舍。[vLLM 当前优化文档](https://docs.vllm.ai/en/latest/configuration/optimization/#chunked-prefill)

我们已有单执行器下的 chunk/round 交错，因此首步应是利用现有事件时间调整公平性，而不是复制 GPU batch scheduler。先保存提交、首次 callback、每次恢复和每轮完成的单调时间，比较 `decodeBurst` 对长 prefill 插入时的影响。当前 `decodeBurst` 计算的是步骤数，首次发布首 token 也算一步；MTP round 可产生多个 token，不能把它误称为 token 配额。

不要在运行中随意改变 prefill chunk：本模型 GDN 的 BF16 回存边界可能随分块变化。若以后做以目标时长驱动的 chunk 选择，应在请求开始前选择一个已验证配置并固定到请求结束，缓存身份也记录它。调度器可以先在固定 chunk 之间调度，避免将“公平性收益”和“算术改变”混为一谈。

vLLM 将跨实例 PD 用于独立调 TTFT/ITL 与减少 decode 尾延迟，而非承诺吞吐自动增加；其现有连接器也有各自兼容范围。[vLLM PD 文档](https://docs.vllm.ai/en/latest/features/disagg_prefill/)

本机先保留一份权重与同进程句柄。未来跨进程时，应传 token IDs 和有版本的完整状态；不重复套模板/分词，不传裸 MLX handle。是否采用独立进程取决于真实隔离收益与状态传输成本。

## 2. MTP：有效 token / 总成本，优先于接受率

vLLM 当前提供每请求的草稿数、接受数、round 数、接受长度直方图，可选记录每轮有效草稿长度；它把含 bonus 的 mean acceptance length 与不含 bonus 的 draft acceptance rate 分开，并可在最后的 usage chunk 中输出摘要。[vLLM 请求级接受指标](https://docs.vllm.ai/en/latest/features/speculative_decoding/acceptance_metrics/)

本项目已有大部分原始计数，应增加方便比较的派生摘要，保持既有字段含义：

- `accepted / proposed` 只反映草稿质量。
- `actual committed output / measured decode seconds` 才是有效 decode 吞吐；首 token 按现有合同排除，EOS/剩余预算按实际输出计数。
- `sum(draft, verify, commit/replay, history)` 与已计时 decode 窗口核对；GPU 重叠或嵌套区间不能简单重复相加。
- 报告每轮产出分布、0 接受 round 比例及浪费草稿数；不把 `1 + accepted/rounds` 当成所有 EOS/预算裁剪情况下的实际输出统计。

SGLang 的 adaptive speculation 使用按 batch size 区间独立维护的 EMA 与滞回，在 round 完成后切换预先准备的运行状态；当前文档限定 EAGLE/EAGLE3、top-k=1。vLLM 也支持按并发范围选择草稿长度。它们不是已证明适配我们原生 MTP 的现成模块。[SGLang adaptive SD](https://docs.sglang.io/docs/advanced_features/adaptive_speculative_decoding)、[vLLM dynamic SD](https://docs.vllm.ai/en/latest/features/speculative_decoding/dynamic_speculative_decoding/)

本机建议先离线找 depth 0/1/2 的成本表；队列中有多个请求不等于 GPU batch size 增大，不能直接套 vLLM 的并发区间。若之后加入在线控制，依据单请求接受率、整轮耗时和其他 decode 请求的等待，限制切换频率，并保留显式静态配置作为参考。切到 AR 再切回 MTP 时仍须补齐真实 head history；跳过历史维护不能悄悄污染后续草稿状态。

## 3. ReplaySSM：值得探测，但先核对节省占比

SGLang 的 Qwen3.8 官方技术文章描述了验证时保存 recurrence 输入、接受后从已提交 checkpoint 只回放小状态更新的方案，以减少逐位置完整 GDN 状态捕获；它同时强调 GDN recurrent、卷积历史与 KV 的一致交接。这篇文章针对 Qwen3.8-2.4T-A95B，不能视为我们的 Flash-Next 实现或性能证据。[SGLang Qwen3.8 技术说明](https://www.lmsys.org/blog/2026-08-12-qwen3-8-day0-support)

本地 `VerificationCapture.recurrentStates` 每层每位置为 `[1,48,128,128]` BF16，即 1,572,864 字节（1.5 MiB）。depth2 的 S3 验证每个 GDN 层写 4.5 MiB 捕获张量，另有卷积输入和提交拷贝。该算术只估计逻辑 payload，不等于真实 DRAM 流量；其占整轮权重读取的比例必须实际测量。

候选实现应保存当前内核已得到的 recurrence 输入，复用相同加法顺序与逐 token BF16 回存，只回放 accepted prefix 的 recurrence；不要重跑大矩阵、MoE 或整个主干。PLE 卷积/hash、Attention/QSA、MTP head 继续按原事务规则提交。先通过单层所有接受长度的逐位状态对照，并证明临时字节下降且额外 kernel 没抵消收益，再决定整模型试验。

上游功能组合也在发展：SGLang 有关于 ReplaySSM 与 checkpoint 位置导致前缀复用下降的近期问题报告；这是报告者的场景证据，不能推广为所有版本都有问题。它提醒我们：未来测试必须包含“先 MTP 生成，再分叉复用前缀”，不能只验证单独 decode。[SGLang issue #37834](https://github.com/sgl-project/sglang/issues/37834)

## 4. 前缀缓存：一个 token 前缀，多个不同的状态规则

SGLang Unified Radix Cache 让 FULL、SWA、MAMBA 使用同一 token 树，各组件独立检查可复用边界；MAMBA 需要前沿 checkpoint，并在请求修改共享状态前恢复到私有槽。当前树实现会对候选节点执行所有组件的 validator；仅命中 FULL 不代表整个混合模型可以复用。[Unified Radix 设计](https://www.lmsys.org/blog/2026-08-11-unified-radix-cache)、[固定版本的匹配实现](https://github.com/sgl-project/sglang/blob/2c05ed4e7776c876478f4b2db61acb12b9a27d01/python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py#L754)、[Mamba 组件](https://github.com/sgl-project/sglang/blob/2c05ed4e7776c876478f4b2db61acb12b9a27d01/python/sglang/srt/mem_cache/unified_cache/components/mamba_component.py#L142)

vLLM 也按缓存组协调共享前缀。要注意其旧 Hybrid KV Cache Manager 设计页仍写 Mamba prefix 为 WIP，但本次核对的主分支 `MambaManager` 已包含 align、checkpoint 与 speculative block 管理。前者可学组织方式，不能据它断言当前不支持。[旧设计页及版本提示](https://docs.vllm.ai/en/stable/design/hybrid_kv_cache_manager/)、[当前固定代码](https://github.com/vllm-project/vllm/blob/6865e67f0be02d53694517f6f71d7fb96492792d/vllm/v1/core/single_type_kv_cache_manager.py#L1408)

我们第一版可以采用比 radix tree 更小的实现：有界 checkpoint 表，先支持精确公共系统提示命中。每份 checkpoint 记录以下同一提交边界的信息，再逐步加最长前缀树：

| 状态 | 本机复用约束 |
| --- | --- |
| Attention KV / QSA raw、pooled key | 逻辑长度、池化边界、绝对位置一致；尚未落地的 page table 不参与首版 |
| GDN recurrent / conv3 | 必须有该前缀位置的 checkpoint；不能从最终状态截取到任意更早位置 |
| PLE conv9 / 两 token hash history | 与 token 前缀一致，否则下一 token 可能查错 SSD n-gram 行 |
| MTP head KV/QSA / history / previous stream | 与固定 history 策略和主干前缀一致；MTP 配置变化须拒绝命中或显式重建 |
| 模型与数值身份 | 权重、tokenizer、布局/dtype、kernel 配置及 prefill chunk 策略一致 |
| 输出游标 | 缓存已消费输入的状态；待输出 token、已发布数量、EOS/预算按新请求重建 |

不要把现有单次消费 `QwenPrefillResult` 直接放进多请求缓存。它还持有固定请求和 pending token，生命周期是“消费一次”；缓存需要不可变共享 checkpoint、显式私有恢复和引用释放。初期只在原有 chunk 边界保存，命中后沿原位置继续分块，不人为改变数值边界。

内存准入应至少区分：按 token 增长的 KV/QSA、每请求固定 GDN/PLE、MTP history、验证临时状态与共享模型。逻辑字节总和仍须与 MLX active memory/峰值一起观察，view/共享存储不能重复算成真实占用。Radix/paging 是索引与存储管理方法，不会消除 recurrent checkpoint 的需求。

## 5. SSD 与会话：适配统一内存，别复制三级容量模型

SGLang HiCache 区分 device、host 与外部存储，支持不同写入和预取策略；其文档明确单实例 L2 的共享范围与 L3 不同。[HiCache 配置与分层](https://docs.sglang.io/docs/advanced_features/hicache_best_practices)

对 M5 Max 的设计判断：CPU/GPU 使用同一统一内存容量，另拷贝一份“host cache”不自动增加容量。先做 RAM checkpoint → SSD 文件的两层方案。恢复前比较实测 SSD 读取/恢复成本与跳过 prefill 的收益，小前缀可直接重算；冷热盘分别测，不能把文件页缓存命中称为物理 SSD 性能。

写入应有独立磁盘额度与临时文件提交，校验身份、长度、tensor layout 和内容摘要，失败退回正常 prefill。写入和预取必须有界、可取消，优先保障模型已有的 PLE n-gram 读取。第一版不引入 Mooncake/分布式存储；本地文件格式成熟后再考虑共享存储。

SGLang 的 session-aware eviction 用 session 关联作为淘汰偏好，仍允许内存压力下回收；关闭会话移除引用，不等于立即删除内容。[SGLang 会话淘汰设计](https://www.lmsys.org/blog/2026-08-11-unified-radix-cache)

这个思路适合 agent 多轮，但晚于正确的命中/恢复与普通 LRU。首版完整传入 prompt，以 token 内容决定复用；session ID 只是复用概率提示，不能替代前缀校验，也不能无限 pin 住活跃对话。

## 6. 服务：复用已有事务与取消，不在回调里等网络

vLLM 在异步生成被取消/关闭时将 abort 传播到输出处理与 engine；SGLang 的 tokenizer manager 也处理客户端断开后的请求终止。应借鉴生命周期闭环，而非照搬 Python 异步实现。[vLLM 固定版本取消实现](https://github.com/vllm-project/vllm/blob/6865e67f0be02d53694517f6f71d7fb96492792d/vllm/v1/engine/async_llm.py#L700)、[SGLang 固定版本断连清理](https://github.com/sgl-project/sglang/blob/2c05ed4e7776c876478f4b2db61acb12b9a27d01/python/sglang/srt/managers/tokenizer_manager.py#L2175)

本机 HTTP 首版建议一个推理执行器，网络线程仅提交、取消、读取事件；token callback 写有上限的输出缓冲，避免在同一执行器等待慢客户端。缓冲满时按已定义策略暂停调度或取消该请求，不能丢已提交 token 后继续宣称响应完整。排队、ready、prefill、decode、流式结束分别测试取消，并保证额度只释放一次。

先提供请求级 JSON 摘要，服务上线时再聚合队列、运行数、阶段 token/s、TTFT/输出间隔和内存。SGLang 已区分生产队列指标及估计的读写字节；估计带宽仍应明确为模型估算，不能当硬件 DRAM counter。[SGLang production metrics](https://docs.sglang.io/docs/references/production_metrics)

## 核对与许可

本次只读核对了官方文档和以下主分支快照，未运行 vLLM/SGLang，也未移植代码：

- vLLM：`6865e67f0be02d53694517f6f71d7fb96492792d`。核对 `kv_cache_coordinator.py`、`single_type_kv_cache_manager.py`、`sched/scheduler.py`、`engine/async_llm.py`。
- SGLang：`2c05ed4e7776c876478f4b2db61acb12b9a27d01`。核对 `unified_radix_cache.py`、`unified_tree_core.py`、`components/mamba_component.py`、`mamba_radix_cache.py`、`managers/tokenizer_manager.py`。

两者所核对仓库的根许可均为 Apache-2.0：[vLLM LICENSE](https://github.com/vllm-project/vllm/blob/6865e67f0be02d53694517f6f71d7fb96492792d/LICENSE)、[SGLang LICENSE](https://github.com/sgl-project/sglang/blob/2c05ed4e7776c876478f4b2db61acb12b9a27d01/LICENSE)。本文只归纳思路。若后续实际复制/翻译实现，应逐文件核对版权和 vendored 许可、保留要求的声明与 NOTICE，并在本项目 provenance 中记录源路径、提交和修改；不能用仓库根许可覆盖所有依赖来源。
