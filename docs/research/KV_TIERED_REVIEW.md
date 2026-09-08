# 分层状态缓存：LMCache、DwarfStar 与 MLX LM 设计审阅

审阅日期：2026-09-09。范围是为本项目制定能力和取舍；本次没有复制上游运行代码，没有构建或执行上游模型。上游结论来自当日官方文档和固定提交的源码；下文“本项目应”是设计建议，不是已经交付或测得的性能。

## 结论

本项目应把“缓存”定义为**可复用的完整推理状态服务**，包含 Attention KV/QSA、GDN、PLE、准确 token 边界和执行配置。下一步重点是页/块共享与检查点协同、取消时明确的资源所有权、有界异步传输和成本驱动的保留策略。不能把纯 Attention 项目的 token 分块直接等同于混合状态的可恢复分块。

当前本地 [KV_CACHE_RELIABILITY.md](../KV_CACHE_RELIABILITY.md) 已记录 RAM/SSD 两层、联合逻辑额度、异步 IO、同前缀合并和故障回退。这里补充这些能力下一步应满足的契约；不重做存储目录、校验和既有生命周期。

## 固定来源

| 项目 | 本次 SHA / 许可 | 主要用途 |
| --- | --- | --- |
| [LMCache](https://github.com/LMCache/LMCache/tree/47da378cae7fce8efbd14a3cf5ab78e19e63ef3c) | `47da378cae7fce8efbd14a3cf5ab78e19e63ef3c`，Apache-2.0 | 异步层级、传输事务、对象锁、背压、配额 |
| [antirez/ds4，当前 README 名称 DwarfStar](https://github.com/antirez/ds4/tree/6289c516273979173abbc062209a81dd3706b804) | `6289c516273979173abbc062209a81dd3706b804`，MIT | 专用本机 runner、会话检查点、保留价值 |
| [MLX LM](https://github.com/ml-explore/mlx-lm/tree/95fdd057b101eb79f02da2940a8ec153d1761f1b) | `95fdd057b101eb79f02da2940a8ec153d1761f1b`，MIT | Apple 缓存类型、可裁剪能力和会话前缀策略 |

许可在各提交的根目录 LICENSE 核对。未来若复制、改编源码，应逐文件保留所需声明，并核对其第三方来源；本次仅吸收设计，不引入依赖。

## LMCache：取当前 MP 的对象生命周期

注意版本：当前官网已把旧 `async_loading` 页面归入 deprecated 的 in-process 模式，推荐 MP。旧模式的 `MemoryObj` refcount/pin 和 weighted semaphore 可帮助理解历史问题，但不能当作当前 MP 接口。以 [MP 概览](https://docs.lmcache.ai/mp/index.html) 和下面固定源码为依据。

| 观察到的机制 | 固定源码链 | 本项目的转化 |
| --- | --- | --- |
| `reserve_write/finish_write`、`read_prefetched_results/finish_read_prefetched` 分开，发布与读取都有明确终点 | [StorageManager](https://github.com/LMCache/LMCache/blob/47da378cae7fce8efbd14a3cf5ab78e19e63ef3c/lmcache/v1/distributed/storage_manager.py#L182) → [L1Manager](https://github.com/LMCache/LMCache/blob/47da378cae7fce8efbd14a3cf5ab78e19e63ef3c/lmcache/v1/distributed/l1_manager.py) | 显式 `reserve → fill → validate → publish`；半写入状态永不命中。读取 lease 持有到设备真正消费完成。 |
| LOOKUP 异步预取、query/wait 查询就绪；L2 加载前预留目标空间，分开记录在途字节 | [StorageManager prefetch](https://github.com/LMCache/LMCache/blob/47da378cae7fce8efbd14a3cf5ab78e19e63ef3c/lmcache/v1/distributed/storage_manager.py#L410) → [PrefetchController](https://github.com/LMCache/LMCache/blob/47da378cae7fce8efbd14a3cf5ab78e19e63ef3c/lmcache/v1/distributed/storage_controllers/prefetch_controller.py#L936) | 区分“索引可找到”“数据已到达”“完整状态已恢复”。调度器等待通知，期间运行其他 decode；取消时先令请求失效，再由 IO 完成路径释放缓冲区。 |
| 取消只释放该 lookup 实际取得的对象、组及读锁数，避免减掉另一个请求的锁 | [LookupModule.free_lookup_locks](https://github.com/LMCache/LMCache/blob/47da378cae7fce8efbd14a3cf5ab78e19e63ef3c/lmcache/v1/multiprocess/modules/lookup.py#L502) | 令牌化 transfer ID 和 generation/epoch，完成或取消只能结算一次；一位等待者退出不能取消其他消费者共享的生产任务。 |
| 独立 Store/Prefetch/Eviction 控制器，支持分层存储策略和按水位回收；按 `cache_salt` 管理配额 | [StoreController](https://github.com/LMCache/LMCache/blob/47da378cae7fce8efbd14a3cf5ab78e19e63ef3c/lmcache/v1/distributed/storage_controllers/store_controller.py) → [EvictionController](https://github.com/LMCache/LMCache/blob/47da378cae7fce8efbd14a3cf5ab78e19e63ef3c/lmcache/v1/distributed/storage_controllers/eviction_controller.py) | 读恢复优先于后台写；软水位提前回收、硬上限拒绝可选缓存工作。后续按受信任的服务端 cache domain 管理份额。 |
| token chunk 使用 rolling prefix hash；每块身份依赖之前的前缀 | [TokenHasher](https://github.com/LMCache/LMCache/blob/47da378cae7fce8efbd14a3cf5ab78e19e63ef3c/lmcache/v1/multiprocess/token_hasher.py#L183) | 使用稳定内容摘要和规范编码，绑定前缀、模型/布局/数值 ABI、准确位置、状态组；重复文本片段处在不同前文中不能直接复用。 |
| engine-driven 路径存在 PREPARE/COMMIT，取消注册会清理尚未结束的 SHM transfer | [EngineDrivenTransferModule](https://github.com/LMCache/LMCache/blob/47da378cae7fce8efbd14a3cf5ab78e19e63ef3c/lmcache/v1/multiprocess/modules/engine_driven_transfer.py#L267) | 先定义本机 PD handoff 的 prepare/commit/abort 契约；由 runner 控制 MLX tensor 生命周期，CPU IO 只接收归档字节。 |

MP 的 TTL 读写锁针对失联客户端恢复。**对象有效期、消费者 lease 和设备执行完成是三件事**：本项目不能因为墙钟超时就释放仍被 Metal 命令引用的存储；跨进程租约恢复后置，当前进程内依靠明确所有权和完成通知。

本机统一内存下，GPU tensor、CPU 归档、Metal/MLX allocator cache 仍可能重复占用同一 RAM 容量。“迁移到 CPU”不能当作释放独立显存。继续保留 request/cache/workspace 联合账本，并用实际 MLX/RSS/系统内存压力分别验证；不照搬 pinned-host、CUDA IPC、cuFile/GDS、RDMA 的数据路径。

## DwarfStar：把会话的保留价值纳入策略

当前仓库仍是 `antirez/ds4`，README 名称是 DwarfStar，目标是少数模型的本机专用 runner。它将共享检查点文件格式和 server/agent 的缓存策略分开，适合作为本项目专用设计的对照。

- **兼容性分层**：文件 envelope 版本与 graph payload ABI 分离。借鉴这一点，让“格式能解析”与“状态能安全恢复”各自判定。我们应保留严格模型/量化/执行配置身份，不能采用其可选跨量化恢复。[格式与 ABI](https://github.com/antirez/ds4/blob/6289c516273979173abbc062209a81dd3706b804/ds4_kvstore.c#L29)
- **价值密度和衰减**：淘汰评分综合衰减命中数、可复用 token 数和文件字节；保留有意义的冷启动/退出锚点，对被新长会话覆盖的旧 continued 检查点降权。这个思想优于无限保留所有续写位置。[评分](https://github.com/antirez/ds4/blob/6289c516273979173abbc062209a81dd3706b804/ds4_kvstore.c#L532)
- **会话与自动缓存分开**：agent 可保存、列出和加载显式会话，server 自动前缀缓存采用自己的保留规则。本项目也应区分“请求已结束”和“会话可复用状态仍值得保留”，但显式 session 不能无限 pin、不能绕过 quota。[agent session load](https://github.com/antirez/ds4/blob/6289c516273979173abbc062209a81dd3706b804/ds4_agent.c#L4528)
- **稳定边界**：检查点边界对齐 prefill/compressor 更新时机。ds4 的 2048 对齐、尾部 trim 和 continued 间隔针对自己的模型；本项目应使用已验证的 GDN/PLE/QSA 边界，不能照抄这些数字。[边界选择](https://github.com/antirez/ds4/blob/6289c516273979173abbc062209a81dd3706b804/ds4_kvstore.c#L701)
- **不照搬可读文本键**：ds4 在 byte-prefix 命中后处理后续 suffix tokenization，有自己的 exact sampled transcript/tool replay 语义。我们继续以准确 token 前缀为准；同文本但不同 token、模板、工具渲染或前文，都不能自动视为同一混合状态。[text-prefix restore](https://github.com/antirez/ds4/blob/6289c516273979173abbc062209a81dd3706b804/ds4_kvstore.c#L1220)

不要把上游存在某策略等同于完整耐久保证。所审查 ds4 写入段使用临时文件、flush/close/rename；本项目已有 file fsync、directory fsync、错误时保留磁盘账目的路径应继续保留，不为贴近上游而削弱。[写入实现](https://github.com/antirez/ds4/blob/6289c516273979173abbc062209a81dd3706b804/ds4_kvstore.c#L1054)

## MLX LM：不能把所有缓存都当作可裁剪 KV

官方 MLX LM 将 cache 的 tensor `state`、`meta_state`、具体 cache 类型一起持久化；`can_trim_prompt_cache` 要求每一层都声明可裁剪。当前 `LRUPromptCache` 用 trie 做 exact/shorter/longer 匹配，只有全层可裁剪时才从更长状态回退、或者删掉被长状态覆盖的短前缀；缓存有条目和字节上限，并区分 system/user/assistant 类型。[状态与裁剪契约](https://github.com/ml-explore/mlx-lm/blob/95fdd057b101eb79f02da2940a8ec153d1761f1b/mlx_lm/models/cache.py#L43)、[LRUPromptCache](https://github.com/ml-explore/mlx-lm/blob/95fdd057b101eb79f02da2940a8ec153d1761f1b/mlx_lm/models/cache.py#L1635)

这对本模型的直接约束是：GDN/PLE 检查点不支持仅靠截掉 Attention KV 就回到任意历史位置。Attention 页可以不可变共享；递归状态需要准确边界快照，分支时生成私有可写状态。较长检查点也不能无条件替代一个有复用价值的较短 GDN/PLE 锚点。system 边界作为保留价值信号可以吸收，业务角色不能授权跨隔离域共享。

## 建议固化的七项能力

| 优先级 | 关键能力 | 必须可验收的行为 |
| --- | --- | --- |
| P0 | 完整状态与稳定身份 | 恢复单位绑定所有状态组和准确边界；缺一个组、位置/ABI 不符即从有效完整前缀重算。跨进程重启不能仅依赖非稳定的语言内置 hash。 |
| P0 | 资源所有权 | 每个 request、snapshot、IO transfer、PD handoff 有唯一负责释放的 owner；重复完成、取消、clear、producer 退出都不双重释放或遗留额度。 |
| P0 | 统一准入和背压 | 在启动复制/加载前同时考虑目标状态、staging、元数据、在途读写、待写文件；有界队列、读优先、可取消等待、软/硬水位；被跳过的缓存仍能正常推理。 |
| P1 | Attention 块共享 + 混合状态检查点 | 固定物理块与逻辑前缀索引分开，完整页只读共享、尾页私有或 COW，GDN/PLE 在可靠边界保留；共享块的引用数为零才可回收。 |
| P1 | 分块持久化与清单 | 内容键描述前缀与状态组，manifest 原子发布一套完整可恢复状态。块可去重和增量保存，孤儿块有界清理；导入过程不能暴露部分状态。 |
| P1 | 成本与会话策略 | 以近期复用概率 × 实测节省 prefill 时间减去保存/恢复成本，结合驻留字节做准入/淘汰；高价值 system 锚点与会话末状态有限保留，LRU 作为基准与回退。 |
| P1 | 隔离和运维 | 单机默认一个服务端隔离域；共享 API 前补显式 domain、配额、清除范围和观测，domain 不能由无认证客户端任意冒用。提供在途字节、实际恢复 token、净节省时间、写放大和驱逐原因。 |

这里的 P0/P1 是本次审阅建议，由项目主计划统一安排。当前已有能力不重复列为“待实现全部”；应将其作为契约和验收项加强。

成本策略先只采样和离线回放，再与 LRU 对比，保留手动关闭。不能只优化 hit rate：短前缀 hit 可能不抵归档导入成本，频繁 SSD 写可能挤占 PLE n-gram 读取并拖慢 decode。必须分别记录 prefill、decode、cache wait、恢复有效 token 和总物理 IO。

CacheBlend/非前缀拼接、跨机共享、RDMA/GDS、压缩或量化 KV、失联跨进程租约，以及 MTP 缓存留到后续。它们分别需要新的数值语义、硬件路径或故障边界；当前没有理由把这些复杂度作为本机工业缓存的先决条件。

## 针对上述设计的增量验收

1. 同一长系统前缀的多会话分叉：跨 RAM/SSD、清除、淘汰、取消任一分支后，其他分支的原始混合状态和完整输出与冷计算一致。
2. 在 lookup、目标额度预留、IO 完成、导入、publish、PD commit 每个边界取消；注入迟到/重复完成；排空后 request/workspace/transfer 都归零。
3. 不同前缀持续超过 RAM 与 SSD 容量，观察系统锚点、会话继续点和大对象的公平性；统计复用节省、写放大和 PLE/decode 尾延迟，不能只证明额度没超。
4. 丢一个块、错一个状态组、manifest 截断、写中进程退出、写入失败及删除失败均不发布不完整状态；重启后有界回收孤儿，正确回退。
5. 实际系统内存压力下的 admission 和恢复，再做小时/天级 SSD churn。现有约 16 分钟 RAM 主导与约 93 秒强制 SSD 记录不替代这些门槛。

本审阅没有引用上游营销吞吐数字，也没有将上游通过的测试当作本项目已经通过的测试。
