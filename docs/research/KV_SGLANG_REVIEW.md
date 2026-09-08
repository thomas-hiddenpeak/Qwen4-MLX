# SGLang KV cache 机制评审

日期：2026-09-09。范围：官方文档与当前主线源码核对；未运行 SGLang 或本项目推理，本文是设计输入，不是性能或生产验收证据。

固定上游版本：[SGLang `30e7a3072d3f1e9bd70cd5e44146ca27c80522c4`](https://github.com/sgl-project/sglang/commit/30e7a3072d3f1e9bd70cd5e44146ca27c80522c4)，GitHub API 返回提交时间 `2026-09-08T15:56:40Z`。以下源码链接固定此 SHA，不依赖持续变化的 `main`。

## 优先吸收的八项机制

| 机制 | 上游具体实现与证据 | 本项目要求及适配边界 |
| --- | --- | --- |
| 1. 一个逻辑前缀，多种物理状态 | Unified Radix 将 Full、SWA、Mamba 放在同一树中，匹配边界须通过所有组件校验；组件分别管理锁、资源及物理 I/O。[组件设计](https://github.com/sgl-project/sglang/blob/30e7a3072d3f1e9bd70cd5e44146ca27c80522c4/python/sglang/srt/mem_cache/unified_cache/components/README.md#L1-L127) | 定义 `PrefixStateManifest`：token 边界、Attention KV、QSA 原始/池化状态、GDN、PLE history/conv 必须构成同一可恢复边界。树上找到 token 前缀只是候选，不能提前算有效命中。树的决策与存储/MLX 执行分层；无需照搬通用组件框架的全部复杂性。 |
| 2. Attention 分块共享，递归状态保留检查点 | Mamba 将可分叉点对齐 checkpoint grid；命中后为请求分配私有 state；路径检查点有软上限，保留分叉、尾端、被锁节点。[Mamba 匹配/复制与路径上限](https://github.com/sgl-project/sglang/blob/30e7a3072d3f1e9bd70cd5e44146ca27c80522c4/python/sglang/srt/mem_cache/unified_cache/components/mamba_component.py#L155-L307) | Attention 的 immutable page 可共享，写入 tail 要私有；GDN/PLE 不能从最终状态切回任意较短前缀。第一步按稳定 chunk 边界及高价值分叉保留**完整混合检查点**；再引入 KV 页共享。检查点间隔、每条路径数均纳入字节预算。不能把 Radix 树已有能力等同于已有 paged KV。 |
| 3. 活动引用与会话偏好分离 | session 引用是可淘汰的软保护；运行请求的 component lock 是硬保护。会话关闭解除偏好，不立即删除内容；session ID 不补全历史文本。[session 语义](https://github.com/sgl-project/sglang/blob/30e7a3072d3f1e9bd70cd5e44146ca27c80522c4/docs/docs/advanced_features/session_radix_cache.mdx) | 活动/正在恢复/正在写入的状态必须持有唯一资源 lease；会话近期使用、TTL、优先级仅影响淘汰次序，不能无限 pin。支持有界会话软保留和显式释放。会话 ID 是缓存偏好，租户隔离需要独立 namespace，不能拿 session 代替授权。 |
| 4. 混合状态恢复是一次事务 | HybridCacheController 为 side pool 分配资源，任一分配失败即回滚已分配部分；取消后仍完成必要 ack，避免同步队列失配。[多池原子分配](https://github.com/sgl-project/sglang/blob/30e7a3072d3f1e9bd70cd5e44146ca27c80522c4/python/sglang/srt/mem_cache/hybrid_cache/hybrid_cache_controller.py#L829-L895)、[取消 ack](https://github.com/sgl-project/sglang/blob/30e7a3072d3f1e9bd70cd5e44146ca27c80522c4/python/sglang/srt/mem_cache/hybrid_cache/hybrid_cache_controller.py#L656-L676) | 统一 `reserve → read/verify → allocate/import → publish`，任何组件失败、超时、取消、clear epoch 改变都释放全部临时额度，不发布半套状态。异步完成只能交回所有者执行器，GPU 状态不在 I/O 线程修改。故障回退需验证冷 prefill 与未缓存 oracle 一致。 |
| 5. SSD 恢复必须可停止、有收益门槛 | HiCache 提供 best effort、wait complete、timeout；按连续前缀和阈值预取。设计页将 timeout 推荐用于有 SLO 的服务。[HiCache 工作流](https://github.com/sgl-project/sglang/blob/30e7a3072d3f1e9bd70cd5e44146ca27c80522c4/docs/docs/advanced_features/hicache_design.mdx) | 建议生产默认有界 timeout；比较预计 SSD read+verify+import 与重算代价，短命中可跳过。只能在**完整混合检查点**完成后启用，不能把读取了一半的 archive 当部分前缀。单机统一内存中 MLX resident tensor、CPU staging 与 SSD 是不同生命周期/格式，不必重复建立 GPU/CPU 两个同容量副本池。 |
| 6. 写回按重用收益和 SSD 成本决策 | 上游支持全量、热度选择、淘汰写回三类策略；file evictor 有 reserve/commit/abort、容量上限、最小剩余空间、水位回收。[写回说明](https://github.com/sgl-project/sglang/blob/30e7a3072d3f1e9bd70cd5e44146ca27c80522c4/docs/docs/advanced_features/hicache_design.mdx)、[文件容量管理](https://github.com/sgl-project/sglang/blob/30e7a3072d3f1e9bd70cd5e44146ca27c80522c4/python/sglang/srt/mem_cache/storage/file/lru_file_evictor.py#L1-L24) | 保留本项目原子文件、校验和、目录隔离；增加 min-free-space、写入字节/秒上限、队列水位、热度选择及显式高价值系统前缀策略。淘汰写回要先拿暂存预算，不能在内存压力最严重时制造无界副本。缓存写入可丢弃，前台 decode 不应等待 SSD 写回。 |
| 7. 同前缀合并与公平调度结合 | 上游 waiting queue 建临时 radix，对冷却又共享长前缀的请求暂时降序；同时有 LPM、DFS 和 token aging 的 HRRN 策略。[队列匹配与 HRRN](https://github.com/sgl-project/sglang/blob/30e7a3072d3f1e9bd70cd5e44146ca27c80522c4/python/sglang/srt/managers/schedule_policy.py#L342-L454) | 保留现有 single-flight 与取消接管，进一步让等待 I/O/leader 的请求不占可运行 GPU 槽。调度成本区分未缓存 prefill、恢复字节、预计输出；加入最大等待/aging，不能只按命中长度让冷长请求饿死。PD 两阶段单独统计，decode 间隔与 prefill TTFT 各设护栏。 |
| 8. 命中、读取、真正复用三套指标 | 指标分别记录 storage 查询命中、成功读取、最终未形成可用预取的原因；请求 cached tokens 按 device/host/storage 拆分。[存储指标](https://github.com/sgl-project/sglang/blob/30e7a3072d3f1e9bd70cd5e44146ca27c80522c4/python/sglang/srt/observability/metrics_collector.py#L1902-L1972)、[请求有效命中](https://github.com/sgl-project/sglang/blob/30e7a3072d3f1e9bd70cd5e44146ca27c80522c4/python/sglang/srt/observability/metrics_collector.py#L1790-L1837) | 核心 KPI 是有效复用 token 比率、节省 prefill 工作与 TTFT；辅助指标含 read bytes、discard bytes、save/restore p50/p95/p99、队列等待、取消回收、预算拒绝、校验失败、淘汰原因。所有 hit 指标注明单位和分母；逻辑额度与进程 footprint/MLX allocator 分列，避免把索引命中或逻辑预算当实际内存安全。 |

## 当前主线、未验证能力与不能照搬的部分

- **Unified Radix 已进主线且当前代码标记为默认。** session 文档仍演示旧环境开关，`environ.py` 已明确该变量废弃，说明文档与实现有版本差异；采用机制前以锁定源码与该版本测试为准。[默认状态说明](https://github.com/sgl-project/sglang/blob/30e7a3072d3f1e9bd70cd5e44146ca27c80522c4/python/sglang/srt/environ.py#L1836-L1839)
- **上游已有 MLX auxiliary state 适配，但不能据此宣布本模型已受支持。** 它保存原生 cache state、meta state 及 offset 等属性，并明确配置不同情况下只有一个释放 owner；`enable_mamba_extra_buffer` 明确未实现。值得学习所有权接口；本项目需独立覆盖 GDN、PLE、QSA 的语义与实际 buffer 隔离。[MLX snapshots](https://github.com/sgl-project/sglang/blob/30e7a3072d3f1e9bd70cd5e44146ca27c80522c4/python/sglang/srt/hardware_backend/mlx/kv_cache/auxiliary_state.py#L1-L90)、[释放 owner](https://github.com/sgl-project/sglang/blob/30e7a3072d3f1e9bd70cd5e44146ca27c80522c4/python/sglang/srt/hardware_backend/mlx/kv_cache/auxiliary_state.py#L220-L235)、[extra-buffer 限制](https://github.com/sgl-project/sglang/blob/30e7a3072d3f1e9bd70cd5e44146ca27c80522c4/python/sglang/srt/hardware_backend/mlx/kv_cache/auxiliary_state.py#L329-L346)
- **HiCache file 后端是学习材料，不是本项目持久化验收标准。** 官方设计页称其为示范后端；当前写入使用临时文件和 replace，所查路径没有 fsync/checksum；evictor 未配置时允许无界增长。本项目不能为了与上游一致而退化已有容量/耐久/校验保护。[文件写入](https://github.com/sgl-project/sglang/blob/30e7a3072d3f1e9bd70cd5e44146ca27c80522c4/python/sglang/srt/mem_cache/hicache_storage.py#L511-L557)、[无界默认说明](https://github.com/sgl-project/sglang/blob/30e7a3072d3f1e9bd70cd5e44146ca27c80522c4/python/sglang/srt/mem_cache/storage/file/lru_file_evictor.py#L1-L24)
- **CUDA 层间 H2D 重叠、TP all-reduce、RDMA 后端当前不列为本机必须能力。** 本机取其异步计划、容量预留、可取消完成、数据布局原则。Swift/MLX 的 buffer 生存期、同步点与 macOS memory pressure 必须本机验证。分布式/远端存储保留接口，待本机 PD 和两级缓存成熟后实施。MTP 性能继续后置。

## 推荐进入主计划的最小闭环

1. **可靠性阶段**：真实 memory-pressure / SSD ENOSPC / 超时 / restart / crash / 多前缀淘汰长稳测试，核对 lease、临时字节、文件和请求回收；把状态一致性与资源有界设为硬门槛。
2. **复用效率阶段**：完整混合检查点布局与路径预算；session 软保留；SSD 恢复超时和收益门槛；写回队列及最小剩余空间；有效复用 token 指标。
3. **容量效率阶段**：Attention immutable pages + tail COW 与 GDN/PLE/QSA 边界配合，降低完整快照重复容量；按单会话、共享 10k+ 系统前缀、多分支、多租户压力分别验收。
4. **调度阶段**：冷前缀 single-flight 继续正确，等待 I/O 不占可运行槽，命中收益与 aging 兼顾；prefill/decode 的工作量、排队与延迟分开考核。

以上能力可分步落地；不能仅凭“采用了 SGLang 同名算法”或短压力测试宣称工业可用。
