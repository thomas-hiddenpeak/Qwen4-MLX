# 路由 MoE prefill 分块实验

2026-09-07，M5 Max，原模型 affine Q4/group64，MTP 关闭。起因与前序数据见 [prefill 热点分析](PREFILL_HOTSPOTS.md)。

## 共享与路由专家

每层共享专家的中间维度为 640，每个 token 都执行，并乘以独立 sigmoid 门控；它不参与 top10。512 个路由专家由 router 每 token 选择 10 个。本 runner 的共享专家为常驻 BF16，路由专家的 Q4 权重库也已载入内存；SSD n-gram embedding 是另一条路径。路由表示选择计算，不表示本实验按需从 SSD 载入专家。

## 候选实现

仅对排序后的多 token routed affine Q4/group64、BF16 输入/scale/bias、512 个专家，且 `(K,N)=(2560,640)` 或 `(640,2560)` 的 RHS NAX 矩阵路径调整几何。原 BM32/BN64/BK64/WM2/WN2 改为 BM16/BN64/BK64/WM1/WN2。共享专家、权重表示和单 token decode 保持原实现。

原 kernel 已在每个 BM tile 内按专家连续区间循环，且只执行与该专家相交的 16 行 SIMD 子块。两种几何保持相同的 16×32 MMA 片段，不能按平均 8.125 次路由分配与 BM32 的比例推算空算。BM16 增加并行 threadgroup，也可能增加同一专家权重的重复加载。

`scripts/build_mlx_moe_tiling.py` 在独立目录构建实验库，不修改作者库。`ANERUNNER_MOE_QMM_BM=0` 保留原调度，16 选择候选，32 为显式对照。此环境变量是串行实验开关，不是并发请求配置。导出计数器只统计已编码目标矩阵 dispatch，不代表物理 DRAM 流量。

## 真实输入与检查

`capture-gpu-moe-prefill` 在同模型 producer.prefill → consumer.decode 中抓取 MoE 前的原生 BF16 输入。提示词 11,057 token，chunk416；层 0/23/47 与 offset 0/4992/9984 的笛卡尔积共九组。保存之后仍完成 128-token AR 输出、终止原因、callback、PD 交接及最终 offset11184 检查，全部通过才提交输入。

`probe-gpu-moe-prefill-tiling` 对每组输入比较路由 ID、权重、专家输出、路由归约、共享输出及完整输出的字节与有限性。另取首组输入的 205 行前缀检查不整除 tile 的尾部。之后测常驻完整 MoE forward，包括共享专家，每种模式预热三次，再运行四组 ABBA（每模式八个计时样本）。模式切换、加载、诊断和 host readback 不计入样本时间。

## 初次执行

[v1 capture](../results/moe-prefill-tiling-v1/capture.json) 的九组输入及完整生成检查通过。初次微测因预编译 Metal 库缺少 BM16 实例而停止，没有获得候选输出或性能数据。实际 MLX 构建为 `MLX_METAL_JIT=OFF`；已有 JIT 源码不意味着该构建会自动编译新实例。失败记录保留于 [v1 micro](../results/moe-prefill-tiling-v1/micro.json)。

后续修复在独立 Metal 库中加入单个 BF16/group64/Q4/BM16/WM1/WN2 实例，保留原 42 个 AIR 编译对象，沿用 `-fno-fast-math`。局部构建命令设置 Xcode 的 `DEVELOPER_DIR`；没有更改系统工具链选择。最终产物为 `results/moe-prefill-tiling-v2/mlx-tuned-fixed`，原输入哈希均不变。

## 实测结果与决定

[原始微测](../results/moe-prefill-tiling-v2/micro.json)、[汇总](../results/moe-prefill-tiling-v2/summary.json)。实际加载的独立 dylib 和 Swift executable 哈希均已核对。十组输入每组十一项比较全部逐位一致且有限，包括强制 BM32 对照、诊断/普通调用对照及共享分支。每组正式计时各模式恰好编码 24 次目标矩阵 dispatch，BM64 为零。

下表为完整 MoE 的八次热样本平均耗时，已包含共享专家；不是全模型 prefill 时间。

| 层 | Prompt offset | 原 BM32，ms | BM16，ms | BM16 耗时增加 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0 | 5.111 | 5.761 | 12.7% |
| 0 | 4992 | 5.593 | 6.228 | 11.4% |
| 0 | 9984 | 5.340 | 5.990 | 12.2% |
| 23 | 0 | 4.448 | 5.137 | 15.5% |
| 23 | 4992 | 5.209 | 5.702 | 9.5% |
| 23 | 9984 | 4.919 | 5.376 | 9.3% |
| 47 | 0 | 4.289 | 4.877 | 13.7% |
| 47 | 4992 | 4.558 | 5.136 | 12.7% |
| 47 | 9984 | 3.965 | 4.577 | 15.4% |

九组均值相加为 43.432ms → 48.784ms，同样工作量的耗时增加 **12.32%**，吞吐下降 **10.97%**。这是这九组 MoE 微测的聚合，不是整模型加速比。派生 S205 尾部样本也更慢：4.122ms → 4.318ms，耗时增加 4.75%。

真实每块活跃专家为 236～425 个，最集中的一组有单个专家接收 409 次分配。全部 512 个专家的平均 8.125 不能描述这种偏斜。对排序后每个专家连续区间 `[a,b)`，跨 BM 边界的区间数为 `floor((b-1)/BM)-floor(a/BM)+1`；本次 BM16 的专家矩阵遍历数量比 BM32 增加 **21.4%～34.1%**，相交的 16 行 MMA 子块数保持相同。

这解释了为什么缩小分块存在重复加载代价，但只是代码结构与实测相符的解释；没有直接测量物理 DRAM 流量、缓存命中或 GPU 占用率来证明唯一原因。下一候选应考虑按专家边界组织工作，减少跨边界重复遍历，而非继续只缩小全局行分块。其调度成本与端到端收益仍需单独验证。

**BM16 不纳入默认。** 所有真实微测均退步，因此没有继续完整模型候选 A/B。完整 128-token golden 检查仅属于原路径的输入捕获；候选当前证据是上述十组局部张量，不声称已通过整模型候选输出检查。Release 构建及 30 项 CPU 契约/阶段/调度测试通过；参考服务恢复并确认关闭 MTP/drafter。
