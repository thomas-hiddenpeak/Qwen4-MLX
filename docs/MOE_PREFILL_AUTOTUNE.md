# Prefill MoE 融合与小范围自动调参

2026-09-07，M5 Max。使用原模型 affine Q4/group64 与 BF16 运算边界，MTP 关闭。前一轮 [BM16 实验](MOE_PREFILL_TILING.md)退步之后，本轮建立可复用的候选搜索、数值淘汰、配置保存和完整 PD 对照。

## 本轮实现

`GPUMoEPrefillReduction` 将 sorted 专家输出的逆排列 gather、路由权重乘法及 top10 sum 合并为单个 Metal kernel。普通路径不再构造 originalOrder 和 products 两份逻辑中间张量；S416 时每份 21,299,200 字节。此数字不等于实测 DRAM 流量或缓存节省。

严格复制原 MLX `col_reduce_small` 的 BF16 顺序：先做八个 partial（slot0+8、slot1+9、slot2…7），再按 partial0…7 相加；产品和每次相加均保留 BF16 舍入。该顺序只针对本模型 H2560、top10 的当前归约几何。诊断时专家输出走额外旁路，routed 结果仍来自候选，防止诊断悄悄回退原计算。

这次没有合并 gate/up/SwiGLU，也没有新建按专家边界对齐的 GEMM。当前两个工作项是归约融合，以及原 NAX 专家矩阵的几何搜索；路由与共享专家计算保持原实现。

## 搜索空间

| Native ID | BM | BN | BK | WM/WN |
| --- | ---: | ---: | ---: | --- |
| 0 | 原启发式，本次为32 | 64 | 64 | 2/2 |
| 1 | 64 | 64 | 64 | 2/2 |
| 2 | 32 | 128 | 64 | 2/2 |
| 3 | 32 | 64 | 32 | 2/2 |
| 4 | 32 | 32 | 64 | 2/1 |

每个矩阵配置与原归约、融合128/256/512线程组组合，总计20种，包含原实现。独立构建脚本 `scripts/build_mlx_moe_autotune.py` 复用原 native 对象与42个 AIR，只新增缺失的3种 Metal 实例，不修改作者运行库或重排/复制完整权重库。

`autotune-gpu-moe-prefill` 复用已通过完整 golden 的九组真实 S416 输入，另取首组205/240行前缀检查尾部。每个候选比较九项张量/输出，包括候选诊断与普通调用；任何不一致即淘汰该组合。通过者每次对照各预热两次，再运行两组 ABBA，各四个计时样本。计时为 fresh 完整 MoE forward 加 y.eval，不含诊断、host readback、配置切换或加载。

按九组真实输入的配对基准均值之和/候选均值之和排序，两个派生样本不参与评分。至少观察到2%局部提升才选非参考配置；这个门槛只是决定是否进入完整 PD 测试，不是发布门槛或置信区间。

## 局部结果

[原始搜索](../results/moe-prefill-autotune-v1/autotune.json)、[保存的候选配置](../results/moe-prefill-autotune-v1/selected-config.json)。19个非参考组合在11个输入上均通过逐位一致与有限性检查，未淘汰数值候选；矩阵 dispatch 计数均与所选配置吻合。

自动胜者为 **native4 + 融合512线程**：九组配对均值合计 47.288ms → 46.092ms，吞吐提升 **2.59%**。九组中八组变快，一组退步；这是小范围搜索的候选结果，不能直接称为整模型提升。原矩阵配融合256线程为 +0.98%；单独 native4 为 +1.36%。这些组合各有自己的配对基准，不可将百分比相加。BM64 或 BN128 的组合整体退步约14%～17%。

## 阶段接口与配置边界

`QwenGenerationRequest.prefillMoEConfiguration` 是请求局部的归约表，当前候选保存205/240/416行的线程组选择；未列出的长度回退原归约。decode、verification 和最后 S1 不使用该表。配置可持久化再读取，默认请求仍为 nil。

矩阵几何的 `ANERUNNER_MOE_QMM_CONFIG` 当前仍是**串行实验开关**，由测试命令在 producer.prefill 前选择，完成并 drain 后恢复0，再交给 consumer.decode。它还不是可用于交错请求/独立部署的请求局部原生后端接口。配置保存库 SHA，完整对照必须验证所加载的库。不能仅把该 JSON 传给普通 generation request 就宣称同时启用了其中的 native 几何。

`probe-gpu-hotspots --detail moe-fusion --moe-config PATH` 从保存文件选择参数，运行 baseline/candidate 预热后再做 ABBA，全部关闭同步阶段采样。分别保存 prefill、decode 和交接指标。P11057/chunk416 的27个多 token 块应有3888个目标矩阵 dispatch；融合表覆盖416/240时应有1296次归约建图，decode 两类计数必须为零。每轮保留完整128-token golden、终止原因、最终offset11184、callback及PD交接检查。

Release 构建和31项 CPU 契约/阶段/调度检查通过。

## 完整 PD 结果与决定

[完整请求](../results/moe-prefill-autotune-full-v1/full.json)、[汇总](../results/moe-prefill-autotune-full-v1/summary.json)、[候选状态](../results/moe-prefill-autotune-full-v1/candidate-status.json)。六轮均完成128-token AR输出，共768 token逐项匹配 golden；终止、offset11184、callback与交接检查全部通过。每轮3888个目标矩阵 dispatch 的配置正确；候选每轮1296次融合归约建图，基准为0。所有 decode 的两类新增调用计数均为0，没有开启同步阶段采样。

首轮 baseline/candidate 作为预热，剩余四轮如下：

| 顺序 | 配置 | Prefill 秒 | Prefill token/s | Decode token/s |
| --- | --- | ---: | ---: | ---: |
| A | 原实现 | 13.018 | 849.36 | 32.81 |
| B | 自动候选 | 13.781 | 802.35 | 30.89 |
| B | 自动候选 | 15.083 | 733.07 | 29.37 |
| A | 原实现 | 16.600 | 666.09 | 27.61 |

按每种配置的两次阶段耗时均值计算，prefill 为 14.809s → 14.432s，746.64 → 766.14 token/s，观察到 **+2.61%**；decode 为29.99 → 30.11 token/s，观察到+0.42%，没有证据表明修改了 decode 性能。

**这轮不接受为稳定性能收益。** 两次基准 prefill 自身漂移27.51%，两组相邻对照为−5.53%和+10.05%，方向相反。不能把中间候选与最后变慢的基准单独相比，也不能直接用平均+2.61%推广默认。保留实验配置，正常默认不变；参考服务已恢复，MTP/drafter关闭。

本轮完成了融合归约、真实输入自动调参、配置存取及完整PD正确性验证。后续更大的候选是按专家边界组织矩阵工作，以及双指针gate/up/SwiGLU融合；两者都需要新shader和独立收益验证，不把本轮的小幅观察外推到它们。

随后完成的双指针融合及独立验证见 [Prefill gate/up/SwiGLU 融合](MOE_PREFILL_GATEUP.md)。它使用stock基础库及原归约，没有叠加本轮native4/归约512配置。
