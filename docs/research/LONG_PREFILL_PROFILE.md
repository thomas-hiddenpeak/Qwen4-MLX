# 长 prefill 的成本与末段 profiling

当前 reference QSA 限制了可见 token，但仍对完整历史执行密集 SDPA。已完成 32K 末段同步诊断：多 token 块的 attention 内部以 SDPA 耗时最高，下一项 attention kernel 工作应先围绕它验证。本文的字节数与算术量来自源码和形状推导；实测只覆盖下述 32K 窗口，没有 kernel 提速结论。

## 成本来自哪里

[GPUAttention](../../Sources/ANERunnerGPU/GPUAttention.swift) 的 QSA 在历史超过 2051 行后，选择 512 个四 token 块及最多三行尾部，构造 bool mask。reference 路径把这张 mask 交给完整长度的 SDPA，稀疏可见性没有减少 QK/PV 的矩阵维度。

三项随历史长度增长的工作值得分别测量：

- **密集 SDPA。** 本模型为 24 个 query heads、2 个 KV heads、head dimension 256。reference 的 BF16 query 缩放和 QK score 先落到 BF16，再应用 mask、softmax 和 PV。单层单块 QK+PV 约为 `4 × 24 × 256 × S × T` FLOPs。固定 chunk 时，整个冷 prefill 的这部分计算约随 prompt 长度平方增长。
- **完整排序。** [GPUAttention](../../Sources/ANERunnerGPU/GPUAttention.swift) 的 argpartition 调用进入绑定 MLX 的 GPU `ArgPartition::eval_gpu`，该版本实际执行完整 merge sort；Runner 随后才截取最后 512 个块。每个 query 排序约 `T/4` 个分数，`qsa.select_mask` 同时还包含选中块的 scatter、展开和 causal/tail mask 构造。
- **完整历史复制。** reference K/V、raw index 和 pooled index 每次 concat 都生成新的完整输出。绑定 MLX 的 `concatenate_gpu` 分配 `out.nbytes()`，逐个 input 复制到输出切片，没有 donation 分支。固定 chunk 下，这些历史复制累计约为 `O(P²/S)`。新 token 的投影、MoE、GDN、PLE 处理主要随新增 token 数量增长，不能与这些历史成本混为一项。

在真实 262K 最后一个完整块 `S=416, T=262080`，每个 attention 层的一个 BF16 score plane 为 **5,233,213,440 B**，QK+PV 约 **2.679 TFLOPs**。selector 的四头 FP32 products 为 **436,101,120 B**；求和后的 scores 和按 head 广播的 bool mask 各为 **109,025,280 B**。这不是把 mask 物理复制成 24 份。

同一块的 K/V、raw、pooled concat 输出合计每层 620,605,440 B，12 层共 **7,447,265,280 B（约 6.936 GiB）**。理想输入读取加输出写入约 13.872 GiB，只是逻辑传输量，不能当作实测 DRAM 流量。每四层求值的生命周期也不允许把 12 层 score plane 直接相加作为峰值内存。

完整 checkpoint 的保存是少数 publication 点上的另一项复制，不是每块都保存；RAM 恢复共享 attention handles 后，后续 reference suffix concat 仍需生成完整历史。保存/恢复能力与实测边界见 [长上下文结果](KV_LONG_CONTEXT_RESULTS.md) 和 [AR prefix cache](../AR_PREFIX_CACHE.md)。

## 源码身份与定位

Runner 的原生库位置由 [Package.swift](../../Package.swift) 中的 `ANERUNNER_MLX_ROOT` 决定。本次检查的绑定 MLX 头文件版本为 **0.32.2**，默认安装根为仓库相对路径 `../qwen38-ssd/runtime/mlx-serve/lib/mlx`，对应源码在同级 `mlx-src`。这些结论针对这份绑定实现；版本号相同但源码或库不同仍需重新核对。

主要原生定位（路径均相对于该 `mlx-src`）：

| 实现 | 定位 | 源文件 SHA-256 |
| --- | --- | --- |
| BF16 reference fallback | `mlx/fast.cpp:826–887`；Metal D256 dispatch 在 `mlx/backend/metal/scaled_dot_product_attention.cpp:719–768` | `fast.cpp: 29b31f96bdb02afdadbde3f2bcf42cf458fc69385a908bbde9fc3a14d23de077` |
| ArgPartition 全排序 | `mlx/backend/metal/sort.cpp:118–264,342–353`；稳定比较见 `mlx/backend/metal/kernels/sort.h:40–81,130–156` | `sort.cpp: 4f5b55d2b753900bdeb7eeeb5652aaea3cb24fabd1297500c83514eebc1532d7` |
| concat 分配与复制 | `mlx/backend/metal/slicing.cpp:14–42` | `604d3bea4ce0e87eda3ca550200fc06eb78d949de49db42ef8bf12fb331e0b6c` |

Runner 的实现入口是 [GPUAttention](../../Sources/ANERunnerGPU/GPUAttention.swift)、[GPUProfiler](../../Sources/ANERunnerGPU/GPUProfiler.swift) 和 [generate-gpu](../../Sources/ANERunnerCLI/GPUGeneration.swift)。源码形状推导不等于实际 kernel 占比；完整运行仍需绑定 binary、原生库、源文件及输入 hash。

## 只测真实末段

`generate-gpu` 提供两个显式诊断参数，均要求开启 `--profile-stages`：

- `--profile-attention true|false`：开启已有 attention stage 细分，默认 false，不切换 attention kernel。
- `--profile-from-token N`：只记录 forward **起点 offset >= N** 的调用，默认不限制。N 必须非负且小于实际 prompt 长度；跨过 N 的 chunk 不会被拆分。报告字段为 `profiler.minimumPosition`，旧报告缺少该字段时按 nil 处理。

窗口以前的 `measure` 只执行原 body，不收集输出、不增加 stage 同步；`isRecording` 不依赖位置，以便模型继续设置 forward context。新窗口模式仅在最终 prefill S1 评估和记录 `mixer_and_head`，避免把中间块不使用的词表 head 算入诊断；未指定窗口时保留原行为。

以下命令需要包含这两个参数的构建，在 [独占实验窗口](../EXPERIMENT_CONTROLLER.md) 运行。输入必须是完整 **32766 个 Int32 ID 的 JSON 数组**；路径替换为实际模型和 fixture，输出使用新目录：

```sh
.build/release/ane-runner generate-gpu \
  --model-dir /path/to/Qwen3.8-Flash-Next-MLX-SSD-Stream \
  --tokens-file /path/to/tokens-32766.flat.json \
  --context 32768 --max-tokens 2 --repeat 1 --mtp-depth 0 \
  --prefill-chunk 416 --prefill-eval-layers 4 --prefill-attention reference \
  --decode-mode reference --gdn-gemv-mode reference --ssd-prefetch nextChunk \
  --profile-stages synchronizedStages --profile-phase prefill \
  --profile-attention true --profile-from-token 32032 \
  --output results/long-prefill-profile32-v1/model.json
```

基准 fixture 的完整 ID 数组来源 SHA-256 为 `d041c43bc334d5b0bcf57a9e30887bfc8a4dd1605ab3c5e93d0b5808524d79a1`。此次规范化空白后的 flat 文件 SHA-256 为 `2864df396c56525dd134019855d9f204980971800520abf5b5dfb271e179891a`，32766 个 ID 已逐项核对相同。只有使用这些相同 ID 时，下述已知输出 `[16,11]` 才是适用的连续性检查；任意同长度输入不能沿用该 oracle。

P32766 的最后两个 body 块是 `32032/S416` 和 `32448/S317`，之后保留 `32765/S1`，仍属于 prefill。这个窗口包含三次 forward；若从 32448 开始，则只有 S317 和 S1，必须另行标记。不能把 S1 平均进 S416 或称为 decode 吞吐。

## 读报告与验收

要求完整 48 层、AR、reference、chunk416/eval4，输出 `[16,11]`、length finish、final offset 32767；另核对 `trials[0].decode_steps == 1`、一个 `decode_step_seconds` 和正的 `phase_metrics.decode_seconds`。输出连续性不替代完整 121 状态正确性 oracle。

`profiler` 必须报告 `synchronizedStages`、`phaseFilter=prefill`、`attentionBreakdown=true`、`minimumPosition=32032`、`droppedRecords=0`。所有 record 成功且时长有限非负；位置只能为 32032、32448、32765，各对应 tokenCount 416、317、1。每个位置的 12 个 attention 层均须有完整八个子阶段，合计 288 个 attention record；不应再包含重复计时的外层 `attention`。只有最终 S1 有一个 `mixer_and_head`。

按位置、阶段汇总 `elapsedMilliseconds` 与 `evaluationWaitMilliseconds`，比较 `qsa.select_mask`、`attention.sdpa`，再看 `attention.kv_append`、`qsa.history_pool` 和 `qsa.score`。elapsed 已含 evaluation wait，不能再次相加；`precedingStreamDrainMilliseconds` 不在 stage sum 内，应单列。select_mask 包含 mask 构造，不能标成纯排序时间。默认 stage allocator snapshot 为 nil，不能读作内存为零。

同步诊断会改变正常 overlap 和复用。这份报告不代表 GPU-only 时间、无干扰吞吐或服务延迟，也不能用于解释 HTTP 与 CLI 的时差。退出状态、完整 report、源/库/输入冻结及服务恢复仍需验证。

## 运行结果

**2026-09-14：32K 末段同步诊断通过。** 独占控制器先完成 large SSD import，再运行本 profile；两 case 均 exit 0。实际 profile 在北京时间 04:42:59–04:44:15 运行，macOS 26.6.2、绑定 MLX 0.32.2。根控制器 postflight 577 个冻结文件和 102 个模型文件 stat 零差异，参考服务以完全相同 argv 恢复并确认 idle，MTP/drafter 均未加载。

独立分析器核对全部 32766 个输入 ID、输出 `[16,11]`、length finish、实际 final offset 32767 和一次 decode。报告共 **1123 条成功 record**，含 **288 条唯一 attention 子阶段**，恰好覆盖三位置 × 12 层 × 八阶段；无 drop、非法时长、重复外层 attention 或窗口外位置。唯一 vocabulary head 位于 32765/S1。

以下原始文件保存在本机 `results/night-final-small-v1/`，该 ignored 目录不随仓库发布；身份用于复核本次结果：

| 项目 | SHA-256 |
| --- | --- |
| Runner binary | `91d626264dbfdbbd40d5c22bc6c4292a9ead873ec0c89ac7a71f2d01e1861117` |
| `libmlx.dylib` | `fc7cbc4002ecfe90cb1f7b73f21f02a733bf7f536c6486030a04dc5fdf000469` |
| 原始 `profile32.json` | `7f18301a2bf24e38b4475bb1ac507799f1aee709f6a4e42b0b8f71f950eefcce` |
| `independent-profile32-analysis.json` | `1be0c39df6d0be2ce4b3c6eee641ccfcf983dcfe05ad12fd54bc5a147fc447c1` |
| `profile32-relative-costs.json` | `d4e3a1a3219898877af4d5de4773fae8ca3770bb8472a4c8c110fa4dae870a40` |

每格为 **12 个 attention 层的毫秒总和（该列占比）**。三个时间列分别以同位置八个子阶段的对应总和为分母，不能互加。evaluation wait 已在 elapsed 内；前置 drain 另列。位置之间也不混合平均。

**offset 32032 / S416：** attention 子阶段 elapsed 合计 641.315 ms，evaluation wait 639.051 ms，前置 drain 2.774 ms。

| 阶段 | elapsed ms（占比） | evaluation wait ms（占比） | 前置 drain ms（占比） |
| --- | ---: | ---: | ---: |
| `attention.qkv_projection` | 28.223 (4.40%) | 27.956 (4.37%) | 0.286 (10.31%) |
| `attention.kv_append` | 38.563 (6.01%) | 38.485 (6.02%) | 0.351 (12.66%) |
| `attention.index_projection` | 4.485 (0.70%) | 4.424 (0.69%) | 0.344 (12.38%) |
| `qsa.history_pool` | 12.842 (2.00%) | 12.269 (1.92%) | 0.321 (11.56%) |
| `qsa.score` | 25.737 (4.01%) | 25.599 (4.01%) | 0.310 (11.17%) |
| `qsa.select_mask` | 52.663 (8.21%) | 52.044 (8.14%) | 0.359 (12.96%) |
| `attention.sdpa` | 380.432 (59.32%) | 380.067 (59.47%) | 0.300 (10.81%) |
| `attention.output` | 98.369 (15.34%) | 98.208 (15.37%) | 0.504 (18.15%) |

**offset 32448 / S317：** attention 子阶段 elapsed 合计 525.562 ms，evaluation wait 523.457 ms，前置 drain 2.696 ms。

| 阶段 | elapsed ms（占比） | evaluation wait ms（占比） | 前置 drain ms（占比） |
| --- | ---: | ---: | ---: |
| `attention.qkv_projection` | 23.411 (4.45%) | 23.149 (4.42%) | 0.333 (12.37%) |
| `attention.kv_append` | 37.407 (7.12%) | 37.311 (7.13%) | 0.347 (12.86%) |
| `attention.index_projection` | 4.330 (0.82%) | 4.271 (0.82%) | 0.319 (11.85%) |
| `qsa.history_pool` | 13.895 (2.64%) | 13.331 (2.55%) | 0.307 (11.38%) |
| `qsa.score` | 18.021 (3.43%) | 17.895 (3.42%) | 0.359 (13.31%) |
| `qsa.select_mask` | 44.286 (8.43%) | 43.776 (8.36%) | 0.318 (11.82%) |
| `attention.sdpa` | 303.105 (57.67%) | 302.767 (57.84%) | 0.334 (12.38%) |
| `attention.output` | 81.106 (15.43%) | 80.957 (15.47%) | 0.379 (14.05%) |

**offset 32765 / S1：** attention 子阶段 elapsed 合计 63.950 ms，evaluation wait 62.781 ms，前置 drain 2.422 ms。

| 阶段 | elapsed ms（占比） | evaluation wait ms（占比） | 前置 drain ms（占比） |
| --- | ---: | ---: | ---: |
| `attention.qkv_projection` | 12.081 (18.89%) | 11.886 (18.93%) | 0.277 (11.42%) |
| `attention.kv_append` | 19.826 (31.00%) | 19.760 (31.47%) | 0.321 (13.27%) |
| `attention.index_projection` | 3.381 (5.29%) | 3.339 (5.32%) | 0.310 (12.79%) |
| `qsa.history_pool` | 4.590 (7.18%) | 4.382 (6.98%) | 0.293 (12.09%) |
| `qsa.score` | 4.541 (7.10%) | 4.410 (7.02%) | 0.280 (11.56%) |
| `qsa.select_mask` | 6.992 (10.93%) | 6.600 (10.51%) | 0.344 (14.22%) |
| `attention.sdpa` | 5.247 (8.20%) | 5.182 (8.25%) | 0.303 (12.51%) |
| `attention.output` | 7.292 (11.40%) | 7.222 (11.50%) | 0.294 (12.13%) |

多 token 块的 `attention.sdpa` 分别为 **380.432 ms / 59.32%** 和 **303.105 ms / 57.67%**；`qsa.select_mask` 为 8.21% / 8.43%，`attention.kv_append` 加 `qsa.history_pool` 为 8.02% / 9.76%。这些比例只描述 attention 子阶段。同期所有已记录模型阶段的 elapsed 合计分别为 1888.115 ms / 1810.799 ms，其中 MoE 为 **862.855 ms / 969.755 ms**；不能把 attention 的 59% 写成整个模型的 59%。

最后 S1 另看：KV append 加 history_pool 占 attention elapsed 的 38.18%，SDPA 为 8.20%；这不是长块的同一种工作形态。该 S1 的模型阶段合计 1017.666 ms，不能与记录在业务 decode 中的一步约 0.043 s 比速度：前者逐阶段同步、包含诊断扰动，后者没有相同的 observer。唯一最终 head 为 33.496 ms，已排除在上表 attention 总和之外。

**下一项 kernel 优先级：在 attention 内先研究 SDPA。** 首先用同一 Q/K/V/mask 做单层数值诊断，评估减少未选中历史行计算的实现，并保留 reference 的已知 BF16 中间边界。按可见行 gather 会改变归约，不能仅凭这一耗时占比就替换默认实现；现有 fusedQSA 的跨模式失败仍未放行。精确 top512 可作为随后风险较小的选块优化，但本轮多 token 窗口里 selector 约占 attention 的 8%，优先级低于 SDPA。

项目总体仍优先完善 KV streaming / SSD cache 管理。以上仅确定后续 attention 实验的顺序，不改变当前产品目标，也不从这三个同步采样外推 262K 的阶段占比、整模型收益或加速倍数。

如果 select_mask 占比值得优化，下一实验可只替换完整排序为精确 top512，保留现有 FP32 score、bias、scatter 与 bool mask。稳定排序的等值规则、±0/±inf/NaN、未来块均需覆盖；1e-7 bias 可能被舍入，不能假定没有 tie。目标先逐 query 比较选中集合和完整 mask 字节，再比较原 SDPA 输出及全状态。

若密集 SDPA 占主导，再评估按 query gather 可见行；它改变 GEMM tile、softmax 与 PV 归约顺序，保留 BF16 中间边界也不能保证逐位一致。[32K fusedQSA 跨模式筛选](KV_LONG_CONTEXT_RESULTS.md) 已有 58/121 状态超出固定门槛，不能放宽阈值后放行。单层同 Q/K/V/mask 的精度诊断及完整 cold/warm 回归仍是前提；本文没有接受 kernel 替换。
