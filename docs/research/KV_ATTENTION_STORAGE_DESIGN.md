# K07 第一增量：AR decode 的 K/V 容量追加

审阅日期：2026-09-09。本文只做代码设计审查，未改运行源码，未 build、运行 GPU 或测量性能。依据是本地 runner、实际安装的 MLX 对应源码，以及已经固定版本的上游研究。

建议先做一个默认关闭的 **单 token AR decode K/V capacity append**：保留容量 buffer，追加新行，现有 SDPA 读取逻辑前缀 view。先证明 MLX 确实复用了 buffer，再考虑 QSA 辅助数组和分页 reader。它有独立的收益机会，但不等于 K07 的跨请求页共享、尾页 COW 或 K08 增量持久化已经完成。

## 1. 当前实际复制路径

| 路径 | 当前行为 | 首个增量 |
|---|---|---|
| `GPUAttention.forward`，`attention.kv_append` | 每层 K、V 各一次 axis-2 concat；BF16 `[1,2,T,256]`，每个 decode token 重建全部历史 | 改为容量追加，逻辑 shape 仍为 `[1,2,T,256]` |
| `GPUAttention.qsaMask`，`qsa.history_pool` | raw index 每步 axis-1 concat，BF16 `[1,T,128]` | 暂时保留，作为下一独立增量 |
| 同上 pooled index | 超过 2051 行后，每完成一组 4 行追加一个 `[1,1,128]` pooled key | 暂时保留，不改变池化与选择顺序 |
| `QwenModel.privatePrefixStateCopy` / archive export | 对逻辑状态做独立紧凑复制，完整保存 Attention、QSA、GDN、PLE | 保留，继续充当 oracle 和缓存格式边界 |

当前 `GPUAttention.State.retainedRowCount` 同时代表 K/V 与 raw allocation extent。这在二者每次一起 concat 时成立；只给 K/V 加容量后必须拆分 extent，不能让 raw 的保留量诊断跟着 K/V 容量一起虚增或遗漏 K/V padding。

本地作者 runner 并非只有 concat：`../qwen38-ssd/runtime/mlx-serve/src/transformer.zig` 的 `KVCache.updateDense`（约 3553 行）会释放旧 views、增长完整 buffer、`writeAtOffset` 使用 `mlx_slice_update`，再创建逻辑 views。正常 `gatedFullAttnWith`（约 14193 行）通过 `cache.update` 使用该实现。其 QSA raw/pooled 路径仍有 concat。这个实现提供了具体所有权处理参考；本次未证明本地作者源码等于某一已固定 commit，也未把它当作本模型性能对照结果。

## 2. 为什么暂时不用新分页 attention kernel

安装 stamp 为 `mlx=1f8e74e3f12f mlxc=56b2d39fc831 target=26.2`；完整 MLX 上游 SHA 为 [`1f8e74e3f12f31365464a6867c6579f0e9b29d85`](https://github.com/ml-explore/mlx/commit/1f8e74e3f12f31365464a6867c6579f0e9b29d85)，参见 [已有本地核对](../MOE_BRANCH_OVERLAP_FEASIBILITY.md)。本地 MLX 有项目补丁，以下路径均以实际 `../qwen38-ssd/runtime/mlx-serve/lib/mlx-src` 内容为准，固定上游链接只用于定位原始实现。

现有 scalar decode SDPA 已接受所需布局。`mlx/backend/metal/scaled_dot_product_attention.cpp:776` 的 vector 路径对 B=1 的 K/V 只要求最后一维 stride=1；`[1,2,C,256]` 容量 buffer 的 `[1,2,T,256]` slice 满足要求，不必先 `contiguous_copy_gpu`。kernel 使用实际 head/sequence strides，单次与两次归约的选择仍看逻辑 `T`，不是容量 `C`。因此可以保持当前 attention kernel、逻辑长度、mask 和归约边界。

首个增量只在明确 `.decode`、S=1、普通 AR、positionBase=0 下生效。仅凭 S=1 不够：最后一个 prompt token 也是 S=1，MTP head/verify 也会调用相同 attention 类。prefill 先保留原路径；本地 causal matrix SDPA 的 `has_backing_rows` 补丁会根据底层 padding 调整读取长度，贸然把容量策略扩展到 prefill 会扩大数值验证范围。

真正分页要让 SDPA 直接读取页表，并同时处理 QSA 访问；若每 token 先把页 concat 回连续 KV，主要复制成本仍在。当前容量方案省去了这个 reader 改造，代价是不同请求仍各自持有完整 K/V。

## 3. 最关键的门槛：slice_update 不自动等于原地写入

公开 C API 已有 `mlx_slice_update`；runner 的 `Tensor.swift` 尚未包装。实际 Metal 路径：

1. `mlx/backend/metal/indexing.cpp:733` 的 `SliceUpdate::eval_gpu` 先调用 `copy_gpu(in, out)`，然后写更新区域。
2. 只有输入为完整 contiguous allocation，才走可 donation 的 Vector copy；逻辑 slice 可能走 General copy，必然重新分配。
3. `mlx/backend/common/copy.h:25` 只有 donation 成功才复用输入；否则先复制全部容量。
4. `mlx/array.h:294` 同时要求 descriptor 和 data 唯一所有权。旧 view、旧完整 Tensor、外部 State/checkpoint alias 都可能阻止 donation。

相应固定原始实现：[SliceUpdate](https://github.com/ml-explore/mlx/blob/1f8e74e3f12f31365464a6867c6579f0e9b29d85/mlx/backend/metal/indexing.cpp#L733)、[copy 分配](https://github.com/ml-explore/mlx/blob/1f8e74e3f12f31365464a6867c6579f0e9b29d85/mlx/backend/common/copy.h#L25)、[donation 条件](https://github.com/ml-explore/mlx/blob/1f8e74e3f12f31365464a6867c6579f0e9b29d85/mlx/array.h#L294)。

实现时应对**完整容量 buffer**调用 slice-update，并在其 lazy graph 求值前释放本请求旧逻辑 views 和旧 backing handles。不能一边保留旧 `var next = state`，一边期待 donation。外部 alias 必须继续生效：它存在时退回安全的 functional copy，旧 checkpoint 保持不可变；不能绕过 MLX 的检查强写旧输入。

还有一个容易造成误判的位置：当前 `attention.kv_append` 的同步 profiler 会在闭包返回后立即 eval。若仅在 `forward` 尾部替换旧 state，profiler 模式可能保留旧 views，使本来正常模式可 donation 的操作变成全容量复制。追加 helper 必须在返回可 eval 输出前完成旧句柄交接；普通、同步 profiler 和 async submit 三条路径分别检查。Swift 代码看起来已经置空不是运行证据，ARC 和 MLX graph 的最终引用状态仍须验证。

## 4. 建议的最小接口和状态迁移

以下是拟议接口，不是已提供的 API。

| 文件/函数 | 小范围改动 |
|---|---|
| `Tensor.swift` / `MX` | 增加 checked `sliceUpdate(source:update:starts:ends:)`；不开放任意输入 pointer 写入 |
| `GPUAttention.swift` / `State` | 增加私有 K/V 全容量 Tensor 和 `kvCapacityRows`；保留公开逻辑 `keys/values`；分别记录 K/V、raw、pooled extent；reset 释放全部句柄 |
| 同文件 / `appendKV` helper | 单独管理 grow、旧 views 释放、whole-buffer update、逻辑 views 重建。使用值字段和 MLX functional graph，避免共享可变 storage class |
| 同文件 / `forward` | 接收可选 `KVAppendPolicy` 和容量上限；既有 concat 为默认/reference；先更新 K/V 存储，随后 QSA 仍按原 offset 计算，成功后统一推进 logical offset |
| `QwenModel.forward` | 显式传递策略，只允许普通 AR decode 生效；所有 prefill、verification、MTP、capture 调用默认 reference |
| `QwenGenerationRequest` / AR `decodeStep` | 请求级固定策略，校验与 MTP 互斥；仅 AR 分支传入。容量上限来自已批准的 `prompt.count + maxTokens` 范围 |
| `QwenModel` 状态复制、导出和预算相关 helper | 逻辑快照仍紧凑；必要时拆分 evaluation roots 与 logical payload 统计，不改变 archive schema |

一次普通追加的所有者转换：

`已 eval 的当前 backing + logical views → 释放本请求旧 views → 创建 whole-buffer 更新 graph → 移交新 backing、重建新 logical views → eval/stream 完成 → 下一 token`。

在创建 graph 的小作用域内释放旧完整 Tensor 临时引用；不要把旧 State 持有到外层输出 eval 后。可以在 exclusive `inout` 访问中更新内部 backing；若随后投影/QSA/eval 抛错，沿用 `QwenModel.forward` 已有的 `state.valid=false` 和 generator 同步恢复/丢弃游标，不把更新一半的状态放回可重试队列。不能为维持“失败前 state 不变”的假象而永久持有全量旧 State，使 donation 始终失败。

容量先用固定增长步长 256 行作为实验值：`C = min(roundUp(requiredRows, 256), admittedRowLimit)`，溢出先拒绝，不能分配到模型最大 262k。首次由紧凑 prefill 状态转换、容量不足增长和外部 alias 触发 fallback 都允许全量复制；正常无 alias 且未增长的 decode 才要求只写新行。第一步的转换费用算入 decode，不隐藏在准备阶段或 warmup。

原 `GPUAttention.prefixState` 需要仍能处理 capacity state：K/V extent 与 raw extent 分开，超出原小尾部保留范围就用 `GPUVerificationCopy.tensor` 紧凑复制，并丢弃多余 backing 句柄。相同 count 返回 alias 继续安全，但后续续写应自然触发 COW fallback。MTP 初版仍禁用新模式，不能借此省略公共状态接口的不变性检查。

`State.tensors`、`namedTensors`、`prefixStatePayloadBytes` 与 archive 只能描述逻辑数据；若为了及时 detach graph 将完整 backing 加入 joint eval，应建立单独 `evaluationTensors`，不得直接加入逻辑 payload 集合造成双计数和格式漂移。诊断新增容量指标也不得参与数值状态 hash。

## 5. 预估收益与资源代价

12 个 attention 层，BF16 K/V 每层每历史 token 共 2048 bytes；raw 为 256 bytes，pooled 每 4 行产生 256 bytes。历史 concat 的读+写工作量按实际逻辑形状推导：

| 历史长度 T | K/V 每步读+写 | raw 每步读+写 | pooled 每步平均读+写 | 合计 |
|---|---:|---:|---:|---:|
| 10,000 | 468.750 MiB | 58.594 MiB | 3.662 MiB | 531.006 MiB |
| 11,232 | 526.500 MiB | 65.813 MiB | 4.113 MiB | 596.426 MiB |

pooled 平均按每 4 步追加一次估算；未计新行、激活或其他 kernel。K/V 占该历史复制量约 88.3%，因此先优化 K/V 比同时重做四种数组更划算。上述不是硬件 DRAM 字节，也不是已测节省时间；权重读取仍可能占主导，整模型 TPS 收益可能较小。

256 行线性增长带来的 K/V 最大 padding 为 255 行，12 层合计约 **5.98 MiB/请求**。相比几何倍增，容易控制统一内存和并发额度。代价是初次转换与每次增长的历史复制/清零，短输出 16 tokens 可能不能很好摊薄；所以必须同时测 16、128、512 tokens，不能只报已经增长完毕的内层 append。

当前 `requestStateReservation` 为 AR 预留 `2 × estimatedPrefixStateBytes(P+maxTokens)`，用于 functional old/new。若新容量严格不超过该批准长度，两份容量状态可落在相同形状上界内；但 grow 的 zeros、中间 slice-update、外部 alias、延迟图可能形成额外同时存活的 buffer。**必须先证明实际增长图的峰值上界；证明不了就额外 reserve 临时额度，eval 完成后再释放，不能直接宣称原 2 倍一定覆盖。** 观测分别列出逻辑 payload、K/V backing/slack、grow/fallback 临时 bytes、MLX active/cache/peak 和进程 footprint，不能把 ledger 当物理上限。

QSA raw/pooled 是后续收益较小的增量。raw 可复用相同容量机制，但投影切片本身可能保留 640 列 backing；pooled 更新每 4 步一次，收益更小。首个增量不丢弃完整 raw 历史，不改变 FP32 mean → BF16 → RMSNorm → RoPE 的顺序、2051/2052 启用边界、4 行分组、top512、tie bias 和未完成尾组的可见性。

## 6. 最小验证门槛与停止条件

**第一关：小张量机制，先确认值得接整模型。** 使用 B1/Hkv2/D256 的真实 MLX buffer，不加载权重。覆盖首次转换、连续追加、容量边界前后、非整 256 长度、上限 clamp、保留旧 State/view 再追加、reset。逻辑 K/V 每个 BF16 原始 bit 都与 concat oracle 一致，旧 alias hash 不变。通过可核对的 native 诊断或 Metal trace 证明：未增长且无 alias 时 backing 确实复用、没有全容量 copy；alias 时 fallback 确实保留旧内容。allocator active 值不变、C handle 地址相同或没有崩溃都不充分。诊断不得通过持有旧 view 本身破坏 donation。

**第二关：现有 SDPA 与整状态。** 对逻辑长度 255/256/257、1023/1024/1025、2051/2052、4095/4096/4097 及 11k+ 测 K/V capacity view 与紧凑 reference 的输出；QSA 分组边界与有限真实 mask 单独覆盖。保持同一 kernel/归约时要求逐位一致，不能给“缓存变更”放宽容差。完整模型从相同 prefill 状态分两条普通 AR 路径，检查 raw IDs、finish reason、逐步/边界全部逻辑 mixed state 与 host offsets；继续用独立紧凑快照，不让两条路径共享可写 backing。K02 的 416 checkpoint、最后 prompt token 重算和完整 cache oracle 不变。

**第三关：资源与业务边界。** 两个 decode 游标交错，一支保留 checkpoint 后继续、一支取消，clear 与 cache copy 交错，另一支仍正确；容量不足拒绝和设备错误后游标失效；request/workspace lease 最终归零。测试 snapshot copy/export 后源继续 decode、compact restore 后首次启用新模式。检查 async submit 与同步 profiler 分别不发生所有权破坏；性能测量关闭逐状态 readback 和同步细分 profiler。

**第四关：收益。** 本机受控串行 ABBA，固定输入、原始 token、prefill/decode 策略和冷/热状态，10k+ prompt 配 16/128/512 output。报告独立 prefill 时间、包含初次转换/增长的 decode 时间与 TPOT 分布、grow 尖峰、实际 copy/allocate 证据、内存峰值。只有显著大于运行噪声、可重复的整体 decode 改善，且短请求与资源成本可接受，才建议进入默认候选；否则保持实验关闭。

若第一关发现正常路径 donation 始终失败，先修正句柄生存期；若 SDPA 又隐式复制逻辑 views，先定位 layout predicate。不要为了让实验显得成功而换成未经所有权证明的 custom kernel 输入写入。若公开 API 在当前所有权结构下确实无法达到目标，再评估小型 native Primitive；这时应作为另一个明确增量审查，而不是立刻扩建通用页池。

## 7. 与已固定上游机制的关系

- [vLLM 研究](KV_VLLM_REVIEW.md)已固定 release `2cf0a6915ce544dc493a0990f2ea38d81601128a` 与 main `1b2c591cd0c3bb5a85ac7f3d6cbaa2fa7df6bc7d`。吸收的是不可变共享、COW 源/目标引用留到完成、direct reader 才有分页收益；本增量仅验证追加/读取布局，不实现 block pool 或跨请求共享。
- [SGLang 研究](KV_SGLANG_REVIEW.md)已固定 `30e7a3072d3f1e9bd70cd5e44146ca27c80522c4`。共同可恢复边界和 recurrent 私有化仍适用于 Qwen GDN/PLE；Attention 容量不产生任意 token 的 GDN checkpoint。统一内存无需照搬 CUDA 的完整 CPU/GPU 双池。
- K07 后续真正页化应接不可变完整页、私有尾页及直接读页的 SDPA/QSA；快照仍可在选定 checkpoint 时 materialize。整请求页共享、减少 system+tail 重复快照、降低 SSD 写放大都是后续目标，不能由本次 append 优化的收益代替验收。
