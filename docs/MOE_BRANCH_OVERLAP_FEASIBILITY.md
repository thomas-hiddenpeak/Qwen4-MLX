# MoE shared/routed 重叠：pinned MLX 可行性

2026-09-07。本节结论来自本地固定源码、安装库符号和真实 fixture 元数据。独立 native library 与 Swift release 已构建，reference/fused 两组真实单层 GPU 测试均通过 finite/bitwise 和编码计数检查，但 native pair 均未达到预设性能门槛。探针只改调度，保留原量化、投影和 BF16 回存。

**本轮双 down 探针数值正确，收益不足，保留为诊断入口，不接入生产推理。** 当前 MLX 本身已使用 concurrent encoder；实测的小幅差异无法归因于硬件并行，不能把这个实验写成已实现的并发提速。完整支路扩展也不凭本轮结果推进。

## 固定来源与现状

- DwarfStar：[`9ab705347c1775e7599ede7eb81a6255ec7dccb5`](https://github.com/antirez/ds4/commit/9ab705347c1775e7599ede7eb81a6255ec7dccb5)，本地 `results/research/upstream/ds4`；[原调研](research/REDIS_AUTHOR_RUNNER.md)。
- MLX：安装 stamp 为 `mlx=1f8e74e3f12f mlxc=56b2d39fc831 target=26.2`，上游完整 SHA [`1f8e74e3f12f31365464a6867c6579f0e9b29d85`](https://github.com/ml-explore/mlx/commit/1f8e74e3f12f31365464a6867c6579f0e9b29d85)。本地 pinned tree 位于 `../qwen38-ssd/runtime/mlx-serve/lib/mlx-src`，判断以实际本地内容为准。
- 本项目：[GPUMoE](../Sources/ANERunnerGPU/GPUMoE.swift)、[GPUMoEFused](../Sources/ANERunnerGPU/GPUMoEFused.swift)、[现有 prefill plugin](../native/moe_gateup_bridge.cpp) 与 [builder](../scripts/build_mlx_moe_gateup.py)。

## 数学依赖：shared expert 是独立支路

对一层输入 `x`，router 计算 top-10 的 IDs 与 scores；routed gate/up → SwiGLU → down → 加权归约得到 `r`。shared gate/up → SwiGLU → shared down 得到 `s`，另一个常驻小投影 `sigmoid(x × shared_router)` 得到 `g`，最后 `y = r + BF16(s × g)`。实际实现保留 gate/up、SwiGLU 和加权乘法的 BF16 舍入位置。

shared 的 `g` 不是专家 top-k 路由，不依赖 selected IDs 或 scores；两支路直到最终加法才互相依赖。当前模型 routed 是 E512/top10、H2560、中间宽度 640、affine Q4/group64，shared gate/up/down 是 BF16，实际 shared 宽度 640。S1 decode 已有两个 routed custom kernels；shared 投影走 pinned MLX GEMV，N=1 的 shared gate 投影走独立 dot-product 路径。

## DwarfStar 实际做了什么

`ds4_metal.m` 的 [`ds4_gpu_parallel_ffn_start_range`](https://github.com/antirez/ds4/blob/9ab705347c1775e7599ede7eb81a6255ec7dccb5/ds4_metal.m#L9595) 关闭当前 encoder，打开 concurrent encoder，保存 shared gate/up/down 参数。它先编码 routed 与 shared 的 gate/up-SwiGLU，随后在同一 encoder 内加资源 barrier；两个 down 随后编码，最后关闭 encoder，后续消费者通过 encoder 顺序加入依赖。[两个阶段及 finish](https://github.com/antirez/ds4/blob/9ab705347c1775e7599ede7eb81a6255ec7dccb5/ds4_metal.m#L9918)。

它硬性限定可审计的 resident、fused producer/direct consumer 几何；不同 quant、TP 与 SSD streaming 有明确拒绝条件。该实现的 Q8 shared、IQ2/Q2 routed、H4096/top6 等条件不适用于我们，不能移植 kernel body 后直接声称等价。可吸收的是“先编码一层独立 producer，解决所有依赖，再进入下一层”的排列方式。

## MLX 已有能力与真正限制

| 核对项 | 实际接口和含义 | 对实验的影响 |
| --- | --- | --- |
| Metal dispatch type | `device.cpp:576` 的 `get_command_encoder()` 已使用 `MTL::DispatchTypeConcurrent`。 | 普通 MLX dispatch 本来就有重叠机会。 |
| 自动屏障 | `set_input_array` 检查前序输出；`set_output_array` 检查读写 hazard；每次 dispatch 前 `maybeInsertBarrier()` 根据整块 MTL buffer 资源插 `BarrierScopeBuffers`。 | 两个独立分支不一定串行，也不保证处于同一个 barrier epoch。 |
| `start_concurrent()` | `device.h:32` 的 RAII context 暂存输出集合，退出时合并；用于 `slicing.cpp:35` 向同一 buffer 的互不重叠切片写入。它不是打开 Metal 并发。 | 本实验两个输出各自分配，无需这个接口；不要把依赖链或隐式 copy 包在其中。其布尔状态也不支持任意嵌套。 |
| `barrier()` | 发出 Metal barrier，但不清理 hazard 集合。 | 单独调用它不能作为 bookkeeping 的阶段提交；后面可能再次插入 barrier。 |
| Native 边界 | C++ `CommandEncoder` 和 `Primitive::eval_gpu` 可用；Swift/C custom-kernel API 只构造 lazy graph。原生 compute encoder getter 是 private。 | 用一个小 C++ primitive 控制顺序，继续使用 MLX 的 encoder、stream、fence 和临时资源所有权。 |
| 图顺序 | `transforms.cpp:180` 构造宽度受限的 BFS tape，执行时从尾部取出。 | Swift 先写 routed、再写 shared，不足以证明 GPU 按支路顺序串行。必须与现有图实际计时比较。 |

固定链接：[device.cpp](https://github.com/ml-explore/mlx/blob/1f8e74e3f12f31365464a6867c6579f0e9b29d85/mlx/backend/metal/device.cpp#L346)、[device.h](https://github.com/ml-explore/mlx/blob/1f8e74e3f12f31365464a6867c6579f0e9b29d85/mlx/backend/metal/device.h#L32)、[slicing](https://github.com/ml-explore/mlx/blob/1f8e74e3f12f31365464a6867c6579f0e9b29d85/mlx/backend/metal/slicing.cpp#L35)、[tape](https://github.com/ml-explore/mlx/blob/1f8e74e3f12f31365464a6867c6579f0e9b29d85/mlx/transforms.cpp#L180)。

## 最小探针：复用两个原 child primitive

第一版只处理 routed down-reduce 与 shared down，输入 activation 已由原有路径产生。新增一个返回两个 lazy sibling outputs 的 C ABI，接受两个**尚未评估的单节点 recipe**：

```c
int anemlx_moe_down_pair(mlx_vector_array *results,
    mlx_array routed_recipe, mlx_array shared_recipe,
    int serial_control, mlx_stream stream);
```

这里 recipe 是原 custom down kernel 的 `[2560]` BF16 输出，以及原 `matmul([1,640], [640,2560])` 的 `[1,2560]` BF16 输出；不是待遍历的任意图。2D 同 dtype 的 matmul factory 直接产生一个 Matmul 节点；3D 外层 reshape 不传进来。routed recipe 的输入必须是 `[10,640]` BF16 activation、原 Q4 down bank/scales/biases、U32 `[10]` IDs、BF16 `[10]` scores 和原 K/N 标量。

原生 factory 通过 `array::primitive_ptr()` 保留这两个 child，按固定 8+2 个输入构造外层依赖并使用 `array::make_arrays` 返回两个新输出。recipe 自身不是外层 input，否则图调度器会先执行它，失去实验意义。只允许 CustomKernel/Matmul、固定几何、GPU 同 stream、单输出、未调度节点；输出不能别名输入或另一个输出。factory 不修改 recipe。C ABI 是受控内部入口：原 kernel 身份、K/N 标量值与合法 router IDs 由 Swift 固定 recipe 保证，native 检查 primitive 名称/arity/layout，不反射任意 CustomKernel shader 或回读标量。

外层 `eval_gpu` 的顺序是：

1. 核对评估后的实际 layout。routed 八项输入必须 row-contiguous；shared activation 是 row-contiguous，权重必须是原 row-major `[2560,640]` 权重的转置 view。严格拒绝要隐式 copy、cast、fill 或其他节点的情况。
2. 用 `encoder.set_input_array` 预声明两节点全部实际输入，再调用 **`encoder.maybeInsertBarrier()`**。这把所有输入依赖在第一次 dispatch 前解决，防止第二个 down 的输入检查在第一个 down 后插入全局屏障。重复绑定的 slot 随后会被 child 正常重绑。
3. 通过已持有的 `Primitive` 基类虚函数，调用第一个原 child 的 `eval_gpu(inputs, outputs)`；然后调用第二个。控制组只在二者之间插 `encoder.barrier()`。不调用 `eval`、`async_eval`、synchronize 或 scheduler，不直接提交 command buffer。
4. 外层的两个输出及全部 inputs 走既有 `gpu::eval` 资源保持流程；保留 plugin handle 到进程结束。用固定计数记录成功编码，不能称为完成次数或硬件流量。

这些调用有明确本地来源：[array backend API](https://github.com/ml-explore/mlx/blob/1f8e74e3f12f31365464a6867c6579f0e9b29d85/mlx/array.h#L195)、[Primitive 虚接口](https://github.com/ml-explore/mlx/blob/1f8e74e3f12f31365464a6867c6579f0e9b29d85/mlx/primitives.h#L48)、[gpu::eval 的 child 调用与保活](https://github.com/ml-explore/mlx/blob/1f8e74e3f12f31365464a6867c6579f0e9b29d85/mlx/backend/metal/eval.cpp#L30)、[CustomKernel allocation/encode](https://github.com/ml-explore/mlx/blob/1f8e74e3f12f31365464a6867c6579f0e9b29d85/mlx/backend/metal/custom_kernel.cpp#L13)。安装库 `nm -gU` 确认 `array::make_arrays`、`matmul`、CommandEncoder 的 input/output/barrier 符号；具体 Matmul/CustomKernel 的 eval 实现不作为外部链接目标。**独立 native、Swift 集成及下述有限真实 fixture 已通过；这不构成完整模型或任意输入的运行验证。**

已新增一个 C++ primitive/factory、一个小 C ABI header、一个独立 builder、Swift loading wrapper 和仅限探针的两个方法。原 `GPUMoE.forward` / `GPUMoEFused.decode` 函数保持原状。复用现有 native plugin 的 pinned 编译 recipe、stock-install 核查、异常转换、显式 stream 与 dylib 保活；不复用 prefill NAX tiling。编译与有限运行已证明这条双节点路径无需修改 stock MLX、复制 Metal shader 或创建新 queue。

## 一层实验与停止条件

使用 `fixtures/moe-real/converted/decode.json`：来自 layer 0 的真实第一个 decode token，H2560，真实 top10 为 `[333,109,214,249,351,90,315,88,261,148]`；权重仍从同一 checkpoint 读取。原始 capture 的两个 prefill 行也可分别以 S1 运行增加输入和路由覆盖；它们的多 token 原输出不是 S1 逐位参照，应逐个与当前 S1 baseline 比较。

先比较完整 MoE 的 routed_sum、shared_down 和 y 等 13 项诊断：全部有限，与原图逐位一致，再做无 detailed profiler 的有限交错热测。四条路径都记录：现有 lazy graph、相同 recipe 的未包装图、native serial control、native pair；每轮正序/反序交替，真实行顺序轮转。只胜过人为加 barrier 的 serial control，不足以证明改善当前 runner。初始化、JIT、首次权重加载放在热测外，每轮重新构造输出，不重复 eval 同一已缓存输出。保存每轮 wall time、路径和校验。首版限定 stock MLX，未接需要替换库的 GPU command timing overlay，也不增加 kernel 内同步。

单层反复使用同一专家容易命中 GPU cache，其数字不能直接解释整模型 DRAM 利用率；应交错真实行/路由并明确仍是热单层筛选。小门槛为全部数值一致、**现有图中位耗时 / native pair 中位耗时 ≥ 1.05**（固定工作量的速率比例改善 5%，等价耗时下降约 4.762%），才进入 11k/128、MTP 关闭的实模 ABBA；整模型 decode 至少改善 3% 且完整 IDs 相同才考虑保留。更慢、偶发分叉或只改善人为 serial control 即停止。

完整支路若继续，需要固定 producer 层级：routed gate/up 与 shared gate/up；shared SwiGLU；两个 down；原 shared sigmoid/final combine。shared SwiGLU 比 DwarfStar 的 fused producer 多一层依赖，贸然把 routed down 提前排在 shared activation 后，会被随后 shared down 的全局 barrier 截断。先把每一层全部输入 hazard 合并，再编码独立节点；不要泛化为任意图并发执行器。当前探针不改变 prefill、MTP verification、PD 调度或 API 默认行为。

## 为什么可能没有收益

并发不会减少专家权重字节。如果 routed 和 shared 各自已接近同一 DRAM 的可持续上限，时间下界仍由两支路总读量决定；两个分支还争用相同 GPU 算力、寄存器和缓存。只有 occupancy 不足、dispatch 空隙或访存/计算互补留下空闲时才可能获益。理想的 `Tserial / max(Trouted,Tshared)` 仅是资源互不竞争时的上界，不能当作预测。

本轮有限单层结果未支持扩大此 native 路径。即使不新增同步、复用原始运算，包装也可能增加 CPU 建图/资源登记成本；不能只凭理论重叠空间重做后端。

## 2026-09-07 单层实测结果

原始报告为 `results/moe-down-pair-v1/reference.json` 与 `fused.json`。两组使用同一 executable、native plugin、stock MLX、模型 config/index 和三个真实输入行；记录的可执行文件 SHA256 为 `ad7717f28a03015c46a203acd51fc0720da9e7349918b790e523a388c713776b`，plugin SHA256 为 `d1c5d1315307223b70960859e1af61febc3a7ddd3c3dca8b0023f9f8a0f7a245`。环境为 macOS 26.6.2（25G83）、MLX 0.32.2；只有 layer-0 MoE 权重驻留，不是完整模型测试。

每行每模式 warmup 3 次、计时 24 次，每种模式有 72 个样本，每份报告 288 个。下表是全部原始样本重算的中位时间（毫秒），不是 kernel GPU 时间或带宽。

| Shared tail | 原 forward | Recipe graph | Native serial | Native pair | 原 forward / pair | 门槛 ≥1.05 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| reference | 0.2773540 | 0.2714790 | 0.2707710 | 0.2691665 | 1.030418 | 未通过 |
| fused | 0.2752915 | 0.2502715 | 0.2576255 | 0.2642920 | 1.041619 | 未通过 |

相对原 forward，native pair 的中位耗时分别下降 **2.95% / 4.00%**。但是 reference 中 pair 仅比人工 serial control 低 0.59%；fused 中 pair 反而比 serial 高 **2.59%**，比未包装的 recipe graph 高 **5.60%**。逐行差异也不一致：reference 的真实 decode 行中，pair 比原 forward 慢约 2.04%。这些结果不支持把整体差异归因于 shared/routed 硬件并行；本轮没有 GPU dispatch overlap 或物理 DRAM 计数证据。recipe graph 的表现只能提示另一个值得独立验证的图构造问题，不能替 native pair 通过门槛。

独立复核了 JSON 内全部 cases、比较结果、计数和原始计时：

- 每组 3 个 case，每个 case 46 个比较项（13 项诊断 × 3 条候选路径，加各自 plain/diagnostic 输出对照与 baseline 自检），共 138 项；全部 `exact=true`、`all_finite=true`。两组共 276 项通过。
- 两种 native 模式各编码 87 对，均符合 `3 × (2 次正确性求值 + 3 次 warmup + 24 次计时)`；报告实际值与预期均为 `[87,87]`。
- 样本 index 连续，round/row/mode 的正反顺序和轮转规则与源码一致；每个 row/mode 正好 24 个正且有限的样本。总中位数、逐行中位数和性能门槛标记均由原始样本复算一致；fixture 文件 SHA256 与报告一致。

**结论：正确性小门槛通过，性能小门槛未通过。** 保留构建助手与显式单层探针，未接入 generation/prefill/MTP 默认路径，不扩大为完整 shared/routed 调度器，也不发布生产性能收益。两份报告的 `passed=true` 只表示数值/计数检查通过并完成计时；`layer_performance_gate_passed=false` 才是本轮是否进入下一步性能集成的判断。

## 许可

DwarfStar 与 MLX 根许可证均为 MIT；本笔记仅借鉴调度设计。若以后复制具体源代码，需在文件内保留对应作者版权与许可证，并记录固定来源。最小双节点 probe 复用进程内原 MLX primitive，不复制 DwarfStar 或 GEMV kernel body；其 array/encoder 使用模式应按现有 native bridge 标注 Apple MLX/MLX C 来源。

## 当前复现入口

[native bridge](../native/moe_down_pair_bridge.cpp)、[builder](../scripts/build_mlx_moe_down_pair.py)、[Swift loader](../Sources/ANERunnerGPU/GPUMoEDownPairProbe.swift)、[CLI probe](../Sources/ANERunnerCLI/GPUMoEDownPairProbe.swift)。builder 在 `results/moe-down-pair-v1/native` 的 provenance 保持 `built_not_executed`，仅记录编译/链接、C 导出、stock 依赖、签名和原文件不变检查；builder 本身不加载库或启动 GPU。后续 Swift/GPU 结果单独记录于上述 reference/fused 报告。

```sh
python3 scripts/build_mlx_moe_down_pair.py --output results/moe-down-pair-v1/native
.build/release/ane-runner probe-moe-down-pair-gpu \
  --model-dir /absolute/path/to/model \
  --pair-library /absolute/path/to/libanemlx_moe_down_pair.dylib \
  --output results/moe-down-pair-v1/probe.json \
  --shared-elementwise fused --warmups 3 --runs 24
```

输出必须是新路径。`passed` 仅指 finite/bitwise、编码计数与计时采集完成；性能是否达到小门槛看 `layer_performance_gate_passed`。默认使用真实 layer-0 三行。`--shared-elementwise reference` 可检验另一条原有 tail 路径，两次结果分别报告；不要混合为同一基线。
