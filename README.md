# Qwen4-MLX

面向 Apple Silicon 的独立 Swift 推理工程，当前适配 Qwen3.8 Flash-Next，已实现完整 **48 层文本生成**。原始 Q4 专家与 BF16 主干常驻统一内存，51.2B n-gram 表通过 SSD 按需读取；主计算使用 MLX C / Metal，Swift 负责分词、状态、读取和生成循环。推理无需 Python 进程或作者 HTTP 服务。

本目录作为独立 Git 仓库管理，提交源码、测试、脚本、文档及文本测试样本。模型权重、大型张量样本、`results/` 实验结果和构建产物仅保留在本机；文档中的本地结果链接不会随克隆一起提供。构建依赖外部 MLX，默认位置为 `../qwen38-ssd/runtime/mlx-serve/lib/mlx`，可通过 `ANERUNNER_MLX_ROOT` 指定兼容构建；具体步骤见 [GPU runner 使用与验证](GPU_RUNNER.md)。

**当前生成默认：chunk416、SSD 跨块预取 `nextChunk`、1 个 SSD worker，并开启符合条件的融合 causal attention。** `ANERUNNER_FUSED_PREFILL=0` 可关闭融合 attention；GDN blocked 仍关闭，每 4 层同步。MTP 默认关闭，原生 head 和生成会话 API 已接入实验路径，详见 [MTP 与会话接口](MTP_AND_SESSIONS.md)。

源权重格式和 `reference` BF16 prefill 累加不变，另提供 FP32 累加实验选项；chunk416 已改变跨块状态舍入边界，因此不宣称保留作者全部数值语义。新默认配置的输出一致性验证目前限于同一份 1,217-token 文档提示、64-token 生成。构建、普通使用示例和验证范围见 [GPU runner 使用与验证](GPU_RUNNER.md)。已有 Core ML / ANE 探针继续保留，以下记录的是此前局部 ANE 实验，不代表整模型使用 ANE。

普通单 token decode 的性能诊断入口见 [采样、GPU 时间轴与指标口径](TELEMETRY.md)：`generate-gpu --telemetry-dir NEW_PATH` 可保存进程／系统采样，离线分析按载入、预填充与解码分开；有无采样的整模型配对已经保存，GPU trace 本模型结果仍单独核验。物理 DRAM 带宽不可用时保持 `null`。

Agent 长提示基准已加入 [11k system/user 输入](fixtures/gpu-agent-11k/provenance.json)，使用 `--context 16384`；无前缀缓存的首次处理、预热复跑、QSA 路径和已知波动见 [GPU runner 长提示记录](GPU_RUNNER.md#agent-长系统提示基准11k)。1,217-token 输入保留为快速回归，不能代替 agent 长上下文测试。

最新的 [模型专用解码优化](GPU_SPECIALIZATION.md)已加入取回 token、共享专家融合、固定投影合并及同进程交错实验。30 轮完整模型输出一致；组合融合在本轮配对中约快 2.2%，但增加 3.67 GB 内存，且绝对速度存在明显漂移，因此保留可选，单 token decode 仍默认使用 `reference` 路径。

后续的 [主要瓶颈审计](GPU_BOTTLENECK.md)已取得完整 GPU 命令缓冲时间，并用真实 GDN / head 权重做独立微基准。GDN 约占 42% 逻辑权重，其固定矩阵形状成为下一步重点；微基准差值不直接等于整模型提速，物理 DRAM 带宽仍未知。

Prefill MoE 的[专家分组融合](docs/MOE_PREFILL_EXPERT.md)已可通过配置显式使用；后续[叠加路由归约的对照](docs/MOE_PREFILL_COMPOSITION.md)在单层微测有额外收益，但本轮11k整模型prefill基本持平，因此组合保持实验选项。两个阶段分别计时，MTP默认关闭。

## 早期 Core ML / ANE 验证（2026-09-05）

- 本机 macOS 26.6.2 / Swift 6.3.3：独立 Release 编译成功，36 项 XCTest 全部通过（[日志](results/moe-concurrency/swift-tests.log)）；另有 MoE 调度集成验证。
- Swift 原生 SSD 读取：真实表的固定样本、随机与边界行、重复行顺序共 9,280 个数值，与 PyTorch 的 FP8→FP32 缩放→BF16 参考逐位一致；负数与越界请求均被拒绝。[结果](results/ssd-verification.json)
- n-gram 哈希：通过上游 80 个已知索引、EOS 与分块连续性、溢出和负数余数测试；真实模型配置的 CLI 结果保存在 [hash-golden.json](results/hash-golden.json)。
- Swift 原生 Core ML 调用：固定 PLE 的三个输出共 112,640 个值，与同一模型包的 Python Core ML 预测完全一致。[验证摘要](results/coreml-ple-swift-validation.json)

以上 PLE 比较验证调用、输入布局和输出读取的一致性，并非与高精度原模型完全一致。新的 MoE 结果同时对照真实作者输出和独立 FP32 参考，并单独记录硬件证据。

## 完整 MoE 的真实输入验证：初始 FP16 串行基线

Swift 现在执行完整的 `512 路 router → 动态 top-10 → 全部选中专家的 SwiGLU → 加权求和 → 带 sigmoid 门控的共享专家 → 相加`。专家图采用固定容量 32；按专家分组实际 token、补零、分块调用，再恢复 token/slot 顺序。单 token decode 同样只取有效位置，不丢弃任何分配。

从固定作者实现的隔离捕获版本取得实际中文请求的 layer 0 输入。完整原始 prefill 共 26 token，保留从 0 编号的位置 14（汉字内容）和位置 25（模板），以及 offset 26 的 decode 形状步骤作为对照。该步骤的 token `[271]` 是作者延后处理的最后一个 prompt 换行，并非第一个生成内容 token。捕获是在服务就绪后由控制器启用；暖机数据没有进入 fixture。

| 检查 | 结果 |
|---|---|
| Swift 动态路由 | 3 token 的 1,536 个 logits、30 个专家 ID 与 30 个权重全部精确匹配作者实现 |
| 完整 Swift MoE 对独立 FP32 参考 | prefill 2 token 相对 L2 误差 0.337%；decode 1 token 0.288% |
| 完整 Swift MoE 对作者原生 BF16 输出 | prefill 0.602%；decode 0.519% |
| 暖运行完整单层 MoE | prefill 2 token 5.366 ms；decode 1 token 2.922 ms（3 次暖机后 10 次中位数） |
| 首次执行，含编译、载入 | prefill 约 1.69 s；decode 约 0.90 s |
| 已检查的专家包 | 29 个实际 routed experts + shared；360 个非 const 运算全部 preferred ANE |
| ANE 硬件证据 | Instruments 记录到 10 个实际 decode 专家和 shared 各 104 次具名 Prediction，共 1,144 次；CPU_ONLY 对照为 0 次 |

误差均为整块输出的相对 L2，不能推导整模型质量。Swift 使用独立动态路由，并未读取 fixture 中的期望专家来替代路由。FP32 参考保留 Swift 路由权重，使用原始 affine Q4 解码权重、原始共享权重及 FP32 SwiGLU/累加。分支数值、时间拆分和限制作了单独记录：

- [本轮汇总](results/moe-real/milestone.json)、[decode 报告](results/moe-real/decode-fp16-linear.json)、[prefill 报告](results/moe-real/prefill-fp16-linear.json)
- [decode 数值验证](results/moe-real/decode-validation.json)、[prefill 数值验证](results/moe-real/prefill-validation.json)
- [ANE 硬件事件与 CPU 对照](results/moe-real/ane-hardware-evidence.json)、[原始 Instruments trace](results/moe-real/decode-coreml.trace)
- [路由对照](fixtures/moe-real/converted/routing-comparison.json)、[捕获来源](fixtures/moe-real/converted/provenance.json)、[原始结果内部一致性](results/moe-capture/integrity.json)
- [小模型调度集成测试](results/moe-scheduler-test/report.json)：5 token、容量 2，验证跨块、补零、回填、缓存容量 1 的淘汰重载，以及缺失专家拒绝；缓存大小改变时所有输出完全相同。

硬件追踪证明这里的专家计算实际使用 ANE；router、CPU 输入整理、共享标量门控和加权合并仍在 CPU。追踪运行有额外开销，表中速度来自不带追踪的独立运行。运行报告中的 `physicalFootprintBytes` 只覆盖 Swift 进程，不包含系统 ANE/Core ML 服务的全部资源，不能据此声称整模型只需几十 MB。

初始 ANE 成功路径使用 FP16 专家图。保留源 Q4 解码值的 LUT 包约为 FP16 包的一半，但测试过的 S32/S128 计划仍选择 CPU；更小的 affine Q4 包也选择 CPU，并新增少量权重误差。详细对照见 [导出报告](results/moe-export/report.md)。仅全模型 routed experts 的 FP16 权重就约 241.6 GB，不能在本机 128 GiB 内全量常驻。现有有界 LRU 限制已载入的模型对象数量，编译缓存和系统服务内存不受这一数量直接约束。

**局部数值检查通过，性能目标尚未通过。** 使用同一层完整 Q4 专家 bank 和相同真实输入，Python MLX 对照得到：

| 同一层的暖运行 | Swift / Core ML / ANE | Python MLX GPU，输入已驻留 |
|---|---:|---:|
| prefill 中选取的 2 token | 5.366 ms | 0.945 ms |
| decode 1 token | 2.922 ms | 0.692 ms |

初始串行实验约慢 4–6 倍。GPU 对照使用公共 `argsort/gather_qmm`，不是作者融合 Metal 内核的精确计时；包括设备同步，另外保存了 CPU 输入/输出往返的独立系列。decode 的 GPU 数值相对原 capture 也有约 0.611% 差异，因此不能把两种实现看成逐位相同内核的速度比赛。短样本受调度和频率影响；这些结果足以否定初始实现已有速度优势，不能直接推算整模型 token/s。[GPU 对照原始样本](results/moe-mlx/report.json)

## 新进展：有界并发与压缩专家

Swift 增加了 `--expert-concurrency`（默认 1，当前建议实验值 4）。不同专家可以同时提交预测；每个专家独占自己的输入缓冲并在锁内完成预测和输出复制。所有工作结束后才处理缓存淘汰，返回结果仍按原 token/slot 顺序合并。缓存容量小于并发度时自动减小每批调度数量；容量 1 保持串行。错误也等待全部工作结束后返回，避免释放正在使用的模型。

使用同一 FP16 bank，按 1/2/4/8 路正序、逆序各测一次，每组 3 次暖机、10 次计时。全部 9 个输出张量与串行逐位一致，实际专家调用数也一致：

| 完整 MoE 中位耗时 | 1 路 | 2 路 | 4 路 | 8 路 |
|---|---:|---:|---:|---:|
| decode 1 token | 2.610 ms | 2.116 ms | 1.808 ms | 1.815 ms |
| prefill 中的 2 token | 4.827 ms | 3.905 ms | 3.223 ms | 6.480 ms |

4 路在这一轮分别降低约 31% / 33% 耗时；8 路不稳定，不作为推荐值。[配对并发实验](results/moe-concurrency/report.json)。`maximumScheduledConcurrency` 记录提交宽度，不代表 ANE 内部实际并行度。`predictionMilliseconds` 是各调用耗时之和，重叠时可能超过墙钟时间；`expertPredictionWallMilliseconds` 单独记录包含输入整理和结果复制的调度耗时。

压缩路径采用公开 Core ML LUT8 接口，每 4 个输出通道共用一组 256 项码表。29 个 routed experts 从源 affine Q4 解码后的 FP16 权重重新近似编码，shared 保持 FP16。它不是原 Q4 的无损封装，也没有证明 ANE 使用原生 8 位算术。

| 压缩 bank 的完整 MoE 检查 | decode 1 token | prefill 中的 2 token |
|---|---:|---:|
| 对原权重独立 FP32 参考，相对 L2 | 0.330% | 0.360% |
| 对作者 BF16 capture，相对 L2 | 0.555% | 0.612% |
| 新一轮 4 路 FP16 配对中位耗时 | 1.872 ms | 3.300 ms |
| 同轮 4 路 LUT8 配对中位耗时 | 1.741 ms | 3.062 ms |

精度继续通过原定 FP32 0.5% / capture 2% 的局部门槛；路由 logits、专家 ID 和权重完全一致，LUT8 的串行/并发输出也完全一致。同轮压缩版耗时降低约 7%，两轮顺序互换，每组累计 20 个暖运行样本。不同轮次的绝对耗时有波动，应优先使用同轮配对结果。

匹配的 29 个 routed 专家、shared 与两份路由权重文件合计从 **300.50 MB 降至 172.22 MB（减少 42.7%）**。这是未编译模型文件大小，不含报告、编译缓存、运行内存；尚不能证明全模型能常驻 128 GiB。原始源 Q4 和 SSD 表均保留。

新的 LUT8 bank 在 4 路调度下完成独立 Instruments 硬件追踪：10 个实际 decode 专家及 FP16 shared 各 104 次具名 ANE Prediction，共 **1,144 次**，CPU_ONLY 对照为 **0 次**。复制的 shared 包被系统添加了编译 UUID；解析器仅归一化严格 UUID 后缀并保留完整原始标签，旧 FP16 trace 的回归检查也通过。prefill 的其他专家已有计算计划和完整输出验证，尚未单独记录其硬件事件。

- [本轮机器可读汇总](results/moe-optimization-v2/milestone.json)、[新压缩 bank 的 ANE 硬件证据](results/moe-optimization-v2/ane-hardware-evidence.json)
- [压缩 bank 和文件哈希检查](results/moe-lut8-group4-bank/assembly-validation.json)、[manifest](results/moe-lut8-group4-bank/manifest.json)
- [decode 精度](results/moe-optimization-v2/decode-lut8-validation.json)、[prefill 精度](results/moe-optimization-v2/prefill-lut8-validation.json)、[配对计时和逐条命令](results/moe-optimization-v2/benchmark.json)
- [压缩方案对照](results/moe-compression-v2/report.md)：原始 Q4 分组、INT4、INT8 与 LUT8 的精度、大小和设备计划分开记录。

其他尝试保留为诊断工具：动态传入 top-10 权重仍需约 98.3 MB/次，现成权重输入预测约 14.8 ms；把真实专家合并成一个稠密大图会计算未命中的专家，decode 约 2.1 ms；中间维度补到 1280 虽可让 S1 选择 ANE，单专家计时没有独立收益。目前都未接入原生 runner。相关证据分别在 [动态权重](results/moe-dynamic-v2/)、[合并专家](results/moe-grouped-v2/)、[补齐中间维度](results/moe-padding-v2/)；这些探针的计时边界不同，不能作为完整 MoE 的等价速度比较。

**这一 Core ML 阶段的局部正确性和相对旧 runner 的加速通过，超过 MLX GPU 的目标未通过。** 随后新增的 GPU 路径已实现 GatedDeltaNet、QSA、连续状态、Hyper Connection、全部层和输出 head；完整证据单独列在 [GPU runner](GPU_RUNNER.md)，不与这里的 ANE 子图结果混合。

## 实现分工

- **Swift**：完整模型与生成循环、分词、GPU 张量生命周期、GDN/KV/QSA/PLE 状态、SSD 按需读取，以及已有 Core ML 子图调用。
- **MLX C / Metal**：原始 Q4 MoE、BF16 矩阵计算、融合路由/GDN/HC 和 GPU 上的采样选择；部分内核复用固定上游，保留许可和来源。
- **Python**：离线权重转换、参考计算、fixture 准备和结果验证；继续使用工作区已有 uv 环境。
- **Core ML**：当前系统可运行的第一个计算后端。允许 CPU 与 ANE；仅配置 `CPU_AND_NE` 不证明所有操作都实际由 ANE 执行。
- **CoreAI**：macOS 27 / Xcode 27 上后续添加的适配层。当前工程没有伪装成已实现的 CoreAI 后端，也不链接父工程要求 macOS 27 的 Swift package。

作者的 `mlx-serve` 保留为固定版本的服务与参考实现。本工程复用其已构建的原生 MLX 库，但独立执行模型。完整 GPU 路径读取约 77.84 GB 源权重，51.2 GB 表仍按行读取；无需重复下载或复制整套资源。

## 构建

独立 package 最低目标为 macOS 26.2，与当前已构建的 MLX 原生库一致。本机 macOS 26.6.2 / Swift 6.3.3 可以直接构建，不需要升级到 macOS 27。

在本目录运行：

```bash
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun swift build -c release
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun swift test
```

无第三方 Swift package 下载依赖；链接本机已有的 MLX C、MLX 动态库及 Metal 库，路径覆盖方式见 GPU 使用文档。构建目录与父工程相互独立。测试命令仅对本次调用选择已安装的完整 Xcode，不修改系统工具链设置。

## 检查模型与读取 SSD

以下命令均从本目录执行：

```bash
.build/release/ane-runner inspect \
  --model-dir ../qwen38-ssd/models/Qwen3.8-Flash-Next-MLX-SSD-Stream

.build/release/ane-runner lookup \
  --model-dir ../qwen38-ssd/models/Qwen3.8-Flash-Next-MLX-SSD-Stream \
  --rows 0,10000048,320001535 \
  --output results/rows.json

.build/release/ane-runner hash \
  --model-dir ../qwen38-ssd/models/Qwen3.8-Flash-Next-MLX-SSD-Stream \
  --tokens 10,20,30,248044,40 \
  --output results/hash.json
```

`lookup` 最多接受 256 行；通过 `pread` 只读取文件头及请求的行。输出为 FP8 解码、乘 scale 后按 BF16 舍入的数值，再以 Float32 表示，以便与作者的输入语义对照。逻辑读取量不等于实际 SSD I/O，系统文件缓存仍可能命中。

`hash` 接受该模型分词器产生的 token IDs；默认历史填充 EOS，也可以用 `--previous a,b` 显式传入两个前序 token。`tokenize` 与 `generate-gpu` 提供文本入口。哈希 seed 从配置读取，未提供时使用作者默认值 1234；此模型仅有一个 PLE 表，表序号为 0，与 decoder 层位置不同。

## Core ML 固定子图

```bash
.build/release/ane-runner probe-coreml \
  --model ../qwen38-ssd/results/coreml-ple-probe/ple_scale64_minimal_s1.mlpackage \
  --fixture fixtures/ple-s1-actual-rows.json \
  --output results/coreml-block.json
```

fixture 提供有名称的输入张量、形状和数据类型，可选参考输出。此命令验证 Swift→Core ML→Swift 的调用链，并记录误差与调用耗时。JSON 用于离线对照，不作为将来每个 token 的内部张量传输格式。

目前复用前期真实 PLE 权重与真实表行样本来验证这一调用链；其余 PLE 输入仍包含合成状态。该小实验不替代完整 MoE、GatedDeltaNet 或整模型验证，也不证明 ANE 已优于 GPU。

## 运行 MoE

FP16 bank 已导出所需的 29 个实际专家以及早期探针 expert 0；新的 LUT8 bank 含这 29 个实际专家。bank 内其余专家没有导出；任意新输入仍按完整 512 路矩阵计算路由，选中缺失专家会明确报错。可先执行 `route-moe` 获取新输入需要的专家 ID，然后按需追加导出。

从本目录运行：

```bash
.build/release/ane-runner route-moe \
  --manifest results/moe-export/layer_0_fp16_linear_s32/manifest.json \
  --fixture fixtures/moe-real/converted/decode.json

.build/release/ane-runner probe-moe \
  --manifest results/moe-lut8-group4-bank/manifest.json \
  --fixture fixtures/moe-real/converted/decode.json \
  --expert-concurrency 4 --warmups 3 --runs 10 \
  --output results/moe-optimization-v2/decode-rerun.json
```

默认 `--precision bfloat16Boundaries` 复现 router 的 BF16 舍入边界；这批真实数据的路由完全一致，但不承诺任意输入与 GPU 的 fast-exp 和归约树逐位相同。专家输出为 FP16，加权求和用 Float32；共享 gate 与最终输出按所选精度舍入。`--precision float32` 可用于独立精度诊断。`--compute-units cpuOnly` 可用于设备对照；`--cache-experts N` 限制 routed expert 的 LRU 容量，shared 另占一个常驻模型。

离线导出和独立验证从仓库根目录使用现有 uv 环境：

```bash
.venv/bin/python experiments/ane-runner/scripts/prepare_moe_fixture.py

.venv/bin/python experiments/ane-runner/scripts/export_moe.py \
  --experts 88,90,109,148,214,249,261,315,333,351 \
  --modes fp16 --capacities 32 --layout linear

.venv/bin/python experiments/ane-runner/scripts/validate_moe.py \
  --fixture experiments/ane-runner/fixtures/moe-real/converted/decode.json \
  --report experiments/ane-runner/results/moe-real/decode-fp16-linear.json \
  --output experiments/ane-runner/results/moe-real/decode-validation-rerun.json

.venv/bin/python experiments/ane-runner/scripts/verify_moe_scheduler.py
```

`export_moe.py --all-experts` 只导出指定层的全部专家，会增加磁盘占用；默认按需导出。模型转换、JSON fixtures 和 Python 高精度参考都属于离线验证流程。这里的 Core ML `probe-moe` 不启动 Python，也不使用 MLX 算子执行专家；`probe-gpu-moe` 和 `generate-gpu` 则使用原生 MLX C / Metal。

重建压缩 bank 时，以已验证 FP16 bank 为输入，在一个新的输出目录执行以下命令。导出器按专家独占文件系统 claim，已有包会明确拒绝覆盖；并行 worker 必须使用不重叠 ID。汇总步骤按唯一 ID 复核全部当前文件哈希，再生成 runner 可用的 manifest。

```bash
.venv/bin/python experiments/ane-runner/scripts/compress_moe_bank_shard.py \
  --output-dir experiments/ane-runner/results/moe-lut8-rebuild \
  --experts 42,51,58,72,88,90,91,109,123,136,144,148,150,171,214,223,226,233,249,261,302,315,333,351,358,373,391,453,471 \
  --group-size 4 --worker rebuild
```

将原 FP16 bank 内的 `shared.mlpackage`、`router.f32.bin` 和 `shared_gate.f32.bin` 原样复制到这个新目录后运行：

```bash
.venv/bin/python experiments/ane-runner/scripts/assemble_moe_compressed_bank.py \
  --source-manifest experiments/ane-runner/results/moe-export/layer_0_fp16_linear_s32/manifest.json \
  --bank experiments/ane-runner/results/moe-lut8-rebuild \
  --experts-file experiments/ane-runner/fixtures/moe-real/converted/actual-expert-ids.txt
```

随后运行 Swift `probe-moe` 和 `validate_moe.py`，重新验证完整输出。导出成功或计算计划不替代精度和实际硬件执行检查。

独立 SSD 数值验证脚本可从仓库根目录执行：

```bash
.venv/bin/python experiments/ane-runner/scripts/verify_ssd.py \
  --model-dir experiments/qwen38-ssd/models/Qwen3.8-Flash-Next-MLX-SSD-Stream \
  --runner experiments/ane-runner/.build/release/ane-runner \
  --sample-dir experiments/qwen38-ssd/results/ngram-row-sample \
  --report experiments/ane-runner/results/ssd-verification.json
```

## Core ML 后端的后续工作

以下保留 Core ML 子路线的未完成项；当前项目优先级以 [runner 转型与带宽验证](BACKEND_PLAN.md) 为准。

1. 扩大真实输入与层覆盖，测量压缩包的实际编译缓存和常驻内存，继续减少逐专家调用开销；现有真实 fixture、CPU/ANE 对照和硬件追踪继续作为检查条件。
2. 完整 GatedDeltaNet 与连续状态；随后加入 QSA、KV 缓存和 Hyper Connection。
3. 串联全部 48 层与输出 head，验证连续生成的质量、内存、速度、能耗和实际 ANE 执行。
4. 在 macOS 27 上加入 CoreAI 适配，复用数据与参考测试，重新验证编译、数值与性能。

此前 GPU 服务的约 40 token/s 属于作者引擎历史对照；独立 runner 的实测结果见 GPU 使用文档。128 GiB 机器上进行整模型对照时应顺序运行，避免同时保留两套完整权重。

## 来源

SSD 格式、hash 与 BF16 舍入语义参考 [garnermccloud/mlx-serve 的固定提交](https://github.com/garnermccloud/mlx-serve/blob/7dbcba04c98e4fd3bcc533c63e645547f13cc3b1/src/qwen4_exp.zig)。移植文件保留来源及许可声明。模型、完整校验和前期探针记录见 [SSD 基线实验](../qwen38-ssd/README.md)。
