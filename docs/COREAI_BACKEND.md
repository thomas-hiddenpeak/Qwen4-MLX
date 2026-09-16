# Core AI 后端：首批真实权重子图

2026-09-17 起在 macOS 27 系统 Core AI 上开发。目前已从子图、[连续状态验证](COREAI_STATEFUL.md)、[完整48层混合生成](COREAI_HYBRID.md)推进至[完整 CoreAI 生成](COREAI_NATIVE.md)：独立 `coreai-runner` 无 MLX 链接，全部神经网络计算已迁移，原始 Q4 专家保留压缩驻留。当前容量256、逐 token prefill；质量、性能、HTTP 服务及缓存迁移仍待推进。以下保留首批 MoE 子图结果及当时的路线。

## 本轮实测

本机 macOS 27.0（26A428）、CLT Swift 6.4 / SDK 27.0；系统框架报告设备架构 `h17c`，可用 CPU、GPU、Neural Engine。Python `coreai-core 1.0.0b2` 也已选择系统 runtime。独立 package 保持 macOS 26.2 最低版本，CoreAI 代码由 SDK 条件编译及 macOS 27 availability 隔离。

使用本模型 layer 0 的真实共享专家和真实路由专家 333，分别导出 S1/S32、FP16 SwiGLU：`down(silu(gate(x)) * up(x))`。每个资产约 9.83 MB；只读取所需权重切片。路由专家原始 affine Q4/group64 被展开为 FP16，因此本实验不证明压缩权重驻留能力。

每个资产验证实际捕获输入、确定性随机输入、全零输入；共 36 组系统 Swift CoreAI 调用。S32 的 actual 是两个真实 token 加 30 个零，normal 则使用 32 个不同生成输入。参考为 CPU FP32 运算、相同 FP16 权重，初始门槛为 relative L2 ≤ 0.005、max absolute ≤ 0.02；零输入要求输出绝对值 ≤ 1e-6，并检查形状、FP16 类型及有限数值。

| 配置 | 数值通过 | 非零输入 relative L2 范围 | 结论 |
| --- | --- | --- | --- |
| GPU 偏好 | 12/12 | 0.000253–0.000461 | 这两类子图通过初始数值门槛 |
| CPU only | 5/12 | 0.00355–0.01136 | 7 组超标，尚未接受 |
| Neural Engine 偏好 | 4/12 | 0.00606–0.05730 | 仅零输入通过，尚未接受 |

全部调用均完成，但“成功返回”和“数值通过”分别统计。另用 Python 系统 CoreAI 对共享专家 S1 的 actual/normal、三种配置交叉检查，6 组均与 Swift 输出逐位相等；这些差异不能归因于 Swift 张量搬运。正式 `probe-coreai` 又复跑上述 12 组 GPU 偏好检查，全部通过，与独立探针输出逐位相等。

CPU 顺序 FP16 累加模拟能较好重现误差。只将三个投影提升到 FP32、每个投影后立刻转回 FP16，保留原权重值、SiLU/乘法及 FP16 输入输出后，共享专家 S1 的 CPUOnly actual/normal relative L2 分别降至 **0.000378 / 0.000587**，连同零输入全部通过原门槛。GPU 偏好亦通过这三项；Neural Engine 偏好仍有 **0.005697 / 0.049350** 误差。这支持 CPU 投影累加精度的解释，但不证明内部内核实现，也未解决 ANE 偏好问题。候选通过 `--fp32-projections` 保留，默认不启用；尚未验收其他专家与容量。

上述配置中的 GPU/Neural Engine 是调度偏好，允许回退；此次未采集硬件执行时间线，不据此声称 ANE 驻留。参考服务保留运行，本轮没有进行独占条件下的性能对比，也不报告整模型加速。

本轮之后按用户要求停止 mlx-serve，后续开发默认不常驻参考服务，仅需要对照时启动；配置和启停说明见[按需参考服务](MLX_SERVE_SERVICE.md)。

本机原始资产、CPU 参考、模型哈希、36 组输出和跨语言结果在 `results/coreai27-readiness/`，不随 Git 克隆提供。源码入口和离线导出脚本随仓库提交。

## 构建与复跑

使用 macOS 27 SDK 构建，不需要链接父目录 Apple 示例 package。当前 CLT 能直接构建；无需修改全局 `xcode-select`。

```sh
xcrun swift build -c release
```

离线导出使用工作区 Python 环境，需要 `torch`、`numpy`、`coreai-core`、`coreai-torch` 以及复用权重读取模块所需的 `coremltools`。脚本不调用设备 runtime。源权重及捕获 fixture 沿用 `export_moe.py` 的本机安装要求，首次克隆需先准备这些文件。

```sh
../../.venv/bin/python scripts/export_coreai_moe.py \
  --kind shared --layer 0 --output results/coreai-export

../../.venv/bin/python scripts/export_coreai_moe.py \
  --kind routed --expert 333 --layer 0 --output results/coreai-export

.build/release/ane-runner probe-coreai \
  --model results/coreai-export/layer0-shared/s1/layer0-shared-s1-fp16.aimodel \
  --fixture results/coreai-export/layer0-shared/s1/actual.json \
  --compute-units gpu --warmups 3 --runs 5 \
  --output results/coreai-shared-s1.json
```

`--compute-units` 支持 `default`、`cpuOnly`、`gpu`、`neuralEngine`；`--function` 默认 `main`。目前支持 float16/float32/float64/int32 的命名张量 JSON，复用已有 `CoreMLBlockFixture` 格式，名称不表示实际调用 Core ML。函数如有持久状态会明确拒绝；后续状态实现需另行验收。

导出时追加 `--fp32-projections` 可复现 CPU 精度候选，使用新的 `--output` 目录；资产名追加 `-gemm32`。权重常量仍为 FP16，投影使用 FP32 计算并恢复 FP16 输出，不同设备仍可能选择不同实现。

报告分开记录模型加载、输入准备、暖机后的函数调用、输出读取及误差。命令成功退出只说明调用完成，数值是否合格需根据 `comparisons` 与实验门槛判断。JSON 用于离线验证，不作为未来每个 token 的内部传输方式。当前不支持 interleaved 张量布局，会明确拒绝，避免将其误读为普通 strides。

本轮 release 编译、真实子图数值对照、导出器 CPU 小测试和 CLI 的无效迭代次数/形状溢出/数值范围拒绝检查通过。标准 XCTest 未执行：当前 CLT 缺少 XCTest，已安装的 Xcode 27 尚未完成许可初始化；未修改系统工具链或许可状态。macOS 26 兼容性仅验证条件编译和弱链接，未在旧系统实跑。

## 向完整模型推进

1. **GDN / QSA 状态语义。** 首轮真实权重子层的短 prefill/decode、稀疏边界和状态恢复已通过，范围见[连续状态结果](COREAI_STATEFUL.md)。下一步串接完整 decoder 层并检查 PLE、残差与源 BF16 的累计误差；再按 profile 选择设备。
2. **保留量化驻留的 MoE。** 使用 CoreAI 的 GatherMM/量化路线或 Metal 扩展，验证本模型 Q4 分组、scale/bias、top-10 路由及共享门控。整套专家展开 FP16 会超过目标机器的可用容量，不能把本轮单专家导出方式直接推广至全模型。
3. **整层、整模型与现有服务对接。** 保留 SSD n-gram 按行读取、prefill/decode 分离及 runner 的调度；CoreAI 只接收本轮需要的 PLE 行，不把 51.2 GB 表变为模型常量。后端状态需要明确导入/导出及身份版本，不能直接复用现有 MLX 缓存文件。
4. **长上下文和性能验收。** 再恢复前缀缓存、SSD 状态归档、取消、内存压力和 262144 上下文检查。独立统计 prefill/decode，用硬件时间线判断 GPU/ANE/CPU 分工；MTP 性能仍放在后段。

Apple 当前导出工具可将 PyTorch 图转换为 CoreAI，并有 GPU MoE/GatherMM、权重压缩和自定义 lowering 支持；这不等于现有 MLX 模型目录能够原样加载。其通用 SequentialEngine 的输入/状态约束属于示例引擎，并非 CoreAI `InferenceFunction` 的统一硬限制，本项目继续使用自有 Swift runner。

- [CoreAI 转换工具](https://apple.github.io/coreai-torch/)
- [官方模型与压缩选项](https://github.com/apple/coreai-models/blob/main/models/README.md)
- [GPU 模型编写参考](https://github.com/apple/coreai-models/blob/main/skills/skills/model-authoring/references/gpu_rules.md)
- [系统设备偏好](https://developer.apple.com/documentation/coreai/specializationoptions/init(preferredcomputeunitkind:))
- [CoreAI 硬件 profiling](https://developer.apple.com/documentation/coreai/analyzing-model-runtime-performance-with-instruments)

旧 Core ML/ANE 实验继续保留在[历史记录](COREML_ANE_HISTORY.md)，不混作本轮 CoreAI 的硬件证据。
