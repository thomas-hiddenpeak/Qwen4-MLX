# CoreAI：GDN / QSA 连续状态验证

2026-09-17，独立 Swift runner 已在系统 CoreAI 上执行真实权重的完整 GDN 子层和 Attention/QSA 子层，跨 prefill/decode 保持状态，并验证重置、检查点恢复后继续执行。使用 macOS 27.0 / SDK 27、GPU 偏好；没有设备执行 trace，不声明全部运算的硬件驻留。

这一步补齐的是**子层之间连续调用所需的状态能力**。本次子层验证尚未包含完整 decoder 层或48层生成；随后新增的[完整混合生成](COREAI_HYBRID.md)已把所有 GDN/QSA 接入真实文本链路，Q4 MoE 等部分仍用MLX。纯 CoreAI 与服务迁移尚未完成；现有 MLX 服务路径和缓存格式保持独立。原作者 mlx-serve 按需关闭，本次子层验证没有加载完整参考模型。

## 实测范围

| 场景 | 输入与边界 | Swift 执行次数 | 输出/状态比较 | 最大 relative L2 |
| --- | --- | ---: | ---: | ---: |
| GDN layer 0 | S4 prefill + 3 次 S1 decode | 14 | 42/42 通过 | 0.0007081 |
| Attention/QSA layer 3，短序列 | S4 prefill + 3 次 S1 decode，cache32 | 14 | 112/112 通过 | 0.0005281 |
| QSA 稀疏边界 | 已有 2051-token K/V/raw 状态继续到 2056，cache2056 | 18 | 144/144 通过 | 0.0005052 |

每个场景包含一次正常序列、一次 reset 后完整重放，以及同一检查点恢复两次后的尾部续算。**共 46 次 Swift 调用、298 项比较全部通过；重放的逻辑输出和状态摘要逐位一致。** 固定初始门槛为 max absolute ≤ 0.02 且 relative L2 ≤ 0.005；全零参考要求绝对误差 ≤ 1e-6。整数 mask、offset、pooled count 必须精确相等，不适用浮点容差。

另以 Python 系统 CoreAI 先行执行上述三组序列的单轮调用，13 步全部通过；正式验收结果来自 Swift `probe-coreai-sequence`，不是只运行 Python 导出器。

QSA 的 budget 为 2048，池化比例为 4，首次稀疏选择出现在 2052 token。边界组跨过新池化块，五步可见 token 数依次为 2048、2049、2050、2051、2048；CoreAI mask 与 CPU 参考精确匹配。该组从已保存的 K/V/raw 状态开始，首次调用重建池化状态，没有在 CoreAI 中从零预填 2051 tokens，不能称为完整长上下文验收。

## 状态如何传递

[`CoreAIStateSession`](../Sources/ANERunnerCore/CoreAIStateSession.swift) 使用函数的显式 tensor 输入/输出：

- GDN：`conv_history [1,3,10240]` 为 FP16，`recurrent_state [1,48,128,128]` 为 FP32，后两维是 value/key。
- QSA：K/V、raw indexer、pooled indexer 缓存，以及 Int32 offset/pooled count。
- 常规步骤直接保留 CoreAI 输出的 `NDArray`，传给下一个函数；每步 fixture 只提供本次激活，不允许覆盖保留的状态。
- checkpoint、restore、reset 使用按 dtype 和 strides 复制的独立存储；快照绑定所属 session，不能跨 session 混用。
- 所有输出通过名称、形状、类型及有限值检查后，一次提交新状态和 step count；执行期间拒绝再次 step、reset 或 checkpoint。参考误差门槛由诊断命令另行检查。

这是显式状态的函数调用，尚未接 CoreAI 原生 mutable-state 接口。诊断模式每步将输出及全部状态读取到 host 做对照；这些 host 值不会回灌下一步。函数调用时间与读取时间分开记录，但当前固定小 cache、诊断读回和全量池化都尚未优化，**不报告整模型性能收益**。

## 精度与来源边界

权重是当前模型的真实 BF16 dense 权重，导出时转换为 FP16；GDN 与 QSA 均保留本模型的 sigmoid 输出 gate、norm、head 分组、RoPE 等语义。投影及敏感中间量使用 FP32，GDN 持久递归状态也使用 FP32。

输入来自已捕获的 layer0 MoE 激活，在相应子层边界重放，**不是原请求的原生 GDN/Attention 层输入捕获**。源码记录了来源、权重切片和 fixture 哈希。

存在两种不同对照，不能混淆：

1. 上表是 **CoreAI 对本次导出方程的 CPU 参考**，检查图转换、执行和连续状态正确性。
2. 对现有独立 MLX BF16 oracle 的比较另存诊断。GDN 的 26+1 replay 输出 relative L2 约 0.00598 / 0.00521，递归状态约 0.00575 / 0.00603；QSA 稀疏前两步输出约 0.004816 / 0.004884，选择 mask 相等。这些误差包含 FP16/BF16 边界及状态精度策略变化，不能把上表通过解释为源模型全精度等价或整模型质量已验收。

后续整层/整模型对照必须继续检查这项累计误差，而不是放宽本轮 runtime 门槛。

## 复跑

需已有源权重验证记录和 `fixtures/gpu-sequence-reference` / `fixtures/moe-real` 本机捕获。大型资产、状态张量与 `results/` 不随 Git 克隆提供；Python 环境沿用 [CoreAI 后端](COREAI_BACKEND.md)依赖。导出器仅进行 CPU authoring 与参考计算，不启动模型服务。

```sh
../../.venv/bin/python scripts/export_coreai_gdn.py --output results/my-coreai-gdn
../../.venv/bin/python scripts/export_coreai_qsa.py --output results/my-coreai-qsa
xcrun swift build -c release

.build/release/ane-runner probe-coreai-sequence \
  --sequence results/my-coreai-gdn/sequence.json --compute-units gpu \
  --replays 2 --checkpoint-step 1 --output results/my-gdn-report.json

.build/release/ane-runner probe-coreai-sequence \
  --sequence results/my-coreai-qsa/sparse-sequence.json --compute-units gpu \
  --replays 2 --checkpoint-step 1 --output results/my-qsa-sparse-report.json
```

`--checkpoint-step` 是保存前已完成的步数，0 关闭检查点重放。每个 sequence 明确模型路径、初始状态、state input→output 绑定、prefill/decode 阶段、非状态输入和完整参考输出。命令检查失败会保存报告并非零退出。当前 SHA256 针对逻辑形状、dtype 和 Double 表示的值，不是物理 GPU buffer 或 DRAM 读写计数。

本机证据：`results/coreai-stateful/swift-runtime/`、`gdn-final/`、`qsa/v1/`。导出器的独立 CPU 方程、分块状态、head 布局、mask/tail 选择与追加隔离测试，以及此前 MoE 导出测试合计 11 项；Swift release 编译与旧 `probe-coreai` 共享专家入口一并回归。标准 XCTest 仍受当前 CLT 缺少 XCTest 的环境限制。

独立 Swift 调用另通过 7 项状态检查：非连续布局复制后独立存储、外部 session 快照拒绝、状态输入覆盖拒绝、错误 dtype 拒绝、错误后正常执行、reset 全输出一致、预先取消后状态不推进且操作锁释放。CLI 的两个负例也按预期失败：夹具注入参考状态在加载模型前被拒；只篡改一个整数 mask 值时，数值验收返回失败。记录在 `session-guards/` 与 `negative-cases/`。未在正在执行的 GPU 调用中注入故障或取消，也未完成服务长稳验证。

后续的完整连接进展见[混合生成](COREAI_HYBRID.md)。CoreAI 内原始 Q4 专家压缩驻留、prefill/decode 调度和前缀/SSD 缓存仍需逐步迁移；不直接混读 MLX 缓存归档。
