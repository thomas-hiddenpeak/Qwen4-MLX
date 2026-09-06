# GDN verification capture 与 recurrence replay 可行性

2026-09-07，只读核对当前源码、模型配置与既有原始报告；本次核查没有运行 GPU。

**值得安排一个单层快筛，尚不足以优先改造生产 MTP。** 当前 S3 验证确实额外写出 162 MiB 的 recurrent snapshots；保留 recurrence 输入约需 2.13 MiB。可是捕获写入的耗时没有被单独测出，既有记录中的整个前缀提交仅占 decode 墙钟约 1.65%–2.06%。Replay 还要重新做接受前缀的 recurrence，不能按节省字节直接承诺整体加速。

最小快筛可直接复用现有两个 kernel 入口：`applyCapturing`，以及 `apply(roundStateEachToken: true)`。后者验证后再对接受前缀调用同一个 recurrence，不必先新增 Metal kernel、修改生产状态合同或重跑任何大矩阵。

## 1. 当前实际执行路径

模型 `config.json.text_config` 为 48 层，其中 36 层 `linear_attention`，位置为 `i % 4 != 3`。`QwenModel` 对这些层统一创建 fused recurrence + fused prework。当前捕获模式包括 `batchedCaptured`、`batchedScalarMoE`、`batchedScalarLinear`、`batchedTokenMoE`；它们均使用逐 token BF16 回存边界。普通 AR 与 `scalar` verification 不产生捕获 bank。

相关实现：

- [GDN 状态与 forward](../Sources/ANERunnerGPU/GPUGatedDeltaNet.swift)：`VerificationCapture`、`State.tensors`、`committingPrefix`、`captureInputs`。
- [fused recurrence](../Sources/ANERunnerGPU/GPUGatedDeltaNetFused.swift)：`applyCapturing` 声明三个输出，`captureSource` 在每个时间步写 `state_seq`，循环结束仍写独立 `state_out`。
- [MTP 验证与事务提交](../Sources/ANERunnerGPU/QwenMTPDecoder.swift)：先 checkpoint、完整验证和 evaluate，再根据 `MTPCommitPlan.trunkConsumed` 提交；整轮成功后才发布给 caller。
- [模型级 prefix commit](../Sources/ANERunnerGPU/QwenModel.swift)：统一提交 GDN、PLE、Attention/QSA，最后 evaluate 全部返回状态。

每层的具体对象如下。这里的字节是 tensor payload，全部使用二进制 MiB；不是 allocator resident memory 或 DRAM 计数。

| 对象 | shape / dtype | 每层 payload | 所有权与写入 |
| --- | --- | ---: | --- |
| 初始 recurrent checkpoint | `[1,48,128,128]` BF16 | 1,572,864 B = 1.5 MiB | checkpoint 保留不可变 MLX handle，不额外复制整份状态 |
| 最终 recurrent | `[1,48,128,128]` BF16 | 1.5 MiB | recurrence 正常输出；capture 路径也独立分配并写一次 |
| 逐位置 recurrent bank | `[S,1,48,128,128]` BF16 | `S × 1.5 MiB` | capture 的额外输出，不是前两项的 view；每一步都写全状态，包括最后一步 |
| convInputs | `[1,S+3,10240]` BF16 | `(S+3) × 20,480 B` | `concat(history,qkv)`；热状态 fused prework 原本不需要该 concat，所以在此路径是捕获额外工作 |
| 最终 conv history | `[1,3,10240]` BF16 | 61,440 B | fused prework 独立输出；冷状态 composed prework 的 `MX.copy(slice(...))` 可能仍共享 concat backing |

已读固定 MLX `backend/metal/custom_kernel.cpp::CustomKernel::eval_gpu`：每个输出分别 `allocator::malloc(out.nbytes())`，不存在把 `state_seq` 自动别名到 `state_out` 的逻辑。`QwenModel.evaluate` 求值 `outputs + state.tensors`，其中包括 capture bank 与 convInputs；这些不是未求值的纯 Swift 元数据。capture 的写入就在原 recurrence kernel 内，**没有单独的 snapshot copy kernel**。

原状态更新先在 FP32 register 中完成，再计算当前 `y`，随后把状态转成 BF16 snapshot；rounded 模式把该 BF16 值转回 FP32，供下一 token 使用。循环结束的最终 BF16 store 仍保留。因此可避免的内容包括最后位置在 bank 中的重复存储。

## 2. 接受长度与提交代价

验证输入是 `[pending] + drafts`。接受 `a` 个 draft 时，真实提交长度是 **`count = a + 1`**；接受 0 个 draft 仍须消费 pending token，不能恢复成零步状态。depth2 的常规验证为 S3，分别提交 count1/2/3。

- 部分接受：slice bank 的 `count-1` 行并 reshape，然后 `GPUVerificationCopy.tensor` 用 batch-axis gather 分配独立 recurrent；convInputs 的 `[count,count+3)` 行也 gather 成独立 conv history。每层 recurrent copy 的逻辑读/写各 1.5 MiB，36 层各 54 MiB；卷积 copy 的读/写各 2.109375 MiB。后续统一 evaluate 才完成这些拷贝。
- 全部接受：直接复用最终 recurrent/conv，清除 capture 引用，无 recurrent 重算，也没有上述 GDN prefix copy。
- 原 checkpoint 与完整 verified 值仍可被调用方保留；返回 committed state 不修改它们。取消、失败和 EOS/预算处理不能改变这条发布规则。

Replay 在部分接受时可用一次 recurrence kernel **替换原 recurrent gather**，不必天然增加 kernel 数。它依然读一次初始状态、写一次最终状态，但比 gather 多了 recurrence 算术和输入读取。最小复用版本还计算不再需要的 `y`；其开销会诚实纳入快筛。全接受不回放。

## 3. 字节机会有多大

保留 `k/v/decay/beta` 足以重建 state，单 token 每层为 `4096 + 12288 + 96 + 96 = 16,576 B`。为直接复用现有 `apply`，快筛再保留 q，合计 **20,672 B/token/layer**。当前热 prework 已物化这些独立 BF16 输出，保留 handle 会延长寿命，通常不需要再写一份输入副本；冷 composed 路径的 view/backing 和连续化复制需单独观察。

| 验证长度 | 36 层 recurrent bank | 保留 q/k/v/g/beta | bank 减去输入 payload |
| --- | ---: | ---: | ---: |
| S2 | 108 MiB | 1.419434 MiB | 106.580566 MiB |
| S3 | 162 MiB | 2.129150 MiB | 159.870850 MiB |
| S4 | 216 MiB | 2.838867 MiB | 213.161133 MiB |
| S5 | 270 MiB | 3.548584 MiB | 266.451416 MiB |

初始 checkpoint、最终 state、convInputs、输出 y、正常 prework 活跃输入没有从共同成本中扣掉。以上是可去掉的 bank 与需延长寿命的输入 payload 比较，**不是峰值内存实测差值**。例如 S3 的 convInputs 另占 4.21875 MiB，replay 若沿用当前卷积提交方案仍需它。

若完整 replay 计算抵消收益，还有更小的后备候选：只捕获前 `S-1` 个位置，全部接受继续取 `state_out`。S3 可少写 54 MiB/round，无需任何接受后 recurrence。该方案仍要处理 S1 的零快照边界与 capture shape 合同；本次不实现，也不把其收益视为已建立。

## 4. 既有真实测量能说明什么

以下重新读取原始 JSON，均剔除计划中的前两轮 warmup，按同策略两轮 decode 墙钟和阶段秒数聚合。两时间窗口分别汇总，不能将绝对速度相互比较。

| 既有报告 / 策略 | 平均完整 decode | 平均整个 prefix commit | commit 占 decode | verify 占 decode |
| --- | ---: | ---: | ---: | ---: |
| [11k/128](../results/mtp-token-axis-v1/long128.json)，ScalarLinear | 3.392038 s | 55.991 ms | 1.6507% | 87.9737% |
| 同窗口，TokenMoE | 3.273229 s | 59.100 ms | 1.8055% | 87.0468% |
| [11k/256 复测](../results/mtp-token-axis-repeat-v1/long256.json)，ScalarLinear | 8.147857 s | 161.902 ms | 1.9871% | 87.9841% |
| 同复测窗口，TokenMoE | 8.124838 s | 167.389 ms | 2.0602% | 87.3219% |

这里的 commit 来自 `rollbackSeconds`，在 capture 模式并不表示重跑主干。它包含 **所有层的 GDN/PLE/KV/QSA 截取、构图及 evaluate**，不是 GDN recurrent copy 的独占时间。即便把该完整区间消掉，节省也只有表中的约 1.65%–2.06% decode 时间；replay 实际不能消掉所有这些工作。

capture 写入处于 `verifySeconds` 内。既有报告的 profiler 为 disabled，没有 recurrence/capture 单独 GPU 时间，故 **capture 对整轮的占比仍未知**。约 88% 是完整 target verify，包含权重投影、MoE、Attention、HC、输出 head 等，不能拿来当 replay 的优化上限。

`draft + verify + rollback + history` 与 decode 墙钟之间还差约 2.02%–2.19%；这些差额没有分摊给 capture。相应计时是 host 墙钟，不是可加总的独立 shader 时间。11k/256 复测还使用了系统采样，不与未采样窗口混合。

本次只读核查时对照了记录中的 source manifest：GDN、fused recurrence、prework、MTPCommitPlan、QwenMTPDecoder 哈希相同；Attention 文件已有后续改动。随后为 CLI 隔离 probe 只给 `VerificationCapture` 补了公开初始化器，未改变计算、捕获或提交逻辑。因此表格是可追溯的历史阶段基线，不能替代当前完整 commit 的复测。

11k/128 每轮有 55 次 S3 验证，接受直方图为 `[12,14,29,0,0]`。Replay 只在 26 次部分接受中执行，合计回放 `12×1 + 14×2 = 40` 个 recurrence 位置；全部 29 次接受直接复用 final state。这个分布可用于单层快筛的加权比较，不能把所有 case 简单平均。11k/256 记录有一次少于 depth2 的草稿宽度，只有聚合直方图不足以还原该轮宽度与接受分支，不臆造完整逐轮轨迹。

## 5. 最小候选与通过门槛

第一步只扩展已有 `probe-gpu-sequence` 的隔离分支；生产 verification 枚举、MTP 默认和事务代码不动。复用 `fixtures/gpu-sequence-reference` 的 recurrence 张量及 manifest SHA 校验。该 fixture 的输入来自已捕获激活在其他模块边界重放，**不是当前完整生成轨迹中的原生 recurrence 抓取**；先用它验证基本算术和筛掉显然更慢的方案。

隔离分支已经编写，当前未构建、未执行。只需现有小型 sequence fixture，无需加载模型权重；构建更新后的 CLI 后可运行：

```bash
.build/release/ane-runner probe-gpu-sequence \
  --fixture fixtures/gpu-sequence-reference/manifest.json \
  --replay-only true --replay-repeats 16 \
  --output results/gdn-replay-v1/probe.json
```

它覆盖 cold / warm 两个已有 fixture checkpoint、S1…5 的全部接受长度，先验证数值再计时。warm checkpoint 取 fixture 的最终状态并重放起始输入；conv bank 由已有 convHistory 值循环组成以验证裁剪，不冒充新的原生轨迹。每个 case 预热 4 次后交错测量；完整 y/final state、提交 recurrent/conv 和续跑 S1 在计时前及最后一轮后均逐位核对。report 中的内存快照含两种策略和共享 fixture，不代表单策略峰值。

对 S1…5、每个 `count ∈ 1…S`：

1. 原 `applyCapturing(roundStateEachToken: true)` 产生完整 y/final state/bank，以现有 `committingPrefix` 得到 oracle。
2. 候选用原 `apply(roundStateEachToken: true)` 完整验证；部分接受从相同初始 checkpoint 回放前 count 行，全部接受复用最终 state。卷积提交继续遵守相同 `[count,count+3)` 规则。
3. 比较完整验证 y/final state、全部接受长度的 recurrent/conv/offset，以及提交后再执行 S1 的 y/state，要求所有 BF16 值有限且逐位相同。零 draft 接受映射 count1；count0、count>S 必须拒绝。
4. 冷初始状态与非零已有状态分别检查；构造/求值每次使用新图。预热后交错测 capture verify/commit 与无 bank verify/replay，CPU 读回和正确性比较置于计时之外。报告各子阶段与完整单层合计，不仅报告去掉 bank 的一半时间。
5. 明确输出 capture/input/共同状态的逻辑 payload；用已有 MLX memory 接口辅助观察，不能用所有 view 的 `nbytes` 相加冒充真实占用。

快筛若数值失败立即停止。若按照真实 S3 接受分布加权后，单层 verify+commit 合计不快，或优势落在重复抖动内，继续保留现有 capture；不因为省了约 160 MiB 就进入完整 MTP 改造。出现清楚、可复现的收益，才做生产候选并重新跑短生成、状态/取消/EOS/预算，以及 11k 输出完整一致性和阶段性能。

生产候选将涉及的最少接口：

- `GPUGatedDeltaNet.VerificationCapture`：用显式类型区分 snapshots 与 replay inputs；记录初始 offset、长度、rounding 策略及输入 handle，避免含糊的多个可选字段。
- `GPUGatedDeltaNetFused`：第一版可复用现有 `apply`；只有快筛通过且多余 y 成本值得去掉时，才新增不计算 y 的 state-only recurrence。
- `GPUGatedDeltaNet` 的 layer-owned prefix commit：接收 verified state、checkpoint 和 count，调用该层 kernel；纯 `State` 不持有可变 kernel 实例。
- `QwenModel.commitVerificationPrefix`：调用对应 GDN layer 的提交方法，保留所有权/session/offset 校验及统一 evaluate。
- `QwenMTPDecoder.Statistics` 与明确实验开关：分开记录 capture/commit/replay 的可观测阶段；不得把 recurrence replay 记成旧的整主干 `replayedTokens`。

未来前缀缓存还要测试“先 MTP，后分叉命中”，并保留 GDN/PLE/QSA/MTP head 的同一边界；本单层实验不能证明这项完整服务能力。上游设计背景与适用边界见 [vLLM/SGLang 调研](research/VLLM_SGLANG.md#3-replayssm值得探测但先核对节省占比)。

## 首轮单层结果

2026-09-07 Release 构建通过。`results/gdn-replay-v1/probe.json` 完成 30 个 cold/warm、S1…5 的接受长度 case，720 项有限值/逐位输出与状态比较全部通过，20 项非法 count 检查全部拒绝。包含接受后继续一个 S1 的输出和 state；全部接受复用 final Tensor 对象。生产 MTP 的 capture/commit 路径没有改变，仅为隔离探针新增公开的 capture 初始化器。

每个 case 每种模式预热 4 次、交错测量 16 次。按历史 S3 接受分布 `[12,14,29]` 加权各 case 中位数，得到以下筛选近似：

| 初始状态 | Capture verify+commit | Replay verify+commit | 相对改善 |
| --- | ---: | ---: | ---: |
| fixture cold | 275.087 µs | 269.496 µs | 2.07% |
| fixture warm replay | 241.229 µs | 233.026 µs | 3.52% |

这不是一次按该分布执行的完整模型时序，更不是整模型 MTP token/s；输入和卷积 fixture 的人工重放边界仍如上所述。S3 提交两行的 case 有小幅回退，收益随接受长度变化。当前结果说明数值方案可行、临时 bank 有节省机会，但不足以证明值得替换生产 capture；本轮保持为独立探针，不更改默认或据此提升 MTP 发布状态。

构建、计划、简表和参考服务恢复见同目录 `build.log`、`plan.json`、`summary.json`、`run-ledger.json`。该轮参考服务恢复为 PID3782、MTP/drafter关闭，之后已进入下一项驻留诊断；最新进程身份始终以自主接续记录及最新 ledger 为准。
