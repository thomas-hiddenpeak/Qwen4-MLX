# 请求级 MTP 成本摘要

`generate-gpu` 的每个 trial 新增 `mtp_cost_summary`；MTP 关闭时为 `null`。同步/PD/增量库 API 的 `QwenGenerationResult` 同时新增可选 `mtpCostSummary`，旧 JSON 缺字段仍可解码。既有 `mtp_statistics`、`statistics.mtp`、阶段时间和生成逻辑均保留原含义。

摘要由 `QwenMTPCostSummary` 从已完成请求的 CPU 计数与时间派生，在既有 decode 轮计时结束后构造；不访问 tensor、不增加 GPU 求值、同步、计时点或内核调用。它不会选择 MTP 深度或改变默认配置。生产级启用仍受 [既定 MTP 发布条件](MTP_RELEASE_CRITERIA.md) 约束。

## 口径

| 字段 | 含义 |
| --- | --- |
| `committedDecodeTokens` | 首 token 之后实际发布的 token 数，包含实际输出的 EOS |
| `decodeSteps` | 实际 decode 步骤数，包含最终只计算一个目标 token 的收尾 |
| `speculativeRounds` | 真正起草的轮数，即原 `Statistics.rounds`；收尾的无草稿步骤不计入 |
| `targetOnlyDecodeSteps` | 两者之差；当前实现最多一个终止收尾步骤 |
| `draftAcceptanceRate` | 接受草稿数 / 实际提出的草稿数；未起草为 `null` |
| `meanProposedDraftsPerSpeculativeRound` | 实际草稿数 / 起草轮数，反映 EOS 与剩余预算导致的缩短 |
| `meanAcceptedDraftsPerSpeculativeRound` | 接受草稿数 / 起草轮数；不加假定的 bonus |
| `meanCommittedTokensPerDecodeStep` | 实际发布 decode token / 全部 decode 步骤，采用 EOS 裁剪后的输出 |
| `unacceptedDraftTokens` | 已提出但未接受的草稿数；也可能包含首次拒绝后的未采用后缀 |
| `zeroAcceptanceRoundFraction` | 接受数为 0 的起草轮 / 起草轮数；不计 target-only 收尾 |
| `targetEvaluationsPerCommittedToken` | `(verifiedTokens + replayedTokens) / committedDecodeTokens`，是目标输入行数比例，不是 kernel 次数或加速比 |
| `effectiveDecodeTokensPerSecond` | 实际发布 decode token / 原有完整 decode 时间窗口 |
| `decodeSecondsPerCommittedToken` | 同一时间窗口 / 实际发布 decode token，即平均计算 TPOT |

EOS 可以使一轮的实际输出少于 `accepted + 1`。例如两个草稿都接受，第二个是 EOS，则实际只输出两个 token，不再发布 bonus。摘要使用实际输出数，不能以接受率或“每轮加一”代替它。

预算只剩一个 token 时，已有 decoder 直接执行目标路径，不增加 `rounds`、草稿数或接受直方图。因此一个请求可以有 `speculativeRounds = 0`、`decodeSteps = 1`，仍然有正常 decode 吞吐；首 token 即 EOS 或输出预算为 1 时，二者均为 0，吞吐和 TPOT 为 `null`。

## 成本与未知值

`draftSeconds`、`verificationSeconds`、`commitOrReplaySeconds`、`historySeconds` 对应原有 draft、verify、rollback、history 四个顺序区间；`componentSeconds` 为其和。`rollbackSeconds` 在 capture 模式计 prefix commit，在旧重算模式计恢复及重算，所以使用 `commitOrReplaySeconds`，不把两种行为混称为整主干回放。

`decodeSeconds` 使用现有请求/CLI 的 decode 步骤计时，包含这些组成部分和轮内其他工作；排除 prefill、MTP prompt-history、加载、阶段间排队和 callback。它不是纯 GPU 时间，平均计算 TPOT 也不是用户可见逐次 callback 的 p95 间隔。SSD 等待已在相关调用区间内，不能再次加到组件总和。

`decodeSecondsOutsideComponents` 仅为外层 decode 时间减四个组件的记账余量，不是独立测得的 CPU 开销或 GPU 空闲。组件之和大于外层时，`componentsFitDecodeWindow` 为 `false`，余量为 `null`，不会将负差夹成 0。数据非法/非有限时，对应时间和派生值为 `null`。零时间、零输出与无法表示的比值均不会生成无限吞吐或零 TPOT。

`countersConsistent` 核对请求步骤/输出与 decoder 原始计数；`acceptanceHistogramConsistent` 单独核对五个桶的轮数和接受数。发现不一致时相关派生值为 `null`，原始计数仍保留在原字段中供检查。现有数据没有每轮实际输出分布，因此不生成一个假定的 emitted-token histogram。

## 使用与验证

不需要新增开关。现有 `generate-gpu` 命令完成后，可直接查看：

```sh
jq '.trials[] | {repetition, mtp_depth, mtp_cost_summary}' results/your-run.json
```

CPU 检查（不创建模型或 MLX tensor）：

```sh
swift test --filter QwenMTPCostSummaryTests
```

测试覆盖实际输出分母、首 token 终止、接受草稿 EOS、预算缩短和 target-only 收尾、非法时间、计数/直方图和旧结果 JSON 兼容。2026-09-07 Release 构建及相关 46 项 CPU 检查通过，其中本摘要 8 项通过。首次 CLI 报告字典触发编译器类型推断超时，拆成显式局部值后通过；字段含义不变。日志为 [cpu-tests-retry.log](../results/upstream-cost-latency-v1/cpu-tests-retry.log)。

同轮 `long-cost.json` 九个 11,057-token 请求，每个 128 输出，全部与旧 golden 完全一致。深度顺序为 `0/1/2/0/1/2/2/1/0`；前三轮预热，后六轮各深度两次。全部 MTP 成本字段与原始计数、时间、直方图重算一致。另有预算 1/2/16 的 AR-MTP 对照，实际各输出 1/2/8 token，最后一组自然 EOS；预算 1 无 decode 吞吐，预算 2 是零起草轮、一个 target-only 步骤，符合合同。

| 11k 测量窗口 | 两次 decode token/s | 草稿接受率 | Verify 占完整 decode |
| --- | --- | ---: | ---: |
| AR | 29.13 / 20.61 | — | — |
| depth1 | 33.73 / 23.21 | 72.60% | 91.08% |
| depth2 | 34.84 / 31.80 | 65.45% | 88.05% |

后半程存在明显漂移：AR 自身降速约 29%，prefill 也从约 16 秒延长至 48 秒；allocator active 约 81.0 GB、peak 约 82.0 GB 基本稳定，不能据此归因为内存泄漏、其他训练或热降频。现场快照未报告热告警，但不足以排除其他竞争。两次 AR/D1/D2 聚合吞吐为 24.14/27.50/33.25 token/s，**只能作本窗口观察，不能将相对值宣布为稳定加速或默认启用依据**。本次更可靠的结论是输出、边界和统计合同通过；完整 verify 仍是主要耗时。

数据与控制器记录在 `results/upstream-cost-latency-v1/`：`long-cost.json`、`short-budget1/2/16.json`、`cost-summary.json`、`run-ledger.json`。该目录被 Git 忽略，结果摘要保存在本页。

整模型回归复用现有短/11k 的 AR/MTP 对照即可，不另开性能试验矩阵。核对 MTP trial 的 `countersConsistent`、`acceptanceHistogramConsistent` 与 `componentsFitDecodeWindow`，以及有效吞吐等于原 `decode_tokens_per_second`；正常 AR trial 摘要为 `null`。任何算力变化均不应归因于这个纯派生摘要。

研究来源为 [vLLM 请求级接受指标](https://docs.vllm.ai/en/latest/features/speculative_decoding/acceptance_metrics/) 的口径分离思路；实现使用本项目现有计数，未复制上游代码。更完整取舍见 [vLLM / SGLang 研究](research/VLLM_SGLANG.md)。
