# 两个独立长 agent 任务的 MTP 初筛

2026-09-07，本机 M5 Max、macOS 26.6.2、MLX 0.32.2。两个新增任务共 24 次请求的输入 IDs、AR/MTP 完整输出 IDs、冻结 JSON 答案、EOS 与终态边界合同全部通过。它们补齐了 [上线标准](MTP_RELEASE_CRITERIA.md) 中两个独立长任务的本轮正确性证据；**性能仍未通过发布门槛**。

工具任务实际输出 85 个 token，事实任务 110 个，均包含 EOS。128/256 是两个输出上限，同一任务在两个上限下自然结束且完整 IDs 相同；本轮没有生成到 256 个 token，也没有因此覆盖更长 decode 或 head 历史增长边界。

## 冻结任务与运行范围

两个任务完全自写，明确为 synthetic 数据，不含真实用户记录、实际执行的工具调用或模型生成的标准答案。它们分别构造不同的工具目录与项目档案，没有复用原 11k 提示词改尾句，也没有重复段落填长度。

| 任务 | 冻结输入 | 内容与可判定输出 | 实际输出 / decode 计数 |
| --- | ---: | --- | ---: |
| [工具选择](../fixtures/gpu-agent-tools-11k/README.md) | 11216 tokens | 34 个不同工具、租户/服务/事故映射、只读约束；输出唯一 trace 查询工具和精确参数 | 85 / 84 |
| [项目事实检索](../fixtures/gpu-agent-facts-11k/README.md) | 10784 tokens | 28 个项目基线、20 条变更、5 项保留策略；区分草案、撤销与已批准变更，输出当前事实及来源 | 110 / 109 |

答案线索分散于全文。`expected.json`、响应 schema 和源档案在任何生成前写定；随后用现有 runner 的 CPU `tokenize` 入口冻结完整 system + user、no-thinking chat 输入 IDs，并验证解码回原始文本。manifest 保存源文件、tokenizer、输入和 checker 的 SHA256；后述 checker 报告字段修复未改变任务或答案。

输入 IDs 文件 SHA256：

- tools：`b73512d800f78446a3c4b14426ad9bfaf8ec6872eb6f8e32873249d26791091c`
- facts：`8fa0e14806cef623d1eee76a4b9b45c390ebeca75b62bd5c274b5bac55ef875a`

四个 case 按 tools128、facts128、facts256、tools256 顺序运行。每个 case 在同一进程、同一已加载模型中执行 `0,2,0,2,2,0`：前两次分别预热 AR、MTP depth2；后四次为测量 AR/MTP/MTP/AR。24 次均参与正确性检查，性能仅使用 repetition 2、3、4、5。四个 case 属于同一个时间窗口，不能当作四个独立复测窗口。

固定配置：greedy、context 16384、prefill chunk 416、reference decode、`batchedScalarLinear` 验证、初始 head history tail1024、prefill/verify 每 4 层求值、SSD nextChunk/单 worker、wired disabled、默认 fused prefill。未启用实验 MoE/GDN 替代库、详细算子 profiler 或 command-buffer timing。每轮都启用 500 ms telemetry，包含新 GPU 状态通道；这些结果不是无采样开销的基准。

冻结二进制 SHA256 为 `2cbab9da356e373e530fed5d94f468e90620e87999c99b7a2afbf7be0f14cc24`。准确参数、环境、原始 manifest 哈希见 [plan.json](../results/mtp-agent-expansion-v1/plan.json)，进程与服务恢复记录见 [run-ledger.json](../results/mtp-agent-expansion-v1/run-ledger.json)。四组执行于 2026-09-06 22:32–22:42 UTC，参考服务随后恢复 ready。

## 正确性与终态口径

独立离线检查逐轮比较完整 `prompt_tokens` 数组与冻结 IDs、生成 IDs 与同 case 首个 AR、结构化答案与冻结 expected；所有 24 次通过。两个预算间也逐任务比较了完整输出 IDs。checker 允许 JSON 空白及对象键顺序变化，拒绝额外文本、重复键、非有限常量、错误类型、额外字段或错误答案。语义答案正确与 AR/MTP token 一致是两项独立检查。

全部请求自然 EOS，只有最后一个输出 token 是 EOS，没有越过预算，QSA 活跃层数均为 12。decode 分子排除首个输出、包含实际输出的 EOS；不能用 draft 数、验证行数或 round 数代替。

| 任务 | AR 最终 offset | MTP 最终 offset | 原始输出 |
| --- | ---: | ---: | --- |
| tools | 11300 | 11301 | [128](../results/mtp-agent-expansion-v1/tools-128.json)、[256](../results/mtp-agent-expansion-v1/tools-256.json) |
| facts | 10893 | 10894 | [128](../results/mtp-agent-expansion-v1/facts-128.json)、[256](../results/mtp-agent-expansion-v1/facts-256.json) |

AR 终态 offset 是 `input + output - 1`；本轮 MTP 的已接受 draft EOS 被验证路径消费，offset 是 `input + output`。这是 [QwenMTPDecoder](../Sources/ANERunnerGPU/QwenMTPDecoder.swift) 与 [MTPCommitPlan](../Sources/ANERunnerGPU/MTPCommitPlan.swift) 已声明的行为，decoder 终止后拒绝复用，生成接口丢弃状态。不得把相差一位的终态称作 AR 等价、可继续生成的 checkpoint。本次检查的是已发布 token 与终态合同，没有另做全模型最终状态张量逐位比较。

工具 MTP 每次 29 rounds，draft 接受 56/58，接受 0/1/2 的 histogram 为 `[1,0,28]`；事实 MTP 每次 37 rounds，接受 73/74，histogram 为 `[0,1,36]`。两任务合起来实际触发接受 0/1/2，不能替代已有 oracle、取消、失败恢复及其他 EOS 位置的测试。

## Decode 观察值

下表直接由原始每步时间求和：`两轮实际 decode token 总数 / 两轮 decode 时间之和`。平均 TPOT 是它的倒数。AR 漂移为同组后 AR 吞吐相对前 AR 吞吐的绝对变化比例；不删慢轮，不挑最快轮。

| case | AR token/s | MTP token/s | MTP / AR | AR decode 漂移 | 本组判定 |
| --- | ---: | ---: | ---: | ---: | --- |
| tools128 | 30.556 | 48.897 | 1.600 | 4.79% | 单次带采样观察 |
| facts128 | 25.260 | 44.409 | 1.758 | **7.29%** | **漂移超过 5%，无法判定** |
| facts256 | 24.724 | 43.837 | 1.773 | 1.56% | 单次带采样观察 |
| tools256 | 26.358 | 45.709 | 1.734 | 1.87% | 单次带采样观察 |

接受率较高，观察值支持继续推进候选。但只有一个窗口、每 case 一组配对，facts128 还触发预先规定的漂移否决；不能把这些比值宣称为稳定的 60%–77% 加速。其他三组没有触发该漂移阈值，也不代表通过完整性能门槛。

## Prefill、head history 与 decode 成本

`phase_metrics.prefill_target_seconds` 在每个主干 chunk 求值后、调用 `consumePrompt` 前累计，包含主干首 token 选择；`mtp_prompt_history_seconds` 单独记录 head prompt history 和 `finishPrompt` 求值。旧 `prefill_chunk_seconds` / `prefill_tokens_per_second` 包含 chunk 内 head history，不能标成纯主干成本。完整 `prefill_total_seconds` 是 TTFT 阶段包络，还包含阶段间隙与收尾，不能强制等于前两项相加。

以下均为两次测量轮均值。head 初始历史因 ratio-4 对齐实际保留 1027 对，而不是恰好 1024 对，这是 tail1024 既有行为。

| case | AR 主干 prefill (s) | MTP 主干 prefill (s) | MTP prompt history (ms) | MTP 完整 prefill (s) | AR 主干吞吐漂移 |
| --- | ---: | ---: | ---: | ---: | ---: |
| tools128 | 14.953 | 15.148 | 53.545 | 15.204 | 5.91% |
| facts128 | 19.744 | 21.053 | 69.339 | 21.125 | 12.39% |
| facts256 | 20.709 | 20.751 | 60.054 | 20.814 | 2.36% |
| tools256 | 20.733 | 21.442 | 58.467 | 21.502 | 3.06% |

Prefill 变化独立于 decode 判断。本轮使用 `inline-benchmark-no-transport`，没有测量独立进程 PD 传输；模型/head 权重加载位于请求计时之外，不能将上表当作冷启动总成本。

Decode 内的 `mtp_cost_summary.historySeconds` 是逐轮 head 历史更新，与上表 prompt history 不同，必须继续计入 decode。下表为两轮均值，单位 ms；列项相加覆盖完整 decode 时间，未分类余额保留，不混成纯验证内核性能。

| case | draft | verify | commit / replay | decode head history | 其余 decode | 总 decode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tools128 | 133.783 | 1512.627 | 7.605 | 27.076 | 36.789 | 1717.880 |
| facts128 | 175.849 | 2180.907 | 8.388 | 39.316 | 50.004 | 2454.464 |
| facts256 | 181.842 | 2211.485 | 7.226 | 39.094 | 46.854 | 2486.502 |
| tools256 | 138.504 | 1627.055 | 7.508 | 28.497 | 36.142 | 1837.707 |

本轮 replayedTokens 均为 0；commit / replay 是统一统计字段名，不表示发生回放。SSD 等待已处在各自执行阶段内，不能再加一次。verify 占本轮大部分 decode 时间，是成本位置证据，不是任何特定单核优化收益的证明。

## GPU 状态完整性与解释边界

独立重算从每个通道相邻读取端点开始，检查同源链、单调顺序、标签与单位一致。仅当完整 `previous.read_start → current.read_end` 包络落在同一请求阶段内，才累加原始状态计数；跨阶段与窗口外区间排除。覆盖率采用包络并集除以阶段时长，避免相邻包络重叠重复计时。归属结果与汇总中的 raw counts、完整区间数和覆盖率一致。

四组分别有 231、304、315、309 个 interval samples，sample_id 连续；各有一个预期 baseline。每组两个不完整通道区间对应 baseline 的 GPUPH/GPU_CLTM，不是中途漏采；每组排除 28 个跨界或窗口外通道区间。通道端点链均完整，目标 PID/时钟与来源记录一致，collector 完成且 dropped_events 为 0。采样由 controller 结束，sampler 没有向目标模型进程发送终止信号。

测量轮按 AR/MTP/MTP/AR 排列，下表仅列 GPUPH 完整包络覆盖率；GPU_CLTM 独立读取，端点略有不同，精确覆盖率保留在汇总中。

| case | 完整区间数 | decode 包络覆盖率 | thermal category |
| --- | --- | --- | --- |
| tools128 | 5 / 2 / 2 / 4 | 96.05 / 59.61 / 61.42 / 73.03% | 前三轮 nominal，末 AR 为 fair |
| facts128 | 7 / 3 / 4 / 8 | 86.85 / 63.60 / 83.16 / 92.16% | fair |
| facts256 | 8 / 4 / 4 / 8 | 93.02 / 82.47 / 83.45 / 94.39% | fair |
| tools256 | 5 / 3 / 3 / 5 | 79.99 / 84.23 / 84.07 / 81.59% | fair |

tools128 首个测量 AR 中 GPUPH 的 P13 原始计数占比约 74.4%，末个 AR 转为 P11/P12 为主；GPU_CLTM 的 NO_CLTM 比例也发生变化。后续 facts 与 tools256 的状态分布不同。这些变化与运行顺序、阶段和 thermal category 同时被观测到，**不能证明哪一项造成降速**。

[GPUStateSampler](../Sources/ANERunnerTelemetry/GPUStateSampler.m) 记录系统级 `GPU Stats` 下的 `GPUPH`、`GPU_CLTM` 原始档位及系统 thermal category。单位原样保留 `24Mticks`；没有换算成 MHz，不能归因到目标 PID 或推断温度、散热、其他项目训练。thermal 记录是类别采样数，不是持续时间占比。tools128 两个短 MTP decode 窗口各只保留两个完整区间、约 60% 覆盖，其状态分布不能代表整个 decode。

GPU 状态中的 prefill 窗口包含 head prompt history，无法仅凭采样分离主干与 head 的状态分布。这里也没有物理 DRAM 字节计数，不能把状态驻留比例或逻辑权重带宽当作物理带宽利用率。

加载后 active 约 80.63 GB，请求尾部 active 约 81.0 GB，peak 约 82.0 GB；本短批次没有明显增长。尾部快照仍有请求状态，不能说已释放回纯权重基线，也不能当作长时间内存稳定性验收。

## Checker 修复审计

首版 checker 误将报告的 `trial.prompt_tokens` 当作长度，但 [GPUGeneration](../Sources/ANERunnerCLI/GPUGeneration.swift) 实际写入完整整数数组，导致首个 tools128 报告被误拒绝。修复为严格检查 list、每项为非 bool 的 int，再逐项等于冻结 IDs；这是报告 schema 修复，没有依据模型输出来改答案或放宽 JSON 合同。

原始 checker SHA256：`6fd857edf3fc73375623010e2119e5bbe555063da2b15308484376f4e15eebb9`；修复后：`07ff8168331f5dc838008ad09ac010809eec2db5288af7da258e4052bffcb25d`。修复时间 `2026-09-06T22:38:17.402581Z`，发生在首个报告落盘后。两个 manifest 保留原 freeze_utc 和 `checker_corrections` 旧/新哈希、旧 manifest/README 指纹及原因；plan 保留生成前原始 manifest 哈希。

| fixture | 原始 manifest SHA256 | 修复后 manifest SHA256 |
| --- | --- | --- |
| tools | `caa9aefcd69e0d643a4d255d37cbf999b70d2b6f0c0ad73ab159997ce45c19ee` | `ee15aaebdd34c7fbb87de0a977f85c14975f067502ba193fecfdc2910a7331fc` |
| facts | `bdde43b562c1453ff9efef22618b949093708b5a9eef34a641d45345ee72c3a2` | `19aa27b71fd44dc7ad7b2226b55021f07ea26813af6c2955011045363e748f56` |

修复后通过 20 项功能控制检查，以及 scalar 长度、bool ID、错误 ID 三个误输入拒绝检查；四个原始 generation 报告重新运行 checker，24 个答案全部通过。prompt、expected、response-schema、source-data、输入 IDs 与 tokenization 文件哈希保持不变。

## 离线复查与下一步

完整原始路径及 SHA256、逐轮阶段时间、终态合同、采样覆盖与状态分布见 [summary.json](../results/mtp-agent-expansion-v1/summary.json)。该目录是本地原始运行产物；冻结 fixtures 和 checker 保留在仓库内。独立复核没有启动模型、重跑 GPU、构建或修改原始结果。

仅 CPU 功能复查：

```sh
python3 -B scripts/check_agent_workload.py --fixture fixtures/gpu-agent-tools-11k \
  --generation results/mtp-agent-expansion-v1/tools-128.json \
  --generation results/mtp-agent-expansion-v1/tools-256.json
python3 -B scripts/check_agent_workload.py --fixture fixtures/gpu-agent-facts-11k \
  --generation results/mtp-agent-expansion-v1/facts-128.json \
  --generation results/mtp-agent-expansion-v1/facts-256.json
```

本轮可以关闭“两个独立长任务缺失”的正确性待办，并保留自然 EOS 作为新增行为证据。仍需按既定标准完成独立时间窗口和配对组数，所有预定长任务及预算均保留；facts128 不能用本轮比值裁决收益。历史七场景来自不同阶段的二进制，不能与本轮直接合并成当前二进制已通过完整九场景。默认 AR、显式 MTP 默认 scalar 及快速候选的发布状态均不变。
