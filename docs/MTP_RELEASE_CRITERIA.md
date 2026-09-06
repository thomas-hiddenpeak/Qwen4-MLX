# Native MTP 上线标准与候选状态

更新：2026-09-07。适用范围：本机 M5 Max、单请求、greedy、当前下载权重与独立 Swift runner。本文规定发布门槛，不代表已经通过，也不承诺覆盖所有未来输入。

当前对外推荐采用 **兼容现有 AR 的 greedy 输出合同**：固定权重、tokenizer、提示词 token IDs、输出屏蔽、EOS、上下文与 prefill 配置后，MTP 应输出与已验证 AR 相同的 token IDs。先解决 MTP 的正确性与实际收益，再推进后续缓存工作。

用户最新要求将 **prefill 与 decode 在业务接口和执行责任上分离，再分别优化内核**。性能验收相应按阶段进行：MTP 的主要性能门槛是 decode 有效吞吐与 TPOT，prefill 和初始化/交接成本独立报告。端到端时间是辅助指标，不能因 prefill 占比大而否定已经成立的 decode 收益，也不能省略 MTP 额外成本。当前已实现同一模型 / 执行器上的独立 `prefill` / `decode` 作业交接，`generate` 保留组合调用。本机调度器默认按完整阶段执行，可显式开启 chunk / round 合作调度；增量接口通过 26 项 CPU 与 55 项实模检查。GPU 仍串行，没有实现独立进程 PD 服务或 GPU 并行调度。实现与后续边界见 [阶段分离设计](PREFILL_DECODE_SEPARATION.md)。

## 当前状态

部署范围已由用户确认：先完成本机的 prefill / decode 分离，独立部署以后再考虑。本阶段验收不要求跨进程状态传输或独立 PD 服务；本机阶段指标、交接正确性与已实现的准入行为仍需验证。

**已修复主基准的数值分叉；当前为显式启用的实验候选，尚未通过默认启用门槛。** 默认仍为 AR，显式 MTP 的默认验证器仍为 scalar。候选配置为 `--mtp-depth 2 --mtp-verification batchedScalarLinear --mtp-draft-history 1024`，最后一项默认仍是 full。HTTP 实验入口另行验收，尚未上线；这里不推进前缀树或 SSD 状态缓存。

- 旧模式的 S1/S2 dense 投影使用不同加法分组，误差经过层间传播后导致第 57 token 分叉。保留 S1 累积顺序并共享权重读取的新内核，使四个 11k checkpoint 的全部 trace/logits/提交状态对照恢复逐位一致（[数值诊断](../results/mtp-scalar-linear/long-numerics.json)）。这四个位置的诊断不等于所有输入的数学证明。
- 新验证器加 shared MoE、SDPA 两行拆分后，depth1/2 短请求和 11k/128 的两次输出均匹配已有 AR golden（[短输入](../results/mtp-shared-linear/short.json)、[长输入](../results/mtp-shared-linear/long.json)）。
- 固定七场景在 full history 与 tail1024 两种候选版本各执行 AR/MTP/MTP/AR，共 56 次完整请求，token IDs、终止原因和计数均通过。该轮二进制的 tail1024 版本对应其中 28 次；不能把此前版本的 28 次算作该轮二进制的测试。[full history](../results/mtp-seven-scenarios-full-history/seven.json)、[tail1024](../results/mtp-seven-scenarios-tail1024/seven.json)。
- 该轮二进制另外通过 11k/256、11k/128 各四次 AR/MTP/MTP/AR；256 个 token 全量与同轮 AR 相同，前 128 也与旧 golden 相同。[256 输出](../results/mtp-tail1024-long-budget/long256.json)、[128 输出](../results/mtp-tail1024-long-budget/long128.json)。阶段接口拆分前的该候选版本合计 36 次完整请求零分叉，输入仍主要复用七场景，不能当作 36 个独立任务。
- 20 项针对矩阵、GDN、提交计划、KV view 和生成契约的单测通过；[14 项实模状态检查](../results/mtp-seven-scenarios-tail1024/state.json)通过，包括三个取消位置、重试、短预算和合成 EOS。实际长 depth2 出现接受 0/1/2 三种分支；纯值提交计划另穷举 depth1...4 接受长度。合成 EOS 不等于自然终止的质量验证。
- 算术、严格 JSON、代码函数的 [12 项功能检查](../results/mtp-seven-scenarios-tail1024/functional-checks.json)通过；双语解释人工检查通过。两个 QSA 输入是原 11k 文本的精确 token 前缀，用于边界正确性，不视为完整的语义任务。
- 连续七场景结束后 active memory 回到约 80.63 GB，未见持续增长；这不是长时间 soak 或设备故障恢复的证明。
- 两个新增独立 synthetic 长任务（11216-token 工具 JSON、10784-token 项目事实检索）已冻结并完成 128/256 预算各六轮，共 24 次请求的完整输入/输出 IDs、功能答案、EOS 与终态合同检查全部通过。实际分别自然输出 85/110 tokens，两预算完整 IDs 相同；不等于生成满 256 tokens。性能只有一个带 telemetry 的窗口，facts128 的 AR decode 漂移 7.29%，该组无法判定，其他组也未满足完整复测门槛。见 [新增任务实测与 checker 修复审计](MTP_AGENT_EXPANSION.md)。历史七场景不能自动算作本轮二进制的完整回归。

阶段接口拆分前较快的一组原 11k/128：AR 31.20–32.38 token/s，候选 37.90–39.68；按两轮 decode 时间合计计算，有效吞吐观察到约 22% 提升。整请求 wall time 的两轮均值约 17.59 → 17.17 秒，约 2.4% 的描述性净改善，是另一项指标，不能替代 decode 判断。延长到 256 输出的一组为 AR 29.64–32.30、候选 34.63–35.82 token/s；首次 AR prefill 明显更慢，随后 128 复测又出现 AR 16.65 token/s。现有样本不足以证明稳定的阶段收益，须分别检查各阶段漂移，不得把较慢的 AR 轮次挑作加速分母。

上面是候选证据，不是全部门槛已通过。阶段接口拆分前的二进制与原始运行记录见 [汇总](../results/mtp-release-progress.json)。

新增的 `batchedTokenMoE` routed MoE 内核只作为显式定向实验，见 [token 轴验证](MTP_TOKEN_AXIS.md)。它已通过定向正确性检查，但128窗口收益较小且256复测基本持平；未建立稳定性能收益，不自动继承旧候选的全部场景资格，也未改变默认发布状态。

## 算法保证与数值差异

投机解码的分布保持证明要求验证器计算对应前缀的目标条件分布；greedy 可视为只保留目标 argmax 的分布。[Leviathan 等，ICML 2023](https://proceedings.mlr.press/v202/leviathan23a.html)；[Chen 等，2023](https://arxiv.org/abs/2302.01318) 明确说明硬件数值精度的限制。

GPU 批量运算与逐 token 运算可能使用不同归约顺序或计算精度，因此数学相同不等于浮点结果逐位相同。[PyTorch 官方数值说明](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html) 明确区分 batched 与逐 slice 结果；[MLX 数值精度文档](https://ml-explore.github.io/mlx/build/html/usage/precision.html) 也说明部分 FP32 矩阵操作可能采用较低计算精度。MLX 的这条说明不能直接解释本项目 BF16 路径的分叉，仍须核对实际内核。

这意味着：**token 分叉不自动证明功能质量下降，但也不满足当前 greedy 兼容合同。** 不能用“浮点误差正常”跳过定位。vLLM 同样将 greedy 一致性测试与数值稳定性限制分开说明。[vLLM speculative decoding 文档](https://docs.vllm.ai/en/stable/features/speculative_decoding/)

## 必须严格通过的门槛

| 检查 | 通过条件 |
| --- | --- |
| 接受、拒绝及 bonus | 固定 oracle logits 下，depth 1/2 的接受数 0…depth、首次/部分拒绝、全部接受，token、位置及计数全部 exact；未验证的 draft 不得发布。 |
| 完整状态提交 | GDN recurrent/conv3、PLE conv9/两 token hash history、KV、QSA raw/pooled keys 和位置、MTP head 历史均提交到同一边界；无拒绝后缀污染。 |
| 捕获与裁剪 | 裁剪结果与同次验证计算的对应前缀逐位相同；完整接受后清除临时 capture。此项不要求两种不同计算内核的全部浮点张量逐位相同。 |
| 因果关系 | 验证前缀只依赖其可见历史。检查 mask、QSA 池化边界及未来输入隔离；任何超出已独立校验数值差异范围的未来后缀影响都须定位，不能笼统归为舍入。 |
| EOS 与预算 | EOS 出现在 draft、correction/bonus 或首输出时均正确终止，不重复、不漏报、不越过输出预算；剩余 1/2/3 token 覆盖到位。terminal 状态的 offset 口径须明确，不能让已终止 decoder 继续生成。 |
| 取消及失败事务 | draft、verify、commit 阶段取消/抛错，不发布半提交状态；失败 decoder 拒绝复用；GPU/SSD 工作结束后释放请求占用。新请求结果与干净请求一致。 |
| 请求隔离 | 同模型嵌套请求明确 busy；取消、异常、非法输入后重试可恢复；空输入、非法 token、超上下文明确拒绝。 |
| greedy 兼容 | AR 自身重复稳定；scalar MTP depth 1/2 与 AR 全量 token IDs 相同；拟发布快速路径 depth 1/2 在冻结验收集上零 token 分叉，EOS/finish reason 与计数契约一致。 |

分支未实际触发须标为未覆盖，不能因为指定了 depth 就视为已经验证所有接受或拒绝路径。实际请求难以触发的分支应由独立 oracle 测试补齐。

## 数值诊断与功能检查

分叉定位使用同一前缀和 checkpoint，记录第一处差异：层输入/输出、持久状态、logit 最大绝对误差、top-1/top-2 及其间隔。先排除索引、状态和因果错误，再判断计算路径差异。只比较分叉之后的完整文本，无法定位最初原因。

不同内核的中间张量可以采用预先规定的数值容差；容差应来自独立参考与实际 dtype/算子验证，不能为已失败样本临时放宽，也没有通用的“误差低于 1% 就可上线”标准。局部小误差或 top-1 接近不能单独证明完整状态长期等价。

JSON/工具参数合法性、确定性答案、代码测试、长文事实检索和指令约束应独立检查。**允许不同 token 是另一种产品合同**，需要事先明确范围和独立质量验收；当前草案未采用该合同，不能用它把现有兼容性失败改判通过。

## 小型验收集

默认启用前采用一个小而固定的集合，避免无限增加测试数量：

- 保留已跑的七场景：双语、算术、严格 JSON、代码、2051/2053 QSA 前缀与原 11k agent 输入；包含历史分叉回归和生成过程跨 ratio-4 池化边界。
- 两个独立的 10k–12k agent 任务已补齐：工具 JSON 和项目事实检索，分别冻结独立合成资料、token IDs、预定答案与功能检查器，完成本轮 24 次正确性检查。详见 [新增任务实测](MTP_AGENT_EXPANSION.md)；输入不是同一长提示词仅改末尾问题。
- 原 11k 加两个新长任务分别报告 128/256 输出预算。七场景已完成同模型 28 次混合请求与内存恢复观察，不机械追加另一套相同的 20 轮；取消、预算和接受分支也无需无改动重复测试。

上述九场景是首版建议的完整回归集合，两个新长任务已完成本轮正确性检查，性能仍是单个带采样窗口、未全部满足门槛；历史七场景与本轮来自不同阶段的二进制，不能直接合并为当前版本已通过完整九场景。更长上下文、不同 chunk 或新模型配置应有自己的边界用例，不自动继承本次结果。

冻结输入 IDs、检查器、代码/库版本及配置后再验收。首版可先只发布 depth2、tail1024、输出预算不超过 256 的配置；不必为了它验收 depth3/4。AR 与拟发布配置须覆盖全套，scalar oracle 保留在历史分叉与状态分支样本上即可；若同时发布 depth1 或 full history，则须为其补齐同样验收，不能跨配置借用结果。该规模是本机首版的实用回归集，不构成广泛质量或所有上下文长度的保证。

tail1024 只截短初始 head history。未来扩大输出预算时，须专门覆盖 head 再次越过 2051 行 QSA 阈值；当前 128/256 输出没有验证此项。本文件规定单请求门槛；HTTP 实验入口已有的有限网络验收另见 [HTTP 服务实验](HTTP_SERVER_EXPERIMENT.md)，不扩展此处 MTP 合同。随机采样和设备故障恢复不属于这一首版配置的已验收范围。

## 性能与发布决定

**MTP 主要性能门槛为真实 10k+ agent 输入后的 decode 有效吞吐提升至少 10%，对应 TPOT 下降，并在独立时间窗口重复成立。** 这里的计数是目标验证后实际发布的 token，不能使用 draft 数或 round 数。正确性要求不变；prefill 占比不再作为阻塞 decode 优化的依据。

| 阶段 | 必须报告的口径 |
| --- | --- |
| 主干 prefill | 实际输入 token 数、处理耗时及 input token/s；包括本阶段 SSD 等待和必要求值。MTP head 初始化单列。若两者重叠，保留时间窗口，不能重复相加。 |
| 初始化与阶段交接 | 首次 head 构建/加载、head prompt history 求值、状态准备/传递及首 token 选择的耗时各自说明归属。冷/热分开；未实现的 PD 传输填未实现，不能记作零成本。 |
| Decode | `有效输出 token 数 / decode 时间` 与 `decode 时间 / 有效输出 token 数`（平均 TPOT）。包含 draft、verify、提交/回放、逐轮 head history 更新、SSD 等待和轮内调度；不得只计验证内核。现有 API 计数排除首 token、包括实际输出的 EOS，耗时排除 callback，须同时保留这一口径及 callback 时间。 |
| 请求辅助指标 | TTFT、调用总 wall time、首次准备成本、输出预算、接受长度分布、峰值内存。MTP 成批发布 token 时，平均 TPOT 不等同于每次用户可见输出的等待时间。 |

当前 `timeToFirstTokenSeconds` 混合主干 prefill、head prompt history 和首 token 选择；`preparationSeconds` 仅统计首次 head 构建，lazy GPU 求值仍可能发生在请求内；`totalSeconds` 排除该准备但包括 callback。旧结果不能反推精确的纯 prefill 或 PD handoff 时间。新阶段指标落地前，缺失项明确标为未独立测量，原始报告不重写。

同一已加载模型、相同输入/输出预算与资源条件下交错配对 AR/MTP；预热轮数预先固定，至少两个独立时间窗口。后续从相同 prefill 产物单独重放 decode 时，须恢复完整状态，并排除上一轮生成残留。恢复成本单列，不能藏入或移出某个候选的计时区间。存在已知训练等资源争用时，该批性能不用于收益结论。

可执行的最小性能复测：两个时间窗口，每个窗口先各预热一次 AR/MTP，再对冻结的长任务分别运行三组 AR/MTP/MTP/AR；128 与 256 输出分开记录。提前规定组内前后 AR **decode** 漂移超过 5% 时，该组 decode 性能标为无法判定；prefill 采用其独立耗时检查漂移，仅影响 prefill 结论。旧报告只有混合 TTFT 时，不能据它裁决纯 prefill。不得只删慢 AR 或挑最快 MTP，所有组仍保留正确性与原始结果。若 decode 一直波动，就保持 decode 性能门槛未通过，不能放宽阈值解释现有结果。

在拟发布范围内，所有严格正确性检查通过、decode 达到上述稳定收益且持续内存不增长，才通过 MTP decode 候选门槛。所有预先指定长任务及 128/256 预算分别报告，不在看到接受率后挑选有利任务。Prefill 优化按自己的吞吐/延迟验收；阶段交接与初始化必须明确披露，并由业务选择是否采用该组合。这里不再要求每项 decode 内核改进都先证明整请求净收益，也不把混合 TTFT 增加 5% 作为 decode 内核的自动否决条件。

端到端成本仍须完整呈现。设 prefill 为 P、原 decode 耗时为 D、decode 加速比为 s、额外初始化/交接成本为 H，总耗时变化由 `P + D/s + H` 决定。P 很大时，总时间改善比例会被稀释；H 很大时，完整调用甚至可能变慢。这两种结果都应如实报告，但应分别决定 decode 阶段验收与整请求业务选择，不能相互代替。

## 业务分离的正确性要求

业务分离是新增实施要求，先建立可独立调用与验证的 prefill、decode 边界，再分别接入内核候选；不等同于在原 `generate` 内新增计时。当前尚无独立 PD 服务，也未验证跨进程或跨设备的状态交接。

阶段产物必须覆盖 GDN recurrent/conv3、PLE conv9/两 token hash history、attention KV、QSA raw/pooled keys 与绝对位置、MTP head KV/QSA/历史及其绝对位置；还须明确主干 hidden、pending token、已输出计数、EOS/预算与提交边界。只交接 K/V 不成立。交接失败或取消不得发布半份状态，交接后 decode 与同配置连续执行应完整 token 一致，拒绝草稿的后缀不能进入交接产物。新实现完成之前，这些是要求，不能标为已通过。

当前主基准分叉已修复，默认仍为 AR。有限回归、稳定 decode 收益和新增阶段交接验收各有自己的状态；其中任一项的证据不能代替另一项。
