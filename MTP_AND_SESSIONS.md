# Native MTP 与生成会话

2026-09-06。用户调整了开发顺序：现在接入 MTP，随后做前缀树和 SSD offload 等状态缓存管理；前提是 runner 和 API 可稳定使用。无 MTP 的 AR 默认路径保留为对照。

最新要求进一步明确：**prefill 与 decode 先做业务分离，随后独立优化内核，并按阶段统计性能。** MTP 的主要性能门槛是 decode 有效吞吐/TPOT；prefill 和初始化/交接分别验收、披露，整请求耗时保留为辅助指标。当前 Swift API 已拆为独立 `prefill` / `decode` 作业与单次消费的完整状态句柄，`generate` 保留组合调用。同执行器已有有界 prefill / ready decode 队列和 token 准入；默认完整阶段调度，显式 `cooperative` 可按 prompt chunk / 完整 AR 或 MTP round 继续执行。增量接口与调度通过 26 项 CPU、55 项实模检查；GPU 保持串行，尚未实现独立进程 PD 服务。接口与能力边界见 [阶段分离设计](docs/PREFILL_DECODE_SEPARATION.md)。

## 实现范围

用户已确认先在本机拆分 prefill / decode，独立部署后续再考虑。本阶段共享模型权重，围绕阶段接口、完整状态交接、本机调度与 kernel 优化推进；跨进程 PD 服务不作为当前 MTP 验收的前置条件。

`QwenMTP` 使用本模型的 41 个原生 MTP 张量，共 2,113,220,096 源字节：Q4 routed MoE、BF16 HC / attention / shared expert、Q6/group64 draft head。共享主干 embedding，没有另下其他模型，也不调用作者服务。Head 公式与位置语义来自固定作者源码的 `qwen4MtpForward`，许可见 UPSTREAM-LICENSE。

`QwenMTPDecoder` 采用 greedy draft / target verify。每请求维护独立 head KV/QSA 历史。Prompt stream 与下一 token 配对，head 第一行绝对位置为 1；逐轮用真实主干 hidden 提交历史，不能将预测 hidden 永久写入已提交缓存。普通批量模式在拒绝时恢复快照并重算前缀；capture 模式直接提交对应位置的完整状态，包括 GDN recurrent / conv、PLE hash history / conv 和 attention KV / QSA。整轮成功后才向调用者发布 state；取消或失败不发布中间验证状态。

验证方式：

- `scalar`：主干逐 token 验证，保持 AR 运算路径，作为输出参考；草稿不会减少主干调用数。
- `batched`：一次验证 pending token 与草稿。MLX 的批量投影 / MoE 和 GDN chunk 内舍入可能不同于连续 S1；当前未通过输出一致性门槛。
- `batchedRounded`：在批量验证中加入 GDN 每 token 的 BF16 回存，诊断跨 token 舍入差异。它不改变普通 prefill / AR，也不保证消除其他算子的批量数值差异。
- `batchedCaptured`：在 `batchedRounded` 基础上捕获每个验证位置的 GDN 状态与卷积输入。接受前缀时同时截取 GDN、PLE、KV/QSA，无需重新执行主干；全部接受复用最终状态并清除捕获引用。GDN/PLE 前缀使用独立拷贝；当前 KV/raw key 小后缀使用有界 view，较大裁剪仍拷贝，详见下面的更新。`rollbackSeconds` 在该模式记录前缀提交耗时，`replayedTokens` 应为 0。
- `batchedScalarMoE`：保留上述捕获和批量主干投影，将 MoE 按行交给原单 token kernel，诊断 sorted gather 与融合 decode 路径的数值差异。
- `batchedScalarLinear`：新的兼容候选。S2...5 dense 投影共享读取原 BF16 权重，同时保留 S1 GEMV 的累积与归约顺序；MoE 保留每行原 Q4 expert 算法，共享路由和 shared expert 的 dense 计算。S3...5 attention 按最多两行拆分 SDPA，避免 GQA=12 时切换到另一条 attention 路径。当前默认仍不启用 MTP。
- `batchedTokenMoE`：保留 `batchedScalarLinear` 的数值与状态路径，将 S2…5 routed MoE 的逐 token 调用合并为两个带 token 轴的 kernel。显式实验选择，未改变默认值；实现与定向验证见 [token 轴实验](docs/MTP_TOKEN_AXIS.md)。

`--mtp-depth 0...4`，默认 0。显式启用 MTP 时，验证默认 `scalar`；所有批量候选都必须另外指定 `--mtp-verification`。`--mtp-order 0,1,2,1,2,0` 可同进程交错对照；`--mtp-verification-order batchedScalarLinear,batchedTokenMoE,batchedTokenMoE,batchedScalarLinear` 可在相同 depth 下比较验证内核。Decode 吞吐按实际输出 token 计数，不能按 draft 或 round 数代替；平均 TPOT 是同一 decode 窗口的耗时除以有效 token 数。MTP 模式的逻辑权重带宽率置空，因为 AR 固定 bytes/token 不再适用；draft / verify / replay / history 各阶段单独报告。硬件 DRAM 带宽仍未知。

## 会话 API

`QwenGenerator.generate` 为同步 Swift 库 API；当前没有独立 runner 的 HTTP listener、OpenAI 协议或 SSE 服务。它支持：

- Token / context / maxTokens / chunk / MTP depth 的调用前校验；文本路径拒绝多模态特殊输入。
- 同一模型的所有 generator 共用一个请求锁，重叠请求明确报 busy，不无界排队。
- 每请求新建全部状态；合作式取消和 throwing token callback；退出时等待已提交 GPU 工作和 SSD 读取结束。
- 取消 / callback 失败后可重新发起请求；若设备同步恢复失败，在模型层禁止后续请求并要求重载。
- 原模型的两种 EOS 均结束生成。MTP EOS / 预算完成后 decoder 禁止继续调用。

这不是 GPU kernel 可随时中断：取消在 chunk / draft / verify 边界被观察。低层 `QwenModel.forward` 仍要求单一推理执行线程，不应绕过生成 API 并发操作。MTP 后的前缀缓存必须保存整个会话状态，不能仅保存 attention 的 K/V。

## 新要求：阶段业务边界与指标

现有接口已建立可独立调用、验证和管理状态所有权的 prefill 与 decode 边界，内核优化分别接入这两个阶段。把原 `generate` 加上计时标签不能替代业务分离；业务分离也不自动意味着已经具备多进程、跨设备或并发 PD 服务。

交接产物必须保存同一已提交前缀上的完整状态：

- GDN recurrent 与 conv3；PLE conv9 与两 token hash history。
- Attention KV；QSA raw/pooled keys、池化边界及绝对位置。
- MTP head KV/QSA、已提交历史及其独立绝对起始位置；用于后续 head 输入的主干 hidden。
- Pending token、主干 offset、已输出计数、EOS/预算和提交边界；拒绝草稿不能泄漏到交接结果。

交接失败或取消不能发布半份状态；从交接产物启动的 decode 应与同配置连续执行完整 token 一致。真实交接及恢复成本须测量，尚未实现时标为未实现。取消/rollback 与阶段交接分别验证；当前阶段与合作调度证据见阶段设计，旧 MTP 数字不自动覆盖新增调度路径。

性能分开报告：

| 范围 | 指标与边界 |
| --- | --- |
| 主干 prefill | 输入 token 数、处理延迟、input token/s，含本阶段 SSD 等待与必要求值；head prompt history 初始化独立列出。 |
| 初始化/交接 | Head 首次构建/加载、prompt history 求值、状态准备/传递、首 token 选择分别说明；冷/热分开，重叠时间不重复相加。 |
| Decode | 有效输出 token/s、平均 TPOT、draft/verify/commit/replay/history 的耗时，以及接受长度分布。当前 API 排除首 token，包含实际输出的 EOS；计时包含轮内计算、SSD 等待和调度，排除 callback，后者单列。 |
| 请求辅助指标 | TTFT、完整调用 wall time、首次准备成本、callback 与峰值内存。成批输出的平均 TPOT 不等于逐 token 用户可见等待时间。 |

旧 API 的 `timeToFirstTokenSeconds` 含主干 prefill、head prompt history 与首 token 选择；`preparationSeconds` 仅是首次 head 构建，lazy GPU 求值仍在后续阶段；`totalSeconds` 不含该准备时间但含 callback。不能把旧 TTFT 改名为纯 prefill，也不能声称旧记录已测量 PD handoff。

MTP decode 的收益独立判断，不再因长输入 prefill 占比大而阻塞该阶段优化。Prefill 回退与额外交接成本必须同时披露；整请求可能改善较小甚至回退，应由业务按请求长度和使用方式选择，不能隐去成本或拿整请求百分比替代 decode 结论。当前默认仍为 AR，稳定阶段收益与有限回归门槛见 [上线标准](docs/MTP_RELEASE_CRITERIA.md)。

## 首轮实测

[首次完整模型数据](results/native-mtp-first/summary.json)：1,217-token prompt、64-token greedy output。Scalar depth1/2 均与 AR 完整 IDs 一致；原始 batched depth1 在第 30 个输出 token 分叉，depth2 在第 13 个分叉，重复运行可复现。热 decode：AR 35.33–35.67 token/s，batched depth1 35.15，depth2 33.50。既未证明提速，也未通过 AR 输出门槛，因此没有将 MTP 纳入默认。

[会话与长输入复测](results/native-mtp-session-rounded/summary.json)：完整模型复用期间 15 项检查全部通过，包括取消 / callback 抛错后重新请求得到相同输出、同模型的另一 generator 被明确拒绝、非法请求在生成前拒绝、scalar MTP depth1/2 后回到 AR 输出不变。11,057-token 输入下，两轮 scalar depth1 均与 AR / 之前长输入参考完整 128 IDs 一致，12 层主干 QSA 激活，草稿接受 51/75（68%）。本组 AR decode 由 30.47 降至 26.47 token/s，存在运行内速度漂移；不据此给出精确性能收益。

GDN 每 token BF16 回存的小 GPU 测试与连续 S1 输出 / state 逐位一致；扩展捕获后，每位置快照和 count1/2/3 前缀提交也通过。12 项针对本轮改动的测试通过（7 项会话契约、4 项已有 PLE 预取、1 项含多组真实 GPU 断言的 GDN 测试）。整模型 `batchedRounded` depth2 的两轮短输出与 AR 相同，但 depth1 仍在第 30 个 token 分叉，因此舍入修正未解决全部批量差异。

HTTP 层、多请求压力、设备故障注入、验证中途取消以及完整 EOS 分支覆盖尚不属于已验证能力。基础会话验证通过不等于生产服务验收。

[免重算捕获实验](results/native-mtp-captured/summary.json)中，热 depth1/2 分别约 43.58 / 46.61 token/s，AR 约 37.23；草稿拒绝后的整主干重算为 0，depth2 的接受数 0/1/2 均实际出现。所有 MTP 短输出仍在第 30 个 token 与 AR 分叉，因此 11k 快路径测试在短输出门槛失败后未执行，默认保持 AR。这里的较高速度属于不同输出序列上的候选观测，不能称为已验证的无损收益。

随后[按行 MoE 实验](results/native-mtp-scalar-moe/summary.json)的 depth1 两轮短输出均恢复与 AR 一致，热速度 40.78 vs AR 36.34 token/s（约 +12.2%，单输入观测）；depth2 仍分叉。对通过短测试的 depth1 单独进行 [11k 验证](results/native-mtp-depth1-long/summary.json)，两轮在第 57 个输出 token 出现相同分叉；MTP 约 31.72–32.51 token/s，AR 由 32.18 变为 29.86，尚不能据此确认长输入收益。原因没有完全定位，不能仅凭发生在长上下文就归因于 QSA。

以上是早期实验记录；以下进展更新替代其“主基准仍分叉”的结论。[早期结构化汇总](results/native-mtp-milestone.json)保留当时的验证边界。

## MTP 发布前修复与回归

上线条件见 [MTP_RELEASE_CRITERIA.md](docs/MTP_RELEASE_CRITERIA.md)。HTTP 与前缀树 / SSD offload 管理暂不推进。

`results/mtp-release-numerics/long-numerics.json` 从同一 AR checkpoint 比较 S1 与 S2。四个 11k 位置在改变未来草稿后，当前行的全部 trace/logits 逐位不变；S1/S2 则从前几层微小误差逐步扩大，最终 logit 最大绝对差约 0.19–0.31。固定 MLX 的 GEMV / gemv_wide 使用不同的加法分组，已据此实现 `GPUVerificationLinear`。

`results/mtp-scalar-linear/long-numerics.json` 中相同四个位置的全部 trace、logits、提交后状态对照均恢复 bit-exact（KV/raw index 检查新增行，GDN/PLE/pooled 状态检查全部元素）。同目录 `long.json` 的 depth1 两次 11,057/128 完整生成均与旧 AR golden 相同，消除了第 57 token 分叉；当时速度仍接近 AR，未视为优化完成。

`results/mtp-shared-linear/short.json` 与 `long.json` 在加入 shared MoE 批量投影和 SDPA 两行拆分后，depth1/2 的两次短/长输出全部 IDs 与既有 golden 相同。长 depth2 约 33.70–34.18 token/s，但同组 AR 从 31.84 降至 25.09，性能漂移明显，不能据此冻结精确收益。连续请求后的模型 active memory 基线约 80.63 GB 保持稳定。

同目录 `state.json` 的 14 项实模检查通过，新增最终目标求值后、head history 求值后的取消边界；预算/EOS 使用新候选。纯值 `MTPCommitPlan` 已用于实际 batch 提交，测试穷举 depth1...4 的全部接受长度及 EOS/预算。当时 19 项针对矩阵、GDN、提交计划与会话契约的单测通过。

## 有界状态 view 与草稿头历史实验

后续改动避免每轮复制全部 KV：验证产生的新 KV/raw key 缓冲区最多保留 4 个无效尾行，pooled key 最多保留 1 块；重复裁剪累计越界则复制，下一轮 concat 只读取逻辑前缀。GDN/PLE 小状态仍独立复制。新增 GPU 测试确认 view 与 owned copy 的 SDPA 输出逐位相同、未来哨兵不影响结果、重复裁剪触发复制；当前针对性单测共 20 项通过。

`--mtp-draft-history 1024` 仅截短草稿头的初始 prompt history，主干目标模型始终保留完整上下文与位置。首段按 4-token pool 对齐，保留 1024…1027 个配对行；后续生成历史正常增长，并非持续滑窗。head 状态具有独立绝对起始位置，提交后必须与主干的绝对边界一致。默认仍为 full，截短只影响草稿质量/接受率，不允许未经目标验证的 token 输出。

[七场景 full history](results/mtp-seven-scenarios-full-history/seven.json) 与 [当前 tail1024](results/mtp-seven-scenarios-tail1024/seven.json) 各 28 请求均全量匹配同组 AR，包含中英文、算术、严格 JSON、代码、2051/2053 QSA 边界以及 11,057-token agent 输入。当前版本另通过 [11k/256 与 11k/128 延长对照](results/mtp-tail1024-long-budget/run-ledger.json)，合计 36 请求；原 128 token golden 同样保持一致。[算术/JSON/函数执行检查](results/mtp-seven-scenarios-tail1024/functional-checks.json)全部通过。它们是有限回归集，并不覆盖所有输入、采样或服务负载。

七场景中的 11k/128，AR 为 31.20–32.38 token/s，tail1024 为 37.90–39.68，两轮 decode 时间合计对应约 22% 的描述性吞吐提升。草稿头 prompt history 求值约 0.044–0.047 秒，相比先前 full history 约 0.45 秒；状态前缀提交约 0.055–0.060 秒，相比先前约 0.24 秒。它们来自不同轮次，不能单独归因出精确增益。接受 72/110，接受 0/1/2 的 round 分别为 12/14/29，replay 为 0。整请求两轮均值约 17.59 → 17.17 秒是辅助观测，不能因其改善比例较小而否定 decode 观察；稳定 decode 收益仍待独立时间窗口复测。

随后 256 输出对照仍 exact，MTP 34.63–35.82 vs AR 29.64–32.30 token/s；首轮 AR prefill 和后续 128 复测存在明显漂移（后者 AR 低至 16.65）。保留所有结果，不把漂移后的慢 AR 当作加速证据。发布状态仍为显式实验候选，MTP 默认关闭，后续工作仍聚焦本文链接的 MTP 发布门槛。
