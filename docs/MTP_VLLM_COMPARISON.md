# MTP 性能拆解与 vLLM 借鉴

2026-09-06。本次是重新分析既有原始结果和源码，没有运行新性能测试，也没有改变运行默认值。当前发布门槛仍见 [MTP_RELEASE_CRITERIA.md](MTP_RELEASE_CRITERIA.md)。

用户最新要求是先在业务接口上分离 prefill 与 decode，再独立优化各阶段内核。MTP 主要看 decode 有效 token/s 与 TPOT；prefill 吞吐/延迟、初始化与阶段交接成本分别报告，端到端时间作为辅助指标。本文的性能数字来自旧同步调用。后续已增加同进程独立阶段接口，见 [阶段分离设计](PREFILL_DECODE_SEPARATION.md)；仍未实现独立进程 PD 服务，不能把本文旧数字视为新接口的实测。

## 先区分观察到的加速与发布证据

当前候选并非“解码没有提高 10%”。[tail1024 七场景](../results/mtp-seven-scenarios-tail1024/seven.json)中，11,057 输入 / 128 输出的两轮均值如下。解码计数均为首 token 之后的 127 个 token。

| 时间口径 | AR | MTP depth2 |
| --- | ---: | ---: |
| 首 token 前（混合指标，不等同于纯 prefill） | 13.593 s | 13.894 s |
| 后续生成 | 3.997 s | 3.276 s |
| 后续生成平均 TPOT | 31.47 ms | 25.80 ms |
| 完整调用 wall time | 17.591 s | 17.171 s |

生成吞吐比为 1.220，观察到约 22% 提升；生成耗时省 0.721 s，同轮首 token 前耗时增加 0.301 s，最终净省约 0.419 s。AR 首 token 前阶段约占总时间 77%。这解释了为什么 decode 明显变快，整请求均值只改善 2.4%；后一个百分比不能拿来判定前一个阶段“没有达到收益要求”。约 22% 的 decode 观测仍需独立、稳定的配对复测才能用于该阶段发布。

head history 求值本身约 0.045 s，不能把首 token 前全部 0.301 s 差额归因于它；剩余差额尚未分离为调用路径开销或环境漂移。随后[长输出复测](../results/mtp-tail1024-long-budget/run-ledger.json)出现 AR decode 低至 16.65 token/s 的明显波动。上述均值是描述性结果，不证明稳定阶段收益，也不应把更慢的 AR 轮次挑作加速分母。旧报告未独立测量纯 prefill 及 PD handoff，不能从 TTFT 或总耗时相减构造这两个指标。

后续报告固定拆为三类：主干 prefill 的 input token/s 和处理延迟；decode 的有效输出 token/s、平均 TPOT 及各轮 draft/verify/commit/history 成本；初始化/交接的冷、热耗时与内存。初始化不挪入 decode 来混淆内核对比，也不能从请求总成本中消失；callback、首 token 和 EOS 采用明确且一致的计数规则。

## 当时的 MTP 优化建议（当前已后置）

2026-09-08 用户要求将 MTP 性能放到整体计划后段。下述候选顺序与计时保留为专项历史依据，不代表当前主线下一步；AR 服务、完整状态缓存和基础 prefill/decode 工作按[最新计划](UPSTREAM_ADOPTION_PLAN.md#当前实施顺序2026-09-08调整)先行。已经实现的后续 MoE 探针及负结果仍以对应实验文档为准。

实施顺序先建立独立的 prefill 产物与 decode 输入合同，再用相同完整状态重放 decode，最后逐项替换内核。交接状态包含 GDN、PLE、KV、QSA、MTP head 的历史与位置、pending token 和提交边界；只传 attention K/V 不足以恢复本模型。本机业务分离与合作调度现已实现并验证，见阶段分离设计；下面的 MoE/GDN 顺序是后续 decode 优化优先级，不是新增内核的性能承诺。

同一长输入的 MTP 生成耗时分解（两轮均值，不含 prompt history）：

| 阶段 | 时间 | 占生成时间 |
| --- | ---: | ---: |
| 目标模型验证 | 2.880 s | 87.91% |
| 草稿生成 | 0.236 s | 7.22% |
| 目标状态前缀提交 | 0.057 s | 1.75% |
| 草稿头历史更新 | 0.038 s | 1.16% |
| 未计入以上阶段的间隙 | 约 0.065 s | 约 1.96% |

55 轮生成 127 个 token，平均每轮 2.309 个；每轮目标验证约 52.36 ms，整轮约 59.56 ms，AR 每 token 约 31.47 ms。验证一次虽然覆盖多个 token，却比单 token AR 昂贵，因此不能拿 2.309 直接当加速比。上述计时是阶段 wall time，不是纯 GPU 时间或 DRAM 带宽。

[GPUMoE.swift](../Sources/ANERunnerGPU/GPUMoE.swift) 的验证分支仍逐行执行 routed experts：S3 每层调用三次 `fused.decode`，每次两个主要 Metal 内核，48 层合计 288 次。dense 的 [GPUVerificationLinear](../Sources/ANERunnerGPU/GPUVerificationLinear.swift) 已经共享一次权重读取并分别累积多个 token；GDN recurrence 也在一个内核内推进时间步。因此下一项明确候选是 routed MoE，不能把已合批的 dense/GDN 重新描述为待合批。

1. 先给现有 Q4 gate/up 与 down/reduce 增加 token 轴，S3 每层 6 次调度变为 2 次。保持各 token 的原 S1 加法、BF16 舍入和专家归约顺序；减少的是调度，不是逻辑权重字节。
2. 再测跨 token 的专家重合。S3 有 30 个 assignment，若对应 U 个不同专家，按专家合并后的逻辑可复用比例上限为 `1-U/30`。只有重合足够且组织成本可控才值得共享加载/解码；实际物理流量仍受缓存影响。
3. 在这个局部改动上先跑真实形状的 exact 与耗时对照，通过才替换完整模型候选。不能重新启用此前会改变 token 的普通批量 QMM 来获取虚假的收益。MoE 在整个 verify 中占比尚未单独测量，因此暂不预测最终 token/s。

## vLLM 可以借鉴的部分与边界

官方源码核对固定于提交 `f4eccdadefc6501fafeb1a0bf7f171ff24f984b0`，避免将未来 main 的改动混入本次结论。

- **融合专家计算与中间缓冲复用。** [官方 Fused MoE 设计](https://docs.vllm.ai/en/stable/design/fused_moe_modular_kernel/)把 token 重排、两次专家矩阵乘、激活、还原与加权归约组织在专家实现内，并声明可复用的工作区。这支持先优化上述验证阶段；不需要搬入它的分布式 All2All 层。CUDA/Triton 的实现需要针对本机 Metal、Q4 affine 权重和固定数值顺序重写。
- [固定版本专家内核](https://github.com/vllm-project/vllm/blob/f4eccdadefc6501fafeb1a0bf7f171ff24f984b0/vllm/model_executor/layers/fused_moe/fused_moe.py#L120)按 expert 排列 token，并采用促进 L2 复用的 grouped 执行顺序。S3 小批量的排序/padding 成本与大批量不同，因此先做不排序的 token 轴合并，再评估按专家合并。
- **短序列 GDN 融合。** [固定版本的 MTP GDN 调用](https://github.com/vllm-project/vllm/blob/f4eccdadefc6501fafeb1a0bf7f171ff24f984b0/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py#L1758)传入状态索引、接受长度、输出门和 norm 参数，融合 post-conv 后续工作。这是 MoE 之后可评估的验证阶段方向；本机已在一个内核内递推，潜在工作是把周边操作也合入，而非重新实现时间循环。不同模型状态索引与 BF16 边界须保留当前语义。
- **草稿 token 驻留 GPU。** [固定版本 proposer](https://github.com/vllm-project/vllm/blob/f4eccdadefc6501fafeb1a0bf7f171ff24f984b0/vllm/v1/spec_decode/llm_base_proposer.py#L678)将草稿 token tensor 接到下一步，结束后统一 stack。本机可研究消除每个 draft 的 `ints()`，但主干的 SSD n-gram 查表仍需取得最终草稿 IDs；这不是把整个目标验证移成无需 CPU 的图。
- **按成本选择投机深度。** [官方 Dynamic SD](https://docs.vllm.ai/en/stable/features/speculative_decoding/dynamic_speculative_decoding/)当前按并发 batch size 配置 K，主要解决 `BS*K` 验证工作过大的问题；它并非通用的“按接受率自动调深度”。本机固定单请求不能直接套其并发规则。如将来做深度选择，应按本机不同深度的验证成本和实际接受长度估计每个有效 token 的时间，而不是只追求接受率。
- **保留数值验收。** [vLLM 投机解码说明](https://docs.vllm.ai/en/stable/features/speculative_decoding/)同时列出 greedy equality 测试和浮点/批大小造成的输出差异边界。参考成熟框架不能代替当前候选的固定 AR 兼容合同。

在完成业务边界分离后，decode 内核的优先次序是：批量 routed MoE 的一次调度 → 根据实际专家重合研究读取复用 → 再考虑草稿同步与深度策略。草稿只占约 7%，单独优化草稿无法解决占约 88% 的验证成本。Prefill 的大矩阵、SSD 重叠与吞吐另行评测，不借其时间占比否定 decode 改进。上述是有源码和阶段数据支持的实验方向，尚无独立 PD 服务或这些新内核的实测收益。
