# 本机调度、前缀缓存与 continuous batching 的顺序

2026-09-07，只读源码与已有结果后的判断。**先用已有 PD 接口补一个确定到达时点的公平性实验；MTP 发布门槛满足后，优先做精确系统前缀 checkpoint；真正跨请求 GPU batching 后置。** PD 的下一步很小，缓存才是针对重复 10k 系统提示词的下一项主要开发。正在运行的 MTP 性能窗口不计入本文证据，也不因这份排序跳过其发布条件。

本文补充 [实施计划](../UPSTREAM_ADOPTION_PLAN.md) 与 [精确前缀设计](../EXACT_PREFIX_CHECKPOINT_DESIGN.md)，不替代它们。[早期上游研究](VLLM_SGLANG.md) 的“HTTP 尚未实现”是当时快照；当前 HTTP/SSE 和固定 12 轮短测已完成，仍不等于生产服务、连续批处理或缓存已经实现。

## 排序及实际改动量

| 能力 | 处理方式 | 当前落点与收益边界 |
| --- | --- | --- |
| PD 调度策略与计量 | **可直接吸收设计**；优先补现有探针，不换 kernel | 库默认 wholeStages，HTTP 显式 cooperative；都有同一模型、单执行器。已有 chunk/round、FIFO、背压、取消及原始 callback 时钟。先分清 TTFT、输出途中等待和整组吞吐；不能把重新排序称为权重复用。 |
| 精确系统前缀缓存 | **需本机改造**；下一项主要开发 | 已有明确的 K 边界与私有恢复设计，可以保留现有 generator/scheduler。首版 AR trunk checkpoint，MTP 明确冷 miss。它减少重复 prefill 和对其他 decode 的干扰；首版私有复制不承诺减少每请求 KV 占用或增加准入容量。 |
| continuous batching | **暂缓实现** | 当前能动态进入/退出调度队列，但每次只运行一个请求的完整步骤。真正跨请求 batch 需要让一个模型步骤消费多个独立状态，处理不同长度、位置、EOS、MTP 接受数和释放时点；不是调大 max-connections/max-resident 或把多个请求拼成一个 sequence。先确认长期有多个 ready 请求、真实专家路由重叠及共享权重读取机会，再决定改动。 |

固定 vLLM `6865e67f0be02d53694517f6f71d7fb96492792d` 的 scheduler 在一个 token/input budget 内遍历 running，再处理 waiting，输出按请求记录的 token 数和 block 分配；这与本机一次 `runNext` 只调用一个 backend slice 的接口不同。可吸收“保护已运行请求、限制新 prefill”的策略，不能只移植 Python 队列就获得相同 batching 效果。[固定源码 schedule](https://github.com/vllm-project/vllm/blob/6865e67f0be02d53694517f6f71d7fb96492792d/vllm/v1/core/sched/scheduler.py#L521)

固定 SGLang `2c05ed4e7776c876478f4b2db61acb12b9a27d01` 的 Mamba validator 要求相应状态存在，Full KV 命中可以长于可复用 recurrent 边界，命中后登记请求私有槽及复制来源。适合吸收的是组件一致性和私有恢复，不是照搬 CUDA 内存池。[固定 Mamba validator/恢复](https://github.com/sgl-project/sglang/blob/2c05ed4e7776c876478f4b2db61acb12b9a27d01/python/sglang/srt/mem_cache/unified_cache/components/mamba_component.py#L131)

## cooperative 的吞吐与公平性边界

1. **一步是一份请求，burst 是全局步骤数。** `runNext` 从一个队列取一个 ID，再调用一次 prefillSlice/decodeSlice；未完请求放回 FIFO 队尾。首次发布 prefill 已选出的 token 也占一个 decode 步骤，之后 AR 一步输出一个 token、MTP 一步可能输出多个。因此两个 ready 请求获得相近的步骤机会，不代表相同 GPU 时间或 token 份额。[scheduler:264–348](../../Sources/ANERunnerGPU/QwenLocalScheduler.swift#L264)、[decodeStep:665–694](../../Sources/ANERunnerGPU/QwenGeneration.swift#L665)

2. **416-token prefill 中途不能让出 GPU。** 每步包含整块 forward/evaluate、适用的 MTP prompt history 和 PLE lookahead join；完整步骤结束后才调度其他请求。将 burst4 改为 8 可减少 prefill 插入次数，不能压短已插入的单块停顿。cooperative 在每个边界 join，而 wholeStages 可保留 lookahead 到后续块；不能假定切换没有吞吐成本。[prefillStep:553–588](../../Sources/ANERunnerGPU/QwenGeneration.swift#L553)

3. **resident/ready 限制先于公平性。** 默认 resident=2、ready=2。两个 ready 占满时，新的 prefill 不会开始；一份 partial prefill 与一份 decode 已占两个 resident 时，新到达短请求也不能挤入。已有 partial prefill 会绕过因容量阻塞的新请求继续推进。burst 不解决第三份请求的容量等待；有界队列也不提供以秒计的延迟保证。[limits:15–18](../../Sources/ANERunnerGPU/QwenLocalScheduler.swift#L15)、[prefillIndex:274–285](../../Sources/ANERunnerGPU/QwenLocalScheduler.swift#L274)

4. **逻辑 token 配额与缓存命中分开。** 提交立即预留完整 prompt+maxTokens，queued 请求同样占额度；这不是物理内存预算。前缀命中后 context/API usage 仍是完整 P，不能因跳过 K 的计算就退还 K 的准入额度。完整 checkpoint 还会额外持有状态副本，须计入实际峰值。[submit:231–249](../../Sources/ANERunnerGPU/QwenLocalScheduler.swift#L231)、[配额说明:120–124](../../Sources/ANERunnerGPU/QwenLocalScheduler.swift#L120)

5. **HTTP 的 CPU 工作也是下一次 GPU slice 的前置成本。** 网络队列独立，但 worker 每轮先做最多一份 render/tokenize/submit，再调用 runNext；同线程 token callback 还做 UTF-8 和 JSON 编码。CPU/网络不会因此自动和下一 GPU 步骤并行。库探针用既有 token IDs 和轻量 callback，可先隔离调度政策；HTTP 的到达/发送测量另算。[worker:421–468](../../Sources/ANERunnerCLI/GPUHTTPServer.swift#L421)

6. **跨请求 batch 的缺口在模型状态接口。** `QwenModel.forward` 接收一份 `[Int32]` 与一个 State，embedding 明确建 `[1,n]`；State 只有一份 offset/sessionIdentity。GDN 和 Attention forward 都检查 batch=1，QSA 使用该请求的单一 offset/positionBase，PLE hash 则依赖本请求历史。MoE 通用入口虽可展平 batch×sequence，但这不补齐上述状态隔离；其特定 fused-prefill 路径也限定 batch=1。[Model:9、295、343](../../Sources/ANERunnerGPU/QwenModel.swift#L295)、[GDN:167](../../Sources/ANERunnerGPU/GPUGatedDeltaNet.swift#L167)、[Attention:131](../../Sources/ANERunnerGPU/GPUAttention.swift#L131)、[PLE:123–130](../../Sources/ANERunnerGPU/GPUPLE.swift#L123)、[MoE:254–268](../../Sources/ANERunnerGPU/GPUMoE.swift#L254)

这说明当前没有新增的跨请求权重读取合并。未来 batching 可能提高共享投影和重叠专家的权重复用，但收益取决于真实路由、状态搬运和形状；本轮没有测到这类收益，也没有必要为单个重复系统提示的客户端先重做全部状态/算子接口。

## 已有实测支持到哪里

| 已有证据 | 可用结论与限制 |
| --- | --- |
| 最初 cooperative 实模及完整 IDs | 长 AR / 短 MTP 同时排队时，短 TTFT 从 20.322 s 降到 1.679 s；但其 callback p95 从约 60 ms 增到约 613 ms，组墙钟从 22.048 s 到 22.305 s。改善首响应和完成顺序，未证明吞吐增加或全过程平滑。每模式只有一组。 |
| 后续 burst4→8，各 18 项通过 | 长 AR/短 MTP 组合中短完成 11.341→7.317 s，callback p95 947.8→758.4 ms；活动短 decode 中插入长 prefill 次数 8→4。与此同时 wholeStages 对照自身从 42.321→30.908 s，不能把所有时间变化归因于 burst。默认仍为 4，未做可信反序性能验收。 |
| HTTP 19+15 项及 12-cycle/46 项短测 | 已有断连、decode 后恢复、限额、接收期限和重复请求的文本/usage 证据；短测约 13 分 29 秒，31 完成+12 取消。它不证明随机到达公平性、缓存收益、batching 吞吐、完整 IDs 或长期稳定。 |

前两行已核对 [原始 cooperative 汇总](../../results/cooperative-pd-v1/summary.json)、[burst 原始报告与时钟复算](../../results/upstream-cost-latency-v1/latency-summary.json) 和 [指标合同](../SCHEDULER_LATENCY_EXPERIMENT.md)。现有探针在 pump 之前依次提交长、短两份请求；不是“已活跃 decode 后才到达长 prompt”。[实际提交位置](../../Sources/ANERunnerCLI/GPUCooperativeSchedulerProbe.swift#L184)；HTTP/短测范围见 [服务验收](../HTTP_SERVER_EXPERIMENT.md)。

## 仅建议两个后续实验

### 1. 已开始 decode 后再到达长 prefill

只扩展现有 cooperative probe 的到达时点，不改 scheduler/kernel：选择短提示、已冻结完整输出且实际至少生成64 tokens的请求，在第8个callback记录11k长请求到达；当前slice返回后再submit，避免从callback重入scheduler，并分别记录到达/提交时刻。保留相同416分块、数值配置、两份resident和输出预算。先固定AR消费者；MTP若已完成发布门槛，可用一个另列的固定配置复核，不能混合统计。

使用已热身的独立参考，预定 **4→8→8→4** 四组。记录完整 IDs、submit/step/callback 时钟、prefill 活动秒数、decode compute 秒数、消费者到达后剩余完成时间、burst-gap p95/max、生产者 TTFT/完成以及组墙钟。必须从事件证明新 prefill 确实插在消费者第 8 个 token 与终态之间；预取/取消后状态和所有额度归零。

局部筛选建议：每组数值/生命周期先全部通过；两个配对方向的消费者剩余完成时间均改善至少 15%，组墙钟不恶化超过 3%、长请求 TTFT 不恶化超过 10%，max gap 不出现可重复超过 5% 的回退，才考虑将 8 纳入可选调度配置。这些是**拟定的实验判据，不是现有 SLO**。若时间漂移掩盖结果或权衡不成立，保留 4；若 max gap 仍由单块耗时决定，不再扩大 burst 扫参声称能解决它，也不在此实验偷偷改 chunk。

### 2. 单条 AR 精确 checkpoint，两个后缀 A/B/A

按既有设计，仅增加同模型实例的一条不可变缓存及私有恢复，默认关闭；实施依赖仍服从 MTP/生命周期门槛。用完整 chat 编码逐 ID 核对系统前缀，取 `K=416*floor(S/416)` 且 `0<K<P`，在真实冷 prefill 的 K 边界保存。完整 GDN/conv、Attention KV/QSA、PLE conv/hash 都恢复；prefetch 只接收 `tokens[K...]`，其 initialOffset 为 K，不能把整份 tokens 重送。MTP 请求整体冷 miss，不复用 AR checkpoint 后宣称获得了 MTP 缓存。

先冷 A/B，再命中 A/B/A；改变后缀首 token 与长度，穿插一个副本推进后取消。K 处全部持久 tensor/整数 history/offset/nil、最终 logits 和完整输出 IDs 对照；checkpoint 与另一副本不可被污染。复用设计中的小 QSA 边界用例 K=1664、P=2051/2053，不另造通用缓存框架。

性能判据沿用设计：真实约 10k 命中 **TTFT 至少下降 50%**，decode 无可重复超过 3% 的回退；保存/恢复和全部复制成本必须实记，逻辑 payload 与 MLX 实际峰值分开，prefill 吞吐分子只用实际执行的 P−K。首版不能改变完整 prompt 的 API usage/context/准入配额。MTP head 的历史起点依赖完整 P，h[K−1] 还会与新 token[K] 配对；原 chunk 的 trunk tail 重建是后续独立能力，不属于这次 AR 成功的外推范围。[MTP 实际消费关系](../../Sources/ANERunnerGPU/QwenMTPDecoder.swift#L76)、[完整设计与门槛](../EXACT_PREFIX_CHECKPOINT_DESIGN.md)

本次仅新增研究文档，没有运行新模型或修改源码/配置。外部复核只读取上述固定 upstream commit；没有复制实现。vLLM/SGLang 对应来源为 Apache-2.0，若未来移植代码仍须按具体文件保留版权/NOTICE 和修改记录。
