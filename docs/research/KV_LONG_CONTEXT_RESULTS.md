# 独立 Runner 长上下文与 RAM 前缀缓存结果

2026-09-14。**当前已完成32K数值筛选，以及64K和完整262144 reference 冷/热 RAM 缓存验证；真实HTTP长上下文验收仍为 PENDING。** `fusedQSA` 的32K跨模式数值门槛失败，默认继续使用 `reference`。本页汇总同一增量的证据和后续优先级，不把准备好的代码、脚本或静态分析当成实测通过。

这里的模型上限是 **262144 tokens（256 Ki tokens）**，指完整渲染提示词与请求输出预算之和，**不是265000 tokens**。缓存命中不减少该逻辑长度。合成重复文本用于容量与状态一致性检查，不代表真实长文、工具或coding-agent业务质量。

服务与缓存合同沿用[HTTP入口](../HTTP_SERVER_EXPERIMENT.md)、[缓存能力](../KV_CACHE_CAPABILITIES.md)、[可靠性](../KV_CACHE_RELIABILITY.md)和[运维说明](../KV_CACHE_OPERATIONS.md)。[既有HTTP物理页短验收](KV_PAGED_HTTP_RESULTS.md)绑定其自己的二进制与窗口，不能替代本页长上下文验证。

## 增量与固定版本

本轮发布候选包含七个Swift文件：

| 文件 | 增量范围 |
| --- | --- |
| `Sources/ANERunnerCore/QwenHTTPServiceCapacity.swift` | 统一 context、请求体、连接期限和 scheduler 额度校验；启动前及加载模型后复核上限 |
| `Sources/ANERunnerCLI/GPUHTTPServer.swift` | 将显式容量配置接入 parser/request/scheduler，长上下文服务限制为AR，终态提供实际offset及独立prefill/decode字段 |
| `Sources/ANERunnerCLI/GPUGeneration.swift` | `tokenize --prompt-file`，避免大型真实chat fixture受argv限制；与`--prompt`互斥 |
| `Sources/ANERunnerCLI/GPULongContextProbe.swift` | 独立cold/warm全121状态、有限值、实际decode与allocator观察 |
| `Sources/ANERunnerCLI/CLI.swift` | 新入口分派与参数帮助 |
| `Tests/ANERunnerCoreTests/QwenHTTPLongContextCapacityTests.swift` | 容量、body及期限参数的CPU边界 |
| `Tests/ANERunnerGPUTests/QwenLongContextSchedulerTests.swift` | 无设备工作的真实scheduler准入、额度及超限边界 |

32K、64K、262144和本轮CLI负控使用相同release二进制SHA256：

```text
da457be460df0e5158b71f4f9d1e3c924f4616e25f33dd1bbfbe888e9a741e00
```

本轮 **182项相关Swift CPU测试、9项CLI负控通过**。CLI负控使用不存在的模型路径，实际在载入权重前拒绝非法context/额度/resident/body/deadline/attention参数，以及缺失或冲突的tokenize输入。CPU通过不等于262K设备容量或HTTP响应通过。

原始证据在 `results/long-context-screen32-v1/{cpu.log,cli-validation.json,run-ledger.json,plan.json,postflight.json}`。32K后核对602个冻结文件、102个模型payload stat，64K后核对599个冻结文件、102个payload stat，均无变化；不同文件数来自各自冻结清单，不合并为同一清单。这些窗口均恢复参考服务的精确原argv，并核对idle与MTP/drafter关闭。262144后再次核对599个冻结文件及102个payload stat，无变化；参考服务恢复后PID为66360，仅代表该次postflight。模型payload采用stat复核，不能扩大称为全部权重内容SHA复核。

## 已完成结果

三个上下文窗口均为完整48层模型，chunk416/eval4、AR、dense reference KV、24GiB state逻辑预算、8GiB RAM缓存；未启用MTP或物理页模式。O2实际产生两个IDs，其中首输出属于prefill，随后执行一次真正decode。按真实生成器规则，prefill body到P−1，再执行最后S1；所有缓存检查点留在原416网格。

| 场景 | P / O | RAM检查点 / warm实际prefill | 输出与状态 | 结论 |
| --- | --- | --- | --- | --- |
| 32K reference | 32766 / 2 | 32448 / 318 | cold/warm IDs均为`[16,11]`，最终offset32767；B、P、P+1处各121状态/host完全一致且有限，每次1次实际decode | 该模式的冷/热容量检查通过 |
| 32K fusedQSA | 32766 / 2 | 32448 / 318 | 该模式自身cold/warm与输出同样完全一致 | 自身复用通过，跨模式筛选失败 |
| 64K reference | 65534 / 2 | 65312 / 222 | cold/warm IDs均为`[16,11]`，最终offset65535；6组状态、726个张量记录一致且有限，每次1次实际decode | 完整报告通过，独立6105项检查通过 |
| 262144 reference | 262142 / 2 | 262080 / 62 | cold/warm IDs均为`[16,11]`，B/P/P+1全121状态/host一致且有限，每次1次实际decode，最终offset262143 | 完整报告通过，独立6579项检查、0失败 |

32K跨模式在相同P比较121张量：**63项满足预先声明的`RMSE <= 0.02 + 0.02 * reference RMS`，58项失败，仅11项逐位相同**。总体报告为`complete=true, passed=false`，进程非零退出；没有调宽阈值，也不能因为两个输出IDs相同而接受fused策略。该结论与[早期QSA融合实验](../QSA_PREFILL_FUSION.md)的证据边界一致。

原始报告分别为 `results/long-context-screen32-v1/model.json`、`results/long-context-reference64-v1/model.json` 与 `results/long-context-reference262-v1/model.json`；独立记录为各目录下的 `independent-agent-analysis-v1.json`。独立检查重新配对完整hash/host、输出和阶段计数，并从原始样本复算allocator峰值；它没有重新执行模型。32K未保留fused张量的原始payload，因此独立记录检查的是已记录RMSE与门槛，不声称从双方原始张量重算了全部误差。

以下只记录带诊断观察器的阶段墙钟，扣除观察器区间；不是无观察器的吞吐验收，也不从一次decode换算稳定tokens/s：

| 场景 | 冷prefill / warm实际suffix prefill | 冷decode / warm decode（各一次） |
| --- | ---: | ---: |
| 32K reference | 47.911s / 1.509s | 0.930s / 0.045s |
| 32K fusedQSA | 43.484s / 1.096s | 0.934s / 0.040s |
| 64K reference | 158.220s / 1.436s | 0.884s / 1.034s |
| 262144 reference | 2454.232s / 3.115s | 1.034s / 1.016s |

64K的业务allocator prefill峰值为83842793714字节（cold）和83702683508字节（warm），包含已驻留模型；它不是进程RSS或stateBudget上限。诊断复制峰值另列，均不冒充物理DRAM流量。上述完成窗口最终request/cache/workspace及lease均归零，不能据此断言任意压力、超时或长期运行已经合格。


262144冷prefill活动墙钟为 **2454.232353087s（约40.9分钟）**，warm实际62-token suffix为 **3.114880958s**，报告中的warm恢复时间为 **0.274384875s**。cold/warm业务allocator峰值分别为 **95510718306B / 95261783216B**，诊断观察器分别花费 **14.62464s / 14.15351s**，与业务阶段分开记录。最终stateBudget各项及lease均为0。这证明本次完整模型容量、RAM复用和至少一次真实decode；O2不证明持续长上下文decode吞吐，合成fixture不证明业务质量，CLI成功也不等于HTTP请求期限或响应合同通过。RAM共享恢复候选尚未应用，因此0.274s不能归功于该候选。

## 262144 CLI通过；HTTP仍PENDING

默认服务配置保持context16384、预留32768 tokens、2个resident、256KiB body、300秒连接期限。准备验证的显式长上下文profile为context/reserved tokens262144、1个resident、8MiB body、3600秒连接期限、24GiB state逻辑预算、8GiB RAM缓存，`reference`、AR，SSD关闭。stateBudget不计入全部权重、attention激活、HTTP/tokenizer副本和MLX allocator缓存；配置算术可容纳状态不等于实际内存安全。

| 验收门槛 | 状态 | 已有证据或待登记项 |
| --- | --- | --- |
| CLI P262142/O2 cold + RAM warm | **通过** | B262080、warm实际suffix62；B/P/P+1全121状态/host、完整IDs、有限值、每次实际1次decode、offset262143与最终资源账目通过；独立6579项检查 |
| HTTP同边界cold JSON + warm完整SSE | **PENDING** | 原生chat分词与请求body绑定、usage/actual offset、RAM来源、单一终态和完整SSE结束；不能仅以HTTP200通过 |
| HTTP边界拒绝与恢复 | **PENDING** | P262143/O2返回400；body超限413、长context MTP拒绝；无新模型准入/终态及稳定资源计数的限定证据 |
| HTTP P262112/O32，取消前JSON与取消后SSE | **PENDING** | 两次B262080命中、实际prefill32、实际decode31、offset262143，完整正文/finish/usage一致；HTTP不暴露IDs，且此新prompt没有独立cold oracle |
| 取消、资源与独占服务清理 | **PENDING** | 独立约32K prefill取消、短请求恢复、保留长缓存再用、请求/flight/claim清理与统一40秒owned-process清理；不扩大称为262K中途取消或耐久通过 |

物理页配置若另行启用，当前native逻辑上限131072 tokens；更长请求必须在设备写入前选择整游标dense fallback，并核对没有长前缀native导入。无需先扩大页池，才能验证262144的dense服务。现有2GiB SSD归档上限及旧式全量Data路径仍保留；本页不声称支持262K SSD持久化或恢复。

## 后续三项优先级

1. **先完成reference容量与RAM可靠性。** CLI容量已通过，继续收齐上述HTTP证据，再讨论显式profile的使用范围。对已压紧的RAM快照，可试只在restore时共享48个immutable attention Tensor引用，GDN/PLE仍私有复制、新session identity；保存、SSD及预算合同不变。必须单独对照全121状态、分支/clear/取消、恢复时间与真实峰值；capacity首次有效写入必须换入私有buffer。该候选尚未应用或实测，不能宣称warm已省掉约7GiB复制，也不能忽略suffix concat仍可能复制整个前缀。[现有完整状态与缓存合同](../KV_CACHE_RELIABILITY.md)优先于性能调整。
2. **再减少长prefill实际稀疏算术。** 保留现有QSA selector/top512四token块及0–3个因果尾，直接按query gather至最多2051行，按时间升序、validity mask及GQA共享执行QK/softmax/PV。262144/2051约127.8倍仅是这两次矩阵乘的算术减少上限；selector、MoE/GDN/PLE、KV复制、gather和同步成本仍在。Q8的K/V gather有效载荷约33.6MB，但整块416的惰性图可能累计约1.75GB，必须用真实求值与owner边界限制暂存。先核对相同selector成员/score/权重/输出，再过32K/64K已有数值与cold/warm门槛，最后才测无观察器prefill；不绕过本轮58项失败。[阶段分离](../PREFILL_DECODE_SEPARATION.md)与[attention存储边界](KV_ATTENTION_STORAGE_DESIGN.md)继续适用。
3. **最后扩展大SSD流式读写。** 先在现有大小限制内建立单缓冲（起步8MiB）、ack/背压、FD/epoch/SHA及预算所有权，再做小上下文121状态roundtrip；通过后才允许新路径的大归档。完整目标State仍需预约约7GiB，额外host staging才应随chunk有界；把文件上限与在途buffer额度分开，禁止只是把1MiB读取循环追加到全量Data。最终校验前不发布半状态，取消/clear/超时后实际IO仍持有FD和lease；保留`fsync→rename→目录fsync`发布与旧格式隔离。大文件写成功、RAM命中或增大配置都不构成262K SSD恢复通过。[SSD队列](KV_SSD_QUEUE_DESIGN.md)和[超时所有权](KV_SSD_TIMEOUT_VALIDATION.md)是必须保留的合同。MTP继续排在这些工作之后。

32K精度链的静态线索是：reference会落地BF16 QK score和BF16 softmax概率，而M5 fused路径以不同的FP32 split-D/在线softmax归约计算，后续QSA与MoE离散选择可能放大差异；当前未找到具体mask/GQA/缩放错误，也未证明全部误差只是可接受的舍入。最小单层诊断候选将对同一次真实layer3 Q/K/V/mask比较reference、native fused及FP32-score路线，并加相同query分组的BF16控制、原始字节和因果tail隔离。它**尚未运行**，不作为本页通过项，也不修改生产默认或放宽阈值。
