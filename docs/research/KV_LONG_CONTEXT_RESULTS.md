# 独立 Runner 长上下文与 RAM 前缀缓存结果

2026-09-14。**当前已完成32K数值筛选、64K及完整262144 reference CLI冷/热RAM验证，以及RAM恢复优化版本的真实262144 HTTP容量、缓存复用和取消恢复验收。** `fusedQSA` 的32K跨模式数值门槛失败，默认继续使用 `reference`。本页汇总同一增量的证据和后续优先级，不把准备好的代码、脚本或静态分析当成实测通过。

这里的模型上限是 **262144 tokens（256 Ki tokens）**，指完整渲染提示词与请求输出预算之和，**不是265000 tokens**。缓存命中不减少该逻辑长度。合成重复文本用于容量与状态一致性检查，不代表真实长文、工具或coding-agent业务质量。

服务与缓存合同沿用[HTTP入口](../HTTP_SERVER_EXPERIMENT.md)、[缓存能力](../KV_CACHE_CAPABILITIES.md)、[可靠性](../KV_CACHE_RELIABILITY.md)和[运维说明](../KV_CACHE_OPERATIONS.md)。[既有HTTP物理页短验收](KV_PAGED_HTTP_RESULTS.md)绑定其自己的二进制与窗口，不能替代本页长上下文验证。

## 增量与固定版本

`da457…`容量基线的增量包含七个Swift文件；后续RAM/SSD组合版本在下文单列：

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

容量基线的 **182项相关Swift CPU测试、9项CLI负控通过**。CLI负控使用不存在的模型路径，实际在载入权重前拒绝非法context/额度/resident/body/deadline/attention参数，以及缺失或冲突的tokenize输入。CPU通过不等于262K设备容量或HTTP响应通过。

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


262144冷prefill活动墙钟为 **2454.232353087s（约40.9分钟）**，warm实际62-token suffix为 **3.114880958s**，报告中的warm恢复时间为 **0.274384875s**。cold/warm业务allocator峰值分别为 **95510718306B / 95261783216B**，诊断观察器分别花费 **14.62464s / 14.15351s**，与业务阶段分开记录。最终stateBudget各项及lease均为0。这证明本次完整模型容量、RAM复用和至少一次真实decode；O2不证明持续长上下文decode吞吐，合成fixture不证明业务质量，CLI成功也不等于HTTP请求期限或响应合同通过。该`da457…`基线未应用RAM共享恢复，因此0.274s不能归功于后续优化。

## 262144 CLI与真实HTTP边界、RAM恢复：通过

默认服务配置保持context16384、预留32768 tokens、2个resident、256KiB body、300秒连接期限。本次HTTP实测采用的显式长上下文profile为context/reserved tokens262144、1个resident、8MiB body、3600秒连接期限、24GiB state逻辑预算、8GiB RAM缓存，`reference`、AR，SSD关闭。stateBudget不计入全部权重、attention激活、HTTP/tokenizer副本和MLX allocator缓存；配置算术可容纳状态不等于实际内存安全。

| 验收门槛 | 状态 | 已有证据或待登记项 |
| --- | --- | --- |
| CLI P262142/O2 cold + RAM warm | **通过** | B262080、warm实际suffix62；B/P/P+1全121状态/host、完整IDs、有限值、每次实际1次decode、offset262143与最终资源账目通过；独立6579项检查 |
| HTTP同边界cold JSON + warm完整SSE | **通过** | P262142/O2，正文均为`1,`；warm从RAM复用B262080，仅计算62，实际decode1、offset262143，usage/终态/完整SSE一致 |
| HTTP边界拒绝与恢复 | **通过** | P262143/O2明确400、声明body超限413、长context MTP请求400；准入账目/cache counters稳定且终态日志排空，无新增模型终态；不是全GPU tracing |
| HTTP P262112/O32，取消前JSON与取消后SSE | **通过** | 两次B262080命中、实际prefill32、decode31、offset262143，完整正文/finish/usage一致；HTTP不暴露IDs，此新prompt没有独立cold oracle |
| 取消、资源与独占服务清理 | **通过** | 独立约32K prefill的RST对应唯一cancelled终态，短请求恢复完全一致，随后仍可复用长缓存；request/workspace/flights/queues归零，server/client/controller退出0，统一40秒清理未用kill |

本次HTTP使用后述`5b1653ce…`组合binary，独占测试端口11249/PID70501；wrapper于2026-09-13 19:45:41 UTC完成。6个正常请求和1个取消请求、22项客户端门槛全部通过。原始请求body、tokenization fixture、JSON/SSE、7组唯一模型终态及资源快照位于 `results/ram-restore-http262-v3/http/`；独立记录 `independent-http-audit.json` 实际重解析原始响应、fixture count/LCP262080、wire hash、offset、取消后全文及清理，通过后才确认上述结果，并非仅信任22个passed布尔。该独立审计没有重新运行GPU。

| HTTP场景 | 缓存 / 实际prefill / 实际decode | prefill active / decode / 请求墙钟 |
| --- | --- | --- |
| P262142/O2 cold JSON | 0 / 262142 / 1 | 2191.364s / 0.178s / 2193.000s |
| 同prompt warm SSE | 262080 / 62 / 1 | 2.778s / 0.390s / 4.655s |
| P262112/O32 JSON | 262080 / 32 / 31 | 2.818s / 8.762s / 13.031s |
| 取消及短请求恢复后，同O32 prompt SSE | 262080 / 32 / 31 | 2.923s / 8.080s / 12.342s |

O2 warm实际恢复字段为0.005057750s；这不是与旧CLI 0.274s在相同观察器、阶段和profile下的受控A/B，不给RAM共享归因百分比。请求墙钟含网络/分词和调度，prefill active与decode来自匹配的原始terminal，分别列示；两个O32短样本不等于稳定decode吞吐验收。HTTP没有返回token IDs或121个state hash：O32仅证明完整正文、finish与usage重复一致，不能扩大称为新prompt对独立cold模型的全状态/IDs正确性。

最终ready/idle快照的request/workspace、queues、reserved tokens及liveFlights均为0，**仍保留1份7506284552B的有效RAM cache及1个cache lease**，不是总budget0。缓存published1、hits/restoredHits3、restoreFailures0，无SSD或paged arena。state逻辑ledger峰值22522490904B，MLX累计allocator峰值95509985760B，allocator cache256MiB；不等于进程RSS。日志drops/write_failures为0，client/server/controller均退出0，清理未用kill。postflight核对610个冻结文件和102个模型payload stat，无变化；参考PID73579、精确原argv、idle及MTP/drafter关闭均通过。413另由保存的错误body和原始server唯一rejection_status413/simple_sent核实，wire status/headers未单独归档，不夸大该项原始证据范围。

每30秒只读sidecar共有73份有效进程样本，覆盖2161.563s，采样footprint最大94416527136B；74份thermal读取均为fair，最后一次进程读取在正常退出后得到ESRCH并停止。这不是连续峰值、阶段归因、DRAM字节、GPU频率或带宽利用率证据。原始侧采和摘要为 `telemetry.jsonl` / `telemetry-summary.json`。

报告SHA256：`http/client/summary.json` 为 `754f80362034213bbe9293d9f1ebe54caf753f2c9bc3036f09f725a1f304e331`；独立审计为 `03095bed0299bd1c5cfc7efaa1c671634c24498656e17c1bfef5de237111b3d9`。这是一轮合成容量/缓存/传输可靠性验证，不代表真实长文质量、262K中途取消、实际内存压力或耐久通过。

物理页配置若另行启用，当前native逻辑上限131072 tokens；更长请求必须在设备写入前选择整游标dense fallback，并核对没有长前缀native导入。无需先扩大页池，才能验证262144的dense服务。现有2GiB SSD归档上限及旧式全量Data路径仍保留；本页不声称支持262K SSD持久化或恢复。

## RAM恢复优化：短回归与HTTP通过，CLI全状态对照待补

这是前述`da457…`容量基线之后的独立增量。RAM恢复改为只在已压紧的私有cache Snapshot上调用内部`forkCompactRAMPrefixState`：新请求创建新session identity，共享48个immutable attention K/V/QSA Tensor引用，73个GDN/PLE张量仍私有复制。保存与发布仍执行完整私有拷贝；public fork、SSD归档格式和全部request/cache/workspace预算不因共享而减少。该入口拒绝空状态及paged状态，不能用shape/stride相似就把任意外部或capacity state当作compact来源。

旧cache或request持有的Tensor引用必须跨异步求值存活。Reference suffix prefill创建新concat输出；capacity256不沿用源cache的capacity owner，有padding时建立新backing，整容量边界可能暂时别名，但下一次有效append必须先grow成私有存储。M2页附件继续沿用自己的page/claim所有权。按B262080形状计算，本改动可避免restore时约6.936GiB的attention gather，仍需约56.3MiB GDN/PLE私有副本；这只是预期减少的复制payload，**不是已测得的RSS、峰值或耗时收益**，随后suffix concat仍可能复制完整KV。

RAM修改、SSD导入限额窄修复及其探针采用同一组合release二进制：

```text
5b1653ce1c62ff9d86f29e76566cc66a637f157f119303b967181df55dc45ac2
```

**183项相关Swift CPU测试通过。** 该组合版本已完成以下真实短回归，不能与先前182项简单相加：

| 回归 | 实际状态与历史对照 | 当前结论 |
| --- | --- | --- |
| capacity256 / RAM恢复 / 预算回退 / 取消 | 74组状态、8954个BF16张量记录；74组完整host与已接受历史报告严格一致，输出与同轮reference/历史对应结果一致；最终budget/lease0 | 通过，独立记录复核无失败 |
| M2真实paged prefix cache | 133组状态、16093个BF16张量记录、14个完成trial；同轮冷参考及不同suffix oracle一致，历史对应state/host/输出严格一致 | 通过；物理page IDs与时间不作跨运行相等要求 |
| 优化后262144 CLI全121状态cold/warm及对baseline比较 | `results/ram-restore-reference262-v1`（计划） | **PENDING**：HTTP已经通过，CLI诊断全状态重放与受控性能对照仍须单独完成；不能沿用旧binary的baseline通过 |
| 优化后的真实262144 HTTP | `results/ram-restore-http262-v3/http` | **通过**：匹配最终binary的原生chat fixture，cold/warm、O32、取消/恢复、边界拒绝与资源合同完成；HTTP不暴露全121 state hash |
| 合法大于1GiB的SSD实模恢复 | `results/ram-restore-short-v1/ssd.json` | **通过**：7组状态/847张量与历史64K cold匹配，实际归档121个payload hash独立一致，详见下节 |

短回归原始数据为 `results/ram-restore-short-v1/capacity.json`、`paged.json`、`cpu.log` 和 `run-ledger.json`；独立记录为 `independent-capacity-analysis-v1.json`、`independent-paged-analysis-v1.json`、`independent-paged-historical-comparison-v1.json`。独立工作重新配对已记录hash/host/完整输出，未重跑GPU。两类短回归支持本改动没有破坏这些既有分支和回收路径；短回归本身不能证明优化后262K全121状态、实际内存压力或吞吐已经合格；长HTTP另由前节真实验证支持。短回归整批完成后，`postflight.json`核对572个冻结文件和102个模型payload stat，无变化；参考服务以PID69588恢复，精确原argv、idle及MTP/drafter关闭均核实。

前两次HTTP未完成的来源均保留：v1在启动前正确拒绝旧`43de3b6e…`分词fixture与最终`5b1653ce…`binary不一致，wrapper退出2，未启动测试server。重新分词后的v2通过三项400/413边界，随后客户端记录`event("request_start", name=...)`与helper形参`name`冲突，抛出`TypeError: event() got multiple values for argument 'name'`，尚未开始正常推理请求。修复仅将记录helper形参改为`event_name`，没有修改Swift运行时或放宽门槛。v2 client退出1、server退出0并恢复参考服务；不能记为模型推理失败或完整验收通过。最终v3使用同一`5b1653ce…`binary和匹配fixture，通过完整流程。v1和v3均记录610文件/102模型stat不变，分别绑定各自postflight。

## 合法1–2GiB SSD导入回归：通过

旧cache恢复调用沿用模型import的1GiB默认值，与已显式配置并获准读写的1–2GiB归档不一致，可能把有效文件当恢复失败并安排失效。窄修复仅传入当前模型与checkpoint offset推导的确切logical payload上限，保留全部descriptor/shape/offset校验和2GiB绝对上限；不提高默认512MiB pending额度，也不实现流式读写。

```sh
.build/release/ane-runner probe-gpu-large-ssd-import \
  --model-dir /absolute/path/to/model \
  --tokens-file /absolute/path/to/TOKENS_65534.json \
  --cache-dir /absolute/path/to/NEW_PRIVATE_DIRECTORY \
  --output /absolute/path/to/NEW_REPORT.json
```

本次实际使用上述`5b1653ce…`组合binary、P65534/O2/B65312、reference AR、chunk416/eval4；同一48层模型，12GiB state逻辑预算、2GiB RAM、2GiB SSD pending、4GiB磁盘额度。初始空RAM及空专用store产生cold oracle并写SSD；真实IO/回调释放后清RAM、销毁首generator，再由另一个空RAM generator恢复同一归档。原始报告为 `results/ram-restore-short-v1/ssd.json`，独立记录为 `independent-ssd-analysis-v1.json`，**14055项检查、0失败**。

| 实测项目 | 结果 |
| --- | --- |
| Checkpoint大小 | logical1914925064B，实际tensor payload1914925056B，实际`.qpc`文件1915273929B，磁盘计费1915277312B；严格大于1GiB、小于2GiB。约1.915GB不是1.915GiB |
| 输出与真实执行 | 两次完整IDs均为`[16,11]`、callback一致、length结束；每次实际decode1，最终offset65535 |
| 强制SSD恢复 | 第二generator初始RAM空；cacheSource=`disk`，cached65312、computed/actual-prefill222、diskHits1，restoreFailures/diskFallbacks均0 |
| 完整混合状态 | 7组anchor、847个BF16张量记录及逻辑host在B/P/P+1匹配cold，并严格对齐先前reference64 cold；不是只比较最后输出 |
| 实际归档独立校验 | 独立CPU程序以1MiB有界读取现存`.qpc`，复核manifest/metadata/payload摘要及全部121个张量payload hash；它们与reference64 cold B65312一致 |
| 归档保留与IO账目 | 仍保留1条归档；published1、write计数1915273929B、read计数1915277312B；corruptions/writeFailures/evictions均0。read计数采用store磁盘计费口径，不当作精确物理IO字节 |
| 最终资源与关闭 | 两次清RAM后及最终request/cache/workspace/currentLeases均0；pending jobs/bytes/read intents均0；同一30秒close内IO、callbacks与completed全为true |

保存文件payload SHA256为`2706d5f34232cc158b3274c6b7faadf0426316d49554a0b2da660b74474eda95`；独立审计读取实际归档，超出了只相信API报告或内存里state hash的范围，但没有重新执行GPU。全部文件长度、逻辑payload、磁盘计费与allocator数据分别记录，不能互相替代。

**性能解释受配置限制。** 此SSD探针未像旧64K baseline那样把MLX allocator cache限为256MiB；实际cold/warm观察到的cache分别为28472652548B/28472800004B（约28.47GB），累计allocator peak83826949872B。带诊断的cold/warm请求墙钟为303.154s/6.794s，恢复字段0.322210125s；不将这些数值与旧baseline直接横比，不据此宣称RAM共享更慢、SSD更快或RSS下降。state逻辑ledger峰值9587421224B、最终归零，也不能代表该allocator驻留已释放。

本项通过的是现有全量Data路径、同进程同model的新generator恢复，不能改称跨重启、262K SSD、持续decode性能或流式支持。SIGINT/SIGTERM为协作取消，正在执行的GPU/系统调用不会被强行打断；该probe具备有界取消清理分支，但本次成功运行没有执行取消故障注入。HTTP已按前节完成有限窗口验收；优化后262K CLI全121状态及受控baseline性能对照仍待补。

## 后续三项优先级

1. **先巩固reference容量与RAM可靠性。** CLI容量和上述有限HTTP窗口已通过，下一步补优化后CLI全121状态对照、真实业务与更长运行窗口；工业发布目标仍包含262144总窗口与前缀缓存。对已压紧的RAM快照，已按上述增量只在restore时共享48个immutable attention Tensor引用，GDN/PLE仍私有复制、新session identity；保存、SSD及预算合同不变。必须单独对照全121状态、分支/clear/取消、恢复时间与真实峰值；capacity首次有效写入必须换入私有buffer。该修改已进入上述组合版本并通过短回归与262144 HTTP；CLI全状态对照及实际复制/峰值/恢复时间收益仍待单独比较；不能忽略suffix concat仍可能复制整个前缀。[现有完整状态与缓存合同](../KV_CACHE_RELIABILITY.md)优先于性能调整。
2. **再减少长prefill实际稀疏算术。** 保留现有QSA selector/top512四token块及0–3个因果尾，直接按query gather至最多2051行，按时间升序、validity mask及GQA共享执行QK/softmax/PV。262144/2051约127.8倍仅是这两次矩阵乘的算术减少上限；selector、MoE/GDN/PLE、KV复制、gather和同步成本仍在。Q8的K/V gather有效载荷约33.6MB，但整块416的惰性图可能累计约1.75GB，必须用真实求值与owner边界限制暂存。先核对相同selector成员/score/权重/输出，再过32K/64K已有数值与cold/warm门槛，最后才测无观察器prefill；不绕过本轮58项失败。[阶段分离](../PREFILL_DECODE_SEPARATION.md)与[attention存储边界](KV_ATTENTION_STORAGE_DESIGN.md)继续适用。
3. **最后扩展大SSD流式读写。** 先在现有大小限制内建立单缓冲（起步8MiB）、ack/背压、FD/epoch/SHA及预算所有权，再做小上下文121状态roundtrip；通过后才允许新路径的大归档。完整目标State仍需预约约7GiB，额外host staging才应随chunk有界；把文件上限与在途buffer额度分开，禁止只是把1MiB读取循环追加到全量Data。最终校验前不发布半状态，取消/clear/超时后实际IO仍持有FD和lease；保留`fsync→rename→目录fsync`发布与旧格式隔离。大文件写成功、RAM命中或增大配置都不构成262K SSD恢复通过。[SSD队列](KV_SSD_QUEUE_DESIGN.md)和[超时所有权](KV_SSD_TIMEOUT_VALIDATION.md)是必须保留的合同。MTP继续排在这些工作之后。

32K精度链的静态线索是：reference会落地BF16 QK score和BF16 softmax概率，而M5 fused路径以不同的FP32 split-D/在线softmax归约计算，后续QSA与MoE离散选择可能放大差异；当前未找到具体mask/GQA/缩放错误，也未证明全部误差只是可接受的舍入。最小单层诊断候选将对同一次真实layer3 Q/K/V/mask比较reference、native fused及FP32-score路线，并加相同query分组的BF16控制、原始字节和因果tail隔离。它**尚未运行**，不作为本页通过项，也不修改生产默认或放宽阈值。
