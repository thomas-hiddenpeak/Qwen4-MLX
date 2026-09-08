# 自主研究与开发接续

## 2026-09-09 八小时自主窗口：KV 管理批次 B 已验证，准备持续淘汰

用户重新授权八小时开发与阶段推送，窗口北京时间 **01:56:04–09:56:04**（UTC 截止 `2026-09-09T01:56:04Z`）。已更新并恢复本线程 `qwen4-mlx` heartbeat，每20分钟接续；最后20分钟收尾，到期暂停，不自动进入下一窗口。当前主线 `d449ad0`，根代理唯一 build/GPU/Git owner，MTP 性能后置。

K02 tokenizer计划接口已提交推送 `30c847b`（7项CPU通过），完整会话运行时尚未接线。批次A的最终54项相关CPU检查和release61.96秒构建通过，`results/kv-night-a2/` pressure89checks、timeouts50checks、44成功HTTP+4日志确认取消均通过。实模20请求/229IDs、独立跨状态23组/2567张量及host状态；HTTP21次SSD恢复/7.70GB归档读回，无残留request/workspace/pending。二进制`fc02358d...38502e`，215文件/102模型stat postflight通过。详见可靠性记录，勿把注入pressure或70秒HTTP窗口说成真实物理压力/小时长稳。

最新参考恢复ledger为 `results/kv-night-a2/run-ledger.json`，PID52289、精确argv、idle、MTP/drafter关闭已独立复核；无GPU实验，冻结解除。a1失败因短fixture合法3-token EOS被错误的maxTokens==16假设拒绝，保留原始失败报告，a2已修正并通过。

批次A已提交并推送 `d449ad0`。批次B已验证：K02完整lookup/系统producer/双checkpoint、默认512MiB保留共享system策略及actualForward/recomputed统计；有限close/drain包括callback同一deadline，HTTP唯一关闭入口默认30秒；HTTP完整conversation接线与Prometheus /metrics。165项相关Swift CPU检查、release70.58秒通过，二进制`8b694fcd...57dc2`。B1原生conversation71checks/12对照生成/6冷参考/33状态事件全部通过，pressure89、timeouts50复测通过。B1 HTTP第二进程创建前端口bind失败，保留报告；B2仅将harness改为独立端口，全部9次会话/重启/低空间及36次mixed成功，4次metrics对账、45次完整usage/实际token守恒与3次RST唯一cancelled终态对账通过。B1的219、B2的222文件及各102payload postflight通过，冻结解除。最新参考56391、精确argv、idle、MTP/drafter关闭已复核，ledger为 `results/kv-night-b2/run-ledger.json`。root正阶段提交推送；后续先600秒churn预检，再7200秒窗口，1GiB SSD/160MiB RAM/8长前缀/2短前缀。wrapper仍由tiered agent在ignored目录修复，不要提前启动。

新churn脚本9项根侧CPU控制测试通过，尚无真实长窗。macOS `memory_pressure -S`不执行：为系统级purge/通知，异常退出不能保证复位；真实压力验收仍保留，依据已入research文档。不要重复开发已准备接口、另启GPU控制器或重用旧PID。

## 2026-09-09 当前方向：KV cache 关键能力与开源设计吸收

用户明确要求 KV cache 为当前重点，吸收开源项目优势并制定本项目关键能力。已核对 `cb75be3` 源码与 vLLM、SGLang、LMCache、DwarfStar、MLX LM 的固定提交，形成[统一能力计划](KV_CACHE_CAPABILITIES.md)与三份源码研究。现有 SSD/额度/同前缀合并作为基线；HTTP 自动复用当前限系统/工具前缀，完整会话历史和真正 KV 分页仍有缺口。README 中过期的“SSD 状态缓存尚未实现”已修正。

下一批按计划 A：恢复所有权、SSD 最小空间/有界恢复等待/读写控制、真实内存压力及有效 token/成本指标；B 紧接完整多轮工具/分支复用。维持原 chunk416 数值边界，分离查找范围与共同检查点 single-flight 身份。之后再做 Attention/QSA 页共享与 COW、增量存储、成本淘汰；MTP 性能继续后置。

明确 P0 的状态/故障/实际容量/分阶段延迟门槛及候选 2 小时、发布配置 24 小时压力目标；这是新计划，尚未运行或通过。本轮只改文档，完成交叉审阅、88 处本地链接及 42 个上游 URL/源码行范围检查，均通过；不构建、不运行 GPU、不更改服务或供电。上一轮运行证据和参考恢复 ledger 仍见下一节，历史 PID 不代表本轮重新核对。后续完成的有效增量及时合入并推送主线，不积压实验分支。

## 2026-09-08 当前完成：KV cache 两层管理与可靠性回归

继续按用户要求集中 KV cache，MTP 性能后置。联合 request/cache/workspace 额度、完整混合状态归档、可选持久化 SSD、异步读写/校验/TTL/LRU、同前缀等待合并、取消接管和 HTTP health/429 处理已接入。核心存储 `1c5e3aa` 与运行时集成 `d097657` 已分阶段推送。

175 项 Swift CPU 与 7 项新增 Python parser 测试通过。完整状态/生命周期 27 请求、282 IDs，扣除自身 anchor 后为21组/2361条跨状态张量比较及对应host状态。服务长窗966.47秒：1004成功请求/8032 tokens、84次RST日志确认取消，正常重启与空闲SIGKILL重启均SSD命中且输出相同。另以160MiB RAM额度强制SSD：104成功请求/832 tokens、9次RST、51次SSD恢复/约18.7GB读回、25次同前缀等待合并，全部通过。详见[缓存可靠性](KV_CACHE_RELIABILITY.md)。

当前是候选版本，不将约16分钟RAM窗口和约93秒SSD窗口称为小时/天级工业发布验收；下一重点是持续SSD、多前缀淘汰和实际系统内存压力。最终保留的cache leases是有效缓存，不是未释放请求；request/workspace及SSD pending均归零。新HTTP parser首轮误把冷miss缺省统计当失败，已修复并保留失败报告。

最新参考恢复ledger：`results/cache-reliability-ssd-v1/run-ledger.json`，本次核对PID17735、原argv、MTP/drafter关闭且idle；252文件/102模型payload postflight通过，冻结解除。无在途GPU实验。heartbeat继续暂停，未改供电设置。接续先检查实际PID/命令，以下历史PID不再有效。

## 2026-09-08 前一阶段：AR 服务与前缀缓存

用户要求先处理计划前两项。本轮已完成完整混合状态快照/私有恢复、radix 最长前缀、LRU 和条目/字节/key token 额度，以及 HTTP function tools、调用历史、工具结果续答。HTTP 默认缓存512 MiB/8条，库需显式开启；MTP仍冷prefill。完整范围与原始结果见 [本轮验收](AR_PREFIX_CACHE.md)，尚未开始 SSD 状态卸载或 MTP 性能调优。

133项Swift CPU、6项Python解析测试通过；23个带状态诊断请求、5个独立计时请求共1419生成IDs通过，25组/2785项跨状态tensor比较通过。新HTTP套件40检查/9次推理通过；旧HTTP回归第一次在启动前遇端口占用，保留失败，独立空闲端口重跑19项通过。二进制为 `2941a0dd…5a9a54`。索引阶段提交 `e53480e` 已推送，其余运行时/工具/文档随本轮提交；不重复运行已完成套件。

最新参考恢复 ledger：`results/ar-prefix-cache-v1/http-regression/run-ledger.json`，本次核对PID10167、MTP/drafter未加载且idle。238文件与102模型payload stat核对后已解除冻结。此PID仅为当前快照，接续先验证实际命令/服务；此前74990及下文旧PID均已过期。heartbeat保持暂停，未改供电设置。下一项是后续授权下的SSD状态缓存，不自动恢复旧MTP实验计划。

## 2026-09-08 优先级调整记录：MTP 性能后置

用户明确要求将 MTP 性能优化放到整体计划靠后。此要求覆盖下文旧记录中的“MTP 先行”“缓存等待 MTP 门槛”和“不提前实现共享前缀或 SSD 状态缓存”等接续约定；旧实验结果、失败与性能门槛保持原样，不重新解释为通过。

后续先推进 AR 服务与完整混合状态缓存，再做前缀树、淘汰、SSD 状态卸载及普通 prefill/decode 基础 kernel 优化，最后评估 MTP 的增量收益。AR 缓存仍需验证 Attention KV、QSA、GDN、PLE/n-gram 的完整状态、独立恢复、取消清理与容量限制，但不依赖 MTP 性能验收；MTP 专属状态与缓存兼容性单独验证。AR 缓存不适用于 MTP 时沿用原有 MTP 冷 prefill，只有无法安全执行的显式组合才拒绝，不静默改为 AR。现有 MTP 保持可选，涉及其路径的改动仍维护正确性回归。

此段记录最初的优先级调整，实际缓存实现已在顶部更新；既有 heartbeat 保持暂停。当前方向见 [综合吸收计划](UPSTREAM_ADOPTION_PLAN.md)。以下保留历次进展及当时约定，优先级冲突以用户最新决定为准，运行状态以顶部最新 controller ledger 为准。

用户在2026-09-07凌晨授权至少八小时自行推进，研究vLLM、SGLang和Redis作者的runner，吸收合适特性，并阶段性commit、推送GitHub。首轮工作窗口截至北京时间2026-09-07 13:30（UTC05:30）；到期完成在途实验的收尾、恢复参考服务并整理成果，不再自动启动新实验。

## 当前执行约定

- 独立仓库：`experiments/ane-runner`，远程`thomas-hiddenpeak/Qwen4-MLX`；外层`coreai-models`是另一个仓库。仅提交本项目源码、测试、文档和小型样本，权重、构建与大实验结果保持忽略。
- 用当前工作分支做阶段提交并推送，不强推、不替换已有历史。每次接续先看Git状态及本文件，接手已有工作，不重复开同一项。
- 单个GPU实验所有者；研究与CPU工作可并行，模型加载及GPU测试串行。使用既有控制器核对参考服务身份和空闲请求，测试结束恢复原参数；不停止无关训练或其他项目。
- Prefill/decode分别计时；普通AR与MTP分别比较。现有默认不因局部微测收益自动提升。先跑小数值门槛，再做真实11k完整生成；保存没有收益的结果。
- 当前集中 AR 缓存管理：必须完整保存 Attention KV、QSA、GDN、PLE/n-gram。MTP 性能后置，MTP 专属缓存未支持，显式 MTP 继续冷路径；共享生命周期改动仍回归已有 MTP 正确性。
- 优先采取局部可验证改动。引用原始项目文档、代码版本与许可；借鉴设计和复制实现分别说明。

## 当前状态

- 2026-09-07 20:15：实际专家重叠采集及复算已提交推送`a8fd716`并核对远程SHA。首个shared-load gate/up候选最终release/CPU构建60.13秒、13项CPU通过；两层六组原route pattern + synthetic激活的288项逐位/有限值检查、54暖/216测量完整，独立复算一致。A/C六组倍率0.948471/0.973175/0.956120/0.937033/1.000941/0.966475，未通过性能筛选，不接生成，仅保留显式operator probe。随后同二进制现有AR/D2各一次完整11k/128回归通过，D2计数一致；146文件SHA+102payload stat复核释放，最新参考52066的精确argv、监听、空闲及MTP/drafter关闭已核对。当前无GPU实验，heartbeat仍暂停；root完成阶段提交与推送后交回用户。见MTP_GROUPED_GATEUP_EXPERIMENT.md。

- 2026-09-07 19:50：默认关闭的验证routing capture已release构建并通过19项选定CPU检查。原11k/128普通与采集两请求完整golden一致，55轮全部S3、2640层记录完整无丢失；实测同轮同层重复选择26.7184%，gate/up-only逻辑节省上限17.8123%，未测实际DRAM或宣称生成收益。141文件SHA+102payload stat已复核释放，参考49817的精确argv、监听、空闲及MTP/drafter关闭已核对。当前无GPU实验；gdn agent正在ignored目录实现共享gate/up局部候选，root保持唯一build/GPU/Git所有者。见MTP_EXPERT_OVERLAP.md。

- 2026-09-07 16:28：MTP验证诊断已提交推送`0b24db9`，远程SHA核对通过；独立事件扫描复算245,474,613ns及9项分析CPU控制通过，有效报告为run/analysis-v2.json（首版golden元数据设置错误保留）。GDN QKV S3 TM2候选随后release46.10秒、CPU dispatch契约及42项实权重逐位检查通过，48暖/192测量完整；S3四层倍率0.957720/1.009808/0.898387/1.012265，S2未改参数对照也有方向性波动，未建立可重复收益，不扩大整模型或改默认。140文件SHA+102payload stat已复核释放；参考41069精确argv、监听、空闲及MTP/drafter关闭已核对。当前无GPU实验，heartbeat仍暂停；最终源码保留显式算子probe，未接生成mode。见VERIFICATION_QKV_TM2_EXPERIMENT.md。

- 2026-09-07 16:20：用户继续授权MTP验证热点研究。增加profiler phaseFilter与uptime时间戳，MTP原生跟踪限定synchronized verification，粗round明确无graph boundary；release46.89秒、23项Swift CPU和6项Python通过。原11k、D2/max16的普通/诊断完整IDs及MTP计数相同，2037阶段/22901原生命令无丢弃；诊断GPU跨度MoE32.89%、GDN25.19%、Attention19.78%，不是普通执行的带宽/速度指标。275文件SHA+102payload stat已复核释放，参考40188恢复、空闲及身份已核对。当前无GPU实验；GDN agent在ignored目录准备S3 QKV TM2局部候选，尚未应用/构建/实测。见MTP_VERIFY_HOTSPOTS.md。

- 2026-09-07 15:58：用户白天授权的三项推进已完成首轮。真实SSE溢出通过并推送`8b8f702`；固定11k五轮AR联合诊断已推送`9750d70`，暖decode首末下降3.777%，新增时间90.95%在GPU跨度内，不归因于适配器。共享专家S2/S3融合候选release及23项CPU通过，83项真实层逐位比较通过；四组局部倍率1.038665/1.047143/1.023566/1.020307，含时序波动，未达约5%可重复收益筛选量级。停止该候选的整模型扩展，仅保留三文件算子探针，撤下未验证的新生成模式。实测结果在`results/daytime-mtp-v1/run`，139文件SHA与102payload stat复核释放；参考PID37847已恢复并核对空闲、精确argv、监听和MTP/drafter关闭。当前无GPU实验，heartbeat保持暂停；16:01已确认缩小源码范围后的release重新构建通过（44.80秒），最终二进制`045b7bcc`；新二进制未重复微测，实测证据仍绑定`9cb13917`，见MTP_SHARED_ELEMENTWISE_EXPERIMENT.md。

2026-09-07 15:31用户授权继续。新的真实AR SSE溢出单次测试已经通过4项检查；恢复读取后收到唯一slow_consumer错误、DONE和EOF，后续AR/MTP及资源清理通过。141文件/102模型payload核对后解除冻结，结果在`results/http-sse-overflow-public-v1/`。

随后同进程固定11k AR五请求联合command timing与500ms telemetry已完成，完整输出一致、775个步骤窗口、226045条原生命令记录且无丢失。287文件/102payload核对后解除冻结，参考35851按原参数恢复并确认空闲；当前无GPU实验运行。数据在`results/daytime-drift-v1/run/`，分阶段分析正在进行。共享专家S2/S3尾部融合仅在ignored目录准备，未应用或构建。以下保留此前阶段记录，接续以本段及最新ledger为准。

已推送`9a60fbd`到`codex/moe-composition`。当前自主开发分支为`codex/upstream-adoption`，初始接续提交`8e427b2`已推送。上一轮专家+归约组合330项局部比较、6轮11k生成和5组边界回归通过；单层+5.21%，完整prefill828→824 token/s，保持可选。最新完整记录见[组合回归](MOE_PREFILL_COMPOSITION.md)。

最新参考服务PID31650，`http://127.0.0.1:11235`，MTP/drafter关闭，10:56:42 ready；root精确argv/meta/listener/idle核对通过，ledger与postflight在`results/http-async-logger-regression-v2/`。没有其他GPU实验运行，其他项目未停止。

A/B共192请求完整IDs与阶段/功能范围检查通过，正式36组为23通过/13漂移未定，双窗口性能门槛未通过，已提交推送d432caa。HTTP修复c93011f完成release build、36项Swift CPU/6项Python解析控制、19项live、15项edges、12周期46项soak和6项终态日志检查。首批edges在服务启动前端口检查失败，保留原报告，续批三个独立端口完成。真实非流式output_limit已触发并验证恢复；SSE实际溢出仍未覆盖。90文件postflight与恢复核对通过，旧冻结已解除，历史结果仍绑定c930。

PD和async8已提交推送d887b64：13请求PD正确性通过，短等待改善33.39%/42.05%，但首对整体耗时和长TTFT超限、外基线漂移18.59%，默认burst4保持。async一步完整logits/121持久张量/host信息/原checkpoint的13项检查通过，8轮完整输出与计数通过；原始倍率1.186397伴随30.4455%外基线下降，性能未定，默认0。历史120文件/102模型payload postflight已完成。供电观察及日志失败记录e744e5c已推送核对；历史快照为40W和离散AC电量100/73/45。用户返回后明确低功率适配器是预期配置，撤回供电核对作为性能复测的前置要求；漂移原因尚未确定。

有界logger五case已全部通过：live19、edges15、固定12cycle soak46、terminal6、真实未读日志pipe3。二进制`b0391391b4af4dbdcd31bb16cffbf268e790112129e4887c3ac84ef547be1bc1`，133文件/102模型payload postflight通过后解除冻结；源码已完成验证，随本次阶段提交。此前SIGPIPE13与首次Swift限定名编译失败均保留，最终release51.66秒/43Swift通过。普通文件中的日志归因要求零loss；pipe实际128条满额、52926字节保留、419字节in-flight、内核未读65536字节，written固定165，AR/MTP正确且TERM后0.7258455秒退出。pipe故意丢日志，不能宣称完整终态归因。

包装器的ignored测试集路径已在解冻后改为同SHA生产文件；仅Package.swift与5个生产脚本的隔离目录中11项CPU通过，没有results/模型依赖。历史133文件计划不改写；新SSE真实溢出脚本与小型输入已接入公开目录，9项解析与2项实际main退出控制通过；尚未运行模型尝试。
用户已于11:04左右返回并要求总结。本轮heartbeat `qwen4-mlx`已暂停，不再自动接续新实验；root已核对并结束本任务临时`caffeinate -i -t 30000`进程，系统设置未改。当前没有GPU实验或源码冻结，参考31650保持原参数运行。

## 进行中的工作

1. 三项目调研、吸收计划、MTP成本与输出延迟统计已实现并回归。
2. GDN prefetch、Replay及MoE双down完成局部筛选，均未提升为默认。
3. 固定AR命令缓冲诊断及GPU档位/系统热压力采样已完成。两个独立长任务的24轮AR/MTP回归通过；初轮性能单窗口且一组AR漂移超5%，尚未通过MTP稳定性能发布门槛。
4. loopback实验HTTP服务通过29项CPU、首轮19项live、补充15项网络边界及固定12周期的46项短soak检查。窗口A实际完成12进程96请求，完整IDs/配置/阶段通过；18组性能11组通过、7组漂移待定，不改默认。完整逐组和独立prefill数据见[两窗口执行记录](MTP_RELEASE_WINDOWS.md)。窗口B已完成并合并分析：12组通过/6组漂移未定；A/B的13个未定组全部保留，包括B原256第三组原始倍率0.840651，不认定稳定加速或失败。
5. 控制器安全中断已完成：每个case使用自有进程组，TERM宽限45秒、必要时KILL后等10秒，确认整组清空才恢复参考；SIGTERM/INT只置flag，在安全点转入finally。4项CPU控制与真实controller中断smoke通过，详见[控制器合同](EXPERIMENT_CONTROLLER.md)。窗口A以`results/controller-interrupt-v1/run-ledger.json`作为predecessor；91文件preflight通过。嵌套分析plan仅用于分析，实际执行为12case扁平controller-plan。A/B使用的Swift二进制f95565c及91文件已完成最后身份检查；旧冻结于B完成、恢复核对后解除，记录见`results/mtp-release-window-b/source-freeze-release.json`。历史计划仍保存原指纹，不随新构建改写。

## 接续记录

先查看本页“当前状态”和最新controller ledger，不重复执行历史草案。PD与async8已完成构建、正确性和有限性能筛选并提交d887b64，HTTP输出原因/终态日志已提交f77f58e；旧c930/5f1321b结果仍绑定原二进制，不改写历史指纹。A/B的13个性能未定组也不通过替换数据消除。

Logger五case和参考恢复已经完成，不能重复运行旧草案。本次推送包含有界日志源码、CPU/真实服务结论和公开SSE回归入口；SSE入口只有CPU证据，没有真实overflow通过结果。下一步可先用当前binary与公开脚本/fixture重新冻结一次慢客户端尝试（旧ignored plan-draft不能直接启动），随后直接做性能漂移诊断与有限对照。用户已返回，后续依当前交互推进。

供电不作为已确认性能问题或后续复测前置条件。继续完成有明确边界的服务正确性/活性工作，并针对观测漂移独立分析；不放宽MTP性能门槛，不改默认burst4/async0，不提前实现共享前缀或SSD状态缓存。各agent只有明确分配的文档或ignored候选写权限，root保持唯一build/GPU/Git所有者。

以下是当时的历史记录；其中“当前”“待运行”等措辞描述该时间点，以本页顶部和最新ledger为接续依据。

- 05:20左右：分派三路研究/实现；参考服务保持运行；GDN agent只允许独立编译，尚未获得GPU运行权。
- 05:31：三份上游固定版本已核对，两个调研文档和[综合吸收计划](UPSTREAM_ADOPTION_PLAN.md)完成。开始请求级 MTP 成本摘要和调度输出延迟统计，不改变在线策略。
- 05:31：GDN prefetch4 / prefetch4Vector 独立库编译成功；四层真实权重、确定性 BF16 输入的小门槛全部逐位通过，指定 QKV dispatch 计数正确。QKV 中位墙钟 0.2480 / 0.2510 / 0.2531 ms，两个候选没有收益（-1.17% / -1.98%）；不进入本版本整模型测速，不更改默认。其他矩阵未命中新 kernel，其时差只作为测量波动。原始数据见`results/gdn-prefetch-v1/matvec.json`、`summary.json`；服务已恢复。

- 05:38：调研提交`22d2f85`及 GDN 负结果提交`abe3a6e`已推送。MTP 成本摘要、调度 callback 分位数及 decodeBurst 参数已构建，46项 CPU 检查通过（含8项新成本测试）。当前唯一 GPU 控制器正在执行`results/upstream-cost-latency-v1/plan.json`，先9轮11k的0/1/2深度对照，再预算1/2/自然EOS，最后burst4/8混合请求；在该controller结束和恢复前不得启动另一个模型或重建二进制。运行ledger同目录，完成后其restoration将替代上一轮ledger。GDN agent正在写隔离Replay快筛，不执行GPU。

- 05:51：上述controller全部完成，参考PID3226已恢复。9轮11k完整输出、预算1/2/自然EOS、所有成本字段通过；burst4/8各18项gate通过，短输出期间长prefill插入次数8→4，但最大gap仍约0.9s。明显持续降速使本次速度比较仅作观察，详见MTP_COST_SUMMARY.md和SCHEDULER_LATENCY_EXPERIMENT.md。下一步：正在构建GDN Replay单层快筛；随后固定AR做disabled/fit驻留+现有telemetry诊断，不能将本次漂移直接归因于SSD或训练。MoE双down primitive处于独立源码准备阶段，不得与主GPU实验并行执行。

- 05:55：GDN Replay已通过30cases、720逐位比较、20边界，S3加权单层约+2.07%/3.52%，暂不进生产MTP。提交`98f7daf`正在推送。其恢复ledger为`results/gdn-replay-v1/run-ledger.json`（PID3782），当前又由唯一controller暂停用于`results/residency-drift-v1/plan.json`。该诊断在跑，禁止另一GPU任务/重建二进制；配套只读VM采样进程写`vm-stat.jsonl`并在controller恢复ready后自动退出（最多1200秒）。六轮顺序：D1暖、AR fit暖、AR disabled/fit/fit/disabled，全部500ms telemetry。完成后先分析分页/footprint与阶段漂移，再决定是否需要GPU命令缓冲诊断。MoE双down probe由vllm_sglang_research继续写独立native/Swift，暂未编译。

- 06:10：驻留六轮已完成并独立复核，fit未消除漂移，进程内采样pageins为0且footprint稳定；VM时钟基准不匹配，保留整体数据、不作阶段归因，见RESIDENCY_DRIFT_DIAGNOSIS.md。MoE双down已构建并完成reference/fused两组：三行真实输入逐位及各87/87次原生调用通过，完整单层墙钟约+3.04%/+4.16%，未达5%门槛，pair不稳定胜过serial/recipe，不进入完整模型。服务已恢复PID5357，当前无GPU实验运行。

- 06:24：`aed3e86`已推送并核对远程SHA。固定AR trace三轮完整128 IDs一致，135848 buffers完整无丢失；暖轮prefill新增耗时97.71%、decode76.80%落在GPU跨度内，见GPU_DRIFT_TRACE.md。服务恢复PID6275。GPUStateSampler读取两路原始state及thermalState，空闲采样6个delta/端点通过，不映射MHz。Core SSE缓冲10项+UTF8 8项测试通过，日志results/service-core-v1/tests.log。redis_runner_research正在写HTTP服务；vllm_sglang_research写独立CPU协议层；gdn_pipeline_probe写两个新冻结synthetic长任务及功能checker。三者均不得启动模型或自行build；root协调唯一GPU与统一编译。当前尚无GPU实验运行。

- 07:00左右：两个独立长任务24轮通过并已提交推送`cb23860`，见MTP_AGENT_EXPANSION.md；GPU状态出现nominal→fair及档位分布变化，仅作关联。HTTP release构建49.66s、29项CPU与19项live全部通过，controller已退出并恢复参考PID9437。新入口保持实验性、AR默认、MTP2最多256预算。下一批由vllm_sglang_research准备网络边界脚本（无GPU权），gdn_pipeline_probe准备既定MTP性能窗口分析与草案（无GPU权），redis_runner_research只读设计完整prefix checkpoint（不接生产）。root持有唯一GPU启动/构建/Git权。

- 07:03：HTTP主提交`0f89393`已推送并核对远程SHA。补充网络边界15项全部通过：4连接上限与回收、header/body约15.185s接收408、截断body400、AR/MTP预算1/2/4的length/usage、MTP收到两个content后的RST（日志同ID明确decode）及新AR请求恢复。取消清理本次观测0.109s，不承诺最大延迟；未命中verify内部取消。controller exit0、实验服务graceful退出，参考PID11001 ready。MTP窗口A草案正在做CPU审阅，禁止先启动；实际运行前root确认冻结。后续窗口B反转case顺序。服务12周期soak由agent准备CPU脚本，不与窗口并行加载模型。前缀设计发现MTP跨chunk next-token及full-prompt长度依赖，必须显式处理，不能直接共享decoder。

- 07:35：控制器自有进程组清理已通过4项CPU控制与一次真实SIGTERM恢复smoke。controller13573收到TERM后进入finally；case leader13605先退出，子进程13606保留6秒清理，整组6.0846秒后消失，未发KILL。随后才创建参考13611，07:35:54核对精确argv、11235监听者和MTP/drafter关闭；总中断到退出22.333秒。controller exit1及InterruptedError是这次预期中断结果，恢复成功另由smoke/ledger确认。原始证据为`results/controller-interrupt-v1/smoke.json`、`run-ledger.json`；独立只读复核通过。没有运行实验模型推理或重建Swift二进制，root下一步准备冻结MTP窗口。

- 07:41：root完成窗口A冻结（UTC2026-09-06 23:41:14）与91文件preflight，启动唯一controller PID13960，exec session33813。`results/mtp-release-window-a/plan.json`保存分析合同，SHA256为`9c56790c0d6dfe0d09fc8459d725d65ad3691e069353d6619e5f90a02f37f2a6`；`controller-plan.json`保存平铺执行计划。共12进程96请求、每case三组、全窗口18组，gdn agent已独立只读核对参数与路径。预估约50分钟；参考13611由controller接管，后续恢复PID以新ledger为准。当前禁止并行GPU、修改被冻结源码/controller/helper或重建二进制。

- 07:44:49：窗口A首进程10轮完整输出ID通过。root独立手算前两测量组：G1 AR30.4056/MTP35.4031 tokens/s，倍率1.16436，AR漂移2.4416%，该组通过；G2 AR27.9262/MTP35.8636，倍率1.28423，AR漂移12.3403%，按冻结门槛应为无法判定。保留该组，不重划或补换；这是首进程局部观察，不能作为整窗或发布结论。

- 07:50：控制器收尾提交`84a7436`已推送并核对远程SHA。窗口A的original128两进程共16轮完整IDs/阶段合同通过；三组比值1.1644/1.2842/1.2815，AR漂移2.4416%/12.3403%/0.4213%，第二组仍为indeterminate。增量分析保存在`results/mtp-release-window-a/partial-after-original128.json`；窗口未完成时其全窗口all_correct=false包含缺失报告，不能误读为已完成请求错误。controller已进入tools128part1，继续按原计划保留所有组。新增调度优先级文档，仅研究与两个后续实验设计，没有实现跨请求batch或prefix cache。

- 08:06：完成独立GPU状态阶段分析器与HTTP输出边界核查。新工具只读已有报告，不改冻结的91文件/运行版本；两真实报告逐值重算一致，30控制及2次坏/好文件CLI保留检查通过。原始AR从30.786降到21.108 token/s期间，GPUPH主要标签由P12/P11变为P3/P4，thermal从nominal变为fair；仅作关联，不推算MHz/因果，原MTP门槛不变。见GPU_STATE_PHASE_ANALYSIS.md。HTTP已确认非流式文本超限被泛化generation_failed，关闭路径缺少15/300秒原因；见HTTP_OUTPUT_BOUNDARIES.md。解冻后优先补本地错误原因/结构化终态日志，再选择真实可确定触发的短fixture；本轮未捏造overflow测试或运行另一GPU任务。窗口A此时已完成6/12进程48/96轮，继续原256及工具/事实256，先做完A/B再改源码。

- 08:40：窗口A于08:29结束，12进程96请求全通过完整IDs及阶段检查，正式分析18组为11通过/7漂移待定/0倍率失败，独立只读重算一致；不认定两窗门槛通过。参考17773恢复并核对后，于08:32启动B controller18230/session45809。B正式plan SHA256为`3409095d585414783a82cc9bb26c8955ff14944a39a41824d903ab9a01d48b77`，91文件同A，preflight通过。B已完成事实256的两进程16轮，正在工具256；继续保持冻结。HTTP候选v3仅通过静态审阅与patch应用检查，尚无新构建/运行结果。当前已推送源码阶段为`17164f3`，远程SHA已核对。

- 09:04：阶段记录`634f1fb`已推送并核对远程SHA。B已完成8/12进程64/96请求，全部完整IDs通过，正在tools128part1（PID21030），controller18230/session45809继续唯一GPU任务。HTTP修复v3、日志验证v2、PD公平性候选及对应draft均已静态审阅，尚未应用/build/live。新增限定decode提交研究，避免把历史23.20%增量误当总CPU可优化空间；每8层提前提交仍只是后续候选，不改默认。

- 09:25：A/B全部完成、独立复核和正式合并分析通过其正确性范围，共23性能通过/13漂移未定，含B原256第三组raw0.840651；不改默认。B最后91文件身份检查及参考21912恢复核对后，root解除旧冻结，应用HTTP v3与日志脚本v2。release49.91s、36 Swift CPU、6 Python控制通过，二进制`c93011f804dd287a7758568d69373089959a0b92408379390ea71d7294b0568a`。四case正式controller22649/session27178启动，90文件冻结，当前live；源码提交待真实回归完成，双窗文档先阶段提交。async8最新7文件候选和一步deep-state gate仍ignored，Redis正独立只读审阅；不与HTTP并行build/GPU。

- 09:34：首批live19/19通过，随后edges在Popen服务前的端口bind检查报`OSError: [Errno48] Address already in use`，零gate/无service PID；controller按finally恢复22840。失败时未记录内核TCP状态，不能据此确证TIME_WAIT，也没有停止不明监听者。root核对同90文件及参考idle，改用预先检查空闲的11237/11238/11239独立端口启动v2 controller23104/session15161，只跑剩余edges/soak/logs；edges15/15已通过，soak进行中。新live/edges日志格式只读检查分别194/150条schema、14/8个legacy请求通过，但这项格式检查不代替逐请求终态门槛。async7文件候选已经root apply-check和Redis独立审阅；root另发现性能模板单值mtp-order与repeat2冲突，正让agent仅改ignored模板为mtp-depth0，不能带此错误直接运行。

- 09:46：HTTP c930续批全部完成并恢复24434。soak797.792秒、46/46、12cycle，五次FD11不变、RSS净增1344KiB；终态6gate/8request通过，真实非流式输出超限及fresh AR/MTP恢复通过。root90文件postflight与参考核对完成，旧freeze解除。发现同步日志在满stderr pipe会阻塞的活性缺口，另准备有界writer候选。async模板已修正为合法的`--mtp-depth 0 --repeat 2`，v3 manifest核对通过，生产7文件仍未改。

- 09:58：f77f58e HTTP阶段已commit/push/远程SHA核对。PD+默认关闭async8候选整合构建通过，3项纯CPU策略测试通过。唯一PD controller session80786已启动，112文件/二进制5f1321b冻结，env删除async8以保持AR默认路径。参考predecessor是HTTPv2恢复24434；结束后先检查PD correctness与性能分别结果，再决定启动async同状态门槛。

- 10:02：PD控制器已exit0并恢复25852，112文件postflight通过；默认burst4维持。进入async8同checkpoint门槛（session78710），119文件SHA及102payload大小/mtime核对通过，未开始4process性能筛选。必须先查state-gate.json complete/passed及所有逐位检查，不能仅按controller退出0进入下一步。

- 10:05：async同状态门槛root以独立helper复核通过（121张量+完整logits，0/5次提交），并完成119文件/102payload postflight。5项汇总CPU控制通过；现跑固定0/8/8/0，每process暖0测1，完整128 IDs检查所有8轮。Logger v2已准备并由另一agent审阅，尚未应用/构建；后续五case草案在`results/http-async-logger-regression-v1/plan-draft.json`，不得提前执行。

- 10:12：async4process控制器exit0并恢复26788。正式summary complete/all_correct true、errors[]、outcome indeterminate_drift；保留原始表，不把1.186倍率当收益。120文件/102payload postflight及参考argv/meta/listener/idle完成，旧freeze解除。Logger v2 root实际apply-check、7文件baseline/candidate哈希及11CPU Python通过，待阶段提交后应用与构建。PD小型冻结fixture将随源代码提交以方便复跑。

- 10:16：PD/async结果及可复跑fixture已阶段提交并push d887b64，远程SHA一致。Logger v2已原样apply，release build session91604；待42项相关Swift CPU和Python控制通过后才冻结五case服务回归。参考保持26788，尚未开始logger模型测试。

- 10:20：42项HTTP CPU在真实closed-pipe logger测试处被SIGPIPE13终止，v2日志器未通过；原始build/cpu log保留于`results/http-async-logger-regression-v1/`，未加载模型。Apple XNU证实EPIPE会向进程发SIGPIPE，dup+NOSIGPIPE又会修改共享fileglob，故不采用。Root明确改为自有独立serve-gpu进程入口的SIGPIPE忽略策略（不是其他应用/共享FD修改），Core不隐式改global；redis准备v3 delta与实际closed-pipe测试，待root构建验证。另发现当前协商40 W和离散AC电量100/73/45，见POWER_ENVIRONMENT_FINDINGS；后续不再追加受漂移影响的优化测速。参考26788仍空闲。

- 10:58：logger五case全部exit0，19/15/46/6/3检查通过，参考31650已核对。133文件/102payload postflight后解除冻结；包装器仅路径修复并以不含ignored文件的隔离目录重跑同11CPU通过。实际pipe写入堵塞仍支持health、AR/MTP与0.726秒TERM退出。SSE修正版候选11CPU和独立复核通过，正在准备公开脚本/fixture，无新模型运行。

- 11:06：用户返回要求总结；停止新增实验并暂停八小时heartbeat，结束本任务临时防睡眠进程。SSE公开3脚本/5输入文件已准备，root复跑9+2CPU通过，未启动GPU。正在完成最后源码、结果与公开回归入口的阶段push；参考31650仍运行。

- 11:07：用户说明低功率适配器是有意使用且不影响性能。Root修正文档和计划，不再把40 W当作性能问题或复测前置条件；原始环境记录及性能漂移保留，不编造因果。

- 11:08：有界logger及全部实测结论、供电归因修正已提交83c84f3并推送；公开SSE入口与固定输入纳入下一阶段提交，其9+2CPU通过，真实模型未运行。参考31650保持原参数，自动接续已暂停。
