# 自主研究与开发接续

用户在2026-09-07凌晨授权至少八小时自行推进，研究vLLM、SGLang和Redis作者的runner，吸收合适特性，并阶段性commit、推送GitHub。首轮工作窗口截至北京时间2026-09-07 13:30（UTC05:30）；到期完成在途实验的收尾、恢复参考服务并整理成果，不再自动启动新实验。

## 当前执行约定

- 独立仓库：`experiments/ane-runner`，远程`thomas-hiddenpeak/Qwen4-MLX`；外层`coreai-models`是另一个仓库。仅提交本项目源码、测试、文档和小型样本，权重、构建与大实验结果保持忽略。
- 用当前工作分支做阶段提交并推送，不强推、不替换已有历史。每次接续先看Git状态及本文件，接手已有工作，不重复开同一项。
- 单个GPU实验所有者；研究与CPU工作可并行，模型加载及GPU测试串行。使用既有控制器核对参考服务身份和空闲请求，测试结束恢复原参数；不停止无关训练或其他项目。
- Prefill/decode分别计时；普通AR与MTP分别比较。现有默认不因局部微测收益自动提升。先跑小数值门槛，再做真实11k完整生成；保存没有收益的结果。
- MTP稳定性、取消、状态一致性和服务接口是缓存开发的前置条件。前缀缓存必须包含Attention KV、QSA、GDN、PLE/n-gram及MTP适用状态，不能仅缓存KV就宣布可复用。
- 优先采取局部可验证改动。引用原始项目文档、代码版本与许可；借鉴设计和复制实现分别说明。

## 当前状态

已推送`9a60fbd`到`codex/moe-composition`。当前自主开发分支为`codex/upstream-adoption`，初始接续提交`8e427b2`已推送。上一轮专家+归约组合330项局部比较、6轮11k生成和5组边界回归通过；单层+5.21%，完整prefill828→824 token/s，保持可选。最新完整记录见[组合回归](MOE_PREFILL_COMPOSITION.md)。

最新已验证参考服务：PID13611，`http://127.0.0.1:11235`，MTP/drafter关闭，核对时刻为北京时间07:35:54。最新恢复ledger为`results/controller-interrupt-v1/run-ledger.json`。这是控制器中断smoke结束时的快照；执行前必须与`../qwen38-ssd/results/experiment-status.json`及实际进程重新核对。

北京时间07:41，参考13611已交由窗口A控制器接管。当前唯一GPU任务是`results/mtp-release-window-a/controller-plan.json`，controller PID13960，exec session33813，ledger启动UTC23:41:19。首个`original-128-part1`（PID/PGID13990）已于07:44:49完成，控制器继续`original-128-part2`。分析计划为同目录`plan.json`，运行状态以同目录`run-ledger.json`为准。root持有唯一GPU所有权，不得并行启动另一模型。

本任务 heartbeat `qwen4-mlx` 已启用，每20分钟接续至北京时间13:30；到期应暂停，避免用户醒来后继续无界运行。临时 `caffeinate -i -t 30000` 防止空闲睡眠，允许显示器休眠，不更改系统设置。接续依赖本机和应用保持运行。

## 进行中的工作

1. 三项目调研、吸收计划、MTP成本与输出延迟统计已实现并回归。
2. GDN prefetch、Replay及MoE双down完成局部筛选，均未提升为默认。
3. 固定AR命令缓冲诊断及GPU档位/系统热压力采样已完成。两个独立长任务的24轮AR/MTP回归通过；初轮性能单窗口且一组AR漂移超5%，尚未通过MTP稳定性能发布门槛。
4. loopback实验HTTP服务通过29项CPU、首轮19项live、补充15项网络边界及固定12周期的46项短soak检查。controller2249已退出，随后控制器中断smoke也完成并恢复参考13611。[两窗口分析计划](MTP_RELEASE_WINDOWS.md)及分析器已完成CPU审阅并推送`c30b699`。窗口A于北京时间07:41正式冻结并启动，12进程96请求，预估约50分钟；不能把旧窗口分析控制算作这次发布结果。
5. 控制器安全中断已完成：每个case使用自有进程组，TERM宽限45秒、必要时KILL后等10秒，确认整组清空才恢复参考；SIGTERM/INT只置flag，在安全点转入finally。4项CPU控制与真实controller中断smoke通过，详见[控制器合同](EXPERIMENT_CONTROLLER.md)。窗口A以`results/controller-interrupt-v1/run-ledger.json`作为predecessor；91文件preflight通过。嵌套分析plan仅用于分析，实际执行为12case扁平controller-plan。生产Swift二进制维持f95565c，推理源码及controller/helper身份冻结；窗口A/B完成或root明确解除冻结前不得修改这些代码或构建。

## 接续记录

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
