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

最新参考服务：PID6275，`http://127.0.0.1:11235`，MTP/drafter关闭。最新恢复ledger为`results/gpu-drift-trace-v1/run-ledger.json`。执行前必须与`../qwen38-ssd/results/experiment-status.json`及实际进程重新核对。

本任务 heartbeat `qwen4-mlx` 已启用，每20分钟接续至北京时间13:30；到期应暂停，避免用户醒来后继续无界运行。临时 `caffeinate -i -t 30000` 防止空闲睡眠，允许显示器休眠，不更改系统设置。接续依赖本机和应用保持运行。

## 进行中的工作

1. 三项目调研、吸收计划、MTP成本与输出延迟统计已实现并回归。
2. GDN prefetch、Replay及MoE双down完成局部筛选，均未提升为默认。
3. 固定AR命令缓冲诊断已完成，主要漂移在GPU跨度内；已补原始GPU档位/系统热压力采样。下一GPU工作是两个独立长任务的AR/MTP回归，同时采集状态分布。
4. 有界SSE缓冲和增量UTF8已通过18项CPU测试；正在写仅loopback的实验HTTP服务，固定推理线程、独立网络队列。尚未完成网络/真实模型验收。

## 接续记录

- 05:20左右：分派三路研究/实现；参考服务保持运行；GDN agent只允许独立编译，尚未获得GPU运行权。
- 05:31：三份上游固定版本已核对，两个调研文档和[综合吸收计划](UPSTREAM_ADOPTION_PLAN.md)完成。开始请求级 MTP 成本摘要和调度输出延迟统计，不改变在线策略。
- 05:31：GDN prefetch4 / prefetch4Vector 独立库编译成功；四层真实权重、确定性 BF16 输入的小门槛全部逐位通过，指定 QKV dispatch 计数正确。QKV 中位墙钟 0.2480 / 0.2510 / 0.2531 ms，两个候选没有收益（-1.17% / -1.98%）；不进入本版本整模型测速，不更改默认。其他矩阵未命中新 kernel，其时差只作为测量波动。原始数据见`results/gdn-prefetch-v1/matvec.json`、`summary.json`；服务已恢复。

- 05:38：调研提交`22d2f85`及 GDN 负结果提交`abe3a6e`已推送。MTP 成本摘要、调度 callback 分位数及 decodeBurst 参数已构建，46项 CPU 检查通过（含8项新成本测试）。当前唯一 GPU 控制器正在执行`results/upstream-cost-latency-v1/plan.json`，先9轮11k的0/1/2深度对照，再预算1/2/自然EOS，最后burst4/8混合请求；在该controller结束和恢复前不得启动另一个模型或重建二进制。运行ledger同目录，完成后其restoration将替代上一轮ledger。GDN agent正在写隔离Replay快筛，不执行GPU。

- 05:51：上述controller全部完成，参考PID3226已恢复。9轮11k完整输出、预算1/2/自然EOS、所有成本字段通过；burst4/8各18项gate通过，短输出期间长prefill插入次数8→4，但最大gap仍约0.9s。明显持续降速使本次速度比较仅作观察，详见MTP_COST_SUMMARY.md和SCHEDULER_LATENCY_EXPERIMENT.md。下一步：正在构建GDN Replay单层快筛；随后固定AR做disabled/fit驻留+现有telemetry诊断，不能将本次漂移直接归因于SSD或训练。MoE双down primitive处于独立源码准备阶段，不得与主GPU实验并行执行。

- 05:55：GDN Replay已通过30cases、720逐位比较、20边界，S3加权单层约+2.07%/3.52%，暂不进生产MTP。提交`98f7daf`正在推送。其恢复ledger为`results/gdn-replay-v1/run-ledger.json`（PID3782），当前又由唯一controller暂停用于`results/residency-drift-v1/plan.json`。该诊断在跑，禁止另一GPU任务/重建二进制；配套只读VM采样进程写`vm-stat.jsonl`并在controller恢复ready后自动退出（最多1200秒）。六轮顺序：D1暖、AR fit暖、AR disabled/fit/fit/disabled，全部500ms telemetry。完成后先分析分页/footprint与阶段漂移，再决定是否需要GPU命令缓冲诊断。MoE双down probe由vllm_sglang_research继续写独立native/Swift，暂未编译。

- 06:10：驻留六轮已完成并独立复核，fit未消除漂移，进程内采样pageins为0且footprint稳定；VM时钟基准不匹配，保留整体数据、不作阶段归因，见RESIDENCY_DRIFT_DIAGNOSIS.md。MoE双down已构建并完成reference/fused两组：三行真实输入逐位及各87/87次原生调用通过，完整单层墙钟约+3.04%/+4.16%，未达5%门槛，pair不稳定胜过serial/recipe，不进入完整模型。服务已恢复PID5357，当前无GPU实验运行。

- 06:24：`aed3e86`已推送并核对远程SHA。固定AR trace三轮完整128 IDs一致，135848 buffers完整无丢失；暖轮prefill新增耗时97.71%、decode76.80%落在GPU跨度内，见GPU_DRIFT_TRACE.md。服务恢复PID6275。GPUStateSampler读取两路原始state及thermalState，空闲采样6个delta/端点通过，不映射MHz。Core SSE缓冲10项+UTF8 8项测试通过，日志results/service-core-v1/tests.log。redis_runner_research正在写HTTP服务；vllm_sglang_research写独立CPU协议层；gdn_pipeline_probe写两个新冻结synthetic长任务及功能checker。三者均不得启动模型或自行build；root协调唯一GPU与统一编译。当前尚无GPU实验运行。
