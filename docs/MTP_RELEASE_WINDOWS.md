# MTP 两窗口配对的执行记录

2026-09-07。此批使用已冻结的原11k、工具JSON与项目事实检索三项输入，落实既有[MTP发布条件](MTP_RELEASE_CRITERIA.md)的性能复测；不引入新内核或改变默认。窗口A已完成：96次完整输出通过，18组中11组通过、7组因AR漂移超过5%无法判定。窗口B于北京时间08:32以相同版本、反转case顺序启动。当前未通过两窗口性能门槛，不修改默认。

## 预先固定的工作量

每窗口同时保留三个任务的128/256输出预算，共六个case。CLI单进程最多10次请求，所以每case使用两个进程：

- 第一进程按 `AR, D2, AR, D2, D2, AR, AR, D2, D2, AR`，前两次预热，后八次为两组配对。
- 第二进程按 `AR, D2, AR, D2, D2, AR`，前两次预热，后四次为第三组配对。

每窗口共12进程、96次完整生成（24次预热、72次测量）、18组配对。所有输出参与正确性检查，预热不参与性能门槛。窗口A顺序为原128、工具128、事实128、原256、工具256、事实256；窗口B反转六个case顺序，单独排期。不能把同一窗口内的不同进程当成独立时间窗口。

固定配置为chunk416、context16384、reference decode、D2 `batchedScalarLinear`、初始draft history tail1024、默认fused prefill、GDN blocked关闭、SSD nextChunk/一worker、wired disabled。每轮同样启用500ms telemetry；没有详细算子profile或命令缓冲追踪。这是带采样条件下的配对，不能隐去采样边界。主干prefill、prompt head history、decode、decode内head history分别报告。

原输入的128/256历史AR golden分别实际生成到预算。新工具/事实任务自然输出85/110 tokens（包含EOS），两个预算下长度相同，不强行屏蔽EOS延长输出。原输入的开放式文字只做历史文本/ID回归；新任务另用冻结JSON答案检查。两种功能验收范围不混为一谈。

## 判定与分析

[`analyze_mtp_release_window.py`](../scripts/analyze_mtp_release_window.py)只读正式分析计划与指定原始报告。实际decode吞吐为两次测量输出总数（排除各自首token、保留实际EOS）除以两次完整decode时间之和；草稿、验证、提交/回放和逐轮head history的成本均留在分母里，不拿round数冒充输出token。

每组前后AR **decode** 吞吐漂移超过5%，该组标为`indeterminate_drift`；MTP/AR小于1.10则标为未通过比值门槛。prefill漂移另列，不代替decode判定。不删慢轮、不在看到结果后改顺序、追加替换组或放宽阈值。两窗口全部预定case及三组成立，分析器才报告`two_window_decode_gate_passed=true`；完整发布、持续内存稳定、HTTP和故障恢复仍有独立要求，脚本从不自动修改默认。

分析器还核对完整输入/输出IDs、配置、EOS/预算、终态offset合同、MTP成本覆盖、冻结模型/二进制身份，以及阶段单调时钟与实际token/step对应。坏golden、坏报告或第二窗口失败会保留已有结果和失败来源，不把它们从摘要中消失。请求尾部的MLX active/peak数据只作观察，此时状态可能仍存活，不当作空闲内存或soak证明。

## 正式启动与接续

本机草案位于`results/mtp-release-window-a/plan-draft.json`，它是**嵌套的分析计划，不能直接传入实验控制器**。窗口A已经生成正式`plan.json`与扁平`controller-plan.json`：UTC23:41:14.942275冻结，91份源码/库/元数据文件身份核对通过，采用控制器中断恢复smoke后的参考PID13611及其ledger。正式分析计划SHA256为`9c56790c0d6dfe0d09fc8459d725d65ad3691e069353d6619e5f90a02f37f2a6`。启动与接续由当前GPU所有者执行：

1. 等当前controller退出并恢复参考服务，核对最新ledger/PID和空闲身份；替换分析计划的predecessor，检查当前source/library/metadata指纹与权重payload stat。
2. 保存新的`plan.json`，设`status=frozen`、真实freeze UTC与已确认的ledger；不在生成后改输入、golden、阈值或分组。
3. 从六个case的两个process依次生成扁平`controller-plan.json`，12条case保留原command/report路径，并设置对应`generation_report`和`golden_report`。控制器会在每进程结束立即核对完整输出IDs，失败后恢复参考。
4. 只运行扁平controller计划；GPU运行期间不改已冻结推理源码或重建二进制。完成后离线分析，再单独冻结窗口B，反转case但保留同身份/配置/答案。

```bash
python3 scripts/analyze_mtp_release_window.py \
  --plan results/mtp-release-window-a/plan.json --validate-plan-only \
  --output results/mtp-release-window-a/preflight.json

python3 scripts/run_specialization_experiment.py \
  results/mtp-release-window-a/controller-plan.json

python3 scripts/analyze_mtp_release_window.py \
  --plan results/mtp-release-window-a/plan.json \
  --plan results/mtp-release-window-b/plan.json \
  --output results/mtp-release-windows-summary.json
```

单窗口分析或未通过双窗口门槛时，脚本退出码1是预期，同时保存完整结论；不能当成无报告，也不能把缺失窗口补成通过。`--validate-plan-only`只核对计划和当前文件身份，不执行模型。

本机桌面应用不是受控空载环境：只读启动检查见到其他应用有CPU活动，未停止它们，也不能仅凭CPU百分比认定GPU争用。原始环境观察与每轮GPU状态保留在本地结果目录，结论继续受固定漂移门槛约束。供电检查为AC供电；未更改性能或散热设置。

CPU复核已使用旧真实报告检查decode分子、漂移、错误字段、阶段标签错序和失败结果保留，并完成独立审阅；这些控制不是新的GPU请求或性能证据。实际执行状态及最新恢复ledger始终见[自主接续记录](AUTONOMOUS_PROGRESS.md)。

## 窗口A实际结果

原始报告及正式分析保存在`results/mtp-release-window-a/`，分析结果为`window-analysis.json`。12个进程均正常退出、自有进程组清空；96次完整IDs一致（24次预热、72次测量），配置、功能范围与阶段合同全部通过。第一子进程于UTC23:41:21.315589开始，最后于00:29:17.552156结束，历时2876.237秒；参考PID17773于00:29:32.854301就绪。报告中的`capture_utc_range`仅是各进程capture起点范围，末值00:26:28不是实验结束时间。

下表为预定18组，吞吐单位为实际decode tokens/s；保留全部漂移组。所有原始倍率均高于1.10，但这不能代替漂移门槛。独立只读复核逐读12份报告、ledger及硬件采样，重算分子、分母与漂移，与分析器一致。

| 任务/预算 | 组 | AR | MTP D2 | MTP/AR | AR漂移 | 判定 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 原11k/128 | 1 | 30.406 | 35.403 | 1.1644 | 2.442% | 通过 |
| 原11k/128 | 2 | 27.926 | 35.864 | 1.2842 | 12.340% | 漂移待定 |
| 原11k/128 | 3 | 21.152 | 27.106 | 1.2815 | 0.421% | 通过 |
| 工具/128 | 1 | 25.082 | 40.491 | 1.6143 | 7.043% | 漂移待定 |
| 工具/128 | 2 | 24.549 | 41.967 | 1.7095 | 0.773% | 通过 |
| 工具/128 | 3 | 23.819 | 42.769 | 1.7956 | 8.764% | 漂移待定 |
| 事实/128 | 1 | 24.143 | 42.108 | 1.7441 | 0.134% | 通过 |
| 事实/128 | 2 | 24.482 | 38.788 | 1.5844 | 1.249% | 通过 |
| 事实/128 | 3 | 25.009 | 44.465 | 1.7780 | 0.783% | 通过 |
| 原11k/256 | 1 | 24.739 | 28.720 | 1.1610 | 0.069% | 通过 |
| 原11k/256 | 2 | 23.804 | 29.101 | 1.2225 | 5.911% | 漂移待定 |
| 原11k/256 | 3 | 24.235 | 28.779 | 1.1875 | 10.733% | 漂移待定 |
| 工具/256 | 1 | 25.889 | 39.542 | 1.5274 | 13.470% | 漂移待定 |
| 工具/256 | 2 | 26.506 | 46.833 | 1.7669 | 0.793% | 通过 |
| 工具/256 | 3 | 24.577 | 42.454 | 1.7274 | 3.155% | 通过 |
| 事实/256 | 1 | 23.684 | 40.729 | 1.7196 | 0.077% | 通过 |
| 事实/256 | 2 | 23.897 | 40.724 | 1.7041 | 11.589% | 漂移待定 |
| 事实/256 | 3 | 24.279 | 43.675 | 1.7989 | 4.086% | 通过 |

Prefill单独列出。以下是每case六次AR、六次MTP测量的最小—最大秒数，不含预热；范围不是配对收益或TTFT。主干prefill与MTP prompt head history分开，decode内head history继续包含在上表decode分母中。完整逐请求的total prefill等字段保留在原始分析里。

| 任务/预算 | AR主干prefill秒 | MTP主干prefill秒 | MTP prompt head history秒 |
| --- | ---: | ---: | ---: |
| 原11k/128 | 14.458–26.944 | 14.605–27.193 | 0.053–0.103 |
| 工具/128 | 23.644–28.032 | 22.916–24.264 | 0.061–0.094 |
| 事实/128 | 20.443–22.420 | 20.188–22.931 | 0.057–0.081 |
| 原11k/256 | 19.861–24.312 | 20.266–24.443 | 0.064–0.070 |
| 工具/256 | 20.553–40.944 | 20.474–43.105 | 0.056–0.219 |
| 事实/256 | 21.416–24.278 | 21.673–24.369 | 0.063–0.104 |

窗口A的`complete=true`、`all_correct=true`，但`window_decode_gate_passed=false`。共11组通过、7组漂移待定、0组倍率失败。既定的两窗口全部组门槛因此尚未通过；窗口B将按计划完成，不用于替换A的未定组。

## 窗口B启动

窗口A恢复后，重新核对参考17773的精确argv、11235监听归属、ready和空闲指标，以及同一91文件身份、102项模型payload stat；未调整机器供电、散热或其他应用。UTC2026-09-07T00:32:04.878487+00:00独立冻结`results/mtp-release-window-b/plan.json`，SHA256为`3409095d585414783a82cc9bb26c8955ff14944a39a41824d903ab9a01d48b77`。窗口B preflight通过；控制器PID18230于00:32:11.259809启动，按事实256、工具256、原256、事实128、工具128、原128执行。报告/telemetry路径改为B，输入、golden、配置、分组及二进制保持A的身份。两窗分别排期且不重叠，不宣称形成了独立或恒定的硬件条件。执行期间继续冻结推理源码、分析器、controller/helper与二进制。
