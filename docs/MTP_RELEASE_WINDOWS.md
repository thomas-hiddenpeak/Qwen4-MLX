# MTP 两窗口配对的执行记录

2026-09-07。此批使用已冻结的原11k、工具JSON与项目事实检索三项输入，落实既有[MTP发布条件](MTP_RELEASE_CRITERIA.md)的性能复测；不引入新内核或改变默认。当前仅完成计划、离线分析器和CPU复核，GPU窗口尚未执行，不能据此认定发布门槛通过。

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

本机草案位于`results/mtp-release-window-a/plan-draft.json`，它是**嵌套的分析计划，不能直接传入实验控制器**。正式启动时由当前GPU所有者执行：

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
