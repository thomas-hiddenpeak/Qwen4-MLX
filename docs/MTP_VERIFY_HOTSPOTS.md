# MTP 验证阶段的 GPU 热点

2026-09-07。原 11,057-token agent 输入、D2 `batchedScalarLinear`、输出预算16。普通与诊断运行的完整16个输出、最终位置11072及MTP计数相同。同步诊断下，选中阶段的GPU命令跨度主要分布于MoE（32.89%）、GDN（25.19%）和Attention（19.78%）。这是选择下一项局部实验的依据，不是普通执行下已测得的可消除时间或带宽占用率。

## 诊断改动

`generate-gpu` 新增可选 `--profile-phase prefill|decode|verification`，须与非disabled的 `--profile-stages` 一起使用。默认仍记录所有阶段；未选阶段及无forward上下文的工作直接执行原body，不读取profiler时钟、不收集输出或增加同步。上下文设置与 `isRecording` 语义保留，避免verification永远无法被选中。

每个已记录阶段新增可选uptime起止时间戳，与现有原生MLX命令时间戳对齐。区间从preceding stream drain之后开始，结束于输出/状态求值和同步完成；历史JSON缺少新字段仍能解码。

MTP原生命令跟踪仅在 `--profile-stages synchronizedStages --profile-phase verification` 组合下放行。粗decode标记代表整轮MTP，明确写出 `graph_boundary_available=false`；公共分析器保留整轮GPU跨度，将无法提供的forward/evaluation拆分置空。AR与旧报告的图边界解释不变。

## 实测

两次请求均使用chunk416、prefill/verify每4层求值、完整目标上下文、draft history1024、原始量化权重、reference decode/GDN及关闭async。先普通运行，再进行带阶段同步和原生命令记录的诊断；未拿跨进程耗时比作为优化收益。

两次均为7个decode步骤：13个草稿、8个接受、20个验证token、15个decode输出，接受直方图 `[1,4,2,0,0]`，replay为0，length终止。全部输出匹配原AR golden的前16个ID；没有把这次短生成称为完整128/256任务验收。

| 阶段 | 调用数 | 分段墙钟合计 ms | GPU跨度合计 ms | 选中GPU跨度份额 |
| --- | ---: | ---: | ---: | ---: |
| MoE | 336 | 156.154 | 80.732 | 32.89% |
| GDN | 252 | 161.016 | 61.830 | 25.19% |
| Attention | 84 | 108.674 | 48.553 | 19.78% |
| HC MLP read | 336 | 86.276 | 17.816 | 7.26% |
| HC attention read | 336 | 90.250 | 16.433 | 6.69% |
| Mixer和输出头 | 7 | 17.826 | 15.523 | 6.32% |
| PLE | 7 | 8.558 | 2.575 | 1.05% |
| 其余embedding/HC write | 679 | 128.702 | 2.013 | 0.82% |

GPU列是逐段与原生GPU命令区间求交后取并集，分母合计245.475ms；不是将evaluationWait当成GPU时间。2037个阶段、22901个原生命令记录完整，无丢弃、未完成或错误。6个S3和1个S2验证forward全部保留；不同shape的首次编译/初始化和后续调用须分开观察。

普通运行target prefill为19.317秒、MTP prompt history为0.193秒、decode为0.472秒；诊断对应19.668秒、0.195秒、0.935秒。分段同步明显扰动decode，这些GPU份额只能用于诊断排序，不能直接套回普通decode来预测加速。

GDN包含投影、卷积/门控前后处理、recurrence以及完整capture状态写入，尚未把61.830ms全部归给大矩阵。Attention同时包含projection、QSA/缓存与attention计算。选中的forward阶段不包括外层checkpoint、目标选择/读回、prefix commit、draft/head history，也不覆盖单独标记为decode的target-only收尾。本次7步均进入验证，没有以此宣称覆盖所有停止分支。

## 验证和证据

Release构建通过；23项profiler、阶段、MTP成本CPU测试和6项命令时间分析测试通过。4项CLI无效组合在权重加载前正确拒绝；测试首次使用缺失模型目录时先被config检查拒绝，后改用现有模型元数据确认组合错误，原测试设置失败记录保留。

原始数据位于 `results/mtp-verify-hotspots-v1/run/`：`normal.json`、`profile.json`、`commands.json`、`plan.json`、`run-ledger.json`和`postflight-and-release.json`。实测二进制SHA为 `ac2774982896b467d0f1be6be9f38ec3ea07350b9c662171937f4bb59b03b37f`。冻结275个文件SHA及102个模型payload大小/mtime，运行后全部核对通过；原参考服务PID40188的参数、监听归属、空闲状态及MTP/drafter关闭已核对。

下一项选择GDN的S3 QKV投影作局部筛选：尝试TM4→TM2以减少每线程累加器数量，同时保留每个输出的原K/FMA/归约顺序。它不复制权重，也不代表已证明存在寄存器瓶颈；先要求真实四层权重下逐位正确和约5%可重复收益，再考虑接入整模型。
