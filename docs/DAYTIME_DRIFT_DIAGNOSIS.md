# 同一窗口中的 AR 降速、GPU 命令跨度与状态观察

2026-09-07，M5 Max、macOS 26.6.2。**固定五轮请求再次出现较温和的降速：排除预热后，decode 从 31.719 降到 30.521 token/s，下降 3.777%；增加时间的 90.95% 落在 GPU 命令跨度内。** 本次将现有命令跨度与 GPU 状态采样放在同一组请求中，支持继续定位和局部候选筛选；它不是无采样性能基准，也没有建立供电、温度或物理带宽的因果解释。

## 条件与完整性

单进程 PID35734 连续五轮，每轮重新创建状态，固定11,057个输入token、chunk416、context16,384、128个输出。AR/reference，SSD worker1/nextChunk，wired disabled，prefill每4层求值；MTP head未加载，async提交实验为0且调用数0，详细profiler关闭。命令计时使用已有独立MLX库，同时启用现有500ms telemetry。第0轮固定预热；第1、2、3、4轮全部保留，没有替换较慢样本。

独立检查12项通过，包括完整输入与输出、配置/分母/结束状态、原生时间、全部步骤跨度，以及状态计数的原始重算：

- 五轮完整11,057个输入ID和128个输出ID均与冻结golden一致，finish=length、offset11184、QSA活跃12层。首个输出来自prefill，因此每轮decode分子为127；decode秒数等于127个原有步骤时间之和。
- 每轮28个prefill步骤（26×416、240、1）和127个decode步骤，共775个有序且不重叠的CPU窗口。另有telemetry的load窗口。全部775个步骤从原始GPU区间重新裁剪、求并集，与现有分析器逐项一致。
- 原生记录226,045/262,144槽，序号1…226,045唯一连续；complete=true，errors/dropped/pending/失败状态/缺时间/时钟不匹配均为0。每条状态为Completed，GPU时间落在commit/completion包络内，采用原有1微秒容差。分析器complete=true、errors=[]。
- 生成、native、telemetry的目标PID一致，collector完整结束、零丢事件。hardware有baseline0及205条连续sample；两种状态通道均可用。独立重算11个阶段×2通道的原始计数、完整端点包络并集、覆盖比例，以及11个阶段的thermal观察次数，全部与分析器一致。

实际runner SHA为`b0391391b4af4dbdcd31bb16cffbf268e790112129e4887c3ac84ef547be1bc1`，诊断MLX库SHA为`6615954290a25fe76e783212f92c25786874ead5a7ceaa04d02f5c3421d36fb5`。controller记录运行07:36:52.051至07:38:39.384 UTC，正常退出、owned group清空；reference PID35851在07:38:55.727 UTC确认恢复。该PID只表示本轮交接时点。

主代理完成287个冻结文件和102个模型payload的大小/mtime检查。此计数**不覆盖所有模型metadata**：本次plan没有将`config.json`、`model.safetensors.index.json`、`tokenizer.json`、`chat_template.jinja`列入该287项。生成报告保存这四项加载前SHA；本次独立分析又读取四项，均一致，且模型目录与执行argv一致。这个加载前/分析时对照不追溯扩大原冻结清单，后续实验应将四项补入。

## Prefill 单独统计

单位为秒。GPU跨度是各步骤内裁剪后的命令缓冲区间并集；跨度外使用同一CPU步骤窗口作差。目标prefill计时与命令窗口的时钟调用相邻，二者只有数微秒差异，不能强制视为同一数值。

| 轮次 | target / total prefill | 步骤墙钟 | GPU跨度 | 跨度外 | PLE残余等待 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0，预热 | 17.574774 / 17.579488 | 17.574780 | 11.073270 | 6.501510 | 0.516176 |
| 1 | 13.370874 / 13.374952 | 13.370882 | 12.304672 | 1.066209 | 0.543446 |
| 2 | 13.739777 / 13.744957 | 13.739784 | 12.703736 | 1.036048 | 0.551759 |
| 3 | 14.580049 / 14.581824 | 14.580057 | 13.504385 | 1.075672 | 0.552374 |
| 4 | 14.869257 / 14.871036 | 14.869266 | 13.794769 | 1.074497 | 0.538429 |

暖1→4，步骤墙钟增加**1.498384秒**，GPU跨度增加**1.490097秒（99.45%）**，跨度外只增加8.287毫秒。28个相同chunk位置中，27个GPU跨度变长、26个墙钟变长；GPU跨度比值中位数1.12458。PLE残余等待减少5.017毫秒，不能解释本次prefill增加。

所有暖轮prefill的`buffer_ops`同为150,175，命令缓冲数在14,284–14,294之间；这只是不变的MLX操作记账，不能当成shader数或物理流量。prefill中已有每四层求值，所以forward时间同时包含求值和等待，不是纯CPU构图时间。

## Decode 单独统计

decode计算吞吐沿用生成报告的127/原有步骤秒数，排除加载、prefill和逐轮报告工作。表中的“步骤墙钟”来自独立command marker，与该分母相差每轮约40–50微秒，不改写原吞吐定义。

| 轮次 | decode token/s | 步骤墙钟秒 | GPU跨度秒 | 跨度外秒 | PLE残余等待秒 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0，预热 | 32.183797 | 3.946036 | 3.419685 | 0.526351 | 0.142543 |
| 1 | 31.718976 | 4.003861 | 3.452522 | 0.551339 | 0.144246 |
| 2 | 31.297722 | 4.057764 | 3.520940 | 0.536825 | 0.155534 |
| 3 | 30.472128 | 4.167701 | 3.585226 | 0.582474 | 0.157870 |
| 4 | 30.520956 | 4.161026 | 3.595468 | 0.565557 | 0.156222 |

暖1→4，步骤墙钟增加**157.164毫秒**，GPU跨度增加**142.946毫秒（90.95%）**，跨度外增加14.219毫秒。127个相同token位置中，125个GPU跨度变长、124个墙钟变长；GPU跨度比值中位数1.04218。全部暖轮`buffer_ops`同为374,084；首末命令缓冲数30,810→30,958，不能用操作数量变化解释主要增量。

单步中位数31.165→32.468毫秒，GPU跨度中位数27.084→28.260毫秒，forward中位数3.201→3.292毫秒，evaluate/readback中位数28.162→29.315毫秒。forward和evaluate区间可与GPU重叠，不能额外相加。

跨度外的增加中，“下一buffer未提交”的间隙增加9.059毫秒，“已提交”的间隙增加4.914毫秒；其余包括最后buffer后的步骤尾部。连续decode窗口与步骤和之差3.336→3.271毫秒，未随降速增加。这里没有线程栈，不能把间隙全算成可删除的Swift开销。

PLE残余等待增加**11.976毫秒**。它围绕`pending.wait()`计时，之前已有`asyncEval([h])`，所以这部分等待可能与GPU重叠，不能在157.164毫秒上再次相加、或直接扣除后声称CPU/SSD各占多少。五轮prefill/decode逻辑请求字节分别恒为28,305,920/325,120；这不是物理SSD流量。

## 同一窗口中的 GPU 状态

下表展示GPUPH主要原始状态权重；分母包含所有有效状态，包括OFF。括号为完整来源端点包络对该阶段的覆盖率。P标签保持原名，原单位为`24Mticks`，没有换算MHz。

| 轮次 | prefill GPUPH主要权重（覆盖） | decode GPUPH主要权重（覆盖） | decode thermal观察 |
| --- | --- | --- | --- |
| 0，预热 | P13 47.10%、P9 21.85%（99.47%） | P13 86.60%、P12 5.36%（78.22%） | nominal 7次 |
| 1 | P8 54.80%、P9 18.88%（96.25%） | P13 52.60%、P12 25.49%、P11 12.73%（89.86%） | nominal 8次 |
| 2 | P8 60.75%、P7 18.17%（97.43%） | P12 38.58%、P13 33.96%、P11 18.13%（88.76%） | nominal 8次 |
| 3 | P7 54.09%、P8 24.91%（95.52%） | P10 28.48%、P11 26.96%、P12 25.24%（86.37%） | nominal 8次 |
| 4 | P7 51.90%、P6 17.16%（96.96%） | P12 31.41%、P11 27.77%、P10 21.21%（86.71%） | fair 8次 |

另一个GPU_CLTM通道的decode `NO_CLTM`原始权重依次为96.45%、62.48%、46.15%、21.78%、21.79%；它只是系统报告的通道与状态名称，不能据此单独认定某项限制造成降速。第4轮prefill记录nominal28次/fair1次，其余prefill均为nominal。thermal是离散观察次数，不是温度或各档持续时间；第3→4轮decode速度略回升，也不能将一次类别变化当成速度的确定解释。

分析排除28个跨阶段/窗口外通道delta、2个没有delta的baseline通道，以及4个跨阶段/窗口外thermal观察。有效状态包络只归属完全包含它的阶段，不按相交时长分摊。它们是系统范围计数，无法唯一归因于runner；覆盖率不是GPU活跃率。当前结果支持“跨度增加与状态分布变化同时出现”的观察，仍不能区分GPU时钟、访存停顿、驱动调度或外部资源竞争的因果份额。

五轮MLX allocator active均为78,872,152,378字节，peak均为79,839,639,326字节。这是稳定的容量快照，不能用来证明没有分页、没有其他进程竞争或不存在系统内存问题。

## 当时的后续候选（当前优先级已调整）

2026-09-08 用户将 MTP 性能优化放到整体计划后段。以下是本次诊断当时支持的候选与判断，保留作后续研究依据；不再作为立即开展 MTP verify 优化的任务，当前先推进[主计划](UPSTREAM_ADOPTION_PLAN.md#当前实施顺序2026-09-08调整)中的 AR 服务、完整状态缓存和基础性能工作。

**足以开始小候选的真实局部筛选。** 本次暖轮首末decode下降3.777%，全部四个暖轮最大/最小差相对首暖轮为3.931%，比历史30%以上漂移温和。但这是一个带采样的短窗口，没有候选对照，不能把“小于5%”单独写成某项优化通过，也不能证明未来窗口稳定。

下一步可直接在现有真实单层/单步fixture上先过完整数值与状态门槛，再用固定交错顺序评估一个MTP verify候选。需要整模筛选时，用该候选**同一窗口**的AR锚点和实际committed decode token分母，保留既有5%漂移门槛；prefill/history/decode继续分别统计。不拿本次trace充当无采样baseline。3%级收益与本次观察到的漂移接近，必须由局部交错及后续无详细插桩配对支持，不能从历史最快/最慢轮拼出收益。

本次没有直接分解MTP verify的GPU/host成本，不能把90.95%照搬到verify。已有MTP成本摘要已足以确认verify值得优先研究；无需为了继续一个有界候选先完成供电因果或重新跑192轮。async8仍关闭，驻留和kernel默认保持。

## 原始证据与复核方法

执行合同是[最终五轮plan](../results/daytime-drift-v1/run/plan.json)，不是较早三轮建议稿。原始记录包括[generation](../results/daytime-drift-v1/run/generation.json)、[commands](../results/daytime-drift-v1/run/commands.json)、[hardware](../results/daytime-drift-v1/run/telemetry/hardware.jsonl)、[交接记录](../results/daytime-drift-v1/run/run-ledger.json)和[postflight](../results/daytime-drift-v1/run/postflight-and-release.json)。结果目录被Git忽略，本页保留可公开审阅的结论与边界。

本次只运行已有CPU分析器，无新GPU实验：

```sh
python3 -B scripts/analyze_gpu_command_timing.py \
  --generation results/daytime-drift-v1/run/generation.json \
  --commands results/daytime-drift-v1/run/commands.json \
  --skip-first-trials 1 --output results/daytime-drift-v1/run/command-analysis-new.json

python3 -B scripts/analyze_gpu_state_phases.py \
  --report results/daytime-drift-v1/run/generation.json \
  --output results/daytime-drift-v1/run/state-analysis-new.json
```

实际输出为同目录`command-analysis.json`、`state-analysis.json`；完整原始区间/状态独立重算、12项检查和metadata补充身份对照在`independent-audit.json`。generation SHA为`45660918e7fa480642e1d1af814573f5ed49df2a592938a27323e09066e7c9a7`，hardware SHA为`c4357a509ee039fd495562e73bcc2d9570f9b4e4402e15216b8e4205403d9cc8`，其余源文件SHA保存在分析产物中。
