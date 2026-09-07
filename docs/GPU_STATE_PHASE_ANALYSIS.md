# GPU 状态与长请求阶段的对齐观察

2026-09-07。窗口A的前两个完整进程（同一11,057-token输入、128输出预算）显示：普通AR降速时，系统GPU状态分布也明显变化。这个关联值得保留；原始P状态标签不是MHz，系统范围计数也不能单独归因于runner。MTP发布仍由原定配对/漂移门槛判断，不用这里的状态分类筛掉慢轮。

## 可复用的只读分析

[`analyze_gpu_state_phases.py`](../scripts/analyze_gpu_state_phases.py)读取生成报告及其指定的hardware sidecar，复用既有绝对阶段划分；不运行采样、不向进程发信号、不加载模型。输出保留输入文件与分析源码的SHA256。

```bash
python3 scripts/analyze_gpu_state_phases.py \
  --report results/mtp-release-window-a/original-128-part1.json \
  --report results/mtp-release-window-a/original-128-part2.json \
  --output results/mtp-release-window-a/state-analysis-new.json
```

它要求生成进程、telemetry和sidecar的时钟/PID关联一致、完整采样终态、唯一baseline0与连续整数sample编号；当前来源读取还必须落在本次collection sweep内。复用MTP窗口的阶段合同核对load→各请求prefill/decode顺序及叶步骤计数，因而仅接收同时含这两个阶段的完整生成记录，不能把交换后的AR/MTP标签继续当作有效结果。只把“上一来源read开始→本次read结束”的完整端点包络分配给完全包含它的单一phase；跨阶段的delta不按时长分摊。每路通道单独累计原始状态权重，保留OFF，不相加不同通道。无有效样本时不制造零利用率；有效但全零的状态增量保留原始零，归一化比例为null。thermal只计实际观察类别的次数，不当成各等级持续时间。

`envelope_coverage_fraction`是完整归属端点包络的时间并集/阶段窗口，不是GPU活跃率；包络也不是精确硬件读取时刻。prefill窗口包含适用的head history和host间隙，详细target/history时间继续使用生成报告自身的独立字段。该工具不替代token/功能与MTP发布分析器。

独立CPU复核逐项重算两份真实报告的原始counts、端点并集和thermal计数；修订后结果不变。30项坏输入/边界控制通过，覆盖进程身份、读取越出采集区间、阶段错序、baseline与sample编号、断开的端点链、负计数、跨阶段不分摊和全零比例为null；另两次CLI控制确认NaN/重复键坏文件不会丢弃同批正常结果。首次审阅发现的身份/归属检查遗漏及修复后证据分别保留在`independent-state-audit.json`与`independent-state-audit-v2.json`，这些CPU检查没有运行新模型。

## 前两个完整进程的观察

两个进程分别有398和430条采样，加各自baseline，采集完整、无丢事件。前者排除44个、后者排除30个跨阶段或窗口外的通道delta；各有2个无delta的baseline通道。下面仅列四个**测量AR**请求的decode状态，prefill时间另列；不是将连续请求当成独立硬件窗口。

| 进程/rep | AR decode token/s | prefill target 秒 | GPUPH主要原始状态权重 | decode包络覆盖 | thermal观察 |
| --- | ---: | ---: | --- | ---: | --- |
| part1 / 2 | 30.786 | 14.458 | P12 44.36%、P11 30.72% | 87.17% | nominal 8次 |
| part1 / 9 | 26.203 | 17.658 | P9 28.77%、P6 25.61%、P7 22.39% | 85.15% | fair 9次 |
| part2 / 2 | 21.197 | 26.383 | P3 46.46%、P4 25.99%、P2 17.96% | 94.93% | fair 12次 |
| part2 / 5 | 21.108 | 26.944 | P4 41.59%、P3 33.50%、P2 19.44% | 94.67% | fair 12次 |

状态权重分母包含所有有效状态（含OFF），表中只展示主要几项。对应GPU_CLTM通道也发生变化；part2测量AR的主要标签及权重与GPUPH接近。没有把`CLTM-induced`通道名当作本实验独立建立的降速因果，也没有做未经校验的频率、温度或DRAM换算。

这16轮输入/输出IDs与原始golden一致，首个case的三组实际decode比值为1.1644/1.2842/1.2815，前后AR漂移分别2.4416%/12.3403%/0.4213%。第二组继续标为无法判定；第三组在绝对速度较低时仍有稳定的本组收益。**这只完成一个case，不能宣称两窗口或MTP发布门槛通过。**

原始证据位于`results/mtp-release-window-a/`，包含两个生成JSON、各自telemetry/hardware.jsonl、`state-attribution-original128-v2.json`和`partial-after-original128.json`。固定版本与完整实验合同见[MTP两窗口](MTP_RELEASE_WINDOWS.md)。此前[命令缓冲诊断](GPU_DRIFT_TRACE.md)来自另一轮请求，只提供方向性背景，不能拼入本轮时间轴。桌面其他应用仍照常运行，未更改供电或散热设置。
