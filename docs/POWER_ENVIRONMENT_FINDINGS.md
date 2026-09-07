# 供电环境与后续性能复测

2026-09-07北京时间10:17–10:20只读检查。当前M5 Max机器报告外部电源协商值为**40 W**；三个离散接电快照的电量从100%、73%到45%。当前证据不能把此前吞吐下降归因于供电。11:07左右用户明确说明正在有意使用低功率适配器，并指出这不影响性能；40 W是预期使用背景，不应列为已确认的性能问题。

## 实际观察

机器标识为`Mac17,7 / Apple M5 Max`。10:20:28时，AppleSmartBattery报告`ExternalConnected=true, IsCharging=true, CurrentCapacity=45, MaxCapacity=100`，`AdapterDetails`为`Watts=40, AdapterVoltage=20000, Current=1990`。这是系统报告的电源协议信息，未测墙插功率，也未核对充电器铭牌功率。

| 北京时间 | 原始记录 | 电源与电量 |
| --- | --- | --- |
| 09-07 07:41:14.942 | MTP窗口A冻结前 | AC，100%，finishing charge |
| 09-07 08:32:04.836 | MTP窗口B冻结前 | AC，73%，charging |
| 09-07 10:20:28.362 | 本次白名单快照 | AC，45%，charging；协商40 W |

三个端点不能证明全程连续接电或始终使用同一适配器，A/B旧记录也没有40 W字段。不能用新快照补写旧窗口的功率配置。原始来源为`results/mtp-release-window-{a,b}/environment-before-freeze.json`和`results/power-environment-readonly-v1/snapshot.json`，详细只读说明位于后者同目录的`NOTE.md`；未保存设备序列号。

当前Foundation报告thermal为nominal、低电量模式为false；pmset没有记录thermal/performance warning。这不是温度、频率或“不降频”的证明。已看到若干桌面应用的CPU活动，未取得每进程GPU归属或明确并行训练证据；不能由CPU百分比认定GPU竞争，更不能据此停止其他应用。历史GPU状态与阶段的关联仍只按[既有分析](GPU_STATE_PHASE_ANALYSIS.md)解释。

## 对本轮工作的影响

[MTP两窗口](MTP_RELEASE_WINDOWS.md)、[PD到达实验](PD_DECODE_ARRIVAL_EXPERIMENT.md)和[AR提前提交](DECODE_SUBMISSION_FEASIBILITY.md)已经出现超过预设范围的基线漂移。保留全部原始轮次和既有分类，不重新挑选有利样本，不把上述供电观察当成因果结论，也不放宽性能门槛。

10:20时root曾据这些观察暂缓追加优化测速、优先完成服务活性。用户返回并补充低功率适配器的使用背景后，撤回“先确认供电再做性能复测”的前置要求。下一轮直接针对已观测到的性能漂移做固定工作负载和分阶段诊断，不将供电作为默认解释。既有快照只作为环境记录，不证明有影响，也不独立证明完全无影响。此次没有修改任何电源、冷却或系统设置，没有停止其他应用，也没有启动持续采样器。
