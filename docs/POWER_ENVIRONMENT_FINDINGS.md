# 供电环境与后续性能复测

2026-09-07北京时间10:17–10:20只读检查。当前M5 Max机器报告外部电源协商值为**40 W**；三个离散接电快照的电量从100%、73%到45%。在追加性能对照前，应先核对实际充电器、线材和连接路径。当前证据尚不能把此前吞吐下降归因于供电。

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

本轮后续优先完成服务错误恢复、日志和请求生命周期；暂不追加优化测速循环。用户醒后可先核对为何当前路径协商到40 W；如果更换或确认供电条件，再把它记录为新的实验环境做固定对照。接电状态本身不再作为“供电已受控”的充分依据。此次没有修改任何电源、冷却或系统设置，没有停止其他应用，也没有启动持续采样器。
