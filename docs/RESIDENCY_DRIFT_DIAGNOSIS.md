# 连续运行降速与驻留策略诊断

2026-09-07。固定 11,057 输入 / 128 输出、chunk416、reference AR 的四个计量请求中，decode 从 29.79 降到 24.78 token/s；中间开启 `fit` 仍然继续下降。本轮不支持改变默认驻留策略，也没有把降速定位到分页、SSD、训练或热降频。

## 配置与结果

同进程先 D1 暖机一次，让 MTP head 留在相同权重集合内，再 AR fit 暖机一次；随后四次均为 AR，顺序 disabled / fit / fit / disabled。全程使用现有 500ms 进程 telemetry，原始结果为 `results/residency-drift-v1/generation.json`。独立重算为同目录 `independent-resource-summary.json`，六次全部 128 token IDs 与已有 golden 相同，结束原因、offset11184、QSA 状态门槛通过。

| 计量顺序 | 驻留策略 | Prefill token/s | Decode token/s | Decode 进程 pageins | Decode 进程 faults |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | disabled | 690.02 | 29.79 | 0 | 29 |
| 2 | fit | 666.30 | 28.75 | 0 | 42 |
| 3 | fit | 626.70 | 27.58 | 0 | 29 |
| 4 | disabled | 535.75 | 24.78 | 0 | 38 |

进程计数只用完整落在阶段内的采样区间；边界交叉区间排除，不按时间比例分摊。Prefill 覆盖约96%–97%，decode 约89%–97%。两个来源的 pageins 和 COW 均为0；这不排除未采样时间、驱动或系统其他进程的行为。累计 CPU 时间没有随墙钟明显上升，RSS 约24.784GB，decode physical footprint 约81.62GB，没有与降速对应的持续增长。Prefill 约1.8GB进程逻辑读取不能当成物理 SSD 流量。

采集正常结束，dropped_events=0。`rusage.wired_bytes=0` 是另一口径的进程 gauge，不能据此说 Metal residency 没生效。

## 策略含义与时钟限制

固定 MLX `1f8e74e3f12f31365464a6867c6579f0e9b29d85` 源码中，allocator / residency 初始上限为0；CLI 报告与此一致。`disabled` 仍在请求计时外同步；`fit` 还会清理 allocator cache 并设置进程内 residency 上限。因此这是完整策略对照，不能将差别单独归因于 pin 内存。fit 目标约80.897GB，也不能宣称覆盖 active81.005GB / peak81.982GB的全部临时分配。

额外系统 VM 采样使用本机 Python3.9 `time.monotonic_ns`，实际起点相对进程，而 Swift 使用原生 mach absolute 时钟；当时未记录共同锚点。原文件保留，禁止事后凭秒级 UTC 强行对齐。全实验窗口（包括加载和暖机）系统压缩约+345万页、解压+344万页、pageins+350万页、swap无变化，无法归属某个请求或阶段。未来如再采系统 VM，应直接采 mach_absolute_time 与 timebase，并记录调用区间。

## 下一步

使用已有独立命令缓冲计时库，固定 AR、相同输入，比较连续请求中 GPU command-buffer span 与 span 外时间怎样变化。完整性必须确认同进程、时钟有效、所有 buffer 完成、dropped/pending 为0。该 trace 用于缩小降速位置，计时本身有开销，不作为新的无插桩吞吐成绩；buffer span 也不等于 shader 利用率或物理带宽。
