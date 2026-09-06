# GPU 推理采样与结果对齐

可选采样现已接入独立 Swift runner。它记录载入、预填充、单 token 解码的时间窗口，并由独立 `ane-telemetry` 进程读取 CPU、进程磁盘计数、存储驱动计数及可用的 IOReport 分档。Instruments 的 GPU 执行时间轴和计数器另行采集、离线对齐。

IOReport 使用本机私有接口，放在可选的独立采样进程中；启动后的订阅或读取失败会保留错误和缺失值，不改变模型计算路径。缺少采样器可执行文件、无法创建输出目录或无法启动子进程属于启动错误，在加载模型前报告。它不构成跨系统版本的公开兼容接口承诺。

**完整模型计算仍使用 Metal GPU；当前没有已验证的物理 DRAM 字节／带宽读数，也没有本模型 GPU 读写字节的实测结果。** 本页早期基准固定关闭 MTP；现有请求级进程采样也可记录显式 MTP，成本见 [请求摘要](docs/MTP_COST_SUMMARY.md)。命令缓冲逐步插桩仍限定 AR。采样能力、已读到某个计数器、目标进程的执行证据是三项不同检查。

新增的 [独立 MLX 命令缓冲计时](GPU_BOTTLENECK.md)已获得完整的 GPU 起止区间，与每个 token 的 CPU 时间对齐。这是另一套诊断证据，不改变下文旧 Instruments trace 的不完整标记；区间覆盖比例仍不能解释为 shader 活跃率或物理带宽。

## 已完成的验证

2026-09-05，在 M5 Max／128 GiB 机器上顺序执行同一可执行文件、同一 [1217-token 项目文档输入](fixtures/gpu-telemetry/provenance.json)，每个进程重复三轮、每轮输出上限 256 token。关闭逐阶段 profiler，默认 `reference` prefill 累加，单 SSD worker。两次进程的六轮生成 ID 完全相同，均生成 256 token 后因长度停止；最终 offset 1472，未触发 QSA。这只验证该请求的输出稳定性。

| 同一计时口径 | 不启用采样 | 启用 200 ms 采样 |
|---|---:|---:|
| 第二、三轮 decode 步数 | 510 | 510 |
| generation 报告的 decode 步耗时合计 | 13.6404 s | 14.0602 s |
| 步数／合计耗时 | 37.389 token/s | 36.272 token/s |

证据：[无采样基线](results/gpu-telemetry-validation/baseline.json)、[采样生成记录](results/gpu-telemetry-validation/sampled.json)、[执行参数与交接记录](results/gpu-telemetry-validation/run-ledger.json)。这一配对中采样运行的速率低约 **2.99%**、耗时高约 **3.08%**；顺序运行还受缓存、系统活动和热状态影响，不能把这两个数当作稳定、纯粹的采集开销。

[已保存的离线分析](results/gpu-telemetry-validation/sampled-analysis.json)采用 telemetry 叶步骤自己的时间边界，略多包含记录开销，得到暖 decode **36.268 token/s**；若用包括循环间隙的连续 phase 窗口，则为 **36.244 token/s**。不要把这些分母与上表混用。

同一分析的其他暖 decode 结果：

- 510 个步骤的延迟中位数 **27.345 ms**，P95 **31.139 ms**；残余 SSD 等待合计 **0.392910 s**，平均 **0.7704 ms/token**。
- 请求行的逻辑载荷为 **1,305,600 B**。完整落在 phase 内的进程磁盘读取增量为 **55,705,600 B**，相应观察窗口覆盖 **98.94%**；跨边界的增量不分摊。这两个字节数的含义不同，不能直接相除声称 SSD 放大率。
- CPU 累计用时 **8.0482 s**／覆盖窗口 **13.9214 s**，相当于 **0.578 个核心**；不是整台机器的 57.8% 利用率。
- 采样物理内存占用峰值 **79.207 GB**、resident 峰值 **24.522 GB**，均为十进制 GB。这是离散快照的最大值，不是瞬时真实峰值，也不能用 resident 表示全部 GPU 权重占用。

**GPU 时间轴已用本模型采到，但录制完整性未通过。** 首次 xctrace `--launch` 触发 Documents 目录的系统授权提示，未产生模型生成报告。随后正常启动 runner、向本次 PID **37997** 附加追踪，三轮生成成功，256 个 ID 均与无追踪基线一致。原始 15 秒追踪保存超时；用标准 `xctrace import` 从保留的 2.42 GB 原始附件恢复成功，再导出目标进程的 **44,999 条 GPU Active 区间记录**及 **69,050 条 CPU 提交记录**。嵌套记录会重叠，这不是独立内核调用次数。

导出的时间窗口为 **15.480702 s**，时间范围包含 `repetition=1` 的整个 **7.812484 s** decode，其他两轮 decode 未覆盖。该轮 38,500 条完整落入窗口的 GPU Active 记录求并集为 **6.137777 s**，仅代表已观察到的执行。不过原始录制未正常保存，不能确认没有丢事件；恢复导出明确携带 `--incomplete-source-trace`，因此仍有 `coverage.complete=false`。GPU 活跃率、完整空闲时间保持 `null`，不能将两个时间相除声称实测忙碌率。这次导出没有 GPU 带宽计数器样本，物理 DRAM 字节同样未知。独立空载能力探针仅在默认模板读到 `RT Unit Active`；另两个计数器配置被设备明确拒绝，见 [模板实测边界](templates/README.md)。

证据：[附加追踪生成记录](results/gpu-telemetry-attach/generation.json)、[恢复命令及结果](results/gpu-telemetry-attach/recovery-diagnosis.json)、[GPU 原始导出](results/gpu-telemetry-attach/recovered-export-final/gpu_intervals.json)、[同次离线分析](results/gpu-telemetry-attach/analysis.json)。追踪明显扰动执行，这次的 token/s 单独保留作诊断，不混入上表的正常吞吐。作者服务已按原参数恢复，MTP／drafter／PLD 均关闭；[交接记录](results/gpu-telemetry-attach/run-ledger.json)保存本次恢复事实，实时状态以 [experiment-status.json](../qwen38-ssd/results/experiment-status.json) 为准。

## 开始采样

以下命令从本目录运行。构建与完整模型的资源交接见 [GPU runner 文档](GPU_RUNNER.md#环境与构建)；完整 Swift 模型和作者完整模型服务应顺序运行。采样器不会替你管理作者服务。

```bash
cd /Users/tom/Documents/test/coreai-models/experiments/ane-runner

env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  xcrun swift build -c release
```

构建全部产品，确保 `ane-runner` 旁边有 `ane-telemetry`。可先执行不加载模型的短检查；这里的空闲数据不作为推理性能证据：

```bash
.build/release/ane-runner probe-telemetry \
  --telemetry-dir results/manual-idle-sampling \
  --seconds 2 \
  --output results/manual-idle-sampling.json
```

完整模型资源交接后，先保存无采样基线：

```bash
.build/release/ane-runner generate-gpu \
  --model-dir ../qwen38-ssd/models/Qwen3.8-Flash-Next-MLX-SSD-Stream \
  --tokens-file fixtures/gpu-telemetry/prompt-token-ids.json \
  --max-tokens 256 --prefill-chunk 64 --context 4096 \
  --prefill-accumulation reference --ssd-workers 1 \
  --profile-stages disabled --repeat 3 \
  --output results/manual-telemetry-baseline.json
```

退出前一次完整模型进程后，以相同参数另跑采样版本：

```bash
.build/release/ane-runner generate-gpu \
  --model-dir ../qwen38-ssd/models/Qwen3.8-Flash-Next-MLX-SSD-Stream \
  --tokens-file fixtures/gpu-telemetry/prompt-token-ids.json \
  --max-tokens 256 --prefill-chunk 64 --context 4096 \
  --prefill-accumulation reference --ssd-workers 1 \
  --profile-stages disabled --repeat 3 \
  --telemetry-dir results/manual-telemetry-sampled \
  --telemetry-interval-ms 200 \
  --output results/manual-telemetry-sampled.json
```

`--telemetry-dir` 必须是新目录，已有证据不会被覆盖。集成入口的采样间隔为 **50–10000 ms**，默认 **200 ms**；更短间隔会增加查询和文件输出，不能假设更短就更准确。`--telemetry-interval-ms` 不能脱离 `--telemetry-dir` 使用。独立 sidecar 的帮助显示更宽的底层范围；生成入口仍以上述范围为准。

生成期间，phase 事件在内存保存，signpost 实时发出，硬件 JSONL 持续写入。正常完成或可处理的抛错会结束自己启动的 sidecar；强行终止整个进程可能来不及保存 phase／生成报告。采样器启动后的故障由 `collector_status` 和错误字段记录，不应被当作模型计算失败或零流量。

启动会先等待完整硬件 baseline；若 3 秒内未取得，报告初始覆盖缺口再继续。最终目录通过原子创建取得独占权，并发任务不能共享、截断已有证据。上述两项生命周期保护是在本页全模型对照之后补上的，随后重新构建、通过 54 项 Swift 测试及 4 项实际 CLI 生命周期检查；没有重跑全模型。保留对照时的二进制 SHA-256，计算和 token 循环未作更改。最终检查见 [生命周期验证](results/gpu-telemetry-final-lifecycle-tests.json)、[构建记录](results/gpu-telemetry-final-build.log)、[Swift 测试](results/gpu-telemetry-final-tests.log)。

## 文件与离线分析

| 文件 | 内容 |
|---|---|
| 生成 JSON 的 `telemetry` | 绝对时间事件、连续请求窗口、run ID、PID、采样状态 |
| `hardware.jsonl` | metadata、baseline、增量 sample、summary；每种来源保留自己的读取窗口和错误 |
| `phases.jsonl` | metadata＋load/prefill/decode 叶区间；不含完整 request envelopes |
| `session.json` | 与生成 JSON 内同类的完整 telemetry 对象，包含 request windows |
| `collector.stderr.log` | sidecar 的诊断输出 |
| 离线分析 JSON | phase／暖 decode 聚合、原始采样与归属、coverage、输入文件路径和 SHA-256 |

优先使用完整生成报告进行分析。此命令只读取文件，不执行模型、MLX 或硬件采样：

```bash
python3 scripts/analyze_gpu_telemetry.py \
  --generation results/manual-telemetry-sampled.json \
  --hardware results/manual-telemetry-sampled/hardware.jsonl \
  --output results/manual-telemetry-analysis.json
```

默认暖轮次为 `repetition > 0`。`--warm-repetitions 1,2` 可以显式选择；这只是轮次定义，不证明 OS 页面或所有内核缓存命中。每轮会话状态重新创建，不是前缀缓存。

可选 `--phase-file PATH` 用 `session.json` 或 `phases.jsonl` 覆盖嵌入的 telemetry。只有 JSONL 时，分析器使用同 phase 首个叶步骤至最后叶步骤的连续包络，并标明来源；不会根据旧 generation 的相对耗时猜绝对时间。失败运行若只有保存下来的 `session.json`，可以把该同次文件同时传给 `--generation` 和 `--phase-file`，分析已保存的事件；缺少生成报告的来源字段会留空。不要借另一进程的报告补齐失败运行。

## Instruments GPU 执行时间轴

使用已安装的 Xcode 和本机可用模板。先正常启动本次模型，再对**这次实际 runner PID**执行 attach；在 shell 中将 `runner_pid` 设置为该整数，不从旧报告复制 PID。下例 15 秒仅是短诊断窗口，可能只覆盖部分预填充／解码，并不保证包含整个暖轮次：

```bash
python3 scripts/gpu_trace.py record \
  --template 'Metal System Trace' \
  --attach "$runner_pid" --time-limit 15s \
  --output results/manual-gpu-trace
```

`record` 输出目录必须不存在；它保存 `recording.trace` 并尝试导出。它只负责追踪，不负责结束模型或恢复服务。系统可能要求追踪或目录访问授权；本轮已遇到 `--launch` 的 Documents 授权阻碍，不能把等待授权的空 trace 当作推理。不要为采样修改系统保护或根据旧 PID 操作服务。

模型和追踪都结束后，可重新导出到一个新目录；`--target-pid` 应保持为本次模型 PID：

```bash
python3 scripts/gpu_trace.py export \
  --trace results/manual-gpu-trace/recording.trace \
  --target-pid "$runner_pid" \
  --output results/manual-gpu-trace-export

python3 scripts/analyze_gpu_telemetry.py \
  --generation results/manual-telemetry-sampled.json \
  --hardware results/manual-telemetry-sampled/hardware.jsonl \
  --instruments results/manual-gpu-trace-export/counters.json \
  --gpu-intervals results/manual-gpu-trace-export/gpu_intervals.json \
  --output results/manual-telemetry-trace-analysis.json
```

上面三个输入必须来自被 attach 的**同一次**生成。示例使用前节的采样运行；如启动了另一轮，需一起更换 generation、hardware 和 trace 路径。不能把桌面其他进程的 GPU 活动或旧 trace 拼入本次结果。

## 指标含义与覆盖边界

| 指标 | 来源与允许的解释 |
|---|---|
| `successful_step_seconds` / `phase_window_seconds` | 前者累计成功叶步骤，后者包含 token 循环间隙；load、prefill、decode 分开 |
| `per_step` 的 forward／evaluation 时间 | 主机前向包含图构建、SSD 等待和可能的早期 GPU 提交；evaluation 包含求值、同步／读取，不是纯 GPU 时间 |
| `ssd_wait_seconds` | GPU 已有机会与读取重叠后，主机取结果时的残余等待；不能再与计算耗时相加 |
| `ssd_requested_row_bytes` | n-gram 行的逻辑载荷，包括重复行；不是物理磁盘或 DRAM 流量 |
| process disk delta | `libproc` 的目标进程磁盘累计值之差；不是仅 PLE 文件的计数 |
| system disk delta | IOKit 中观察到的存储驱动求和，虚拟设备和背后设备可能重叠，也含其他进程活动；不是物理 NAND 总量 |
| PMP histogram | IOReport 的系统带宽驻留分档趋势；保留分档名、原值、来源与 `estimated`，不换算 DRAM GB/s |
| CPU `one_core_fraction` | CPU user＋system 增量／覆盖墙钟时间；可大于 1，不除以机器核心数，也不称全机百分比 |
| sampled memory peak | 目标进程快照的最大 physical footprint／resident；不等于瞬时峰值或 MLX allocator 峰值 |
| GPU Active union | Instruments 中严格匹配目标 PID 的 `Active` 执行区间求并集，避免嵌套或多个通道重复累计 |
| CPU command-buffer submissions | CPU 侧提交区间，单独展示，不能当作 GPU 执行 |
| GPU counters | 保留 `(counter_id, group_index)`、原单位和设备范围；设备全局或 GPU/cache 接口计数不能直接归属本进程或改名为 DRAM |

所有 phase 使用 `mach_absolute_time` 换算后的纳秒。Instruments 通过导出的 `time-info` 确定 trace-relative 到绝对时钟的映射，保留原始锚点和舍入不确定性；不把启动 xctrace 的时刻当成 trace 零点。

硬件 delta 只有整个来源观察包络落在某一 phase 才被累加。进程、IOKit、IOReport 的保守包络优先用“上一来源读取开始→当前来源读取结束”，避免把采样器整轮结束时间误当作每个计数器的读取时刻。跨 phase 的记录标为 `partial`，保留原值及重叠时间，**不按时间比例分摊字节或 histogram 档数**。完全位于一个 decode phase 内的 200 ms 采样可以覆盖多个 token，不要求塞入某一个 27 ms 步骤。

`null` 表示不可用或证据不足；有效采样得到的 `0` 才是零。GPU trace 还需要独立、完整的 trace 窗口及正常导出证据：已知时间范围并不证明没有丢事件。存在 run issues、导出表缺失或覆盖不完整时，只保留观察到的执行并集；GPU 活跃率、真实空闲总时长和最大空闲间隙保持 `null`。两个已观察执行区间之间的空隙也不能直接称为已证实的 GPU 空闲。

prefill 含最后一个保留的 prompt token 和首次采样；decode 步数为已生成 ID 数减一，ID 可能包含 EOS。`qsa_active_layers` 只反映最终缓存状态，不能用于推算稀疏物理读量。MLX `memory.peak_bytes` 是进程累计 allocator 峰值，不是本轮独立峰值。

`--profile-stages synchronizedStages` 会逐阶段同步并改变正常的重叠与调度，只用于单独诊断；其等待不能直接当设备执行时长。`hostBodyOnly` 也不是 GPU 时间。比较改动前后的正常吞吐时，保持 `--profile-stages disabled`，分别保存有无 telemetry／Instruments 的结果，并核对生成 token。

离线分析器的 28 项测试覆盖 phase 边界、跨步骤窗口、缺失／零值、进程身份、CPU counter reset、时钟不确定性、GPU 嵌套并集及 trace 不完整时拒绝零利用率；它们不执行完整模型：

```bash
python3 -m unittest discover -s scripts -p test_analyze_gpu_telemetry.py
```

另有 9 项 trace 导出测试、8 项本机 sidecar 检查及 4 项 runner 生命周期检查。整合结果、同输入输出校验及证据哈希见 [机器可读汇总](results/gpu-telemetry-validation/milestone.json)。完整 GPU 覆盖和物理 DRAM 带宽两项尚未通过，不与代码构建或单元测试通过混为一谈。
# GPU 状态补充（2026-09-07）

硬件 sidecar 现在附带 `gpu_states`：一次性订阅系统 `GPU Stats / GPU Performance States` 和 `CLTM-induced GPU Performance States`，并读取 `NSProcessInfo.thermalState`。本机初步发现分别为 GPUPH 的16个状态、GPU_CLTM的17个状态，原生单位 `24Mticks`；状态名和增量原样保存。两路订阅在当前用户权限下可用，缺失或失败保留显式错误，不申请提权。

每路保存本次及前次读取的 start/end。做阶段归属时只使用整个端点包络落在阶段内的增量；发生失败会断开差值链。baseline 没有 delta，不能记成零。采样循环仍由目标 PID 身份控制，但这些 GPU 状态是系统范围，不能归属该 PID；多个通道不能相加。P1…P15 尚无经过核对的频率映射，不导出 MHz、shader 利用率或实际 DRAM 带宽。thermalState 是操作系统压力等级，不是摄氏温度；nominal 也不证明 GPU 没有调节频率。

固定 AR [命令跨度诊断](docs/GPU_DRIFT_TRACE.md)把主要漂移收窄到 GPU 执行跨度内，因而补上这两路原始证据。新增采样已构建，并在空闲参考服务旁完成 baseline+3次采样：端点包络有效、两路 delta 完整、thermal nominal。原始记录 `results/gpu-states-v1/idle.jsonl`。这只是读取能力检查，尚不是生成阶段的降速解释；下一轮真实 workload 将同时记录这些字段。
