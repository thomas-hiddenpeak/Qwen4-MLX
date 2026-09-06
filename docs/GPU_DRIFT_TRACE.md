# 固定 11k AR 的 GPU 命令跨度漂移诊断

2026-09-07，M5 Max，macOS 26.6.2。**暖轮变慢主要发生在 GPU 命令缓冲的执行跨度内；不能据此归因为热降频、物理带宽或其他进程。** 本次是有采样的定位实验，不是正常吞吐基准。

## 条件与完整性

同一进程 PID 6141 连续三轮；第一轮作为预热，重点比较 repetition 1→2。固定 11,057 个输入 token、chunk 416、输出 128、context 16,384；AR/reference、wired disabled、prefill 每 4 层求值，关闭 detailed profiler 和其他 telemetry。每轮重新创建状态，未启用前缀缓存、MTP 或新 MoE/GDN 实验。

- [生成记录](../results/gpu-drift-trace-v1/generation.json)、[原始命令](../results/gpu-drift-trace-v1/commands.json)、[计划](../results/gpu-drift-trace-v1/plan.json)、[交接记录](../results/gpu-drift-trace-v1/run-ledger.json)。实验结束，交接记录确认参考服务恢复。
- 原生记录 **135,848 / 262,144** 槽，complete=true；dropped、pending、failed status、missing timestamp、clock mismatch 均为 0。独立重算确认序号唯一且连续、全部状态为 Completed、GPU 时间位于 commit/completion 包络内（原生容差 1 μs）、记录在采集会话内。
- generation/native 的 PID 与 hook 版本一致；现有 `analyze_gpu_command_timing.analyze(..., skip_first_trials=1)` 返回 complete=true、errors=[]。
- 共 **465 个 CPU 步骤窗口**：每轮 28 个 prefill（26×416、240、最终保留的 1 token）和 127 个 decode。三轮各 128 个生成 ID 全部匹配冻结 [golden](../results/prefill-agent-11k-default-check/default.json)，最终 offset=11184、QSA active layers=12、finish=length。
- 三轮 allocator active=78,872,152,378 B、peak=79,839,639,326 B；这是容量统计。纯 AR 未加载 MTP head，不能与上一轮 residency 实验直接配对比较容量或 token/s。

## 分阶段结果

下表累计同一阶段各步骤的墙钟和裁剪到步骤内的 GPU 跨度并集，单位秒。区间外时间为两者之差；CPU forward 与 GPU 重叠，不能额外相加。

| 阶段 | repetition | 步骤墙钟 | GPU 跨度并集 | 跨度外 | 命令缓冲数 |
|---|---:|---:|---:|---:|---:|
| prefill | 0，预热 | 19.638112 | 11.349475 | 8.288637 | 14,289 |
| prefill | 1 | 13.630817 | 12.615104 | 1.015713 | 14,281 |
| prefill | 2 | 14.460378 | 13.425683 | 1.034694 | 14,284 |
| decode | 0，预热 | 3.932765 | 3.439112 | 0.493653 | 31,123 |
| decode | 1 | 3.985815 | 3.464241 | 0.521575 | 31,036 |
| decode | 2 | 4.266935 | 3.680146 | 0.586789 | 30,831 |

暖轮 prefill 增加 **829.561 ms**，其中 GPU 跨度增加 **810.580 ms（97.71%）**，跨度外增加 18.981 ms。相同位置的 28 个 chunk 中，26 个 GPU 跨度变长、27 个墙钟变长；并非只由一个异常长尾解释。两轮 `buffer_ops` 同为 150,175；该字段是 MLX 记账操作数，不是 shader 数或流量。

暖轮 decode 增加 **281.120 ms**，其中 GPU 跨度增加 **215.905 ms（76.80%）**，跨度外增加 65.214 ms。**127 个相同 token 位置的 GPU 跨度全部变长**，相同位置跨度比值中位数为 1.06368。两轮 `buffer_ops` 同为 374,084，命令缓冲数量减少 205；没有“更多算子或更多命令缓冲解释变慢”的迹象。

decode 单步中位数由 31.208→33.483 ms，GPU 跨度中位数由 27.182→28.994 ms，跨度外中位数由 4.005→4.496 ms；forward 墙钟中位数 3.187→3.503 ms，evaluate/readback 28.044→30.040 ms。生成报告的采样吞吐为 32.2925 / 31.8626 / 29.7634 token/s，只用于标识本次诊断。

连续 decode 窗口比步骤之和多 0.877→1.081 ms，增加仅 0.204 ms。步骤内“下一条缓冲尚未提交”的间隙累计增加 61.921 ms，“已经提交”的间隙增加 2.404 ms，前者是跨度外增量的主要位置；它仍可能包含 host 构图/编码、调度等工作，不能全部记作可消除的 Swift 开销。此分类不包括最后一条 GPU 缓冲之后的步骤尾部。

prefill 的连续窗口覆盖为 92.54%→92.84%，decode 为 86.90%→86.23%。这些是命令缓冲跨度覆盖，**不是 shader 活跃率或 DRAM 带宽占用率**。prefill 的 forward 内有分层求值，因此其 forward 中位数 489.616→525.798 ms 也不是纯 CPU 构图时间。

prefill SSD 残余等待为 0.561883→0.545585 s；decode 残余等待为 0.147139→0.146253 s。二者均未随暖轮变慢增加。第一轮首 prefill chunk 达 7.124 s，明确排除于暖轮比较，不擅自归因为某种初始化活动。

## 结论边界与下一步

这次定位将主要漂移收窄到 GPU 命令执行跨度内部，同时存在较小的 host/提交侧增量。原生时间不能再区分 GPU 时钟、访存停顿、GPU 调度或外部共享资源影响；本次没有采集频率、功耗、温度或物理 DRAM 计数。不能凭此建议改变 kernel、认定热降频或认定训练干扰，也不能把比例套用到无采样吞吐。

现有结果不足以声称 wired fit 能阻止漂移，或改变 ANE/GPU 分工就能解决它。后续优化应继续使用相同条件交错比较；若继续追根因，需要补直接证据，而不是将本 trace 的 token/s 混入性能门槛。

## 诊断库与复现

复用 [已有独立库的构建记录](../results/gpu-bottleneck-v1/native/build-provenance.json)，不重建、不覆盖 stock runtime。库只替换复制后的 `device.cpp`，在原有 completion handler 记录 GPUStartTime/GPUEndTime；原算子对象只读复用。当前 stock MLX/MLXC/metallib、原 device.cpp、timing header 和诊断产物均与该构建记录的 SHA-256 一致。

- pinned stamp：`mlx=1f8e74e3f12f mlxc=56b2d39fc831 target=26.2`。
- 诊断 `libmlx.dylib` SHA-256：`6615954290a25fe76e783212f92c25786874ead5a7ceaa04d02f5c3421d36fb5`。
- 本轮 executable SHA-256：`ad7717f28a03015c46a203acd51fc0720da9e7349918b790e523a388c713776b`。
- generation SHA-256：`0adf7ee0980c9c94a0d004b0bd219939caa119fd406f9d1932b0fda29079ded0`。
- commands SHA-256：`bbc80a66d2f092de1dbfe5a089975bae8ffe80290141eda1c7318a08a559125d`。

完成 GPU 资源交接后，在仓库目录运行；输出使用新路径。当前入口禁止 MTP>0 与 command timing 同用，因此不能借 D1 预热来保留 MTP head。

```sh
env -u ANERUNNER_GATEUP_LIBRARY \
  DYLD_LIBRARY_PATH="$PWD/results/gpu-bottleneck-v1/native/lib" \
  ANERUNNER_FUSED_PREFILL=1 ANERUNNER_BLOCKED_GDN=0 \
  ANERUNNER_MOE_QMM_CONFIG=0 ANERUNNER_MOE_QMM_BM=0 \
  .build/release/ane-runner generate-gpu \
  --model-dir ../qwen38-ssd/models/Qwen3.8-Flash-Next-MLX-SSD-Stream \
  --tokens-file fixtures/gpu-agent-11k/prompt-token-ids.json \
  --context 16384 --prefill-chunk 416 --prefill-eval-layers 4 \
  --prefill-attention reference --max-tokens 128 --repeat 3 \
  --mtp-depth 0 --decode-mode reference --gdn-gemv-mode reference \
  --wired-policy disabled --profile-stages disabled \
  --gpu-command-timing-output results/manual-gpu-drift/commands.json \
  --output results/manual-gpu-drift/generation.json

python3 scripts/analyze_gpu_command_timing.py \
  --generation results/manual-gpu-drift/generation.json \
  --commands results/manual-gpu-drift/commands.json \
  --skip-first-trials 1 --output results/manual-gpu-drift/analysis.json
```

分析器 `decode_groups` 汇总暖 decode；其 `steps` 仍含全部 prefill/预热窗口，可按 phase/repetition 累计 `step_wall_ms`、`observed_gpu_buffer_span_union_ms` 和 `outside_observed_buffer_spans_ms` 重现上表。不完整采样必须停止完整覆盖/间隙归因，保留原始记录供排查。
