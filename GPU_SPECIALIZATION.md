# 固定模型解码优化与交错验证

本轮只优化 Qwen3.8-Flash-Next 的固定结构，MTP、drafter、PLD 均关闭。保留原 affine Q4 专家、BF16 主干、FP8 n-gram 表，以及 reference prefill 的舍入边界。下面的候选已接入独立 Swift runner，默认仍为 `reference` / `disabled`；实现完成与速度收益分开判断。

## 候选实现

| 模式 | 单 token 执行变化 | 额外投影缓冲 |
|---|---|---:|
| `reference` | 保留原路径和 UInt32→Int32 GPU 取回步骤 | 0 |
| `scalar` | 直接读取已求值的 UInt32 token，在 CPU 检查并转换为 Int32 | 0 |
| `elementwise` | scalar + 共享专家的激活、门控与合并由六个公共逐元素运算合为两个 Metal 内核 | 0 |
| `projections` | scalar + GDN 的 qkv/z 与 a/b 分成两组矩阵乘；HC 的 down/injection 合并 | 3,674,603,520 B |
| `all` | scalar、elementwise、projections 全部启用 | 3,674,603,520 B |

这些融合只用于 S=1，包括作者调度语义中保留的最后一个 prompt token；较长 prefill 块保留原计算。共享专家融合沿用公共 MLX 生成的完整 BF16 sigmoid 查表，并保留每次 BF16 乘法和加法舍入。GDN 没有把四个投影全部拼成一个矩阵，因为那会改变当前 MLX 为小 a/b 投影选择的归约方式。

投影拼接在载入时执行；保留原缓冲供 prefill 和同进程参考路径使用。新增缓冲独立记账，不计作新增源权重，也不重复计入每 token 逻辑权重 footprint。包含 projections 或 all 的交错进程会在全部 trial 中持有这份内存，即使某轮选择 reference。这样的对照检验同一驻留条件下的执行路径，不能隐藏部署时额外的 3.67 GB 和加载成本。

`--wired-policy fit` 在请求计时前同步、清理分配器缓存、读取活跃内存，设置进程内 MLX Metal 驻留容量。目标为 `min(active + 256 MiB, maxRecommendedWorkingSet − 256 MiB)`；活跃集无法容纳时明确失败。每轮报告旧值、目标、余量和设置耗时。它不是实测物理驻留量，也不等于 DRAM 流量。

## 数值检查

- [Release 构建](results/gpu-specialization-v1/build.log)成功；[72 项 XCTest](results/gpu-specialization-v2/swift-tests.log)全部通过，新增测试覆盖真实投影尺寸、BF16 边界、显式中间舍入、token 越界和驻留限额边界。
- [真实四层捕获](results/gpu-specialization-v2/run-ledger.json)：26-token prefill 后连续推进 8 个固定解码 token，reference 与 all 的 **315 个捕获张量逐位一致**。这不是全部 48 层内部状态的比较。
- [完整模型候选筛选](results/gpu-specialization-v2/component-screen.json)：同一已加载模型正反顺序跑五种模式，共 10 轮，每轮输出 256 token；所有生成 ID 与先前保存的 reference 基线完全一致。输入是 [1217-token 项目文档](fixtures/gpu-telemetry/prompt-token-ids.json)，未触发 QSA。所有轮次无 profiler、系统采样或 Instruments。

以上验证保留原作者数值语义。旧的独立 FP32 prefill 0.5% 门槛失败记录不因此改变；这轮也不声称任意输入或长上下文质量已通过。

## 本轮性能结果：2026-09-05

[完整汇总](results/gpu-specialization-v3/milestone.json)保存当前二进制 SHA、原始报告和服务恢复事实。两组正式交错实验各有 10 轮，排除前两轮预热后，各模式保留 4 轮、1020 个后续 decode 步。

| 同一进程内的对照 | 原路径 | 候选 | 吞吐变化 |
|---|---:|---:|---:|
| [计算融合](results/gpu-specialization-v3/fusion-analysis.json)：reference → all，驻留均 disabled | 25.757 token/s | 26.318 token/s | +2.178% |
| [驻留额度](results/gpu-specialization-v3/wired-analysis.json)：disabled → fit，计算均 all | 32.091 token/s | 32.263 token/s | +0.536% |

计算融合的两组**非重叠** A/B/B/A（trial 2–5、6–9）分别为 +2.260% 和 +2.096%；驻留额度对应 +0.415% 和 +0.649%。分析器还保留滑动重叠窗口，不能把它们计作额外独立样本。完整候选筛选与这两组实验共 **30 轮 × 256 token**，所有生成 ID 与同一份保存的 reference 基线一致。

本轮只观察到小幅正向配对效应，**不修改默认模式**。all 要额外保留 3.6746 GB 投影缓冲；融合对照两边的 MLX active 都约 82.270 GB。fit 还带来独立的设置耗时，首次约 94 ms，其速度差不足以作为明确收益。

绝对速度随持续测试明显下降：最初模式筛选约 35–38 token/s，最后融合对照约 25–27 token/s。不能将上表跨进程两行直接比较，或把 +2.178% 直接乘到此前 37–38 token/s 基线，声称新的稳定峰值。[测试期间的只读快照](results/gpu-specialization-v3/system-observation.json)显示 AC 供电，后台 Steam 约占一个 CPU 核，另有桌面应用活动；系统未报告温度／性能警告。这些记录不足以证明速度漂移的原因，也不能证明没有降频。

已经定位的下一项候选是缓存 GDN 单 token 的固定启动配置和 T=1 常量，减少每层重复的主机侧配置工作，不增加权重副本。它尚未实现或测量，未计入本轮收益。

## 对照方法

先为两种模式各预热一轮，再按 A/B/B/A、A/B/B/A 执行。每个 trial 重建会话状态，复用同一套加载权重，避免不同进程重复载入造成的额外漂移；正反顺序仍不能排除所有系统活动和频率变化。

完成完整模型资源交接后，在本目录运行：

```bash
.build/release/ane-runner generate-gpu \
  --model-dir ../qwen38-ssd/models/Qwen3.8-Flash-Next-MLX-SSD-Stream \
  --tokens-file fixtures/gpu-telemetry/prompt-token-ids.json \
  --max-tokens 256 --prefill-accumulation reference \
  --decode-order reference,all,reference,all,all,reference,reference,all,all,reference \
  --output results/manual-specialization-abba.json
```

不要与作者完整模型服务同时运行。保存的 [实验控制器](scripts/run_specialization_experiment.py)读取显式计划和最新交接账本，检查 PID、完整命令和空闲请求数，依次运行门槛检查，最终恢复原服务；普通 generate-gpu 命令本身不管理服务。复用旧计划前必须换用新的输出目录和当前恢复账本。

速度按 `decode_steps / sum(decode_step_seconds)` 聚合，不对 token/s 做算术平均。首 token、加载和驻留设置耗时各自记录。逻辑权重 footprint 速率不是实测物理 DRAM 带宽，后者继续保持 null。

保存结果后可直接离线分析，无需重新加载模型：

```bash
python3 scripts/analyze_decode_specialization.py \
  --input results/manual-specialization-abba.json --skip-first 2 \
  --baseline-mode reference --output results/manual-specialization-analysis.json
```

分析器先验证完整 prompt / 生成 ID、MTP、数值模式、上下文和采样设置，再按模式与驻留策略聚合；不能满足公平比较条件时不给出加速百分比。9 项纯 CPU 测试通过，见 [测试记录](results/gpu-specialization-v3/analyzer-tests.log)。

最初的 [调度阈值筛选](results/gpu-specialization-v1/run-ledger.json)发现首尾同一基线从约 38.27 降至 31.94 token/s，无法归因于中间的参数变化；没有据此修改 MLX 调度默认值。五种模式的首轮筛选也只作候选选择，不凭单轮最高值宣称收益。
