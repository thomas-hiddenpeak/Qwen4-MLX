# 本机 PD 调度的输出延迟实验

2026-09-07。已有 cooperative probe 保存了完整 callback 时钟和 `runNext` 步骤，本轮利用这些记录补齐服务效果汇总；没有修改调度器核心策略，也没有在调研期间启动模型或服务。

## 已有内容与本轮增量

[原有探针](../Sources/ANERunnerCLI/GPUCooperativeSchedulerProbe.swift) 已记录每个输出 token 的 `timestampNS`、提交时刻、submit 到首/末 callback、相邻 callback 的全部间隔、每步开始/结束以及完成事件的阶段统计。已有 TTFT 可以直接使用，不需要再用 compute 时间拼算。

本轮新增：

- `--decode-burst 1...64`，默认仍为 4，只选择调度器已有参数；prompt chunk 仍固定为 416，取消回归继续使用原固定配置。
- 每个请求 callback 间隔的 count、p50、p95、max；分位数使用排序后 `q × (n−1)` 位置的线性插值，保留零间隔。
- `terminal_observed_ns` 和 `submission_to_terminal_observation_seconds`，覆盖从提交到终态 `runNext` 返回的墙钟时间。只有末 callback 不能代表清理和终态均已完成。
- 独立 [分析器](../scripts/analyze_scheduler_latency.py)：可读取旧报告或多个新报告，输出每请求与整组指标；不加载 MLX、不读取模型权重。缺失/无效时钟显示 JSON `null` 和终端 `unknown`，不以计算时间代填。

这是向现有 v1 JSON 增加字段，旧 raw 字段保留。分析器优先由原始整数纳秒重新计算，因此可以验证改动前的报告。Swift 增量需要由主任务统一构建和实模验证；下面历史结果只验证离线分析，不宣称新探针已运行。

集成更新：2026-09-07 Release 构建与相关 46 项 CPU 检查已通过，离线分析器在系统 Python 上重新处理六个历史请求成功。burst4/8 新实模正在串行执行，未完成前不使用其结果；本页历史数据仍单独保留。

## 指标合同

| 指标 | 分子/分母与含义 |
| --- | --- |
| TTFT | 第一条 callback − submit 前记录的时刻；含准入、排队、prefill、暂停和交接，排除 submit 之前模型加载。 |
| 请求完成时间 | 终态步骤返回时刻 − submit 时刻；不把最后一条 callback 当终态。另保留 scheduler 自己的 `elapsedSeconds`，两者采样边界略有差别。 |
| Callback gap p50/p95/max | 同请求相邻已提交 token 的 callback 差，包含调度暂停；样本数为输出数−1。 |
| Cooperative burst gap | 后一有输出步骤的首 callback − 前一有输出步骤的末 callback；剔除同轮内部密集发布造成的近零样本，更清楚呈现 MTP 成批输出之间的等待。 |
| Compute decode token/s | `decodedTokenCount / result.decodeSeconds`；首 token 在 prefill 已选出，分子排除它、包含实际发布 EOS，未提交草稿不计数；分母排除 callback 与暂停排队。 |
| Active decode service token/s | 相同 decode 分子 / `phases.decodeServiceSeconds`；包含 callback，仍排除暂停。 |
| Wall delivery decode token/s | `(callback_count−1) / (last_callback−first_callback)`；包含实际调度等待，仅描述首输出之后的交付速度。 |
| 整请求 token/s | 完成结果的输出 token 数 / 请求完成墙钟，包含 prefill 和排队，仅作端到端业务指标。 |
| 整组 token/s | 两个完成且 callback IDs 与结果相符的请求输出总数 / group 的原始 start/end 墙钟；没有完整结果时未知。 |

这里的“可见”边界是 **Swift 库 callback**，尚不是 HTTP/SSE 到达客户端、token 解码成文本或 UI 渲染。MTP 一轮可能连续发布 1…3 个 token，近零 gap 并不表示每个 token 都以这个延迟完成推理。`decodeBurst` 是步骤数量，首 token 发布也算一个步骤；它不是输出 token 配额。

只有 cooperative 模式的一步才对应首 token 交付或一轮 AR/MTP，因此分析器可将 callback 唯一映射到步骤得到 burst 指标。wholeStages 的一个步骤包含多个 decode round，burst gap 显示未知；不能凭接近零的间隔阈值猜测轮次。

这些 p95 是单请求内部的有限间隔分布，不是生产请求总体的 p95。脚本保留原报告 complete/passed 和 SHA256，不会把未通过输出检查的旧记录升级成可用性能证据。

## 从已有实模记录得到的观察

读取 [cooperative-pd-v1/cooperative.json](../results/cooperative-pd-v1/cooperative.json)，source SHA256：`f1a23440e4afa3cd4b265526ee689efa2ccd4874678e8ea6a3b8c48b8646ed0e`。原报告 complete/passed 均为 true，来自历史实模运行；本轮没有重测。离线输出为 [cooperative-latency-existing.json](../results/research/cooperative-latency-existing.json)。

每组长提示 11057 tokens、输出 128；短提示 26 tokens、输出 64。每组两请求同时排队、长请求先提交。第一组和第二组长请求 AR / 短请求 MTP depth2；第三组两边都是 MTP depth2。

| 调度 / 请求 | TTFT 秒 | 完成秒 | Gap p50 ms | Gap p95 ms | Gap max ms | Compute decode token/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| wholeStages / 长 AR | 15.802 | 20.124 | 33.665 | 36.290 | 39.845 | 29.402 |
| wholeStages / 短 MTP | 20.322 | 22.048 | 47.576 | 60.344 | 69.400 | 36.521 |
| cooperative / 长 AR | 17.923 | 22.305 | 34.229 | 36.238 | 40.993 | 29.000 |
| cooperative / 短 MTP | 1.679 | 7.828 | 50.766 | 612.636 | 626.311 | 36.009 |
| cooperative / 长 MTP | 18.409 | 22.198 | 0.001 | 71.818 | 78.119 | 33.536 |
| cooperative / 短 MTP | 1.769 | 8.215 | 62.620 | 597.039 | 652.510 | 28.607 |

第二组让短请求显著更早开始和完成，但 callback p95 由约 60 ms 增为约 613 ms。短请求真实交付吞吐只有 10.248 token/s，而活动 compute 为 36.009 token/s；差别主要体现在调度等待，不能用 compute 吞吐宣称用户获得同等输出速度。第二组短 MTP 的 32 个 burst 间隔 p95 为 620.897 ms，max 为 626.311 ms。

三组总墙钟分别为 22.048 / 22.305 / 22.198 秒，整组吞吐约 8.708 / 8.608 / 8.649 token/s。每种只一组，执行顺序固定，不将约 1% 的差异判断为稳定性能变化。

因此下一步应直接比较现有 `decodeBurst`，同时报告首输出、途中最大停顿和长请求完成时间。只追求短 TTFT 会漏掉这一取舍；本轮不因此调整默认调度策略。

## 下一次实模建议

先完成主任务的当前 GPU 实验，再在同一模型/内核配置下串行执行。当前探针会先运行独立参考，随后 wholeStages、cooperative 和 cooperative 长 MTP 三组，最后检查取消恢复；它验证的是“两请求同时排队，随后长 prefill 与短 decode 交错”，没有模拟随机到达或一个已经 decode 很久的请求再迎来新长 prompt。

首轮先比较 burst 4 与 8；若更大 burst 减少短请求卡顿而不显著损害长请求完成，再补反序重复。两个值都保持相同 prompt chunk 与 MTP 配置，不在这次实验同时改 prefill kernel。下例文件必须不存在；这条命令是下一次运行建议，本轮未执行。

```sh
ANERUNNER_FUSED_PREFILL=1 ANERUNNER_BLOCKED_GDN=0 \
  .build/release/ane-runner probe-gpu-cooperative-scheduler \
  --model-dir ../qwen38-ssd/models/Qwen3.8-Flash-Next-MLX-SSD-Stream \
  --tokens-file fixtures/gpu-agent-11k/prompt-token-ids.json \
  --golden-report results/prefill-agent-11k-default-check/default.json \
  --decode-burst 8 \
  --output results/cooperative-burst8-a.json
```

burst4 用同一入口，改变值和输出文件。若只做第一组可用性筛选，不据此发布百分比提升；要比较性能，用 4a → 8a → 8b → 4b，并保留每组原始回调与完成事件。

```sh
python3 scripts/analyze_scheduler_latency.py \
  results/cooperative-burst4-a.json results/cooperative-burst8-a.json \
  results/cooperative-burst8-b.json results/cooperative-burst4-b.json \
  --output results/cooperative-burst-comparison.json
```

小门槛：所有输出/状态检查通过；短请求 max 与 p95 gap 是否下降，长请求 TTFT/完成时间和整组吞吐是否恶化必须一起列出。若想研究“decode 已活跃时新长请求到达”，再为探针增加确定的到达时点，不能把当前 simultaneously queued 两请求结果换个名称当作那项证据。

## 本轮新实模结果

2026-09-07 按 `results/upstream-cost-latency-v1/plan.json` 依次运行 burst4、burst8。两份报告各 18 项检查全部通过；长/短完整 token、阶段计时、取消后恢复、额度与状态释放正常。新增 callback 分位数与终态时间由两份独立离线结果从原始整数时钟复算一致。该轮没有修改生产调度默认。

| 长请求 / 短 MTP 指标 | burst4 | burst8 |
| --- | ---: | ---: |
| 长 AR：短请求 TTFT | 2.134 s | 2.004 s |
| 长 AR：短请求完成 | 11.341 s | 7.317 s |
| 长 AR：短请求 callback p95 | 947.8 ms | 758.4 ms |
| 长 AR：短请求最大 gap | 1062.0 ms | 908.2 ms |
| 长 AR：长请求完成 | 32.512 s | 30.515 s |
| 长 MTP：短请求完成 | 9.849 s | 6.790 s |
| 长 MTP：短请求 callback p95 | 746.1 ms | 640.5 ms |
| 长 MTP：短请求最大 gap | 846.4 ms | 742.1 ms |

两个混合组合中，短请求已经开始输出、尚未完成时插入的长 prefill 步骤都从 **8 次降为 4 次**。这是原始事件中的结构性变化。burst8 减少了短 decode 被打断的频率，但没有缩短固定 416-token prefill 块本身，不能据 p95 下降声称最大停顿已解决。

这只是一次 4→8；同进程的 wholeStages 对照自身从整组 42.321 s 变为 30.908 s，首份长 AR 还出现约 1.357 s 的孤立 callback gap。时间明显有漂移，所以不将表中差异全部归因于调度参数，也不据此更改默认。需要继续比较时先做漂移诊断，再安排反序；不重复长输入只为补样本数。

报告：`cooperative-burst4.json`、`cooperative-burst8.json`、`latency-summary.json`；独立复核为同目录 `independent-latency-both.json`。参考服务已恢复，MTP/drafter 关闭；运行身份见同目录 `run-ledger.json`。
