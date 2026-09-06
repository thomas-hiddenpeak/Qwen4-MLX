# 专家融合与路由归约的组合回归

2026-09-07，M5 Max。接续 [按专家边界组织 prefill MoE](MOE_PREFILL_EXPERT.md)，将已实现的 expert32 gate/up、grouped down 与 [融合路由归约](MOE_PREFILL_AUTOTUNE.md)组合。生产路径已有组合能力，本轮扩展对照入口并验证组合；未修改 Metal shader、量化、共享专家或生成默认值。

## 比较方式

`probe-gpu-moe-prefill-expert --suite composition` 以 expert32/groupedDown、原归约为基准，候选分别叠加128、256、512线程的融合归约。`--suite expert` 保留原先的专家搜索。双方都执行相同的专家分块，因此报告的是相对上一轮优化的额外收益，不能与之前相对stock的百分比直接相加。

复用九组已通过完整生成检查的真实S416输入，以及首组的S205/S240派生尾部。每组每候选检查10项张量或输出；比较有限性和逐位一致性后，进行双方各三次预热、四组ABBA，各八个热样本。计时包含完整单层MoE的建图和最终求值，包括共享分支；不包含权重加载、诊断或CPU回读。由于双方都调用专家插件，原生gate/plan/down计数按双方调用总数校验。微测归约计数只记录带归约参数的成功host调用，不将它表述为原生dispatch计数。

`probe-gpu-hotspots --detail moe-composed` 接收 `--baseline-moe-config` 和 `--moe-config`，使用同一模型、同一提示词、相同生成预算和真实producer/consumer交接。两个配置均校验插件、基础MLX、模型目录和符号归属；基准必须为expert32/groupedDown且无融合归约，候选保持相同专家路径，并至少命中一个实际prompt块的归约。相同配置或只含未使用长度的表在权重加载前拒绝，避免没有计算差异的A/B被标为组合收益。

Prefill和decode各自统计业务耗时。两阶段分别核对host及插件计数；prefill预期由各请求的实际分块计算，decode新增计数必须全部为零。MTP关闭，attention保持reference，阶段同步采样关闭。

## 局部结果

[微测](../results/moe-prefill-composed-v1/micro.json)全部330项比较通过，三个候选均未因数值差异淘汰。九组真实输入参与排名，派生尾部不参与评分。

| 归约线程数 | 配对基准合计ms | 候选合计ms | 完整MoE吞吐变化 |
| --- | ---: | ---: | ---: |
| 128 | 34.900 | 33.770 | +3.35% |
| 256 | 35.389 | 33.635 | **+5.21%** |
| 512 | 34.612 | 34.057 | +1.63% |

256线程在九组中的八组改善，一组约−0.34%；合计耗时下降4.96%。[所选配置](../results/moe-prefill-composed-v1/selected-config.json)只为205、240、416行启用归约，其他长度保留原归约。它继续使用variant2/groupedDown，512行仍可使用专家融合。

## 完整生成与边界检查

Release编译及33项CPU契约、阶段和调度检查通过。另有两项组合配置拒绝检查，分别覆盖相同A/B配置和只有未使用长度的归约表，均在模型权重加载前退出。[拒绝检查](../results/moe-prefill-composed-full-v1/rejection-checks.json)。

[完整PD对照](../results/moe-prefill-composed-full-v1/abba.json)和[汇总](../results/moe-prefill-composed-full-v1/summary.json)包含A/B预热，再按ABBA测量。六轮11057-token输入、128-token输出，共768个生成token全部匹配原始AR golden及同进程基准；callback、结束原因、offset11184和阶段交接均通过。双方每请求gate/up、plan、down各1296次，仅候选多1296次归约；decode全部为零。

| 热样本聚合 | expert32/down | 再叠加归约256 |
| --- | ---: | ---: |
| Prefill平均秒 | 13.353 | 13.413 |
| Prefill token/s | 828.05 | 824.36 |
| Decode token/s | 29.04 | 29.54 |

Prefill观察到−0.45%，实际可视为本窗口基本持平；没有建立额外整模型收益。两个相邻对照分别为−4.32%、+3.35%，基准自身耗时漂移10.33%。Decode仍执行原路径，表中+1.70%是本窗口的计时变化，不能称为decode优化。统计使用`prefill.totalSeconds`与`result.phases.decodeServiceSeconds`；首个输出属于prefill，decode吞吐每请求分子为127。

局部单层微测的+5.21%没有在本次完整请求中体现，**不将组合设为默认**。本轮没有证据将差异归因于温度、频率、内存带宽或某个系统进程。

五组完整模型边界回归使用真实11k提示词的前缀，比较原始MoE与所选组合，输出预算16，MTP关闭。[边界汇总](../results/moe-prefill-composed-boundaries-v1/summary.json)及[执行计划](../results/moe-prefill-composed-boundaries-v1/plan.json)。每组双方输出、EOS结束原因及最终状态位置全部一致；候选合计27个输出token匹配基准，双方合计生成54个token。这里没有强行忽略EOS来凑足预算。

| Prompt长度 / chunk | 实际prefill块 | 每个专家gate/plan/down调用数 | 归约调用数 | 每方输出token |
| --- | --- | ---: | ---: | ---: |
| 205 / 416 | 204 + 1 | 0 | 0 | 8 |
| 206 / 416 | 205 + 1 | 48 | 48 | 7 |
| 513 / 512 | 512 + 1 | 48 | 0 | 6 |
| 621 / 416 | 416 + 204 + 1 | 48 | 48 | 3 |
| 622 / 416 | 416 + 205 + 1 | 96 | 96 | 3 |

表中调用数为候选的逐层总数。原始基准全部为零，所有decode新增计数也为零。该检查覆盖阈值两侧、S512上界、完整块之后的204/205短尾，以及保存配置在普通`generate-gpu`入口的实际使用。每个进程只有一个冷请求，边界回归不作性能比较。

[候选状态](../results/moe-prefill-composed-v1/candidate-status.json)记录正确性通过、没有额外整模型性能收益和不提升默认的决定。[普通runner配置](../results/moe-prefill-composed-v1/runner-config.json)保留205/240/416的归约256选择；原微测选择文件保持不变。

```sh
ANERUNNER_GATEUP_LIBRARY="$PWD/results/moe-prefill-expert-v1/native/lib/libanemlx_moe_gateup.dylib" \
  .build/release/ane-runner generate-gpu \
  --model-dir ../qwen38-ssd/models/Qwen3.8-Flash-Next-MLX-SSD-Stream \
  --tokens-file fixtures/gpu-agent-11k/prompt-token-ids.json \
  --context 16384 --max-tokens 128 --mtp-depth 0 \
  --prefill-moe-config results/moe-prefill-composed-v1/runner-config.json \
  --output results/composed-generate.json
```

参考服务恢复为PID97647，已确认MTP/drafter关闭，见[恢复记录](../results/moe-prefill-composed-boundaries-v1/run-ledger.json)。后续PID以实时状态为准。
