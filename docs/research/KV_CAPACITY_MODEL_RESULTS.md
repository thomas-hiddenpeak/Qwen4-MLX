# 完整模型的 KV 容量追加

2026-09-09，显式实验策略`QwenGenerationRequest.kvAppendMode = .capacity256`。库默认仍为`.reference`，MTP不能启用capacity；只优化普通AR的单token decode，prefill、QSA算法及归档格式不变。按256行增长并受本请求最大行数约束，私有backing提供逻辑长度view；有旧别名时保留MLX写时复制。它还不是跨请求页池、共享页表或增量SSD。

首次转换/增长先由模型联合额度申请额外workspace permit，绑定model/session/offset并只消费一次；额度不足安全回退concat。许可由实际decode步骤保留至同步完成或错误恢复之后。正常生成不会加载独立allocation诊断库；物理机制证据另见[Swift机制验证](KV_SWIFT_CAPACITY_MECHANISM.md)。

## 数值与生命周期

release二进制`cea597c4d4d4ff56a991f78b0142d39e9fa510c80e14712b0280a4abebe08ac6`通过6项新增CPU许可检查、19项既有请求/预算检查及5项实际GPU State测试（无skip）。GPU测试覆盖public K/V替换、nil替换、旧state/view不可变、KV与QSA raw各自extent及重复小裁切。

P11057、chunk416/eval4的完整模型探针完成6次生成（4×16、2×128）及一次第2输出token取消。74个状态记录中17个为参考锚，**57组独立配对、6897条张量记录**的原始BF16 SHA、shape/dtype及host字段一致；O128仅比较完整IDs，不冒称逐步状态覆盖。320完整输出IDs及取消前2个IDs均匹配改动前模型O512参考。RAM实际命中10816 tokens，只前向241；填满联合账本后15次decode真实回退concat。第2输出token回调取消时546,570,240 B workspace仍由许可持有，返回后request/workspace归零，最终全部lease释放。

## 分阶段性能

实际`QwenGenerator`、无observer、无prefix cache、MTP0；每块先各模式预热16输出，再执行四轮。A为reference，B为capacity。速率为`sum(actual decoded tokens) / sum(decode round seconds)`，每个请求的首输出来自prefill，不进入decode分母；转换/增长成本包含在内。

| 输出上限与顺序 | reference tok/s | capacity tok/s | 本块观察增幅 |
| --- | ---: | ---: | ---: |
| 16 / ABBA | 27.91 | 30.03 | 7.61% |
| 128 / ABBA | 24.97 | 25.94 | 3.85% |
| 512 / ABBA | 25.74 | 27.55 | 7.05% |
| 128 / BAAB | 32.17 | 34.18 | 6.24% |

decode service口径增幅依次7.61%、3.85%、7.05%、6.25%，与round口径接近。16个测量请求加8个warmup共3264 IDs，全部与旧模型参考前缀一致、正常length结束，最终额度全归还、capacity无fallback；O512实际覆盖后续增长边界。不同块的绝对速度存在明显变化，不能合并成一个稳定tokens/s承诺；每块仍只有两样本/模式。前三块同模式decode速率首末漂移最大约1.21%、1.68%、1.41%；BAAB128的capacity耗时增长4.45%，原样保留。

prefill模式间速率差约+0.45%、−0.18%、+0.28%、−0.68%，小于相应块内漂移，**不宣称prefill收益**。以上支持继续验证显式容量策略，尚不构成HTTP端到端p95、公平份额、24小时或新版本两小时耐久验收；不更改默认。

## 复跑

在[唯一GPU控制器](../EXPERIMENT_CONTROLLER.md)下，先运行`probe-gpu-kv-capacity-model`，再运行`benchmark-gpu-kv-capacity`；`help`列出完整参数。固定输入为`fixtures/gpu-agent-11k/prompt-token-ids.json`，模型路径沿现有布局。benchmark使用新输出路径，例如：

```sh
.build/release/ane-runner benchmark-gpu-kv-capacity \
  --model-dir ../qwen38-ssd/models/Qwen3.8-Flash-Next-MLX-SSD-Stream \
  --tokens-file fixtures/gpu-agent-11k/prompt-token-ids.json \
  --max-tokens 128 --context 16384 --order abba --warmup true --output NEW.json
python3 -B scripts/analyze_kv_capacity_benchmark.py NEW.json --output NEW_ANALYSIS.json
```

公开分析器含8项CPU自检（`--self-test`），拒绝缺失/错误样本，不过滤坏样本后计算收益。原始资料在本地`results/kv-capacity-model-v1/`与`results/kv-capacity-reverse128-v1/`。两轮各400文件/102模型stat postflight通过；最后参考77076按原参数恢复，MTP/drafter关闭且idle。上述模型结果对应显式API策略；HTTP开关和服务持续负载另行验证。
