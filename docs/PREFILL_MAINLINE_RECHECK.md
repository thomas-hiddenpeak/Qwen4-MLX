# 主线整合后的 prefill 复测（2026-09-08）

expert32 gate/up + grouped down 保持显式启用。两窗共 12 个完整请求全部通过，但 ABBA 基线漂移 27.03%，未满足事先冻结的两窗稳定性门槛。BAAB 在基线漂移 3.93% 的窗口内观察到 44.87% prefill 吞吐提升。这项候选及其重建、配置入口已进入默认分支；不据此宣称所有负载或设备都有同等收益。

## 设置与范围

代码为主线整合提交 `af6a445`，使用现有 runner 二进制、新建 ABI 2 插件及新生成配置。完整 11,057-token agent 输入、128-token AR 输出，context16384、chunk416、eval4、SSD nextChunk/worker1、reference attention/BF16 累加。A 为默认 MoE；B 只在 205…512-token prefill 块启用 expert32/grouped down，保持原归约。MTP、阶段同步采样与实验 async 均关闭。

两个进程都先 A/B 暖机，测量顺序分别 ABBA、BAAB。控制器暂停已核实为空闲的参考服务，串行运行两个进程。预先要求每窗 A/B prefill 时间比至少 1.03，且同窗两次 A 的相对时间漂移不超过 5%；不能用跨窗平均掩盖漂移。

## Prefill

吞吐由 11,057 / 平均 prefill 秒数计算，包含该阶段 SSD 等待；不是纯 GPU kernel 吞吐。

| 窗口 | A 秒 / token/s | B 秒 / token/s | 吞吐变化 | A 漂移 | 决策 |
| --- | ---: | ---: | ---: | ---: | --- |
| ABBA | 17.180 / 643.61 | 15.483 / 714.16 | +10.96% | 27.03% | 漂移未定 |
| BAAB | 34.401 / 321.42 | 23.747 / 465.62 | +44.87% | 3.93% | 该窗通过 |

跨窗口基线平均从 17.18 秒变到 34.40 秒，未定位这种变化的原因，不合并成一个性能承诺。ABBA 相邻配对为 −1.39% / +23.10%，BAAB 为 +47.41% / +42.31%。

## Decode 单独统计

128 个输出中的首 token 属于 prefill；下面按其余 127 个 token / decode service 秒数计算。候选没有进入 decode，不能把 prefill 收益写成 decode 或 MTP 收益。

| 窗口 | A decode token/s | B decode token/s |
| --- | ---: | ---: |
| ABBA | 26.48 | 26.45 |
| BAAB | 21.07 | 20.67 |

## 全部样本与正确性

保留暖机，不按快慢排除样本。每个请求的 128 个生成、回调及返回 token IDs 都精确匹配原 AR golden；全部 length 终止、offset11184、状态交接通过。B 的每次 prefill 均为 1296 次 gate/up、plan32、down32；A 的新增路径计数为 0，所有 decode 新增计数为 0。没有内部 logits 全量有限性或物理 DRAM 字节结论。

| 窗口 | 序号 | 路径 | 暖机 | Prefill 秒 | Decode 秒 |
| --- | ---: | --- | --- | ---: | ---: |
| ABBA | 0 | A | 是 | 19.076394 | 3.896710 |
| ABBA | 1 | B | 是 | 12.143691 | 4.383003 |
| ABBA | 2 | A | 否 | 15.134100 | 4.619294 |
| ABBA | 3 | B | 否 | 15.347045 | 4.783857 |
| ABBA | 4 | B | 否 | 15.618165 | 4.818439 |
| ABBA | 5 | A | 否 | 19.225354 | 4.971860 |
| BAAB | 0 | A | 是 | 27.678178 | 5.064845 |
| BAAB | 1 | B | 是 | 21.355764 | 5.917440 |
| BAAB | 2 | B | 否 | 23.804017 | 6.202522 |
| BAAB | 3 | A | 否 | 35.089886 | 6.106280 |
| BAAB | 4 | A | 否 | 33.711742 | 5.946566 |
| BAAB | 5 | B | 否 | 23.689420 | 6.087914 |

另外用新配置执行了一次普通 `generate-gpu`，完整 128-token 输出、ABI/符号归属、prefill 1296 次调用和 decode 零调用均通过。这是正常入口与配置交付回归，不增加性能窗口。合计 13 请求、1664 个输出 IDs 精确匹配。前后 237 个文件 SHA 和 102 个模型载荷文件状态未变；参考服务恢复为 PID74990，空闲，MTP/drafter 关闭。

## 复跑与产物

先按 [README](../README.md#可选-expert32--grouped-down) 从源码构建插件并生成配置；无需历史 `results/` 配置或张量捕获。普通 `generate-gpu` 即可显式使用。需要本地重跑 A/B 时，先用相同 tokens/context/max-tokens 的默认 AR 生成新 golden，再运行：

```sh
.build/release/ane-runner generate-gpu \
  --model-dir ../qwen38-ssd/models/Qwen3.8-Flash-Next-MLX-SSD-Stream \
  --tokens-file fixtures/gpu-agent-11k/prompt-token-ids.json \
  --context 16384 --max-tokens 128 --mtp-depth 0 \
  --output results/local-ar-reference.json

ANERUNNER_GATEUP_LIBRARY="$PWD/results/local-moe-native/lib/libanemlx_moe_gateup.dylib" \
  .build/release/ane-runner probe-gpu-hotspots \
  --model-dir ../qwen38-ssd/models/Qwen3.8-Flash-Next-MLX-SSD-Stream \
  --tokens-file fixtures/gpu-agent-11k/prompt-token-ids.json \
  --golden-report results/local-ar-reference.json \
  --detail moe-expert --moe-config results/local-expert32.json \
  --max-tokens 128 --ab-order ABBA --output results/local-expert32-abba.json
```

再用新输出路径运行 BAAB。保留 stock selectors，清除其他实验覆盖值。新生成的 golden 用于同机路径回归；本轮实际对照的是冻结的历史 AR golden，二者验证范围不同。

完整本地产物：[冻结计划](../results/mainline-prefill-adoption-v1/plan.json)、[ABBA](../results/mainline-prefill-adoption-v1/abba.json)、[BAAB](../results/mainline-prefill-adoption-v1/baab.json)、[全部分析](../results/mainline-prefill-adoption-v1/summary.json)、[普通 CLI](../results/mainline-prefill-adoption-v1/cli/ar.json)、[检查与恢复](../results/mainline-prefill-adoption-v1/postflight-and-release.json)。这些本地记录不随 Git 提供，上表与身份摘要随主线保留。

| 身份 | SHA256 |
| --- | --- |
| runner | `3c9309fb66966c2a120504e0f4b4aac384bc2ff01740ea6b348c02a1a6690d33` |
| base MLX | `fc7cbc4002ecfe90cb1f7b73f21f02a733bf7f536c6486030a04dc5fdf000469` |
| 新插件 | `6d43559d59f3973e673f837c256819dce078d128ad0b0af71645b305a05ce981` |
| 新配置 | `7dbac833e0fe795f79281bebc08e1da061269f7b5621678804eccdfa894c56af` |
| ABBA 原始报告 | `0a156669ce072128e6ac10af495b07daaf962291f0e298ea964c3c5ef179ff30` |
| BAAB 原始报告 | `50d28377258505bbbe9a62b8c5812f3ad4afc13a5dd16da2f8c49609f681b1e2` |
| 普通 CLI 报告 | `6e15051a2050bce10d2e4246c1f1d6c08215b891ac98f3a82d6772df9b4801e5` |
