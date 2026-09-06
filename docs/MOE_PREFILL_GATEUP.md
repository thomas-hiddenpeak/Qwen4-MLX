# Prefill gate/up/SwiGLU 融合

2026-09-07，M5 Max。延续 [MoE 自动调参](MOE_PREFILL_AUTOTUNE.md)，本轮实现独立 Metal NAX primitive，将路由专家的 gate、up 投影与 SwiGLU 合并。原 affine Q4/group64 权重库直接传入，无拼接、复制完整权重库或精度转换；共享专家、down 与路由归约保持原实现。

## 实现及阶段边界

`Native/moe_gateup_fused.metal` 基于当前固定 MLX 的 `affine_gather_qmm_rhs_nax`，保留原 BM32 块内专家段循环和 FP32 K 累加顺序。gate/up 共用输入片段，各自保留累加器和反量化缓存；在片上完成两次投影的 BF16 舍入，再用原 BF16 sigmoid LUT 和两次 BF16 乘法生成 activation。

两个候选均为 BK64、WM2：variant0 为 BN64/WN2，variant1 为 BN32/WN1。后一种减少每线程组的输出列数与线程数。本次实测比较两者，不预先将其优势归因于寄存器或内存瓶颈。

新插件链接原作者固定的 stock `libmlx`，使用独立 metallib；原库及原源码不变。构建脚本为 `scripts/build_mlx_moe_gateup.py`，产物与源码摘要见 [构建记录](../results/moe-prefill-gateup-v1/native-fixed/build-provenance.json)。`native/` 保存了初次构建在修改动态库依赖路径时失败的记录；实际测试使用 `native-fixed/`。

Swift `GPUMoEPrefillConfiguration.gateUpVariant` 是请求局部选项：nil 保持原路径，0/1 选择插件。仅允许本模型的多 token prefill；decode、MTP verification 和末尾 S1 不使用它。旧 JSON 缺少该字段时仍解码为 nil。本轮归约表为空，旧原生矩阵调参开关为0，未叠加上一轮候选。

插件严格检查输入 shape/dtype 和实际连续布局，使用官方 MLXC 数组所有权辅助函数，保持惰性建图。成功加载的插件句柄常驻，避免待执行图的 C++ primitive 生命周期超过动态库。当前 CLI 同线程执行；未来跨线程 executor 应显式传递 stream，再验证调度边界。尚无独立部署或跨线程服务的运行验证。

## 真实输入微测

[原始结果](../results/moe-prefill-gateup-v1/micro.json)、[所选配置](../results/moe-prefill-gateup-v1/selected-config.json)。复用已提交、通过完整 golden 的九组 S416 输入（层0/23/47，位置0/4992/9984），另取首组205/240行前缀覆盖尾块。

每个候选比较真实 activation、专家输出、路由结果、共享分支及最终输出，并交叉检查诊断与普通调用。**22组候选/输入、220项比较均有限且逐位一致**，没有候选被数值淘汰。候选 activation 确实由新 primitive 生成并被 down 消费。编码计数匹配：诊断/普通各一次，8次候选计时共8次；参考调用均为0。这些计数不代表物理带宽。

每组先各预热3次，再运行4组 ABBA，参考与候选各8个样本。计时包含新建完整 MoE 图及最终 y.eval；不含加载、诊断、host readback。九组真实 S416 的配对均值合计用于评分，两个派生输入不参与选择。

| 候选 | 参考合计 ms | 候选合计 ms | 完整 MoE 微测吞吐变化 |
| --- | ---: | ---: | ---: |
| variant0，BN64 | 43.109 | 42.442 | +1.57% |
| variant1，BN32 | 43.043 | 40.486 | +6.31% |

variant1 的九组真实输入全部变快，单组吞吐变化 +3.63% 至 +10.33%。它通过2%的局部筛选门槛，进入完整 PD 测试；这不是整模型提升结论。省去了两份 gate/up 逻辑中间输出，S416 时合计10,649,600字节；该数字不是测得的 DRAM 流量节省。

## 完整 PD 检查

`probe-gpu-hotspots --detail moe-gateup --moe-config PATH` 读取所选配置及插件/基础库 SHA，在真实11057-token提示词上运行128-token AR生成。先各预热一次，再做 ABBA；可用 `--ab-order BAAB` 反向复测。prefill/decode 分别统计，阶段同步采样关闭，MTP关闭。producer.prefill 返回后交给独立 consumer.decode，两者共享同一模型。

27个多 token 块 × 48层，候选每请求应编码1296次 gate/up primitive；参考为0。两条路径的融合归约和 decode gate/up 调用应为0。down 沿用 stock 实现，本轮没有新增其计数器，不沿用上一轮三矩阵3888次的计数要求。

Release 编译及32项 CPU 契约、阶段和调度测试通过；随后仅为反向复测新增 CLI 顺序参数并重编译，通过两项无模型加载的非法参数检查。GPU 实现和 native 插件不变。

阶段耗时口径：prefill 用 `prefill.totalSeconds`；decode 用 `result.phases.decodeServiceSeconds`。128个输出中的首 token 属于 prefill，decode吞吐分子为127。加载、两轮预热与阶段之间的交接等待不并入下表。

首轮 [ABBA 原始结果](../results/moe-prefill-gateup-full-v1/full.json)及[汇总](../results/moe-prefill-gateup-full-v1/summary.json)：六轮输出、callback、终止、offset11184和调用数检查全部通过。热运行 prefill 均值15.169s→14.612s，观察到+3.81%；但两次参考 prefill 漂移15.10%，两组相邻对照方向相反，decode也有相似变化。因此补一次反向顺序，不能单独把首轮均值当作稳定收益。

反向 [BAAB 原始结果](../results/moe-prefill-gateup-full-v2/full.json)及[汇总](../results/moe-prefill-gateup-full-v2/summary.json)也全部通过。两个窗口合计12个完整请求、1536个输出 token 均逐项匹配 golden；每个候选请求1296次融合调用，参考和所有 decode为0。

| 热运行窗口 | 参考 prefill 秒 | 候选 prefill 秒 | Prefill token/s 参考→候选 | 观察到的吞吐变化 | Decode token/s 参考→候选 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ABBA | 15.169 | 14.612 | 728.94→756.71 | +3.81% | 29.45→30.39 |
| BAAB | 16.285 | 16.069 | 678.97→688.08 | +1.34% | 29.01→29.10 |

BAAB 两组相邻 prefill 对照为+10.63%和−5.97%；参考自身漂移8.01%，候选漂移27.06%。ABBA 对应为−0.87%和+8.25%。反向顺序没有解决运行随时间变慢的问题，且未修改的 decode 也变化。没有采集足以区分温度、功耗、系统竞争或其他因素的同步证据，不将波动归咎于某一个原因。

合并两个窗口的八次热请求，prefill 均值观察到+2.52%，decode观察到+1.72%；这只是描述均值，**不是稳定性能收益的证据**。完整生成只验证输出和状态，以及公开指标有限性，不等于检查了全部内部 logits/hidden 的有限性。

## 决定与后续

[候选状态](../results/moe-prefill-gateup-v1/candidate-status.json)：**正确性与 PD 阶段隔离通过，保留请求局部可选实验，默认不推广。** 局部九份真实输入的一致收益足以保留这项实现，但本轮完整请求不能支持稳定的整模型提升承诺，也不能据此宣称实际带宽利用率提高了相同比例。

两次完整测试均完成独占与恢复流程，最终参考服务 PID86662、MTP/drafter关闭；见 [恢复记录](../results/moe-prefill-gateup-full-v2/run-ledger.json)。

下一项较大的 kernel 候选仍是按专家边界组织矩阵工作，减少当前全局 BM32 块内对短专家段重复运行矩阵循环。该方向尚未在本轮实现；共享专家持续常驻、每 token计算，无需路由。

后续已完成的专家分块、down复用与独立验证见 [按专家边界组织 prefill MoE](MOE_PREFILL_EXPERT.md)。上述gate/up单独融合结果保留为历史对照。
