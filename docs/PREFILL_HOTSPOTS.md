# 11k prefill 热点分析

2026-09-06 至 09-07，M5 Max。当前 11,057-token agent 提示词中，优先优化多 token MoE 的量化投影。QSA 选块和 K/V 拼接占比较小；attention 的整体成本会随已处理上下文增长，不应据此放弃更长上下文的 attention 优化。

## 测量范围

新命令 `probe-gpu-hotspots` 使用同一模型上的 producer.prefill → consumer.decode，按 baseline / profiled / profiled / baseline 跑四轮，每轮重新建立完整请求状态。固定 chunk416、reference attention / decode、MTP 关闭，生成 128 token，完整匹配 AR golden。阶段接口及正常生成默认值不变。

只有两轮 profiled prefill 开启同步计时。第一窗口 `--detail attention` 细分 QKV 投影、K/V 拼接、indexer 投影、历史/池化、QSA 打分、选块/掩码、SDPA、输出投影；第二窗口 `--detail moe` 细分 MoE。细分子阶段替代父阶段，不能与 inclusive 父阶段重复相加。

每条记录带业务 phase、输入块起点和长度，最后一个 S1 仍标为 prefill。各阶段物化后续依赖和必要状态，避免将未执行的懒图归到下一个阶段。中间 prompt 块的输出 head 没有被正常调用方使用，诊断也保持其懒执行，只在最后一个 prompt token 求值 head。记录只保存标量，不保留 GPU 张量。

**以下占比是同步诊断运行的阶段 wall time 分布。** 它包括建图、求值、GPU 执行和同步，改变原本的执行重叠，不是未扰动的 GPU 占比，也不是物理带宽。占比分母为所有 stage.elapsedMilliseconds 之和；前置 stream drain、SSD 剩余等待和未归因的 target residual 单独列出。原始计时不能直接当作 kernel 加速后的端到端收益。报告编码发生在 prefill 完成后的交接间隙，TTFT 可能包含这部分诊断开销。

## 第一窗口：attention 与主要模块

[原始数据](../results/prefill-hotspots-v1/hotspots.json)及[汇总](../results/prefill-hotspots-v1/summary.json)。两轮诊断分别 10,377 条记录，全部 phase/position 正确，无重复记录、失败或丢弃；没有外层 attention 重复计数，每轮只记录一次 offset11056/S1 的 head。

| 部分 | 两轮合计阶段耗时占比 |
| --- | ---: |
| MoE | 47.64% |
| Attention 全链路，包括 QSA、投影和 KV | 19.58% |
| GDN | 16.57% |
| Hyperconnection 读写与混合 | 15.52% |
| PLE、embedding、最终 head 合计 | 0.69% |

Attention 全链路内部，SDPA 占所有阶段耗时的 9.86%，QSA 选块/掩码 1.70%，K/V append 1.17%。三个数字的分母仍是全部阶段，不是 attention 子总和。raw indexer 历史拼接计入 `qsa.history_pool`，没有算入 K/V append。

同样按诊断阶段求和，attention 全链路在未触发 QSA 的前段占 7.41%，在超过 8k 的多 token 块中占 26.53%；后者 MoE 仍占 42.57%。这是不同上下文位置的观察，不是相同工作量的 A/B，也不能外推 32k/64k 性能。

两轮诊断 target prefill 为 15.747s / 16.739s，阶段求和为 15.009s / 15.979s；前置 drain 0.219s / 0.241s，SSD 剩余等待 0.460s / 0.455s。前后未记录 baseline 为 17.418s / 15.387s；首轮包含首次使用与缓存因素，不把后一次更快解释为优化收益。Decode 分开保存在报告中，本次没有修改 decode kernel 或验证其提速。

四轮完整 128-token 输出、length 终止、offset11184、callback 和 PD 交接全部一致。第一窗口完成时 Release 构建与 30 项 CPU 契约/阶段/调度检查通过；逐 token 一致不等于全部内部张量逐位一致。

## 第二窗口：MoE 内部

采用同样的四轮顺序和完整预算，使用 `--detail moe`，attention 保留粗粒度。第一窗口完成后增加了 MoE 诊断边界，因此两个窗口不是同一二进制的性能对照；源码和二进制哈希分别保存在各目录。

两轮分别有 14,601 条阶段记录。六个 MoE 细分阶段各出现 1296 次（27 个多 token 分块 × 48 层）；最后 S1 的 48 个 MoE 仍使用原粗粒度计时。无父子重复计数、漏记录或 decode 采样。gate/up 与 SwiGLU 保持共同求值，共享专家与最终合并也保持共同求值，未开启全量张量 diagnostics。

| 多 token MoE 部分 | MoE 内部阶段耗时占比 |
| --- | ---: |
| Gate/up Q4 投影及 SwiGLU | 52.69% |
| Down Q4 投影 | 20.91% |
| 共享专家与最终合并 | 8.74% |
| 恢复 token/slot 顺序与加权归约 | 8.64% |
| 排序、逆排列与输入 gather | 5.26% |
| Router 投影、top10 与权重 | 3.76% |

这里的分母是两个 profiled 请求中六个 MoE 细分阶段的 elapsed 总和，排除最终 S1 MoE、其他模块、drain、SSD 等待及 residual。Gate/up/激活加 down 合计 **73.60%**，不是纯矩阵 GPU 耗时，也不能乘上第一窗口的 47.64% 来声称普通运行的精确占比。不同细分粒度会改变同步、物化和正常重叠。

两轮诊断 target prefill 为 16.117s / 16.638s；前后无记录 baseline 为 17.204s / 14.655s。后者约 754.50 input token/s，decode 另计约 30.76 token/s；这是本次后段运行读数，不是新 kernel 的提速结果。详见 [原始数据](../results/prefill-moe-hotspots-v1/hotspots.json)与[汇总](../results/prefill-moe-hotspots-v1/summary.json)。

第二窗口 Release 构建和 30 项 CPU 检查通过；四轮完整 128-token 输出与 AR golden 一致。两窗口合计 8 轮、1024 个输出 IDs，全部 length/offset11184、callback、PD handoff 正常。未在本轮重新验收 MTP 或带细分 profiler 的混合 cooperative 请求。作者服务已恢复并核实 MTP/drafter 关闭，实时 PID 以状态文件和最新恢复 ledger 为准。

## 后续选择

下一次定向实验从真实轨迹的多 token MoE 输入开始，优先检查 gate/up 和 down 的 affine-Q4 专家矩阵调用、分块和数据复用；保留本模型 2560/640 维度、group64、BF16 scale/bias 与原输出舍入。先做单层真实输入快测，再回到不启用 profiler 的完整 PD 请求检验收益和输出。

排序/gather、还原归约有次要成本，但目前不足以取代量化矩阵成为第一目标。暂不继续打磨 `forceFused` 开关，也不把 K/V append 或 QSA 选块作为当前 11k prefill 的最高优先级。本轮交付是热点定位及可复用诊断入口，没有新增加速 kernel，也没有测得物理带宽或 NAX 占用率。
