# MTP 验证的 routed MoE token 轴实验

2026-09-06。本机 M5 Max，原模型 Q4 affine / group64 专家权重，BF16 scales/biases。显式选择 `--mtp-verification batchedTokenMoE`；默认 AR 和显式 MTP 的默认 scalar 均不变。

## 改动

新策略继承 `batchedScalarLinear` 的 dense 投影、router/shared expert、GDN/PLE 舍入、attention、状态捕获和接受前缀提交。仅将 S2…5 的 routed experts 从每个 token 各调用 gate/up/SwiGLU 与 down/加权归约，改为两个带 token 轴的 kernel。S3 的每层主要派发数由 6 变为 2，48 层合计 288 变为 96；这不是 command buffer 数或完整阶段的全部 kernel 数。

每个 token/expert 保留原量化读取、四路 FP32 累加、SIMD 归约、BF16 投影/LUT SwiGLU，以及按 slot 升序的 BF16 乘积与求和。新源仅重排索引，不排序专家或改变计算精度。程序按序列长度和诊断开关缓存；token 索引留在 GPU。

此版没有实现跨 token 的显式权重读取复用，逻辑权重读取量不能按派发减少比例折算。未测得物理 DRAM 带宽。单 token decode、MTP draft head 和 prefill 保留原计算路径。

## 本轮验证配置

原始记录位于 [mtp-token-axis-v1](../results/mtp-token-axis-v1/)。单层 probe 使用 layer0 真实捕获的激活，组成 S2/S3/S5 的 mixed 与 repeated 两类输入；重排/重复行是算子测试，不构成新的真实生成轨迹。每例比较全部诊断 tensor、y、无诊断 y；之后各 12 次交错计时，计时包含 Swift forward 构图、GPU 执行和 y.eval。

整模型固定 11,057-token agent fixture，depth2、初始 draft history 1024、chunk416、prefill/verify 每4层求值，先按 A/B 各暖身一轮，再按 A/B/B/A 测量。A 为 batchedScalarLinear，B 为 batchedTokenMoE；128/256 输出预算分别执行。所有轮次保留在报告中，前两轮不参与性能汇总。

输出 token IDs 使用原 AR golden 作完整比较，另核 prompt IDs、finish、最终位置与 MTP 轮数/接受数/直方图。性能只比较本轮同进程 A/B，旧 golden 的性能不参与。本轮是定向 kernel 回归，不代替九场景 MTP 默认发布验收。

## 已完成的第一窗口

Release 构建、20 项 CPU 契约/提交/阶段测试、6例共120组 tensor逐位比较及14项实模停止/取消/预算检查均通过。取消检查包含 token-axis 验证和 head history 之后的边界。EOS 是将真实目标 token 指定为停止符以触发分支；未据此宣称覆盖自然 EOS。

11k/128 与11k/256各6轮均完整匹配原 AR token IDs、length终止与最终位置11184/11312。每个预算内所有轮次的 draft、accepted、verified、histogram 均相同，replay为0。

| 输出预算 | 旧验证器 decode | token轴 decode | 本轮吞吐差异 | 判读 |
| --- | ---: | ---: | ---: | --- |
| 128 | 37.44 token/s | 38.80 token/s | +3.63% | 两次配对分别约+0.06%/+7.38%，小幅观察值 |
| 256 | 28.64 token/s | 21.68 token/s | -24.31% | A自身首末下降17.2%，B有1.58s停顿；性能高度漂移 |

表中吞吐均为两轮有效输出总数除以 decode 总时间，首 token不计入decode。128平均 target prefill 为A14.669s/B15.215s（753.79/726.73 token/s），head prompt history为48.99/51.42ms，另行报告；没有将prefill变化算成验证内核收益。

256全部慢样本保留，原因尚未确定，不能据此宣布稳定提速或把异常归因于SSD/训练/温度。相同未修改二进制的复测记录位于 [第二窗口](../results/mtp-token-axis-repeat-v1/)，开启相同500ms系统采样，暖身后将顺序反转为B/A/A/B；不与未采样的旧窗口混合计算速度。

## 复测与处理决定

第二窗口6轮的完整256-token输出及所有计数/状态合同仍与参考一致。去掉预定暖身后，旧验证器为 **31.30 token/s**，token轴为 **31.39 token/s**（+0.28%），基本持平。平均TPOT为31.952/31.862ms，target prefill为18.950/18.657s（583.49/592.63 token/s），head prompt history为58.68/57.05ms。此处性能只在双方同样开启采样的第二窗口内比较。

首次256窗口的长停顿没有复现；复测测量轮的最大decode round为87.73ms。不能用复测采样解释之前未采样时间窗的停顿，也没有据此断定根因。

**正确性通过，稳定性能收益未建立，保持可选实验路径，不纳入默认。** 合并派发本身还不足以形成稳定的整模型收益；后续应测真实MTP轨迹中各层选中专家的重合，先确认可复用多少权重，再尝试跨token共享读取。单层重排行的专家重合率不能替代该统计。

合计18轮长请求（含6暖身轮）全量3840个输出IDs一致；20项CPU、120组tensor比较、14项实模边界检查全部通过。两窗口使用同一未修改的源码/二进制；采样配置变化已在计划中注明。参考服务已恢复，MTP/drafter仍关闭。完整核验与阶段数据见 [汇总](../results/mtp-token-axis-v1/summary.json)。

复测采样正常收尾、阶段事件未丢失；加载期间有一次进程磁盘计数下降，已标为无效增量，未用于结论。采样没有提供GPU频率/温度或可确认的物理DRAM字节。
