# PD 分离后的 QSA prefill 融合实验

2026-09-06，M5 Max。本次先尝试固定版本 MLX 已有的 masked fused SDPA；没有移植 FlashInfer，也没有实现新的块稀疏 Metal kernel。默认保持 `reference`。

## 实现与阶段边界

`QwenGenerationRequest.prefillAttention` 和 `generate-gpu --prefill-attention` 接受 `reference|fusedQSA`。开关随请求进入 prefill cursor 和 ready 状态，只有主干 prefill、存在 QSA 布尔 mask 且当前块长度大于 8 时设置 `forceFused: true`。QSA indexer、512 个完整四 token 块的选择、causal 尾部、chunk416、BF16 权重与输入均保持原样。融合改变计算的舍入过程，因此相同 mask 不保证相同浮点输出。

`QwenModel.forward` 在执行设备工作前拒绝 decode/verification 使用这个 prefill 策略。单 token decode、MTP verification、MTP head 保持原路径。`QwenPrefillStatistics.attentionMode` 记录请求策略，历史报告可以没有此字段；短请求即使选择 `fusedQSA` 也可能不触发新路径。

新增 `probe-gpu-prefill-attention` 在同一模型上创建两个 generator，每轮实际执行 producer.prefill → ready → consumer.decode。它逐轮保存完整生成、callback、finish、最终状态位置、策略及独立阶段耗时。相同进程的 reference 和既有完整预算 AR golden 均参与比较。完成报告不等于通过：数值/输出比较失败记录 `complete=true, passed=false`。

## 第一轮检查

Release 构建与 27 项 CPU 请求/阶段/调度检查通过。两项 GPU 算子检查覆盖 S416、N2080/11056、24 个 query heads / 2 个 KV heads、D256、BF16。输出均为有限值；修改全局未选块不影响输出，修改最后一个 KV 不影响此前 query，并确实影响最后一个可见 query。

跨 kernel 相对 L2 差异约 0.673% / 0.678%，最大绝对差异为 0.00390625 / 0.0078125。该输入是确定性合成算子输入，默认 kernel 也不是 FP32 oracle；没有设容差把这些差异判为质量通过。

预热后的短 ABBA 算子计时：N2080 为 reference 1.192ms / fused 1.543ms；N11056 为 4.022ms / 4.964ms。计时包括构图、GPU 执行和同步求值，不含输入创建和 CPU 回读。这里只各有两次测量，不能推广为所有真实层的速度。

真实 11,057-token prompt、128-token 输出、MTP 关闭的第一轮整模型 PD 对照中，reference 完整匹配 AR golden；fused 从生成 token 索引 1 起不同。两者均正常 length 结束、最终位置 11184、callback 与生成 IDs 一致，阶段交接正常。候选文本仍是同主题回答，但这不构成质量验证。公共接口没有暴露 logits/hidden，整模型报告未宣称检查其有限性。

第一轮 target prefill 为 reference 17.935s / fused 13.963s，包含首次运行与缓存差异，不能作为稳定收益。原始记录见 [第一轮](../results/qsa-fused-prefill-v1/)。随后用同一未修改二进制进行预热后的交错诊断，见 [复测](../results/qsa-fused-prefill-repeat-v1/)。

## 预热后复测

第二窗口先各跑一次 reference / fused 暖身，然后按 reference / fused / fused / reference 测量；每种策略两次。Prefill 使用相同完整提示词，MTP 始终关闭。以下吞吐为总 token 数除以总阶段时间，不混合第一窗口。

| 指标 | reference | fusedQSA |
| --- | ---: | ---: |
| 平均 target prefill | 15.789s | 15.528s |
| Target prefill 吞吐 | 700.31 token/s | 712.05 token/s |
| 平均 prefill SSD 剩余等待 | 0.490s | 0.501s |
| Decode 吞吐，仅作各自轨迹记录 | 29.65 token/s | 30.42 token/s |

表面 prefill 吞吐差异为 +1.68%，但 reference 的两次耗时由 14.107s 增至 17.471s（+23.85%），不足以建立稳定收益。所有样本保留，没有据此将漂移归因于温度、训练、SSD 或其他具体原因。Decode 已生成不同 token 序列，不能把其速度差异当作同工作量的 kernel 收益；首 token 不计入 decode。

两个窗口合计 8 轮、1024 个输出 IDs。4 次 reference 全部完整匹配 golden；4 次 fused 彼此完整一致，但全部从索引 1 起与 golden 不同。全部正常 length 结束、最终 offset 11184、callback 与交接检查通过。源码和可执行文件未在窗口间改变。作者服务已恢复并检查 MTP / drafter 关闭；当前 PID 以实时状态文件及最新 ledger 为准。详细结果见 [汇总](../results/qsa-fused-prefill-v1/summary.json)。

## 处理决定

融合候选保留为显式实验，默认仍为 `reference`。完整生成一致性门槛未通过；相同可见性和正常状态交接不足以让它成为默认路径。尚未用新 prefill 策略验收 MTP 或多请求 cooperative 混合调度。

后续若开发自定义 kernel，应借鉴 FlashInfer 按阶段和块索引组织工作的方法，针对本模型 QSA 四 token 块与 GQA=12 设计 Metal 读取/计算复用，并单独验证舍入及状态。当前改动只绕过 MLX 的分派启发式，尚未直接消费选中块列表；也没有测量物理 DRAM 字节，不能据此宣称带宽提升。
