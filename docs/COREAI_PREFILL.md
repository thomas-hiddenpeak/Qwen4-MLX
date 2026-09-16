# CoreAI prefill 1000 token/s milestone

目标是完整模型、11,057-token agent 提示词、无 KV/prefix 命中的 prefill 至少1000 token/s。模型加载、第一次编译、prefill、首次 decode 转换及后续 decode 分开记录；不启用 MTP。当前仍处于实验阶段，**尚未达到完整模型目标**。

## 实现路径

`export_coreai_pd.py --prefill-kernels tensor --q4-kernel metal` 导出独立的主预填函数与尾块函数，共享每层一个权重资产。S1处理解码；尾块不补零，也不多推进状态。Swift校验全部入口的形状与状态契约，按照剩余token及系统前缀边界选择已有函数。

- `coreai_tensor_matmul.py`：MPP TensorOps矩阵乘法，FP16输入/权重、FP32累计、FP16输出。
- `coreai_gdn_chunk.py` / `coreai_gdn_chunk_metal.py`：批量投影，单次GPU调用完成时间递推，FP32 recurrent state保存在寄存器并在结束时写出。
- `coreai_moe_chunk.py` / `coreai_q4_grouped.py`：保持原top-10路由和FP16分数边界，按专家排序，GPU生成固定容量tile计划，三项Q4投影共用计划。选中专家只按小tile反量化，避免展开整个专家矩阵；完整原Q4 bank保持压缩。全长topk实现唯一整数键排序，因为当前CoreAI authoring不支持aten.sort。
- `coreai_q4_gateup.py`：可选gate/up/SwiGLU融合，复用输入tile和反量化缓冲。
- `coreai_qsa_chunk.py`：批量投影和增量池化，保留原可见范围、六项显式状态。可选FP16 SDPA只应用于prefill；S1投影和SDPA保留原精度。

新增精度和融合选项均记录在manifest。它们不改变源权重文件或SSD PLE布局，不宣称源BF16模型质量等价。

## 2026-09-17 单机测量

M5 Max、macOS27。以下均为本机设备运行结果，模型/fixture/报告保留在忽略的`results/coreai-prefill-1k/`，不进入Git。单算子与单层时间不是端到端吞吐；CPU导出、首调用编译和JSON读取不计入热态函数时间。

| 测量 | 热态时间 | 范围 |
| --- | ---: | --- |
| GDN recurrence S512 | 1.34ms | 真实尺寸及真实权重预处理输入重放 |
| 完整 GDN S512 | 3.21ms | 真实layer0，含TensorOps投影 |
| 分组Q4 gate投影 S256×top10 | 6.87→3.09→1.75ms | 相同合成输入；逐word反量化及16×32×64 tile |
| 完整MoE S512 | 17.89→11.47→10.82ms | 真实512专家bank；相同随机输入；调tile后再融合 |
| QSA SDPA S512/C16384 | 21.79→6.10ms | 相同FP16输入/可见范围，内部FP32与FP16接口对照 |
| QSA indexer / topk-mask | 2.28 / 1.31ms | offset8192，保持原排序及mask逻辑 |

GDN小尺寸状态连续/重置/恢复检查通过；QSA真实offset8192的chunk+S1、重置和恢复检查通过。完整MoE对照中路由ID完全相同，FP16分数和输出存在小差异。

**数值边界**：长GDN对CPU参考的误差随长度扩大，S2048输出relative L2约0.00867、末态约0.0320；必须与相同GPU路径的小块连续调用进一步区分，不能直接宣布数值等价。FP16 SDPA设备相对误差约0.000460，最大绝对误差0.001953；这只是同输入注意力子算子的误差，不是整模型质量验收。

已保留负结果：cooperative-input Q4受SDK矩阵尺寸限制，实测4.25ms比最佳threadgroup路径慢；flatten权重寻址约3.04ms，没有明显收益；S2048改BM32未改善完整MoE。固定归约GEMV也未解决旧S1/S4最终logits差异，见`COREAI_PD.md`。

随后GDN设备分块对照已完成：相同2048-token输入及非零初始状态，单次S2048、连续4×S512、连续32×S64的全部输出、最终卷积状态和FP32 recurrent state **逐值完全一致**。因此当前CPU差异不能归因于这些分块方式的状态交接；CPU/GPU算术路径的差异仍保留，不据此宣称源模型质量等价。证据 `gdn-chunk-equivalence-device.json`。
