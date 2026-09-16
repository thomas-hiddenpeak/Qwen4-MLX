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

同一组单层输入继续测量，FP16 QSA的S512/S1024/S2048热态时间分别为19.44/26.73/42.61ms，完整融合MoE分别为10.82/17.94/29.85ms。两者每token成本随块增大而下降；默认块仍须由整模型测量确定。

## 完整模型测量与迭代

`coreai-runner generate --profile-prefill true`记录每块token数、起止offset、墙钟时间、SSD读取、分组函数等待时间和48层明细。层时间已经包含在分组时间内，不能重复相加；这些都是宿主等待时间，不是硬件计数器。`--prefill-chunk`可选择任意已导出的块大小，便于在同一份资产上比较。

导出器只对含SDPA的模块启用externalization，避免其他模块被SDK重复追踪。长导出中断后，可保持原参数并加`--resume`续导；它校验已有资产文件和已记录的配置/源码哈希。仅导出器自身修改时需要显式`--resume-exporter-change`并记录版本关系，其他kernel变化不允许混用。未登记的残留资产会报错，需移开后继续，不会自动删除。

## 稀疏QSA候选与当前内存限制

`--prefill-sdpa float16`是主导出器的可选精度配置；独立QSA导出脚本对应`--prefill-sdpa-fp16`，均保持S1的FP32 SDPA。新实验`coreai_qsa_sparse.py`直接读取选中的块，以MPP QK/PV tile和FP32在线softmax/累计避免全局K/V展开；`coreai_qsa_sparse_full.py`仅替换prefill注意力，保留原索引选择、六项状态、诊断mask和S1路径。**稀疏候选未接入默认导出或服务**。大尺寸`--reference-only`仅生成原FP32参考，不代表已执行候选CPU计算。

| 新测量 | 热态时间 | 结论边界 |
| --- | ---: | --- |
| 同输入真实S512注意力 | 稀疏5.2869ms / native FP16复测5.4792ms | 约3.5%孤立算子收益；早先6.10ms基线不能用来宣称稳定13%收益 |
| 完整真实S512 QSA | 稀疏17.63ms / 较早native FP16 19.44ms | 尚未做交替配对复测，不能当作确定收益 |
| 合成S2048稀疏注意力 | 21.1136ms | 真实尺寸、top512规模；无同输入配对基线 |

五种小尺寸设备场景均通过；真实S512孤立算子relative L2为0.00008285、maxAbs为0.001953，整层状态连续、重置、checkpoint恢复检查通过。7项CPU测试包含新增整层封装测试，覆盖早期稠密、稀疏、尾块、未来位置、选择顺序和S1保持。实际层输入重放来自既有layer0 MoE捕获，不能称为原生layer3整模型轨迹。证据为`qsa-sparse-smoke-summary.json`、`qsa-sparse-real-s512-device.json`、`qsa-sparse-full-real-s512-device.json`及`qsa-sparse-synthetic-s2048-device.json`；均位于上述results目录。这里没有整模型质量或吞吐验收。

完整11,057-token测试在观测到10,752个token后停止：进程physical footprint达到139.3GB，未取得完整prefill结果。单层仅加载的对照中，包含10个函数的资产graphics footprint约2.36GB；只保留main为0.211GB、只保留prefill为1.067GB、两者共存为1.278GB。权重文件映射规模基本相同，说明保留的函数执行资源/工作区是更强的排查方向，**尚未证明运行时重复了整份权重，也未定位到具体内存arena**。当前优先研究共享通用图与外部权重的所有权和约束，再恢复整模型测量。证据见`full-agent-11k-v1-stopped.json`、`layer0-constant-memory-diagnosis.md`和`footprint-main-prefill.json`。
