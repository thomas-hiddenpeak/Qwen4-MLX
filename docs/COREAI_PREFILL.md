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


## 共享图与显式常驻权重

`export_coreai_pd_shared.py --baseline-pd <完整v1目录> --output <新目录>`导出v2格式：48份原量化权重、3份通用decoder图（GDN、含PLE的GDN、QSA），各层保持独立状态。Swift同时支持v1/v2；每份v2权重由长期持有的shared MTLBuffer集合拥有，加载时直接pread，不在每个token重读文件。权重格式检查覆盖路径、文件大小、dtype、offset、溢出和重叠；显式完整性验证可检查文件及分片哈希。几何常量仍留在图内，学习权重全部成为命名输入。

首个完整11,057-token v2测试完成两轮：prefill分别129.79s/85.19token/s及121.25s/91.19token/s，加载16.34s，采样physical footprint峰值109.72GiB。没有prefix命中或MTP；第二轮仅OS文件缓存变热。两轮重置后logits一致、两token输出一致，但这不是源模型质量验收。该版本仍未达到1000token/s。

后续独立定位得到两个可重复结果：

- `--flat-q4`将九个专家权重输入改为rank1，Metal显式按原行序寻址；名字、dtype、字节及存储顺序不变。相同S2048量化投影由12.74ms降到7.01ms；完整layer0由约74.7ms降到45.5ms，输出和两项状态逐值一致。这是常驻外部权重场景的收益，不能与之前常量权重寻址实验混为一谈。
- `coreai_head_metal.py`及独立导出器保持HC与LastHead，使用FP16权重、FP32累加/输出的GEMV，避免整份输出权重转换成FP32。十入口head仅加载的graphics footprint由25.56GB降到0.132GB；同输入单次head由12.8ms降到3.75ms。完整候选head的S1/S2048 logits逐值一致；对原head relative L2为`1.6990e-4`、maxAbs为`8.8215e-4`，对CPU参考分别为`4.5772e-4`、`0.0025303`，**未通过预先设定的`1e-5`/`0.0005`门槛**。首选token相同不代替数值或质量验收；证据为`head-metal-fp32-real/device-summary.json`。

随后使用相同真实词表权重和同一份FP16 mixed输入，绕过HC，单独比较原FP32 linear与新Metal投影。设备候选对原投影relative L2为`1.3767e-7`、maxAbs为`7.1526e-7`；对CPU参考relative L2为`1.3046e-7`，通过原门槛。CPU模拟32-lane求和也只有`1.3046e-7`相对误差。12份完整head设备配置及S1 JSON fixture的末尾输入均逐字节一致，排除了不同输入后缀。证据为`head-projection-fp32-real/device-summary.json`、`manifest.json`及`fixture-audit.json`，可用`scripts/export_coreai_head_projection.py`重新导出。该对照把完整head的较大差异收窄到HC或其与投影连接的图上下文，尚未定位具体编译优化、融合或舍入行为；不能把差异直接归因于GEMV求和顺序，也不宣称HC问题已解决。

`external_call_stage_milliseconds`进一步拆分共享图的提交、等待计算及提取NDArray时间，按prefill/decode分别汇总并记录逐块差值。它们是宿主阶段耗时，不是GPU硬件计数器。

PLE大批量SSD读取对重复行去重，并以最多8个worker执行pread；S1维持串行小请求路径。结果和错误顺序保持原请求语义，14项小文件测试覆盖FP8/BF16、重复行、边界、截断及失败恢复。尚不把读取实现变化视为端到端性能达标。


### 权重缓冲区修复后的完整结果

只改变权重的拥有/绑定方式，独立设备对照同一flat layer0：一份整文件缓冲区加多个偏移为158.7ms（复测144.4ms），每个张量独立缓冲区且绑定偏移为0是45.8ms（复测45.7ms）。全部输出/状态逐值一致，进程内存占用基本相同。没有把这个现象归因于已证实的某个CoreAI内部复制机制。

生产`.resident`因此改为逐张量直接pread到独立shared MTLBuffer，文件仍按原offset读取；`.residentFile`保留旧分配方式供对照，`.mapped`保留实验用途。41项实际生产loader检查通过，包含三种模式、视图持有生命周期，以及填充区损坏时的整文件哈希拒绝。

相同11,057-token完整模型两轮prefill为**19.33s/571.9token/s**与**17.64s/626.6token/s**，模型加载8.07s，采样峰值80.31GiB。两轮均重新计算完整prompt，无KV/prefix命中、无MTP；第二轮OS文件缓存热。两轮重置输出一致，decode单步分别0.148s/0.123s，单步不能视为稳态decode吞吐。短尾块S32/S16/S1降到约0.24/0.20/0.16s，消除了原来接近9s/块的固定开销。**完整prefill仍未达1000token/s**；下一步测试更大块及其独立kernel。证据`full-agent-11k-per-tensor.json`、`full-agent-11k-per-tensor-memory.json`及`external-layer0/flat-buffer-layout-summary.json`。


### 更大块与整数专家分组

共享导出器支持`--integer-grouping --chunk 4096`或`8192`，可与`--flat-q4 --metal-head true`组合。三步I32分组通过block histogram、exclusive prefix和原序scatter生成permutation/inverse/sorted IDs，避免FP32唯一排序键的精度上限；路由本身不变。529、40960、81920项分别测试混合ID、单专家与降序重复输入，三项设备输出均与独立整数稳定排序参考完全一致。大块导出重新生成匹配的embedding/head入口，Swift只在manifest明确声明整数分组时接受大于2048的主块。

完整11K测试中，4096主块热态18.739s/590.1token/s，8192主块23.143s/477.8token/s，均未优于2048主块17.645s/626.6token/s。采样峰值分别88.02GiB和101.22GiB；不据此认定具体内存或算力瓶颈。大块能力保留为可选实验，不提升为性能默认配置。所有场景均重算完整prompt，两轮文本输出一致；这仍不是源模型质量验收。

进一步head诊断中，两实现返回的FP16 mixed逐值一致；用实际mixed和真实权重进行FP32 CPU线性重放，Metal logits的relative L2为1.31e-7，而原图内投影为1.70e-4。新kernel符合声明的FP16输入边界；原图内部为何与显式边界重放不同仍未定位，不宣称已证明某种编译融合机制。设备mixed相对CPU HC仍有差异，需独立的整模型质量验收。证据`head-mixed-diagnostic/cpu-device-mixed-replay.json`。


### 量化读取与原生 uint4 实验（仍未达 1K）

可选 `--flat-q4 --contiguous-affine` 将同一 affine group 的相邻 I16 words 分配给同一线程，减少重复读取 scale/bias；原 FP16 解包与 MMA 算法不变。真实 layer0 down 投影 S2048 从 8.113ms 到 7.141ms，S8192 从 27.084ms 到 24.531ms，输出和计划逐 bit 一致。但 fused gate/up S2048 为 12.863ms vs 13.084ms，无收益。完整 11,057-token 两轮为 21.799s 和 17.514s（热态 631.3token/s），文本与重置 logits 一致，采样峰值约 80.35GiB；相对原 626.6 的幅度不足以认定稳定整模型收益。选项默认关闭。7 项 flat CPU 测试与 9 项共享导出 CPU 测试通过。

公开 MPP `half × uint4b_format` 小图已在本机编译运行；原生 uint4 grouped 全 K 候选仍比原 packed-half 慢：同 S2048 合成几何输入 8.402ms vs 7.093ms。该方案使用分组 affine 后校正，省略了原逐权重 FP16 舍入，输出 relative L2 为 0.000312，因此保留为独立实验，不进入生产默认。原生路径边界 CPU 测试通过，GPU 小图对其独立新公式 oracle 逐 bit 一致；这不等于原模型数学等价。

原始证据位于 `results/coreai-prefill-1k/` 下 `q4-loader-down-s2048`、`q4-loader-down-s8192`、`q4-loader-gateup-s2048/device-summary.json`、`full-agent-11k-contiguous.json`、`q4-native-grouped-smoke/device-large-summary.json`。这些结果均未使用 MTP 或 KV/prefix 命中。


### 按有效历史选择 QSA 工作视图

可选 `--qsa-working-sets 2048 4096 6144 8192 10240 12288 14336` 为主 prefill 块增加静态有效 KV 上界。运行时按真实 `offset + count` 选择覆盖全部历史的最小上界；没有合适入口时回到完整容量，S1 仍走原 decode。六个缓存状态始终保持完整容量，快照和 offset 约束不变。

独立真实 QSA S2048 冷前缀为31.200→9.674ms，offset8192为31.361→22.516ms，输出、六状态和可见数均逐 bit 一致；补充非零随机状态、offset8191/pooled_count2047的部分块对照也一致。完整11K两轮为23.140/17.551s，热态约630token/s；QSA累计5.288→4.589s，但GDN计时波动抵消整轮收益，不能据此宣称显著整模型加速。采样峰值约87.11GiB；多函数的额外工作空间需继续控制。

Release 构建通过；Swift工作集选择及非法metadata检查通过直接编译的fixture执行。`swift test`被本机Command Line Tools缺少XCTest阻断，未报告整套Swift测试通过。共享导出与工作集CPU检查16项通过，真实全模型执行完成两轮。证据：`qsa-working-set-s2048/device-*-summary.json`、`full-agent-11k-qsa-workingset.json`。新增 `generate --prefill-logits-output PATH.f32` 在计时区间外保存首轮preﬁll的248320个Float32 logits，供不同候选完整比较；它不改变正常输出路径。


### MoE 数据搬运与融合边界

实际S8192几何下，原生ordered gather为34.426ms，直接索引kernel为1.464ms且逐bit一致；原inverse/weight/reduce图为88.516ms，融合尾部约1.3–2.5ms。原图广播的索引/加权中间态是已通过设备对照确认的重要成本。完整packed MoE S2048为29.205→22.013ms，S8192为196.800→77.247ms，路由IDs/scores一致。

`--moe-direct-transfers` 默认关闭。`--moe-tail-precision native`只换exact gather；`float16`保留显式half-product候选；`float32`指FP32 products、opaque FP16 routed输出，**不等于**独立实验中的FP32 routed输出策略。原生图在实际设备上会消除一些中间half边界，不能用PyTorch源码上的`.half()`推断其融合后的精确算术。暴露routed/shared为图输出会改变最终结果，诊断图必须与未拆图比较。

完整11K的direct transfers候选热态15.085s/733.0token/s，FP32 routed输出候选14.939s/740.2token/s；同期原配置17.979s/615.0token/s。两候选对原配置最终logits relative L2分别0.0954/0.1115，虽首token和两token文本一致，**尚未通过质量验收，不设为默认**。仅QSA工作集版本的全11K logits则逐bit一致。单独第0层FP32 routed输出候选误差降到4.68e-7，仅83/20,971,520个stream元素不同，仍可能在深层MoE路由中放大。

树形FP32-output尾部是另一个独立候选：pad16后stride8/4/2/1，whole MoE S2048只有21/5,242,880元素不同、relative L2 7.98e-7；完整模型还要单独验证。默认导出不会隐式选它。对应证据：`moe-transfer-s8192/*/device-summary.json`、`layer0-transfer-comparison/device-summary.json`、`full-agent-11k-transfers*-logits-comparison.json`、`full-prefill-distribution-comparison.json`。

### 可选GDN ILP与NAX矩阵核

`--gdn-prefill-rows 1|2|4`默认1。每SIMD同时维护4个独立value rows、复用q/k/gates，真实预处理的H48/V128/T2048 recurrence为5.872→3.630ms。最终FP32 recurrent state逐bit一致；y最大差1.91e-6、relative L2 1.28e-6。V7尾行/零decay/零beta的小图输出与state均exact。S1仍使用原kernel。

可选NAX down移植使用MLX的MIT许可寄存器fragment结构、权重stride72和公开MPP每SIMD16×32×16操作，未链接MLX runtime。真实同输入S2048 down为8.170→5.574ms，output/plan逐bit一致。gate/up版本为12.859→10.030ms但存在约4.12e-4输出差异，仍单独调查。`install_nax_moe(..., projections='down')`只启用已验证down，普通构造默认不启用。整模型达到1K与数值验收均仍待完成。

本阶段root复跑7项transfer CPU检查、4项NAX CPU检查、14项共享导出检查，均通过。所有吞吐仍是完整prompt重算，未用MTP或KV/prefix命中；独立算子速度不可直接当作整模型速度。
