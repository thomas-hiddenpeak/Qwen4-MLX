# 按专家边界组织 prefill MoE

2026-09-07，M5 Max。接续 [gate/up 融合](MOE_PREFILL_GATEUP.md)，本轮将全局 BM 块内的专家段循环改成每块只计算一个专家，并让可选 down 投影复用相同的分块计划。原 affine Q4/group64 权重、BF16 运算边界、路由顺序和共享专家保持原实现。

## GPU 分块与计算

`Native/moe_expert_grouped.metal` 包含 BM16/BM32 的 planner、融合 gate/up/SwiGLU、down，共6个 kernel。一个512线程组对已排序的专家 IDs 做 lower_bound，得到各专家区间；扫描每个专家需要的块数，生成 `{expert,start,count,0}` 描述符。全部计算在 GPU 完成，不读回活动专家数，不额外整理完整权重库。

M是排序后的 token-expert assignments 数，本模型为token数×10。计划容量为 `ceil(M/BM)+512` 行，每行4个Int32；首行记录实际块数。后续矩阵kernel使用固定容量dispatch，超出有效块数的线程组直接返回。gate/up和down共享这一个惰性计划，只生成一次。输出保留原sorted行位置，因此down之后的逆排序及top10归约不变。

两种矩阵形状均BN32/BK64/WN1：BM32用WM2，BM16用WM1，保持每个SIMD的16×32 NAX片段。每个块只装入一个专家的权重，输入和输出按该专家真实行数裁剪。gate/up独立FP32累加后，各自转BF16，再按原sigmoid LUT和两次BF16乘法生成activation；down同样沿原K顺序累加并转BF16。

原生桥接升级为ABI2，保留旧0/1融合入口；新planner、planned gate/up和grouped down显式接收runner的GPU stream。官方MLXC数组所有权保证共享计划及输入缓冲随惰性图存活。新插件仍链接固定stock libmlx，未替换作者运行库。

## 请求与短尾策略

`GPUMoEPrefillConfiguration.gateUpVariant`：nil=参考，1=上一轮融合，2=专家BM32，3=专家BM16；0保留上一轮BN64融合。`groupedDown=true`仅允许配合2/3，默认nil。

原GatherQMM合批条件包含`B/E>=4`，本模型top10、E512对应205 token起。新专家路径只在205…512 token块生效；更短或更长块由模型接口回退原计算，避免强制更换原小批次的累加方式。decode、verification与末尾S1不使用新路径。低层直接调用GPUMoE时，不支持的显式组合会报错。

## 本轮验证

微测复用九份已通过完整golden的真实S416输入（层0/23/47、位置0/4992/9984），另用首组205/240行前缀检查尾块。对照包含参考、旧融合1、专家BM32/16仅gate/up、专家BM32/16同时更改down，共6种模式。候选activation确实被后续down消费，诊断不会回退计算。

每个候选/输入检查10项张量或输出，包括activation、专家输出、路由结果、共享分支和诊断/普通调用交叉比较。通过者各预热3次，再做4组ABBA，每种8个样本。计时包含完整MoE建图、planner与最终y.eval，不含权重加载、诊断或host读回；九份真实输入参与评分，派生尾块不参与。相对旧融合的归一化排名是间接比较，不冒充两候选直接配对。

完整测试使用`probe-gpu-hotspots --detail moe-expert --moe-config PATH`，保留11057-token提示词、128-token AR输出、真实producer/consumer交接，MTP关闭，阶段采样关闭。分别记录prefill与decode业务耗时；候选每个请求应有1296次gate/up及plan，开启grouped down时再有1296次down，decode全部为0。

Release 编译和33项CPU契约、阶段与调度检查通过。原生ABI2构建、导出和依赖验证通过，见[构建记录](../results/moe-prefill-expert-v1/native/build-provenance.json)。

## 真实输入结果

[微测原始记录](../results/moe-prefill-expert-v1/micro.json)与[所选配置](../results/moe-prefill-expert-v1/selected-config.json)。五种候选在11个输入上的550项比较全部有限且逐位一致，没有数值淘汰；880个计时样本及gate/plan/down计数均按预期。

| 候选 | 九组配对参考合计ms | 候选合计ms | 完整MoE吞吐变化 |
| --- | ---: | ---: | ---: |
| 原全局分块融合1 | 44.581 | 42.486 | +4.93% |
| 专家BM32，仅gate/up | 45.015 | 34.529 | +30.37% |
| 专家BM16，仅gate/up | 44.984 | 40.268 | +11.71% |
| 专家BM32，gate/up与down | 44.810 | 32.548 | **+37.67%** |
| 专家BM16，gate/up与down | 45.049 | 39.553 | +13.90% |

胜者为variant2/groupedDown=true，九组真实输入全部改善，单组吞吐变化+33.77%至+46.41%。合计耗时下降27.36%，对应吞吐提高37.67%，两者不能混写。它沿用原归约；本轮尚未叠加之前的归约融合。相对旧融合1的间接归一化提升约31.20%，不作为直接配对结果宣传。

## 完整 PD 结果

[ABBA](../results/moe-prefill-expert-full-v1/abba.json)、[BAAB](../results/moe-prefill-expert-full-v1/baab.json)与[阶段汇总](../results/moe-prefill-expert-full-v1/summary.json)。两窗口各6轮（各排除开头A/B预热），共12轮、1536个输出token全部匹配原golden、callback和返回结果，终止为length，最终offset11184，交接检查全部通过。

每个候选请求实际编码1296次gate/up、1296次plan及1296次down；host建图计数对应。所有参考请求、所有decode的这些计数均为0，融合归约未启用，阶段采样关闭。

统计prefill.totalSeconds及decodeServiceSeconds。首个输出token属于prefill，decode吞吐分子为127，未混入模型加载、预热或阶段间等待。

| 热窗口 | 参考prefill秒 | 候选prefill秒 | Prefill token/s 参考→候选 | 吞吐变化 | Decode token/s 参考→候选 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ABBA | 18.086 | 14.749 | 611.37→749.67 | +22.62% | 25.75→26.38 |
| BAAB | 22.994 | 22.302 | 480.87→495.79 | +3.10% | 22.20→22.12 |

八个热请求合并后prefill均值20.540s→18.525s，观察到吞吐+10.87%（耗时−9.81%）；decode观察到+0.93%。这些是描述性均值，不能当成稳定幅度承诺：ABBA参考自身漂移+46.14%，BAAB候选−22.37%；四个相邻prefill对照依次+9.06%、+34.03%、−8.30%、+17.79%。

这轮确认了真实局部MoE的一致收益和完整生成正确性，整模型平均也有正向观察，但持续运行中的波动仍需单独解释。运行时只读检查显示AC供电、无已记录的系统温度告警、少量swap；这些检查无法排除GPU频率、温度、系统竞争等影响。没有同步因果证据，暂不把变慢归因于某一项。

## 使用与决定

[候选状态](../results/moe-prefill-expert-v1/candidate-status.json)：保留variant2/groupedDown为可选优化，默认仍为nil。[普通runner配置](../results/moe-prefill-expert-v1/runner-config.json)复制自原微测选择，并记录已通过完整PD正确性检查；原选择文件保持原样供追溯。

普通`generate-gpu`新增`--prefill-moe-config PATH`。启动时校验配置版本、模型目录、插件与基础MLX哈希、实际动态库符号归属及ABI；拒绝旧非零全局矩阵调参或非reference累加。配置只传给prefill forward，逐请求核对host/native调用数和decode零调用，不更改进程环境。

在本项目目录运行（输出路径须自行选择）：

```sh
ANERUNNER_GATEUP_LIBRARY="$PWD/results/moe-prefill-expert-v1/native/lib/libanemlx_moe_gateup.dylib" \
  .build/release/ane-runner generate-gpu \
  --model-dir ../qwen38-ssd/models/Qwen3.8-Flash-Next-MLX-SSD-Stream \
  --tokens-file fixtures/gpu-agent-11k/prompt-token-ids.json \
  --context 16384 --max-tokens 128 --mtp-depth 0 \
  --prefill-moe-config results/moe-prefill-expert-v1/runner-config.json \
  --output results/expert-generate.json
```

普通入口已通过[真实生成检查](../results/moe-prefill-expert-cli-v1/generate.json)及[汇总](../results/moe-prefill-expert-cli-v1/summary.json)：同一11057-token提示词、128个输出全部匹配golden，offset11184；prefill的gate/up、plan、down各1296次，decode全部为0。另用错误插件哈希和旧非零全局矩阵配置检查，均在模型权重加载前报错。两窗口PD加普通CLI，共1664个生成token通过对照。单次CLI冷运行不是新增性能A/B证据。

该轮参考服务恢复PID90029，MTP/drafter关闭，见[恢复记录](../results/moe-prefill-expert-cli-v1/run-ledger.json)。以上是当时命令行和 typed API 的验证；工程后来新增了 [HTTP/SSE 适配器](HTTP_SERVER_EXPERIMENT.md)，其独立验证不能从本轮 CLI 结果推导。

后续与融合归约的直接组合对照及短尾回归见[组合回归](MOE_PREFILL_COMPOSITION.md)。当前参考服务PID以最新运行记录和实时状态为准。
