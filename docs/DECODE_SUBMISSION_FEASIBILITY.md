# 当前 token 内的 GPU 提交实验

2026-09-07。**单个 AR decode token 每八层提前提交已完成实现、构建和一次固定四进程实验。正确性门槛通过，性能因基线漂移超过门槛而未定，默认仍为 0（关闭）。** 本次只调整当前 token 内的提交时点，没有开发跨 token pipeline。

## 证据支持到哪里

[既有命令跨度诊断](GPU_DRIFT_TRACE.md)中，暖 AR 的127步墙钟增加281.120 ms，GPU跨度增加215.905 ms，跨度外增加65.214 ms。23.20%是**变慢的增量比例**；最后一轮总decode中，跨度外586.789 ms占4.266935秒的13.75%。这两个比例都不是可消除的CPU开销、GPU空闲率或带宽利用率。历史trace使用`ad7717f2`二进制，不能与之后的`f95565c`或本次`5f1321b4`二进制混算。

跨度外增量中的61.921 ms位于“下一条buffer尚未提交”的间隙，但没有线程栈将它分配给Swift构图、MLX遍历/编码、驱动或系统调度。多数增量仍在GPU跨度内；提前提交可能有收益，也可能因更多图切分、event和command buffer变慢。

## 已有执行顺序

- `QwenModel.forward`需要host token计算CPU n-gram行号、更新UInt32 history并准备PLE读取；PLE等待之前已经执行`MX.asyncEval([h])`。下一token尚未读回时，现有接口不能准备其确切SSD行。
- S1不走prefill的每四层同步求值。正常AR每步最后联合求值selected与全部`state.tensors`，保证Attention/GDN/PLE等持久状态完成，再读取标量。普通每步没有额外的`MX.synchronize()`。
- 当前高层generator已经用`uint32TokenID()`读回。CLI reference仍有UInt32→Int32转换，现有`.scalar`模式可以去掉；这不是新的优化，不能和提前提交同时更改后归因。
- 历史CLI trace只在生成结束整体解码文字，没有逐token tokenizer/HTTP callback。服务的UTF-8、JSON和callback在计算decode之后执行，另计service/callback时间，不能用它们解释上述历史65.214 ms。
- 固定MLX `1f8e74e3f12f`、MLXC `56b2d39fc831`中，`async_eval`仍在调用线程遍历和编码图，最终提交而不等待event。把最终`eval`换成`async_eval`后立即读item，不会消除同一数据依赖。

核对路径为`QwenModel.swift`、`QwenGeneration.swift`、`GPUGeneration.swift`、`GPUPLE.swift`、`QwenExecutionPhase.swift`及固定MLX的`transforms.cpp`/`array.cpp`。本机完整只读记录保存在`results/decode-host-gap-v1/analysis.md`。

## 上游可吸收的边界

| 来源 | 可以借鉴 | 本机限制 |
| --- | --- | --- |
| [vLLM固定core.py](https://github.com/vllm-project/vllm/blob/6865e67f0be02d53694517f6f71d7fb96492792d/vllm/v1/engine/core.py#L649) | 有界in-flight、先提交后消费、消费前处理取消 | 同步`forward([Int32])`外包future不能自动得到独立executor的重叠 |
| [SGLang固定scheduler.py](https://github.com/sgl-project/sglang/blob/2c05ed4e7776c876478f4b2db61acb12b9a27d01/python/sglang/srt/managers/scheduler.py#L1944) | 在WAR barrier和grammar依赖下分开提交与消费 | 不能省略状态依赖，或无条件把处理上一batch套到本模型 |
| [DwarfStar固定Metal runner](https://github.com/antirez/ds4/blob/9ab705347c1775e7599ede7eb81a6255ec7dccb5/ds4_metal.m#L9452) | 局部flush提交、最终end统一等待 | 不等于跨过CPU n-gram/SSD、EOS或取消边界 |

这里只借鉴设计，没有复制实现。vLLM/SGLang为Apache-2.0，DwarfStar为MIT；以后复制代码仍需按文件保留许可与声明。

## 实验范围与预设门槛

已应用的实验路径只在明确指定AR decode、S1的第8/16/24/32/40层HC write后，对`[h] + state.tensors`执行`MX.asyncEval`；第48层、最终selected/state求值、标量读回和PLE前已有提交均保持。MTP的scalar verify、replay和target-only收尾也可能标为decode/S1，已在所有trunk forward入口显式关闭，不能只按形状判断。环境变量`ANERUNNER_EXPERIMENTAL_DECODE_ASYNC_LAYERS`每个模型初始化时读取一次，仅接受`0`和`8`；缺省为0，非法值拒绝。配置及成功提交调用计数已进入generate-gpu报告。

同一11057输入、chunk416、AR输出128、同一读回模式：先分别预热，再固定baseline/async8/async8/baseline。完整IDs、EOS/预算/offset及一步持久状态必须先通过。decode改善至少3%、前后baseline漂移不超过5%才进入后续不带详细采样的复核；prefill另列。保留全部失败或无收益结果，不根据结果扩大参数扫描或切换默认。

## 本次实际验证

两窗口冻结解除后，由主代理应用候选并统一release构建，构建日志记录49.11秒；`QwenDecodeAsyncScheduleTests`的3项CPU测试通过，覆盖环境变量字面值、prefill/verification排除和提交层边界。构建同时包含独立PD探针，不能把编译时间归到这项改动。构建与CPU日志为`results/pd-async-integration-v1/{build,cpu-tests}.log`。

一步状态门槛只加载一个完整模型，不加载MTP head；从同一11,057-token私有checkpoint，分别关闭和开启该提交路径处理同一个token。`checkpoint()`及State赋值本身仍为浅拷贝，两个分支使用已求值的独立GPU gather副本。13项检查全部通过：完整logits逐位一致且有限，121个命名持久张量在初始两份副本、一步结果和原checkpoint复核中逐项通过shape/dtype/完整字节/有限值比较；host offset、逐层offset、UInt32 PLE history、nil/capture状态一致，原checkpoint未被分支修改。两分支成功提交调用数为0/5，最终offset为11,058。这是一处AR状态验证，不代表MTP或任意位置均已完成回归。原始结果为`results/decode-host-gap-v1/state-gate.json`。

性能按预先固定顺序`0 / 8 / 8 / 0`运行四个独立进程，每个进程2次请求：第0轮固定预热，第1轮固定测量。全部8次请求的完整11,057个输入ID、128个输出ID均与冻结AR参考一致；均为length结束、最终offset 11,184、12层QSA活跃。每次输出首token来自prefill，因此实际decode均为127步，四个进程累计实验提交调用数依次为`0 / 1270 / 1270 / 0`（开启时为5次 × 127步 × 2请求）；这是host成功调用计数，不是Metal command-buffer或kernel计数。

四进程及状态门槛使用同一二进制SHA `5f1321b44aa9537b3acf50844740f40625116683872623bd971a7be4aa3f8cb7`，模型目录与metadata SHA一致。四个进程退出均为0，ledger记录顺序执行，未删除或替换请求。固定配置为context 16,384、chunk 416、AR、reference decode/GDN、SSD workers 1、nextChunk预取、wired disabled；详细profiler、GPU command timing和telemetry关闭。

### Decode：暖轮保留，只有测量轮参与比较

| 进程顺序 / PID | 每隔层数 | 暖轮0 decode秒 | 暖轮0 token/s | 测量轮1 decode秒 | 测量轮1 token/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 / 26326 | 0 | 3.957161 | 32.093714 | 4.220243 | 30.093057 |
| 1 / 26428 | 8 | 3.865614 | 32.853774 | 4.029662 | 31.516289 |
| 2 / 26488 | 8 | 4.300563 | 29.531019 | 4.641783 | 27.360179 |
| 3 / 26565 | 0 | 5.400835 | 23.514883 | 6.067535 | 20.931069 |

每行速度的分母是该请求全部127个`decode_step_seconds`之和。两侧聚合使用实际token数除以总时间：baseline为254 / 10.287777751秒 = **24.689491 token/s**，async8为254 / 8.671444842秒 = **29.291543 token/s**；不平均请求速度，不混入预热、prefill或模型加载。

原始比值为**1.186397**，但外侧两次baseline从30.093057降到20.931069 token/s，按`abs(last - first) / first`计算漂移为**30.445520%**，超过预设5%。因此正式结果是`complete=true, all_correct=true, passed=false, outcome=indeterminate_drift, errors=[]`。**不能把原始倍率表述为18.6%的优化收益，也不能据此启用默认。** 本轮没有采集能将漂移归因到CPU、GPU频率或温度的详细数据。

### Prefill：单独列出，不进入decode倍率

| 进程顺序 / 每隔层数 | 暖轮0 target秒 | 暖轮0 total秒 | 测量轮1 target秒 | 测量轮1 total秒 |
| --- | ---: | ---: | ---: | ---: |
| 0 / 0 | 20.569431 | 20.571650 | 14.547741 | 14.548770 |
| 1 / 8 | 19.707397 | 19.708355 | 14.984031 | 14.984954 |
| 2 / 8 | 25.900761 | 25.901927 | 18.903760 | 18.904751 |
| 3 / 0 | 25.131246 | 25.132984 | 26.217630 | 26.221577 |

`target`为目标模型prefill时间，`total`为排除加载的首token时间；本组8次请求的MTP prompt-history时间均为0。实验提交路径不会在prefill启用，以上变化不能归为它的prefill收益。

原始四份报告、顺序与清理证据在`results/decode-host-gap-v1/performance/`；正式汇总为`results/decode-host-gap-v1/summary.json`。本次文档更新另从原始报告独立重算完整IDs、127步分母、两侧总时间和漂移，与汇总一致。性能进程运行时间为2026-09-07 02:06:36.372至02:10:49.111 UTC；本轮结束后参考服务PID26788在02:11:04.404 UTC验证ready。这是该次ledger的恢复记录，不代表后续任意时刻的当前服务PID。

保留实验入口，默认继续为0。后续若复核，仍应固定顺序、预热与漂移门槛，不能替换本轮未定结论或挑选较快的相邻结果；HTTP日志阻塞修复及[固定到达时点的PD探针](research/LOCAL_SCHEDULER_NEXT.md)按主代理排期推进。
