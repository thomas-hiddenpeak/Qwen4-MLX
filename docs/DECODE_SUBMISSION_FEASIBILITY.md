# 当前 token 内的 GPU 提交实验

2026-09-07。只读固定源码与已有数据后，值得保留的下一项小实验是：**单个 AR decode token 每八层提前提交，最终仍等待完整状态完成。** 当前只有 ignored 候选准备，没有新构建、GPU 测量或默认变更。不开发跨 token pipeline。

## 证据支持到哪里

[既有命令跨度诊断](GPU_DRIFT_TRACE.md)中，暖 AR 的127步墙钟增加281.120 ms，GPU跨度增加215.905 ms，跨度外增加65.214 ms。23.20%是**变慢的增量比例**；最后一轮总decode中，跨度外586.789 ms占4.266935秒的13.75%。这两个比例都不是可消除的CPU开销、GPU空闲率或带宽利用率。历史trace使用`ad7717f2`二进制，不能当作当前`f95565c` HTTP服务的计时。

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

## 一个固定的小实验

候选只在AR、decode、S1的第8/16/24/32/40层HC write后，对`[h] + state.tensors`执行`MX.asyncEval`；第48层、最终selected/state求值、标量读回和PLE前已有提交均保持。MTP的scalar verify、replay和target-only收尾也可能标为decode/S1，必须显式排除，不能只按形状判断。默认关闭，实验配置需进入报告身份。

同一11057输入、chunk416、AR输出128、同一读回模式：先分别预热，再固定baseline/async8/async8/baseline。完整IDs、EOS/预算/offset及一步持久状态必须先通过。decode改善至少3%、前后baseline漂移不超过5%才进入后续不带详细采样的复核；prefill另列。保留全部失败或无收益结果，不根据结果扩大参数扫描或切换默认。

HTTP修复与既定回归优先，然后是[固定到达时点的PD探针](research/LOCAL_SCHEDULER_NEXT.md)。本候选需等两窗口版本冻结解除后由唯一GPU所有者统一编译与排期。
