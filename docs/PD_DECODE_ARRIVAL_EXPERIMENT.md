# 已开始decode后到达长prefill的本机PD实验

2026-09-07。固定4→8→8→4的四组已完成：13份请求的正确性与生命周期检查通过，**性能筛选未通过，默认decodeBurst仍为4**。两个方向的短请求剩余完成时间分别改善33.3946%和42.0547%，但首对整组耗时增加9.3112%、长请求TTFT增加11.4344%，超过预定3%和10%上限。反序配对的代价方向不同，外侧burst4整组耗时自身增加18.5856%；本轮不证明稳定收益，也不将所有时差归因于burst。

## 固定输入与执行边界

[正式计划](../results/pd-fairness-v1/plan.json)冻结112份文件，服务二进制为`5f1321b44aa9537b3acf50844740f40625116683872623bd971a7be4aa3f8cb7`。PD探针与默认关闭的async候选一同构建，但本次命令显式移除`ANERUNNER_EXPERIMENTAL_DECODE_ASYNC_LAYERS`；本页不测试async8或新的模型kernel。只调用现有cooperative scheduler，单模型、单执行器、AR、reference decode/attention、chunk416、两份resident/ready、11275逻辑token预留。

[冻结参考](../results/pd-fairness-v1/candidate/frozen-reference.json)直接来自历史cooperative报告。短提示26个输入IDs，既定64-token预算下实际输出64、finish=length；长提示11057个输入IDs，128-token预算下实际输出128、finish=length。没有屏蔽EOS或为满足最短长度换提示词。本次先运行独立短/长AR并逐ID核对冻结结果；短结果不足64或任意不匹配会在四组开始前标为inconclusive，本次两份都通过。

每组只先提交短请求。第8个输出callback（索引7）记录长请求到达；callback不调用scheduler。当前slice返回并记下时钟后，外层才submit长请求。四组到达→开始提交分别为8.459、10.083、8.958、7.375微秒；原始事件均证明到达位于短decode slice内，提交在其返回后，且有真实长prefill插入到短终态之前。没有模拟随机到达、HTTP传输、kernel抢占或GPU并行。

实际请求分解为：初始独立短/长参考2份，四组长短请求8份，独立取消检查2份，取消后fresh短AR1份，共13份。**11份完整完成均匹配冻结的完整IDs；2份取消只验证已输出前缀及释放，不能称为13份完整生成。** 短请求完整终态offset=89，长请求=11184，均符合AR的prompt+output−1；所有完成原因均为length。

## 到达后的公平性与总耗时

短剩余完成时间使用“第8个callback→短请求终态runNext返回”；长TTFT使用“同一到达时刻→长请求首callback”。整组墙钟从首次提交短请求之前到pump结束，包含所有请求、排队与清理。下表不混入模型加载或独立参考。

| 组索引 | burst | 短剩余完成秒 | 长到达TTFT秒 | 长到达→终态秒 | 整组墙钟秒 | 短剩余期间插入长prefill次数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 4 | 8.847050 | 15.801284 | 20.025539 | 20.391700 | 14 |
| 1 | 8 | 5.892613 | 17.608062 | 21.883845 | 22.290410 | 7 |
| 2 | 8 | 6.077730 | 18.181582 | 22.512982 | 22.949768 | 7 |
| 3 | 4 | 10.488743 | 19.044683 | 23.712742 | 24.181629 | 14 |

配对在运行前固定为(0基线,1候选)和(3基线,2候选)，第二对反转时间顺序。每对都要求短剩余时间改善至少15%、整组回退不超过3%、长到达TTFT回退不超过10%；最大gap若两个方向都回退超过5%也会失败。这是[研究计划](research/LOCAL_SCHEDULER_NEXT.md)的局部筛选，不是生产SLO。

| 对照4→候选8 | 短剩余时间改善 | 整组耗时变化 | 长到达TTFT变化 | 最大gap变化 | 剩余/整组/TTFT联合门槛 |
| --- | ---: | ---: | ---: | ---: | --- |
| 0→1 | 33.3946% | +9.3112% | +11.4344% | +1.7850% | 未通过 |
| 3→2 | 42.0547% | -5.0942% | -4.5320% | +1.8066% | 通过 |

原始事件支持插入次数14→7的变化；但首对组墙钟和长TTFT超过上限，第二对则分别下降5.0942%和4.5320%。外侧两份burst4从20.391700升到24.181629秒（+18.5856%），长TTFT自身+20.5262%，短剩余时间自身+18.5564%。这些变化限制了归因，不新增事后漂移阈值，也不删首组、挑选反序通过组或继续扩大burst来补成通过。

## 全部56个输出间隔

每组完整保留callback8→9至63→64的56个相邻间隔；所有样本均计入，没有只挑包含prefill的长间隔。AR每个decode slice发布一个token，因此此处callback gap与跨输出步骤的burst gap相同。p50/p95使用排序后q×(n−1)的线性插值；这是单条请求内的间隔分布，不是请求总体p95。

| 组索引 | burst | 间隔数 | p50毫秒 | p95毫秒 | 最大毫秒 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 4 | 56 | 30.521959 | 527.291020 | 957.235750 |
| 1 | 8 | 56 | 32.780729 | 561.050167 | 974.322000 |
| 2 | 8 | 56 | 33.678646 | 584.347083 | 972.593583 |
| 3 | 4 | 56 | 33.938229 | 656.013916 | 955.334958 |

两个方向最大gap分别回退1.7850%与1.8066%，未触发5%的重复回退条件，但也没有改善；最大停顿仍约0.96–0.97秒。首对p95还从527.291升到561.050毫秒。减少prefill插入次数不能据此宣称单次停顿已经解决，更不能当作GPU权重读取合并或kernel加速。

## Prefill、decode与等待分别统计

下表的target prefill来自`phases.prefill.targetSeconds`，scheduler活动prefill来自终态的`prefillStageSeconds`；后者包含该阶段的控制/准备成本。decode compute来自完整`result.decodeSeconds`，scheduler活动decode包含callback和阶段开销。活动时间均排除调度暂停；所有AR prompt head history为0。每份短/长完整请求分别有63/127个实际decode token，首token由prefill选出；不把等待计入compute吞吐。

| 组/burst | 请求 | target prefill秒 | 活动prefill秒 | decode compute秒 | 活动decode秒 |
| --- | --- | ---: | ---: | ---: | ---: |
| 0/4 | 短 | 0.165824 | 0.165881 | 1.943971 | 1.944967 |
| 0/4 | 长 | 13.891331 | 14.055468 | 4.220818 | 4.223211 |
| 1/8 | 短 | 0.186264 | 0.186318 | 2.050514 | 2.050864 |
| 1/8 | 长 | 15.721374 | 15.776213 | 4.272409 | 4.274786 |
| 2/8 | 短 | 0.202048 | 0.202096 | 2.126565 | 2.126923 |
| 2/8 | 长 | 16.240937 | 16.288216 | 4.328171 | 4.330479 |
| 3/4 | 短 | 0.222235 | 0.222290 | 2.171657 | 2.172020 |
| 3/4 | 长 | 17.069846 | 17.118155 | 4.664840 | 4.667099 |

短请求累计ready队列等待为7.102325、4.061968、4.185470、8.563289秒，长请求prefill暂停等待约1.745–1.926秒；这些原始字段单列，不能用compute秒数拼成用户可见TTFT。实际短/长prefill分别执行2/28步：固定chunk上限之外，最后一个prompt token单独处理，并非简单取ceil(prompt/416)。逐步外层时钟覆盖对应活动时间，阶段等待和结果中的compute计量一致。

## 取消、恢复与证据限制

四组后另用burst4运行相同到达流程，长请求处理第一个416-token块后，两份resident与11275预留仍在。随后在slice之外分别取消长prefill与短decode：短只输出冻结前8个IDs，长尚无输出，两份取消终态各一次；终态scheduler健康、idle，队列/pending/resident/reserved全部为零。最后fresh短AR重新输出冻结完整64 IDs、length、offset89。这里覆盖公开接口的输出、offset与生命周期，不宣称逐位比较了隐藏recurrent/conv/KV tensor，也不覆盖MTP内部回放取消。

[原始报告](../results/pd-fairness-v1/probe.json)SHA256为`33e41e3bce3671c28e3c9910468a8859a12df55b24b663e20181b44246b42777`，所有step、callback、提交/到达时钟和终态保留。独立只读复算验证了全部完整IDs/取消前缀、时钟归属、活动阶段计量、224个post-arrival gap及两个配对公式，与探针一致：`complete=true`、`passed=true`、`correctness_passed=true`，但`performance_screen_passed=false`、`outcome=screen_not_passed`。

[控制器记录](../results/pd-fairness-v1/run-ledger.json)显示自有进程PID25577从UTC01:58:23.468127运行至02:00:36.219477（132.751350秒），以0退出、进程组清空；参考PID25852于02:00:51.507063恢复ready，即北京时间10:00:51.507063。本页只分析这次固定本机AR场景，不代表HTTP、持续多请求负载、MTP发布或内存长期稳定验收。结论是保留默认4，已有短请求改善观察值得保留，但本轮未证明代价稳定可接受。

## 复跑入口

冻结的synthetic输入和历史输出已作为小型fixture提交。使用当前模型路径运行：

```sh
ANERUNNER_EXPERIMENTAL_DECODE_ASYNC_LAYERS=0 .build/release/ane-runner probe-gpu-cooperative-scheduler --scenario decode-arrival --model-dir "$MODEL_DIRECTORY" --tokens-file fixtures/gpu-agent-11k/prompt-token-ids.json --frozen-reference fixtures/pd-decode-arrival/frozen-reference.json --output results/pd-decode-arrival-new.json
```

输出路径必须未存在；仍应确保只运行一个GPU模型实验。该命令只执行探针，不会更改服务调度默认值。
