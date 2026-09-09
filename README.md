# Qwen4-MLX

面向 Apple Silicon 的独立 Swift / MLX 推理工程，当前适配 **Qwen3.8 Flash-Next 的完整48层文本模型**，主要在 M5 Max 上验证。原始 Q4 专家与 BF16 主干常驻统一内存，51.2B n-gram 表通过 SSD 按需读取。MLX C / Metal 执行计算，Swift 管理分词、状态、读取、调度与服务；推理不需要 Python 进程或作者 HTTP 服务。

GitHub 默认分支为 `codex/runner-baseline`。实验分支的阶段成果经验证后及时纳入该分支；[主线整合记录](docs/MAINLINE_INTEGRATION.md)区分可用能力、显式候选和默认行为。

当前集中完善 KV cache。[关键能力计划](docs/KV_CACHE_CAPABILITIES.md)依据 vLLM、SGLang、LMCache、DwarfStar 与 MLX LM 的固定源码快照，安排完整会话复用、真实内存压力控制、SSD 调度及有效收益指标，再推进物理页共享与增量存储。现有联合状态额度、同前缀请求合并、可选持久化 SSD 和已验证范围见[缓存可靠性](docs/KV_CACHE_RELIABILITY.md)；各项实现和验收进度在能力计划中分别记录。**MTP 性能优化放到计划后段**，已有显式 MTP 的正确性和状态隔离要求不变。

## 构建与生成

需要兼容的外部 MLX / MLX C 原生库。现有工作区默认位置为 `../qwen38-ssd/runtime/mlx-serve/lib/mlx`，也可设置 `ANERUNNER_MLX_ROOT`；版本、库布局和完整用法见 [GPU runner](GPU_RUNNER.md)。Package 最低目标为 macOS 26.2，已有 macOS 26.6.2 / Swift 6.3.3 构建记录，当前 MLX 路径不要求 macOS 27。

从本仓库目录执行，模型目录按实际安装位置替换：

```sh
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun swift build -c release

.build/release/ane-runner generate-gpu \
  --model-dir ../qwen38-ssd/models/Qwen3.8-Flash-Next-MLX-SSD-Stream \
  --prompt "用两句话介绍这个模型。" \
  --context 16384 --max-tokens 128 \
  --output results/first-generation.json
```

已提供 [11k agent 输入](fixtures/gpu-agent-11k/provenance.json)，可用 `--tokens-file fixtures/gpu-agent-11k/prompt-token-ids.json` 替换 `--prompt`，保持 `--context 16384`。生成报告保存完整输出 token IDs 及分阶段成本。运行整模型实验时应安排独占模型资源；[实验控制器](docs/EXPERIMENT_CONTROLLER.md)只清理其拥有的进程并恢复已登记的参考服务。

## 当前默认与能力状态

| 项目 | 当前行为 |
| --- | --- |
| 生成 | AR、greedy；MTP关闭，decode为 `reference`，保留原量化格式和BF16参考累加 |
| Prefill | chunk416、每4层求值、SSD跨块预取 `nextChunk`、1个SSD worker |
| Attention | `reference` 策略含符合条件的融合 causal attention；`ANERUNNER_FUSED_PREFILL=0` 可关闭它；QSA专用融合另行显式选择 |
| 调度 | prefill / decode独立接口与单次状态交接；库默认 `wholeStages`，HTTP使用 `cooperative`，decodeBurst保持4；同一推理执行器串行计算 |
| 有界资源 | HTTP连接、排队、输出和日志有额度；request/cache/workspace 联合状态预留、取消及终态清理已实现；接入macOS压力通知与恢复滞回，逻辑额度不等于物理内存硬上限 |
| 默认关闭的候选 | MTP、expert32/down、融合归约、QSA prefill融合、blocked GDN、async8与额外decode投影融合；各自按配置选择，未因主线整合改成默认 |
| 前缀缓存 | HTTP默认512 MiB / 8条完整混合状态快照，压缩前缀树与LRU；自动复用完整会话/工具历史，有限保留共享系统与会话尾部检查点；库默认关闭，显式MTP保持冷prefill |
| SSD 状态缓存 | 显式开启的有界持久化层，完整混合状态归档、异步恢复/写入、校验及重启恢复、可用空间水位、请求等待期限及有界关闭等待；与 n-gram SSD 读取分别管理 |
| 工具调用 | function tools、auto/none、非流式/SSE调用及工具结果续答；客户端执行工具 |
| 尚未实现 | 物理 KV 页共享、跨进程PD、跨请求GPU连续批处理、强制/严格约束工具解码 |

chunk416改变过跨块状态舍入边界；固定输入的输出回归不代表与作者任意输入全部逐位等价。初期短输入、后续11k与不同候选的验证范围分别保留在各实验文档中。

## 本机文本 HTTP / SSE

```sh
.build/release/ane-runner serve-gpu \
  --model-dir ../qwen38-ssd/models/Qwen3.8-Flash-Next-MLX-SSD-Stream \
  --port 11236
```

仅监听 `127.0.0.1`。`GET /health` 提供服务及资源状态，`GET /v1/models` 返回实际模型ID，`POST /v1/chat/completions` 支持流式或非流式文字与工具调用。网络队列与固定推理线程分离；各版本分别实测取消恢复、非流式输出超限、真实AR SSE背压，以及日志管道堵塞时的服务活性。

这是**实验服务及有限API子集**：支持字符串内容的 system / user / assistant / tool、function tools、`tool_choice: auto|none`，使用 no-thinking 模板；temperature只能省略或为0，未知字段会被拒绝。工具调用已完成真实非流式/SSE及结果续答验证，细节见[工具协议](docs/HTTP_TOOL_CALLING.md)；required/指定函数、strict=true、多模态与随机采样仍未支持。上下文固定16384；AR输出预算1…4096，工具请求使用AR。显式纯文本 `mtp_depth: 2` 使用 `batchedScalarLinear` / tail1024，输出预算仅1…256。

完整请求经模板渲染后一次分词，系统、工具定义与 user/assistant/tool 历史均参与[准确前缀复用](docs/research/KV_CONVERSATION_VALIDATION.md)。检查点沿用416-token计算网格，每个请求最多发布系统与尾部两个检查点；编辑历史或分叉只恢复实际一致的完整状态。默认512 MiB放不下两份长状态时保留共享系统锚点，尾部可写入显式开启的SSD层。命中返回 `usage.prompt_tokens_details.cached_tokens`；`/health`提供详细统计，`GET /metrics`提供无请求标签的Prometheus文本指标。

可用 `--prefix-cache-bytes 0` 关闭缓存，`--prefix-cache-directory` 启用持久化SSD，`--state-budget-bytes` 配置request/cache/workspace联合逻辑额度，`--prefix-cache-shutdown-timeout-seconds` 设置SSD关闭等待期限（默认30秒）。该期限约束SSD队列与回调排空，不能保证挂起的GPU或系统调用立即终止。实际范围和验证见[使用合同](docs/KV_CACHE_RELIABILITY.md)。缓存收益来自减少重复prefill；不代表基础decode吞吐提升。

启动、诊断、停服与重启见[KV cache运维](docs/KV_CACHE_OPERATIONS.md)；持续SSD读写淘汰、取消与资源归还的公开复跑入口见[HTTP cache churn](docs/HTTP_CACHE_CHURN_REPRODUCIBILITY.md)。

显式容量策略`--kv-append-mode capacity256`仅用于普通AR decode，默认`reference`。完整模型四组对照观察到约3.9%–7.6%的decode增幅，输出一致；prefill没有可信收益，HTTP持续负载另行验收。实现范围、样本和漂移见[KV容量追加](docs/research/KV_CAPACITY_MODEL_RESULTS.md)。

各轮验证版本、请求示例、限额与剩余边界见 [HTTP/SSE服务](docs/HTTP_SERVER_EXPERIMENT.md)和[输出边界](docs/HTTP_OUTPUT_BOUNDARIES.md)。发送期限、连接期限及长时间稳定性仍有未覆盖范围，不把有限回归表述为生产验收完成。

## 阶段接口与性能依据

[Prefill / Decode接口](docs/PREFILL_DECODE_SEPARATION.md)支持完整阶段与 chunk / round 增量推进。调度交错能够改善请求等待，但不等于同时执行GPU kernel，也不等于复用不同请求的权重读取。

- [采样和命令时间轴](TELEMETRY.md)：prefill、decode、载入分开统计；可追加 `--telemetry-dir NEW_PATH`。物理DRAM字节和带宽不可测时保持 `null`。
- [MTP成本摘要](docs/MTP_COST_SUMMARY.md)：使用实际发布token计算有效decode吞吐，分开起草、验证、提交/回放、历史更新及target-only收尾。
- [调度延迟](docs/SCHEDULER_LATENCY_EXPERIMENT.md)：保存用户可见callback间隔 p50/p95/max、TTFT及请求耗时，不把compute时间等同于墙钟等待。
- [验证阶段热点](docs/MTP_VERIFY_HOTSPOTS.md)与[真实专家路由](docs/MTP_EXPERT_OVERLAP.md)：显式诊断可能改变调度和张量寿命；逻辑权重复用上限不是实测DRAM节省。

MTP的[两窗口回归](docs/MTP_RELEASE_WINDOWS.md)保留完整输出检查及漂移未定组，稳定性能发布门槛尚未通过。局部正确但未建立收益的 [QKV TM2](docs/VERIFICATION_QKV_TM2_EXPERIMENT.md)、[共享分支融合](docs/MTP_SHARED_ELEMENTWISE_EXPERIMENT.md)、[共享gate/up](docs/MTP_GROUPED_GATEUP_EXPERIMENT.md)仅保留显式算子实验，不接入默认生成。

## 可选 expert32 / grouped-down

[按专家分组的prefill MoE](docs/MOE_PREFILL_EXPERT.md)已接普通生成入口，只影响符合条件的prefill块；[叠加归约](docs/MOE_PREFILL_COMPOSITION.md)未建立稳定的额外整模型收益。以下显式预设选择 expert32 gate/up和grouped down，保持原归约，**不执行autotune，也不宣称当前设备已获性能验收**。

[主线整合后复测](docs/PREFILL_MAINLINE_RECHECK.md)的 13 个完整请求全部匹配参考。两个性能窗口分别观察到 +10.96% / +44.87% prefill 吞吐变化，第一窗基线漂移 27.03%，因此保持显式选择；decode 单独统计，完整样本随文档保存。

先构建与本机固定MLX匹配的插件，再生成绑定当前模型路径和库哈希的配置。所有输出路径选用新的位置，不依赖作者本机历史 `results/` 文件：

```sh
python3 scripts/build_mlx_moe_gateup.py \
  --runtime ../qwen38-ssd/runtime/mlx-serve \
  --output results/local-moe-native

python3 scripts/configure_prefill_moe.py \
  --model-dir ../qwen38-ssd/models/Qwen3.8-Flash-Next-MLX-SSD-Stream \
  --build-manifest results/local-moe-native/build-provenance.json \
  --preset expert32-grouped-down \
  --output results/local-expert32.json

ANERUNNER_GATEUP_LIBRARY="$PWD/results/local-moe-native/lib/libanemlx_moe_gateup.dylib" \
  .build/release/ane-runner generate-gpu \
  --model-dir ../qwen38-ssd/models/Qwen3.8-Flash-Next-MLX-SSD-Stream \
  --tokens-file fixtures/gpu-agent-11k/prompt-token-ids.json \
  --context 16384 --max-tokens 128 --mtp-depth 0 \
  --prefill-accumulation reference \
  --prefill-moe-config results/local-expert32.json \
  --output results/local-expert32-generation.json
```

插件构建还需要该MLX的构建树与编译记录，只有动态库不足以重建。配置脚本生成version1、`gateUpVariant=2`、`groupedDown=true`、空归约表，并记录显式预设状态；不会自动启用运行时环境变量。205…512-token块使用该路径，其他长度回到reference，decode与verification不使用它。普通运行省略配置即保持默认。

## 文档、历史与来源

- [KV cache关键能力](docs/KV_CACHE_CAPABILITIES.md)、[主线吸收计划](docs/UPSTREAM_ADOPTION_PLAN.md)、[缓存可靠性验收](docs/KV_CACHE_RELIABILITY.md)、[MTP发布条件](docs/MTP_RELEASE_CRITERIA.md)：区分当前计划、实际完成与发布门槛。早期[精确前缀设计](docs/EXACT_PREFIX_CHECKPOINT_DESIGN.md)及[首轮缓存验收](docs/AR_PREFIX_CACHE.md)保留追溯。
- [Core ML / ANE历史实验](docs/COREML_ANE_HISTORY.md)：保留早期局部数值、硬件证据、负结果与命令；完整MLX生成当前不使用ANE，也未实现CoreAI后端。
- [上游许可](UPSTREAM-LICENSE)与[garnermccloud/mlx-serve固定源码](https://github.com/garnermccloud/mlx-serve/blob/7dbcba04c98e4fd3bcc533c63e645547f13cc3b1/src/qwen4_exp.zig)：复用与移植文件保留来源和许可；vLLM、SGLang、DwarfStar的借鉴范围见吸收计划。

本仓库提交源码、测试、脚本、文档与文本fixture。模型权重、大型张量、`results/`和构建产物不随克隆提供，文档中的历史本地结果链接需要对应实验产物。Swift推理不启动Python；Python用于插件构建、离线转换和验证。
