# AR 完整前缀缓存与服务验收

2026-09-08：完整混合状态快照、请求私有恢复、压缩 radix 最长前缀索引、LRU 和容量淘汰已实现。HTTP 工具调用与结果回传也已实测，见 [工具协议](HTTP_TOOL_CALLING.md)。SSD 状态卸载、跨进程 PD、MTP 缓存与 MTP 性能优化留在后续。

## 使用范围

`serve-gpu` 默认启用最多 512 MiB、8 条快照的缓存：

```sh
.build/release/ane-runner serve-gpu --model-dir /absolute/path/to/model \
  --prefix-cache-bytes 536870912 --prefix-cache-entries 8
```

`--prefix-cache-bytes 0` 关闭缓存。HTTP 客户端无需提供 cache key：服务将 system 与 tools 渲染为候选，与完整 chat 编码逐 token 求共同前缀，再向下取整到原有 416-token 分块边界。不足一个完整块时继续冷计算。不从单独编码的 suffix 拼装提示词，也不自动保存用户消息的任意中间状态。

库的 `QwenGenerator(model:prefixCacheLimits:)` 默认仍不缓存；调用方显式提供限额和 `QwenGenerationRequest(prefixCacheMaxTokens:)` 才复用。hint 必须小于完整提示词长度，nil/0 禁用；`clearPrefixCache(resetStatistics:)` 在同一推理执行器上清理。缓存不是 Sendable，每个 generator 独立持有，不能跨模型实例或进程共享。

HTTP 保留完整 `usage.prompt_tokens`，命中时新增 `usage.prompt_tokens_details.cached_tokens`。`/health` 的 `prefix_cache` 提供 hits、misses、published、evictions、entries、logicalPayloadBytes、keyTokens、skippedOversize、restoreFailures。库的 prefill 阶段另有实际计算 token 数、lookup/restore/save 时间；吞吐分子使用实际计算 token，恢复和保存计入 prefill 总时间与 TTFT，decode 单独统计。

## 状态、边界和容量

每个快照包含 36 层 GDN recurrent/conv、12 层 Attention K/V 与 QSA 原始/池化 indexer 状态、PLE convolution 与 UInt32 n-gram 历史、各层及全局 offset。保存、恢复都用 gather 建立独立紧凑存储并完成求值；恢复从 `makeState()` 获得新 session identity，随后仅为后缀建立请求私有 SSD 预取。

缓存只发布真实执行完成的整块边界；GDN 状态不能倒切到 radix 内部分叉点。命中较短条目时沿原 chunk 栅格继续，可在更长的合法边界再发布。始终保留最后一个提示词 token 的计算。数值 namespace 区分 chunk、evaluation interval、attention、fused-prefill、accumulation 和完整 MoE 配置，最终以完整 Int32 token key 匹配。

快照先求值、等待当前 SSD lookahead 并检查取消，再原子插入；发布前取消不留下条目。共享快照不携带请求游标、回调、待输出 token 或预取任务。淘汰/clear 不破坏已恢复的私有请求；模型设备恢复失败会清空该模型关联缓存并沿用不可用状态。

容量同时限制条目数、紧凑张量及 PLE 历史的逻辑字节、完整 token key 总长度。超大单条在复制前拒绝，普通压力按全局 LRU 淘汰。这是缓存持有负载的额度，**不是进程物理内存硬上限**；请求私有副本、短暂复制峰值、权重和 MLX allocator pool 另计。9984-token 快照实测逻辑负载为 342,724,616 B（326.848 MiB），默认 512 MiB 通常只容纳一条这种长前缀。

MTP 请求整体绕过 AR 缓存，继续原有 MTP 冷 prefill/decode，不静默降级成 AR。开启详细 stage profiler 时也绕过缓存。现有 MTP 正确性仍回归，但其性能不作为本轮依赖。

## 本次验证

测试二进制 SHA-256：`2941a0ddb3bb639e7f68ddfdb51c7fc9b6f0056724bcf1d94d1f43e8fe5a9a54`。133 项 Swift CPU、6 项 Python 报告解析测试通过。实模和 CPU 证据分开计量：

| 实模套件 | 结果 |
| --- | --- |
| 完整状态与生命周期 | 23 个完整请求、779 个生成 IDs；非 oracle 输出分别精确匹配对应冷请求；原 11k A 仍匹配既有 AR golden |
| 原始状态检查 | 30 组共 3342 项 shape/dtype/bytes/hash/finite 检查及 PLE host 状态通过；其中 5 组为建立 anchor 时的自身读回，实际跨状态比较为 25 组、2785 项 |
| 边界及所有权 | 2051/2053 QSA 边界；416→832 最长命中；单条淘汰、1 B 超限拒绝、配置/不同前缀 miss、恢复后推进并取消、插入前取消及随后正常恢复 |
| MTP 隔离 | 833-token 提示词，显式 depth2 冷计算，8 个含 EOS 的完整 IDs 与 AR 一致；不作为 MTP 性能结论 |
| 无状态回读计时 | 另一个进程的 5 请求、640 IDs 通过；与诊断合计 28 请求、1419 IDs |
| 新 HTTP 功能 | 40 项检查、9 次推理：非流式/SSE 工具闭环、auto/none、非法组合 400、10k+ 系统前缀复用/隔离 |
| 既有 HTTP 回归 | 19 项通过：长 AR/MTP 输出、中文、并发、429、half-close、RST 取消、暂停读取、后续恢复和关闭 |

HTTP 回归首轮在服务启动前遇到端口占用，未运行任何检查；保留失败报告，改用已确认空闲端口后重跑通过。三个运行阶段都核对冻结源码/二进制和 102 项模型负载 stat，按 controller finally 恢复参考服务。没有把旧版本的完整 soak/背压测试重复计入本轮。

## 本次延迟观察

以下是无张量回读的 11,057-token 提示词、128 输出预算测试；两个后缀共享 9984-token 前缀，命中后只计算 1073 token。

| 请求 | TTFT | 缓存处理 |
| --- | ---: | --- |
| A 冷运行 | 25.094 s | 无缓存 |
| B 冷运行 | 20.981 s | 无缓存 |
| A 首次保存 | 22.325 s | 保存 8.853 ms |
| B 命中 | 2.636 s | 恢复 17.832 ms |
| A 再命中 | 2.623 s | 恢复 17.403 ms |

这是减少重复 prefill 的收益，没有提高模型每 token 的基础 kernel 吞吐。该窗口 decode 从冷 A 的 26.64 到命中 A 的 23.90 token/s，存在顺序漂移；不能据此宣称稳定 decode 加速或无回退。实验 HTTP 默认启用缓存的依据是完整状态/生命周期验收和减少重复前缀计算，仍不代表生产稳定性验收完成。

另一次真实 HTTP 测试的完整提示词为 11,054 token，复用 10,816 token；32 输出预算实际生成 4 token。冷/命中端到端墙钟为 12.631/0.525 s，改变 user 后仍命中（0.533 s），改变 system 开头则 miss。HTTP 墙钟包含 tokenizer、调度及传输，与上表 TTFT 不是同一统计口径。

## 复现

在模型资源独占且参考服务已由受控流程暂停时运行：

```sh
.build/release/ane-runner probe-gpu-prefix-cache --model-dir /absolute/path/to/model \
  --tokens-file fixtures/gpu-agent-11k/prompt-token-ids.json \
  --suite all --max-tokens 128 --state-readback true --output NEW_STATE.json
python3 scripts/validate_prefix_cache_report.py NEW_STATE.json

.build/release/ane-runner probe-gpu-prefix-cache --model-dir /absolute/path/to/model \
  --tokens-file fixtures/gpu-agent-11k/prompt-token-ids.json \
  --suite long --max-tokens 128 --state-readback false --output NEW_TIMING.json
```

服务 ready 且 idle 后：

```sh
python3 scripts/probe_http_tools_prefix.py --base-url http://127.0.0.1:11236 \
  --suite all --output NEW_HTTP.json
```

原始本机结果：`results/ar-prefix-cache-v1/`；最新恢复 ledger 为 `http-regression/run-ledger.json`，恢复 PID 10167（只代表本次快照，接续前重新核对）。模型权重、构建和大结果不入 Git。前缀索引已单独提交 `e53480e`，运行时、工具协议与本次结论随主线提交。
