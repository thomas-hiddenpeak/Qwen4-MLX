# SSD 恢复失败的分类边界

2026-09-14。**状态：修复已应用，release 构建及本轮选定的 80 项 CPU 测试全部通过；真实 64K v1 SSD 正常恢复及独立持久化审计通过；32K 末段 profile 与整组 postflight 均通过。** 本页记录已应用改动的范围，不表示设备恢复已经完整解决，也没有进行真实 OOM 或 Metal 故障注入。完整 262K SSD 流式支持仍是[后续计划](KV_STREAMING_SSD_PLAN.md)。

[当前 MLX 包装器](../../Sources/ANERunnerGPU/Tensor.swift)把 C API 失败状态、空数组和本地参数错误都表示为 `GPUError.invalid`。[缓存恢复](../../Sources/ANERunnerGPU/QwenPrefixCache.swift)在捕获导入错误后调用 `MX.synchronize`，但这个等待成功不能证明原错误来自坏归档：固定 MLX 源码的 `Error::check` 清空消息后抛错，Metal 事件引用同一个编码器错误，先前的求值等待可以已经消费它；数组分配失败还可在提交 GPU 工作前同步抛出，不设置该错误。[模型导入](../../Sources/ANERunnerGPU/QwenModel.swift)会丢弃求值失败的私有 State，但旧 SSD 分支随后仍可能把后端错误当成缓存未命中并删除有效文件。这是源码支持的可达路径，尚未观察或注入这类运行故障。

此修复仅收窄“什么错误能证明这份 SSD 归档应失效”：

- [归档描述器](../../Sources/ANERunnerCore/QwenPrefixStateArchiveDescriptor.swift)保留公开 `validate`、布局校验和导出原有的 `.invalid` 语义。仅在 `decodeAndValidate` 中，先按模型计算必需逻辑字节并检查调用方布局、预期 offset、额度与非负实际字节数；这些本地失败不标记归档损坏。
- 上述条件通过后，空/过大 metadata、本次 JSON 解码、文件声明的版本、offset、tensor 布局、PLE 历史及字节守恒错误才包装为 `.invalidArchive`。缓存仅按这个明确类型调用 `invalidateAsync`，不泛认外层所有 `DecodingError`。正常同步恢复后的其他可选失败保留磁盘文件并沿用 fallback/`restoreFailures` 计数；取消和失败 join 仍先按已有规则传播。
- 旧合法归档的受理集合不变：旧验证成功本来就要求合法模型布局/offset、规范逻辑大小和足够的额度，必然通过新增前检。这里检查的是已匹配 checkpoint 的模型预期；调用方预期 offset 越界是本地错误，而文件声明与合法预期不符才是内容错误。对比合法约 1.9GB v1 归档时无需更改格式、默认额度或哈希。

[Core CPU 用例](../../Tests/ANERunnerCoreTests/QwenPrefixStateArchiveDescriptorTests.swift)覆盖小额度拒绝合法归档仍保留、同归档在准确额度下成功，以及 metadata/tensor/历史错误被标记；[错误值用例](../../Tests/ANERunnerGPUTests/GPUPrefixArchiveInvalidationTests.swift)仅构造 GPU、资源、取消、不可用错误，不创建 Tensor 或提交设备工作。它们验证分类策略，不能代替真实设备失败恢复验收。

仍有两条明确边界：一是 [Core disk store](../../Sources/ANERunnerCore/QwenPrefixDiskStore.swift)的 `lookupOnQueue` 仍在底层读取/解码区间的宽泛 catch 中执行 `removeRecord` 并计入 `corruptions`，普通 I/O 或其他非内容失败可能仍被删除；本次没有修复整层 SSD 错误分类。二是原 MLX 执行错误被消费后，后续同步成功时应如何决定模型级可用性，仍未修复。不能把本次“有效归档不因模型导入错误失效”扩大为“所有设备故障都可安全恢复”，也不能据此把所有分配失败统一视为永久设备故障。

## 本轮验证

运行 binary SHA-256 为 `91d626264dbfdbbd40d5c22bc6c4292a9ead873ec0c89ac7a71f2d01e1861117`。04:36 的 release 构建完成；原始 XCTest 日志记录 **80 tests、0 failures**。这是本轮选定的 CPU 回归集合，不是全仓库测试总数：归档描述器 12、磁盘存储 41、磁盘读取 4、cache policy 9、GPU 目标内的纯错误值测试 1，以及 profiler contract/position/general 共 13。新增分类用例和五个位置窗口用例均有独立 passed 记录，没有创建 Tensor 或提交 GPU 工作。

同一 binary 的两个 [profiling 参数](LONG_PREFILL_PROFILE.md) 早期拒绝检查也完成：单独 `--profile-attention true` 在未启用 profiling 时返回 exit 1；`--profile-stages synchronizedStages --profile-from-token -1` 返回 exit 1。两者 stdout 为空，stderr 分别为预期的 profiling 依赖和非负 offset 错误；parser 在模型构造之前拒绝。这两项验证 CLI 边界，不是 SSD 故障注入证据。

可复核的原始文件为 `results/night-final-small-v1/build-and-cpu-tests.log`（SHA-256 `066cf2fa8cb066bbb2248959ef4fa2129d8e2a6b8dd4eaa852895912ba747f2a`）和 `cli-negative.json`（SHA-256 `397f260c81ef6f4b07a5842e6cef14ac439d4f75458e32c98b8d3335a306fd8c`）。它们是本地运行产物，不随源码文档发布。已应用的四份源码/测试与静态审查候选逐文件 hash 一致。

**真实 SSD 正常恢复回归通过。** 同一 48 层模型使用 P65534/O2、B65312，先冷保存，再由空 RAM 的新 generator 从 SSD 恢复。冷/恢复两次输出均为 `[16,11]`，均有 1 次实际 decode、最终 offset 65535。7 个完整 anchor 共 847 个 tensor 观察与旧 reference64 的全部 121 状态及 host 历史精确一致；独立审计共 **14,055 项检查、0 失败**。

审计还以 1 MiB 缓冲顺序读取实际 `.qpc` 文件：文件 **1,915,273,929 B**，tensor payload **1,914,925,056 B**，含 PLE host 历史的逻辑状态 **1,914,925,064 B**。121 个持久化 tensor hash、metadata hash 和整 payload hash 均与旧成功归档相同；整 payload SHA-256 为 `2706d5f34232cc158b3274c6b7faadf0426316d49554a0b2da660b74474eda95`。实际写入为 1,915,273,929 B，读回计费为 1,915,277,312 B，后者按文件占用字节记录，不是宣称额外 tensor 内容。两者均超过 1 GiB，恢复来源为 disk，disk hit 为 1，`corruptions/writeFailures` 为 0，关闭后合法归档仍保留 1 份。

冷/恢复 generator 释放后以及最终的 request、cache、workspace、totalBytes、currentLeases 均为 0；磁盘 pending jobs/bytes 为 0，IO 与 callback close 在既有 30 秒期限内完成。SSD case 于 2026-09-13 20:42:59 UTC 退出 0，没有 TERM/KILL；审计只在控制器记录该 case 完成后开始。当前 binary 与上一轮成功报告另有 **38 项排除计时的严格对照、0 失败**，没有放宽状态或输出容差。

本地独立结果为 `results/night-final-small-v1/independent-ssd-analysis-v1.json`（SHA-256 `583e18eb1f6dbf30befc26c355ce98f15696f90630790dfcc7e379e74c4924e7`）及 `independent-ssd-historical-comparison-v1.json`（SHA-256 `17eb00207e167be29783f4eb226fb93ad3a6d9057d5b445e870cff4ca98bbcc9`）。这证明当前 binary 的正常大归档路径未回退；它没有注入真实 OOM/Metal 故障，也没有证明进程重启、长时间耐久或 262K SSD 支持。诊断 wall time 不用于归因本次错误分类的性能收益。32K profile 的实际结果与解释另记在上述 profiling 文档。

两项 case 和控制器均正常退出0；04:44:27的整组postflight核对577个冻结文件和102个模型payload stat，无变化。参考服务以PID80349按精确原argv恢复，ready/idle且MTP/drafter关闭。该PID仅为这次恢复快照；此检查不代表全部权重内容重新做过SHA校验。
