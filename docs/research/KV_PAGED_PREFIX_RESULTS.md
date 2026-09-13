# 跨请求 KV 物理页复用

2026-09-14。独立 Swift/MLX runner 的显式实验路径；建立在[完整模型页池](VLLM_METAL_ADOPTION.md)上，MTP 保持后置。

## 实际生成流程

RAM cache 仍保存完整 dense 混合状态，并可附带同一检查点的不可变 KV 页。恢复时私有化 dense 状态，继续原来的 chunk416/eval4 prefill；首次 decode 只把检查点之后的新行追加到已有页。之后单 token decode 直接读写物理页表。GDN/QSA/PLE 仍属于各请求的私有状态，不能只凭 KV offset 恢复其他状态。

附件绑定模型/context、执行配置、完整 namespace 和 token 前缀摘要。新的完整检查点可从旧附件追加后缀；不同 context 的 completed prefill 交接会忽略旧附件，使用完整 dense 状态导入目标 pool。未完成游标仍绑定原 generator。

本阶段保留 dense cache 和 dense prefill 恢复/计算成本，也增加了可选 paged 附件。它尚未移除缓存中的重复 K/V 表示，不能称为完整分页 prefill。HTTP 接入和长上下文验证另行记录。

## 准入与生命周期

每层 arena 的实际分配额度由 native owner 持有，独立于 cache entry。附件的页表和 wrapper 另申请 cache 元数据额度；clear/evict 移除条目时，暂停请求仍可持有附件与额度。

首次 decode 为整个请求申请常量物理页 claim：`ceil((prompt + maxOutput - 1)/32) - floor(reusedPrefix/32) + 1`。最后一项为不可变尾页 COW 的同时存活空间。其他请求和可选 cache 附件都必须在当前真实空闲槽之外扣除已有 claims；已经变成实际页的额度仍保守计入原 claim，因此可能较早回退，不承诺最佳容量利用率。

拿不到 claim 时，该游标全程保留 dense，不在后续 token 重试。paged 步骤在任何层改变前复核空闲尾页，必要时同步一次；仍不足则有界失败。设备失败单独将模型标为不可用。正常完成、显式 discard 和异常清理都在同步及释放请求状态后归还 claim；预算先于 GPU 清理归还的两处旧错误路径也已修正。

实验 context 仅供遵守准入的 generator/cache 使用。手工 model/pool 调用不能混入同一 context，observer 可以同步取 hash/字节/page IDs，不能保留每个 token 的旧 State 或页别名；额外逻辑字节额度不等于额外物理页预留。设备故障尚未经过硬件注入验收。

## 完整模型验证

`results/paged-prefix-model-v1/model.json` 的 all/O8 测试使用真实 P11057、B10816、chunk416/eval4。基础输入和仅改变共享前缀之后一个普通 token 的输入，各自执行独立冷 oracle。覆盖冷填充、热命中、不同后缀、双游标交错、清缓存后继续、取消后命中、347 页竞争下的 sticky dense fallback，以及不同 context 的交接。

133 组完整状态、16,093 个 BF16 张量记录及 host 状态按明确 oracle 配对一致；114 次 token 状态观察、14 次完整生成和一次取消均通过。独立分析核对 148,898 条条件，未发现失败。所有 context/cache/request 释放后总预算、lease、物理活动页和 claims 归零。新版六项 Attention GPU 边界测试以及 142 项相关 CPU 回归通过。

| 观察项 | 结果 |
| --- | ---: |
| 热命中 adopted prefix / 实际 prefill | 10,816 / 241 tokens |
| 每层保留的完整共享前缀 | 338 页 |
| 十二层首次 decode 导入后缀的 encoded row payload | 5,922,816 B |
| 若按本次完整 prompt 导入的相同 payload 计算项 | 271,736,832 B |
| 两个实际暂停 reader 的逻辑 K/V payload 合计 | 543,522,816 B |
| 这两个 reader 的 page-ID union 对应物理页字节 | 278,396,928 B |
| 同时存在的固定 arena 实际分配 | 402,653,184 B |
| clear 后仍被两个请求持有的附件元数据额度 | 1,833,273 B |
| 当时仍持有的未来页 claims | 每层 18 页，共两个请求 |

业务写入/读页计数在显式诊断导出之前取样；普通 decode 没有隐式全量 materialize。上述 encoded payload、活动页、固定分配、逻辑预算与 MLX allocator 指标分别记录，不能冒充物理 DRAM 流量或 RSS 节省。页 union 是实际页号去重，逻辑 K/V 合计不是另一个 dense 模式的分配实测。

状态探针包含同步导出和哈希，其耗时不用于吞吐结论。无 observer 的 warm ABBA/BAAB 单独运行，冷填充独立记录，八次 O128 完整输出均与既有参考 128 IDs 一致。prefill 的执行速率分子是实际 241 tokens，首输出属于 prefill；分页导入、准入、每步检查与结束清理包含在 decode 时间中。

| 顺序 | stock capacity256 decode tok/s | paged32 decode tok/s | 观察变化 |
| --- | ---: | ---: | ---: |
| ABBA | 15.38 | 15.59 | +1.37% |
| BAAB | 17.05 | 16.53 | −3.04% |

每种顺序各有两样本/模式，方向反转，未建立稳定吞吐收益。分页后缀导入为 3.85–5.44 ms；warm TTFT 为 stock 1.146–1.439 s、paged 1.217–2.382 s，亦不构成延迟改善证据。两种模式都保留同一个固定 arena，并使用各自独立 RAM cache；此计时不测 RSS 节省，不改变默认。

完整 probe+benchmark 的独立分析共核对 151,845 条条件通过。固定 runner SHA 为 `44b897dde8b78502f1ebd255ca168750969f6420811e89d74b5f4b4fd4768bd4`；559 个冻结文件与 102 个模型 payload stat 的 postflight 全部一致，参考业务服务按原 argv 恢复为 idle、MTP/drafter 关闭。原始输出、独立审计和恢复记录位于 `results/paged-prefix-model-v1/`。

## 复跑

通过[唯一 GPU 控制器](../EXPERIMENT_CONTROLLER.md)在新输出目录运行：

```sh
.build/release/ane-runner probe-gpu-paged-prefix-cache \
  --library ABSOLUTE_POOL_DYLIB --model-dir MODEL_DIRECTORY \
  --tokens-file fixtures/gpu-agent-11k/prompt-token-ids.json \
  --suite all --max-tokens 8 --output NEW_MODEL_REPORT.json

.build/release/ane-runner benchmark-gpu-paged-prefix-cache \
  --library ABSOLUTE_POOL_DYLIB --model-dir MODEL_DIRECTORY \
  --tokens-file fixtures/gpu-agent-11k/prompt-token-ids.json \
  --order both --output NEW_PERFORMANCE_REPORT.json
```

本测试没有验证新路径的 SSD promotion、进程重启、HTTP 耐久、真实系统压力或完整 262144 上下文；既有业务参考服务和旧路径的证据不能替代这些检查。默认路径保持不变。
