# CoreAI / MLX 完整模型混合生成

此页保留混合阶段结果。后续已完成独立、无 MLX 链接的[完整 CoreAI 计算路径](COREAI_NATIVE.md)，新的生成与迁移边界以该文档为准；本页结果仍仅适用于当时的混合实现。

目标先从真实文本输入走通全部 48 层并连续生成，再逐步迁移和优化。`generate-coreai-hybrid` 是独立 Swift 诊断入口，运行期间不启动 Python 或原作者 mlx-serve。

## 2026-09-17 实际结果

macOS 27.0 / Swift 6.4，M5 Max，GPU 偏好。以下均由 release Swift 可执行文件在真实完整权重上执行，两个请求各运行两遍，第二遍重置全部 CoreAI / PLE 状态。

| 请求 | Prompt tokens | 输出 tokens（含 EOS） | 每遍 CoreAI 调用 | 重置重放 | 独立 MLX top-1 对照 |
| --- | ---: | ---: | ---: | --- | --- |
| 用一句中文介绍自己 | 20 | 18 | 1776 | 输出 IDs 完全相同 | 18/18 相同 |
| 只回答数字：17乘以3等于多少？ | 23 | 3 | 1200 | 输出 IDs 完全相同 | 3/3 相同 |

实际文本分别为“我是通义千问，由阿里巴巴通义实验室独立开发的大语言模型。”和“51”，均正常 EOS。**四次完整生成合计执行 5952 次 CoreAI attention 函数，所有层实际调用次数与状态 offset 一致。** 独立 MLX 对照使用相同逐 token prefill；没有把既有 chunk416 的输出直接当作逐位 oracle。

全词表原始 logits 仍存在差异：自我介绍 relative L2 为 0.05724–0.14467，算术为 0.06119–0.11603；最大绝对差分别 1.78125、1.28467。这远非源模型数值等价验收，仅说明这两条短请求的 greedy 输出相同。源 BF16 与当前 FP16/FP32 边界、递归状态精度和路由累计影响仍需单独诊断；没有放宽此前子层 runtime 门槛，也没有据此宣称完整质量通过。

本轮分阶段耗时仅记录功能版成本：第二遍自我介绍 prefill 2.757 秒（20 tokens），decode forward 1.650 秒（17 步）；算术 prefill 3.131 秒（23 tokens），decode forward 0.190 秒（2 步）。第一次 prefill 分别 10.753 / 11.819 秒，包含首次实际执行的冷启动成本。此入口逐层 host 桥接、逐 token prefill，**尚非性能对比或优化结果**。

首轮试跑暴露并修复了 CoreAI await 后线程切换导致的 MLX stream 错误。固定线程执行器经独立 CPU 实测，633 次线程观测、930 次资源访问及构造/析构均留在同一线程；随后上述四次实际生成通过。部分层 manifest 也在加载模型前按预期拒绝。Release 构建、11 项已有导出器 CPU 回归通过；本轮未运行 XCTest 或服务长期测试。

本机证据：`results/coreai-hybrid-generation.json`、`results/coreai-hybrid-arithmetic.json`、`results/coreai-hybrid-executor/report.json`、`results/coreai-hybrid-partial-rejection.log`。大型资产与本机结果不随 Git 克隆提供。

## 执行范围

| 部分 | 当前执行路径 |
| --- | --- |
| 36 层 GDN、12 层 Attention/QSA | macOS 27 系统 CoreAI，GPU 偏好 |
| 原始 Q4 路由专家、共享专家 | 既有独立 runner 的 MLX 实现 |
| Hyper Connection、残差、分词、embedding、输出头 | 既有 Swift / MLX 实现 |
| 51.2 GB n-gram PLE 表 | SSD 按需读取；PLE 投影/卷积使用 MLX |
| attention 状态 | 原生 CoreAI NDArray；GDN FP32 recurrent、FP16 conv，QSA FP16 cache |

这条路径会执行真实的完整模型，但**不是纯 CoreAI 后端**。无需把全部 Q4 专家展开为 FP16。资产只额外导出每层 attention 权重，36 个 GDN 与 12 个 QSA 约 5.43 GB；不在 MLX 域重复加载这些 attention 权重。

第一版每次处理 1 token，prefill 也逐 token 执行。每层 attention 输入、输出经过 host Float / FP16 / BF16 桥接；KV 和 GDN 状态保留在 CoreAI NDArray 中，不逐步读回全部状态。GPU 是 CoreAI 的调度偏好，未用硬件 trace 验证具体驻留。

本次默认资产容量 **256 tokens**，prompt 加输出预算必须在容量内。该限制仅属于新入口，不改变现有 MLX 服务容量。更长上下文、分块 prefill、服务调度、prefix cache 和 SSD 状态归档均尚未接入这条混合路径。

## 导出和执行

依赖及模型来源同 [CoreAI 后端](COREAI_BACKEND.md)。从独立 runner 仓库执行；需要 macOS 27、SDK 27 与兼容的外部 MLX C 库。

```sh
../../.venv/bin/python scripts/export_coreai_hybrid.py \
  --capacity 256 --output results/coreai-hybrid-full
xcrun swift build -c release

.build/release/ane-runner generate-coreai-hybrid \
  --model-dir ../qwen38-ssd/models/Qwen3.8-Flash-Next-MLX-SSD-Stream \
  --manifest results/coreai-hybrid-full/manifest.json \
  --prompt "请用一句简短的中文介绍你自己。" \
  --max-tokens 48 --repeat 2 --compare-reference true \
  --output results/coreai-hybrid-generation.json
```

导出要求新目录，按 config 的 layer_types 逐层读取真实权重，保存来源/资产 SHA256。`--layers 0,3` 仅用于局部 smoke，默认完整生成会拒绝部分 manifest。每层初始状态只记录零值、shape 和 dtype；不保存巨大 JSON 张量。

生成使用现有 no-thinking 模板和 greedy 选择，保留特殊输出 token 抑制。`--raw-prompt true` 可跳过聊天模板。`--repeat 2` 在同一已加载模型上重置两域状态并重放，要求输出 token IDs 一致。

`--compare-reference true` 先释放混合模型，再单独加载独立 MLX 基线，使用同样的逐 token prefill 和混合路径生成的 token 做 teacher forcing，对每步 logits、top-1 进行比较。它诊断 FP16/FP32 路径相对 BF16 的累计差异；短文本可读、top-1 相同或 reset 重放一致，都不能单独证明整体质量已验收。

报告分开记录 prefill 和 decode 时长、CoreAI 函数执行时间、实际调用次数、状态 offset、输出 token IDs。首个输出使用 prefill 的最后一行 logits，decode forward 次数因此为输出 token 数减一。函数计时不等于物理 GPU 时间；MLX 内存计数不包含 CoreAI 总内存。

## 状态边界

完整入口核对模型路径、config SHA256、48 个层类型、初始 offset 和每步实际 48 次 CoreAI 调用。中途失败将会话标记失效；必须 reset 后重新预填，不能继续使用已经部分推进的状态。调用期间拒绝重入和 reset。

两域状态不实现现有 `QwenModel.State` 协议，不读写旧 MLX prefix / SSD archives。未来迁移缓存时需要独立的后端、dtype 和布局版本，不能把两条路径的状态当作可互换数据。

MLX 的执行流绑定原生线程，普通 Swift async 函数在 CoreAI 调用返回后可能切换线程。完整命令将模型加载、MLX 运算、CoreAI await 后续算、reset 和基线对照放在专用原生线程执行器上。库调用者也须遵守该约束；模型在每次恢复后检查线程，不能仅靠串行 DispatchQueue 或单请求来保证。

后续的 MoE/HC/PLE 与输出头迁移已进入[完整 CoreAI 入口](COREAI_NATIVE.md)。源模型累计精度、性能与服务状态接入继续独立验收，MTP 保持后置。
