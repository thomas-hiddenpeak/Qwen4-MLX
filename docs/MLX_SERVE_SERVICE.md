# mlx-serve 本机业务服务

2026-09-10。此处是原作者 mlx-serve 的常驻本机服务，独立 Swift/MLX runner 继续独立开发。原生模型配置 `max_position_embeddings=262144`，对应 256 Ki tokens；不是 265000 tokens。上下文是输入与输出的总预算。

## 固定配置

实际参数来自 [`config/mlx-serve-business.json`](../config/mlx-serve-business.json)：

| 参数 | 值与用途 |
| --- | --- |
| 地址 | `http://127.0.0.1:11235/v1`，仅本机 |
| 模型 | `Qwen3.8-Flash-Next-MLX-SSD-Stream`，4-bit，外置 SSD n-gram |
| 上下文 | `--ctx-size 262144` |
| Prefill | `--prefill-chunk 512` |
| 热前缀缓存 | `--prefix-cache-entries 8 --prefix-cache-mem 10GB`（本实现按 GiB 解析） |
| 混合检查点 | `--ssm-checkpoint-stride 8192 --ssm-checkpoint-max 2` |
| SSD 前缀缓存 | `--prefix-cache-disk 0`，本轮不启用 |
| 并发 / 无输出超时 | `--max-concurrent 1 --timeout 1800` |
| 投机与观测 | `--no-mtp --no-drafter --no-pld --metrics` |

`/v1/chat/completions` 支持流式输出与 `usage.prompt_tokens_details.cached_tokens`。`/metrics.json` 同时记录实际 prefill token、缓存 token 和分阶段耗时。当前 `/v1/completions` 的流式 usage 未提供缓存明细，验证时使用独占请求期间的指标差值，不伪造请求字段。

模型含 36 层 GDN、12 层 Attention/QSA 和 PLE；缓存必须连同匹配位置的混合检查点一起恢复。满上下文仅 BF16 K/V 约 6 GiB，每个近尾部的混合检查点约 1.04 GiB，故不沿用默认 2 GiB 缓存和最多 32 个检查点。当前限制优先支持顺序追加与重复请求；较早位置的分叉可能找不到保留的检查点而需要重算。缓存配额不是整个进程的物理内存上限。

## 启动与后续实验

检查当前状态与监听后再操作；启动器拒绝在已有监听时启动第二份模型：

```sh
python3 -B scripts/serve_mlx_business.py --print-argv
python3 -B scripts/serve_mlx_business.py
```

启动器在前台 exec 服务，本身不安装登录启动项或重启守护。受控切换后的当前 PID、日志及 `reference_ledger` 写入 `../qwen38-ssd/results/experiment-status.json`。实验控制器每次恢复更新最新 ledger，并保留业务服务的精确 argv；不恢复历史 4096-token/禁用缓存的基线。服务占用期间不得另起模型测试。复跑会占用 GPU 并影响业务延迟，应选择空闲窗口。

## 本地引擎补丁

上游基线 `garnermccloud/mlx-serve@7dbcba04c98e4fd3bcc533c63e645547f13cc3b1`。保持安装的 MLX 与原基线可执行文件不变，业务版单独构建到 `zig-business-20260910/bin/mlx-serve`。

[`prefix-cache-budget.patch`](../patches/mlx-serve/prefix-cache-budget.patch) 修正三个相关问题：追加替换预先计入继承检查点；替换后执行总字节上限淘汰；三处错误返回保留输入检查点所有权，避免 scheduler 再次清理时重复释放。超大的新条目不替换旧条目，仍可退回较短的已有前缀。分配成功后才转移合并列表，错误时保留旧条目。

本机保留已有 `build.zig` 的隔离 libwebp 配置，不覆盖它。复建需先按 `../qwen38-ssd/runtime/BUILD_INFO.md` 准备固定依赖，再在该源码根目录对照是否已应用补丁；不要重复应用。相应命令如下（输出目录必须选新位置，并同步业务配置中的 binary）：

```sh
git apply --check /absolute/ane-runner/patches/mlx-serve/prefix-cache-budget.patch
git apply /absolute/ane-runner/patches/mlx-serve/prefix-cache-budget.patch
env DEVELOPER_DIR=/Library/Developer/CommandLineTools \
  ZIG_GLOBAL_CACHE_DIR=/absolute/runtime/zig-global-cache \
  .zig-toolchain/zig build test -Doptimize=ReleaseFast \
  -Dwebp-prefix=/absolute/runtime/deps/webp \
  '-Dtest-filter=HotPrefixCache: prefix extension'
env DEVELOPER_DIR=/Library/Developer/CommandLineTools \
  ZIG_GLOBAL_CACHE_DIR=/absolute/runtime/zig-global-cache \
  .zig-toolchain/zig build -Doptimize=ReleaseFast \
  -Dwebp-prefix=/absolute/runtime/deps/webp \
  -Dds4-commit=efdadd41e201 -Dmlx-c-version=56b2d39fc831 \
  --prefix NEW_BUILD_DIRECTORY
```

## 本轮验证

原版运行新增配额回归，两项分别失败于 `expected 32, found 64` 与 `expected 1, found 2`。修复版 4/4 通过（3 个具名回归加 import test），包括 token/合并数组分配失败、旧条目保持、输入由调用者释放、临时内存归还以及 LRU 移位。测试使用 CPU 创建的小数组，不运行模型或 GPU 求值。Zig release 构建通过，补丁另经只读复核。

11k HTTP 验证已完成：

| 请求 | 输入 / 缓存 / 实算 token | 首段输出 | 完整耗时 |
| --- | --- | ---: | ---: |
| 冷请求 | 11057 / 0 / 11057 | 12.090 s | 12.281 s |
| 重复 | 11057 / 11026 / 31 | 0.274 s | 0.463 s |
| 追加对话 | 11099 / 11056 / 43 | 0.312 s | 0.503 s |

三次请求各产生 7 个输出 token，返回预期的两个不同标记；SSE、usage 和独占期间的指标增量相符，缓存 token 与实算 token 之和等于 prompt token。这是有限业务 smoke，每种情况一个样本，不是生产分位数或性能 SLO。

满上下文边界实测通过：合成 prompt 经本服务 tokenize → detokenize → tokenize 往返，准确为 **262143 tokens**，两次请求均生成 1 token，usage 总数均为 **262144**，无输入截断。

| 请求 | 输入 / 缓存 / 实算 token | 首段输出 | 完整耗时 |
| --- | --- | ---: | ---: |
| 满长度冷请求 | 262143 / 0 / 262143 | 671.364 s | 671.364 s |
| 满长度重复 | 262143 / 262112 / 31 | 0.600 s | 0.601 s |

满长度两次输出一致，但这只是合成容量 smoke 的观察，不构成模型逐位正确性或长文理解质量证明。原始 completion usage 没有缓存字段，表中缓存和实算 token 来自独占期间 `/metrics.json` 增量；每次严格只有一个成功请求，计数与 API usage 对齐。冷 prefill 约 11.2 分钟，实际客户端需要容纳这一首响应等待；本配置及验证客户端使用 1800 秒期限。

服务日志显示满长度冷请求后缓存 8840.99 MiB，重复后 9857.27 MiB，均低于 10240 MiB 配额；跨请求继承的检查点也计入该上限。两份旧、新状态都可读不等于未发生物理复制，本轮未建立复制字节或峰值的冷/热独立对照。127 次观测样本及 MLX 记录未发现 OOM，采样最少系统可用内存为 11.12 GiB，MLX 记录峰值约 90.01 GiB；这些不等同于真实外部内存压力验收。

超上限 262147-token 请求返回明确 HTTP 400，未截断或启动推理。最终核对 9 个文件哈希与 102 个模型 payload stat 未变，服务 PID 19347、精确 argv、health、262144-token 模型元数据及 idle 均通过，MTP/drafter 未加载。服务保持运行且保留热缓存。

满长度证据位于 `results/mlx-business-long-v1/`；部署、溢出检查及最终状态位于 `results/mlx-business-service-v1/`。

证据：`results/mlx-business-prep/`、`results/mlx-business-service-v1/`、`results/mlx-business-short-v1/`。运行数据留本机，发布源码保留命令、补丁与验证边界。复跑脚本为 [`verify_mlx_business_service.py`](../scripts/verify_mlx_business_service.py)：

```sh
python3 -B scripts/verify_mlx_business_service.py --phase short --output NEW_SHORT_DIRECTORY
python3 -B scripts/verify_mlx_business_service.py --phase long --output NEW_LONG_DIRECTORY
```

满长度使用合成文本，容量成功不证明该长度上的问答质量。上游已记录 hybrid 冷/热分块带来的数值差异，本轮不承诺完整模型逐位相等。尚未进行多小时业务压力或 SSD 重启恢复验收；SSD 本轮关闭是因为该版本的持久化写出预算与空闲排空能力需另行验证，不能仅打开开关就承诺所有长前缀已耐久保存。

业务可执行文件 SHA256：`36c6ba03bc61fc6ded73158d92a0068ab12bf904a6bb2a120940f458d4bd9860`。短请求报告 SHA256：`9852a342c7b72977a08b6d71f1cecaf6e4ee4f1221f1e1b0d22d019b563fae9e`；满长度报告：`00b85b65b55f7e431543943cfa13f85adab54909aef447c560da418ce989abf8`。
