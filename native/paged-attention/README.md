# 原生分页 Attention reader

这是独立 Swift/MLX runner 的显式实验模块。它通过 MLX lazy primitive 读取页表与 K/V，保留本机固定 MLX vector SDPA 的计算顺序；不替换安装的 MLX。

## 当前合同

- 仅 GPU、普通 AR 单 token：BF16 `Q[1,24,1,256]`，KV heads=2，GQA=12，scale=1/16；输出与 Q 同形。逻辑长度限于 `1...131072`（128 Ki tokens），页大小固定 32。不是模型全部上下文范围的支持声明。
- `PageMajor`：`K/V[P,2,32,256]`，允许乱序、有洞的有效物理页号；`HeadMajor`：`K/V[1,2,T,256]`，支持现有 capacity 视图及各自不同的 K/V head stride。页、head、token stride 独立，最后一页按逻辑 T 截断。
- 可选 bool mask 为 `[1,1,1,T]`，按逻辑 token 寻址，支持零 stride 广播。它消费 runner 已生成的 QSA 可见性，不在此模块重做 QSA indexer/选择。
- 当前 Swift 接线使用一次创建的 identity 页表及现有 `capacity256` 状态；reader 不打包、gather 或拼接完整 K/V。capacity 初始化/增长、QSA history 更新仍属于原路径的实际成本。
- C ABI 版本为 1，见 [paged_sdpa_bridge.h](paged_sdpa_bridge.h)：create/free、read、dispatch_info、metadata_bytes、encoded_reads 和错误查询。原生输入由输出图持有；Swift 销毁 context 后不卸载已校验 DSO，避免待执行图的 C++ vtable 失效。
- `encoded_reads` 是 DSO 内成功编码完整 reader 的累计次数，不是图创建次数、GPU 完成次数或带宽。scratch 字节仅含 BF16 partial 与 FP32 sums/maxs，不含输出、allocator 缓存和并存图。

这里还没有全局共享页池、页引用计数或块级缓存淘汰。现有完整混合状态 RAM/SSD 快照与预算合同保持原样；默认 reader 仍是 stock MLX，prefill、MTP 和 verification 不走新路径。

## 构建

从仓库根目录执行；输出目录必须不存在。脚本使用本机固定 runtime 的 compiler database 与 `-fno-fast-math` Metal recipe，并检查 stock 安装身份。需要完整 Xcode；可用 `--runtime`、`--developer-dir` 指定对应位置。

```sh
python3 -B scripts/build_mlx_paged_attention.py \
  --bridge-dir native/paged-attention \
  --output results/paged-attention-build-NEW
```

默认 `--source-dir` 就是此目录。省略 `--bridge-dir` 只构建独立探针；带此参数额外生成只链接 reader 与 C bridge 的 dylib，排除 probe 的 main。产物：

- `bin/paged-reader-probe`
- `lib/paged_reader.metallib`、可选 `lib/paged_reader.dylib`
- `build-provenance.json` 与逐步命令日志，包含输入/产物 SHA、编译参数、退出码和 stock 依赖验证。

构建脚本不运行探针、不做 GPU 正确性或性能验证，也不覆盖原库。

## 受控验证

将探针命令登记到新的 [实验控制器](../../docs/EXPERIMENT_CONTROLLER.md) plan，固定本次源码、二进制和模型身份后串行运行。不要直接复用旧冻结 plan；控制器负责参考服务的暂停与恢复：

```sh
python3 -B scripts/run_specialization_experiment.py /absolute/path/NEW-controller-plan.json
```

plan 中的两个独立 case 可使用以下 argv，并各自保留 stdout 与退出码：

```text
/absolute/path/build/bin/paged-reader-probe --run
/absolute/path/build/bin/paged-reader-probe --run --async-eval --benchmark-iterations 20
```

没有 `--run` 时探针退出且不启动 GPU。可选 benchmark iterations 范围为 1...200，先完成正确性/owner 检查，再对 11232/11233 行执行 reader ABBA；`--async-eval` 使用 async_eval 后 synchronize。

2026-09-09 本机机制 v1 的同步与异步报告各通过 15 个长度、260 个对照 case，并完成 owner 检查。本地产物在 `results/vllm-metal-paged-mechanism-v1/`。探针不加载完整模型，也不执行学习得到的 QSA indexer；这些结果不代表整模型正确性、生产稳定性或性能收益。后续整模入口是显式 `benchmark-gpu-paged-attention`，其状态哈希诊断时间不能用作吞吐证据。

## 来源与许可

shader 算术及 dispatch 策略来自本机固定 MLX `sdpa_vector.h` / `scaled_dot_product_attention.cpp`，采用 Apple 的 MIT 许可，保留 [LICENSE.MLX.txt](LICENSE.MLX.txt) 和源文件版权说明。固定源 SHA 分别为 `de098d50a67a865e2e64fb1700fc4819307861f009a65642679777a9a9cec7cd`、`215ebac6495706cf01a2635c4dcec29802e130165ed8704f212d789679d2e673`；第二阶段 reduce 直接使用 stock MLX。

[vllm-metal 023e544f](https://github.com/vllm-project/vllm-metal/tree/023e544fec59f872f65f66e23232706e4e17ff2e) 提供了页表与 MLX 图依赖/状态所有权方面的设计参考；本模块没有复制其 kernel 算术。页表 reader 的实现不等同于已吸收其 scheduler 或共享缓存管理。
