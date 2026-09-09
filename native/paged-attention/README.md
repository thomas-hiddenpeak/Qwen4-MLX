# 原生分页 Attention reader 与物理 KV 页池

这是独立 Swift/MLX runner 的显式实验模块。它通过 MLX lazy primitive 读取页表与 K/V，保留本机固定 MLX vector SDPA 的计算顺序；不替换安装的 MLX。

## 当前合同

- 仅 GPU、普通 AR 单 token：BF16 `Q[1,24,1,256]`，KV heads=2，GQA=12，scale=1/16；输出与 Q 同形。逻辑长度限于 `1...131072`（128 Ki tokens），页大小固定 32。不是模型全部上下文范围的支持声明。
- `PageMajor`：`K/V[P,2,32,256]`，允许乱序、有洞的有效物理页号；`HeadMajor`：`K/V[1,2,T,256]`，支持现有 capacity 视图及各自不同的 K/V head stride。页、head、token stride 独立，最后一页按逻辑 T 截断。
- 可选 bool mask 为 `[1,1,1,T]`，按逻辑 token 寻址，支持零 stride 广播。它消费 runner 已生成的 QSA 可见性，不在此模块重做 QSA indexer/选择。
- 完整模型的 `GPUPagedSDPAReader` 接线使用一次创建的 identity 页表及现有 `capacity256` 状态；reader 不打包、gather 或拼接完整 K/V。capacity 初始化/增长、QSA history 更新仍属于原路径的实际成本。
- C ABI 版本为 1，见 [paged_sdpa_bridge.h](paged_sdpa_bridge.h)：create/free、read、dispatch_info、metadata_bytes、encoded_reads 和错误查询。原生输入由输出图持有；Swift 销毁 context 后不卸载已校验 DSO，避免待执行图的 C++ vtable 失效。
- `encoded_reads` 是 DSO 内成功编码完整 reader 的累计次数，不是图创建次数、GPU 完成次数或带宽。scratch 字节仅含 BF16 partial 与 FP32 sums/maxs，不含输出、allocator 缓存和并存图。

完整模型的 reader 实验仍使用原有 capacity 状态及完整混合状态 RAM/SSD 快照，默认 reader 是 stock MLX；prefill、MTP 和 verification 不走实验 reader。以下单层物理页池是独立机制入口，尚未替换完整模型或 HTTP 的状态存储。

## 单层物理 KV 页池

`paged_kv_pool` 与 `GPUPagedKVPool` 建立固定容量的 K/V 物理 arena，每份布局为 BF16 `[pages,2,32,256]`。每个 K/V 页对为 65,536 字节。初始前缀导入一次后，分支各持不可变页表；`fork()` 共享页和已有写入依赖，不复制 K/V。

追加保留所有完整页的物理地址。首版对每次部分尾页续写都分配新页，最多复制 31 行旧 K/V，再写入新行；即使调用者未保留其他分支也保持这一规则，避免未完成图观察到原位修改。页表 reader 直接读取 arena，逻辑 QSA bool mask 保持原 token 顺序。整页到达边界时只分配后续新页，不复制完整前缀。

槽位预留、提交和回滚由 CPU 元数据管理，页号复用带 generation 检查。写入 primitive 固定源/目标页租约，并通过前一写入 ticket 建立 MLX 图依赖；读取和显式导出固定对应状态。GPU dispatch 前注册完成回调持有这些引用，避免图 detach 或 Swift wrapper 提前释放后复用活动页。元数据提交不代表 GPU 已完成。容量须同时容纳各分支、旧尾页版本和在途引用，不能仅按一个分支的 token 长度配置。

Swift 入口提供导入、分叉、单行追加、读取、ready ticket、显式 `materialize()` 和统计。`materialize()` 在求值时导出完整连续 K/V，仅供内容对照及后续归档边界使用；稳态 decode 不调用它。成功加载的 DSO 保持驻留，因为 native 图可能比 Swift wrapper 活得更久。调用和状态 handle 限于所属推理执行器。

统计分别报告完整 arena 分配、唯一活动页、单状态逻辑 K/V、已编码复制/写入/导出字节和操作完成/失败。活动页释放后可在 arena 内复用，整个 arena 的分配仍存在；逻辑字节跨分支求和会重复计算共享页。编码字节是 kernel 描述的 payload，不是测得的 DRAM 流量、RSS 降幅或成功完成证明。

pool 在观察到 Metal 命令失败后拒绝后续导入、追加、读取和导出，并在写入/读取求值入口复查；该保护要求丢弃 pool 并恢复运行环境，不在旧 pool 上清除错误继续使用。目前未通过真实硬件故障验证该路径，也不取消已经排队的工作。

CPU 元数据测试以 C++20、`-O2 -Wall -Wextra -Werror` 编译运行，2,257 项检查通过。它们验证 CPU 内容路由、11,057 行批量导入、分支隔离、边界、OOM 回滚、代际复用及并发释放，不构成 Metal 或整模型证据。本轮 GPU 结果及采用边界见[研究记录](../../docs/research/VLLM_METAL_ADOPTION.md)。

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

增加 `--pool` 则构建独立物理页池及其 C ABI；输出库名为 `lib/paged_kv_pool.dylib`，C++ 页管理使用 C++20。两种 dylib 均排除 CPU 测试与 probe main，独立 probe 可执行文件另行构建：

```sh
python3 -B scripts/build_mlx_paged_attention.py --pool \
  --bridge-dir native/paged-attention --output results/paged-kv-pool-build-NEW
```

CPU 元数据测试不调用 GPU，可单独编译；使用新的输出位置：

```sh
env DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun clang++ \
  -std=c++20 -O2 -Wall -Wextra -Werror -pthread \
  native/paged-attention/immutable_page_pool.cpp \
  native/paged-attention/immutable_page_pool_test.cpp \
  -o /absolute/path/NEW-metadata-test
/absolute/path/NEW-metadata-test
```

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

物理页池使用 Swift binding 进行单层机制验证。将以下 argv 的同步、异步两种模式分别放入控制器 case，NDJSON 输出路径必须为新文件：

```text
/absolute/path/ane-runner probe-gpu-paged-kv-pool --library /absolute/path/build/lib/paged_kv_pool.dylib --eval sync --output /absolute/path/NEW-sync.ndjson
/absolute/path/ane-runner probe-gpu-paged-kv-pool --library /absolute/path/build/lib/paged_kv_pool.dylib --eval async --output /absolute/path/NEW-async.ndjson
```

本机 `results/page-pool-mechanism-v1/` 两种模式各通过 1,435 项检查及 71,470,080 个 BF16 元素对照。覆盖 11k 前缀的四分支续写、旧状态不变、直接 SDPA、OOM、页复用、丢弃 lazy 图以及原生图超出 Swift wrapper 寿命；同步与异步都在检查前等待完成。诊断计时含同步，不用来宣称并发吞吐收益。

## 来源与许可

shader 算术及 dispatch 策略来自本机固定 MLX `sdpa_vector.h` / `scaled_dot_product_attention.cpp`，采用 Apple 的 MIT 许可，保留 [LICENSE.MLX.txt](LICENSE.MLX.txt) 和源文件版权说明。固定源 SHA 分别为 `de098d50a67a865e2e64fb1700fc4819307861f009a65642679777a9a9cec7cd`、`215ebac6495706cf01a2635c4dcec29802e130165ed8704f212d789679d2e673`；第二阶段 reduce 直接使用 stock MLX。

[vllm-metal 023e544f](https://github.com/vllm-project/vllm-metal/tree/023e544fec59f872f65f66e23232706e4e17ff2e) 提供了页表与 MLX 图依赖/状态所有权方面的设计参考；本模块没有复制其 kernel 算术。页表 reader 的实现不等同于已吸收其 scheduler 或共享缓存管理。
