# 从小幅融合转向主要瓶颈

## 2026-09-06：六种派发方案快测

保持原 BF16/Q4 权重和 MTP 关闭，直接复用现有探针测试，没有新增权重副本。

| 同批整模型比较 | Reference | 候选 | 聚合变化 |
|---|---:|---:|---:|
| QKV/Z：BM8→BM4 | 34.364 | 35.030 token/s | +1.94% |
| 三种投影形状：BM4/TM1 | 34.364 | 34.813 token/s | +1.31% |
| 三种投影形状：BM2/TM4 | 35.297 | 35.998 token/s | +1.99% |
| 三种投影形状：BM1/TM4 | 35.297 | 33.895 token/s | −3.97% |

后两行来自另一批，排除了首轮预热；不同批次不可直接比较绝对速度。修改 out 形状的方案也覆盖同形状的 Attention out。共 13 轮、每轮 256 个输出 ID 全部匹配已有 reference，但前后速度波动明显，尚未获得稳定的大幅收益。

另试普通 GEMM 与 K 分区为 256 的 NAX split-K。两者在交替微基准中都更慢，直接淘汰，未继续整模型测试。矩阵输出相对 L2 最大约 0.0136%；这次显式允许归约舍入差异进行性能筛选，默认仍要求逐位一致。

结果：[BM4/TM1](results/gdn-gemv-quick/full.json)、[BM2/BM1](results/gdn-gemv-bm12/full.json)、[GEMM/split-K](results/gdn-gemv-gemm/matvec.json)。这些候选保留为 `--gdn-gemv-mode` / `--gdn-gemv-order` 实验开关，须搭配对应独立库，没有成为默认路径。下一步优先修改实际读取和计算内核，不再只扫描 BM 参数。测试后已恢复作者服务，MTP 保持关闭。

## 先前的瓶颈测量

上一轮计算融合在配对中约快 2.2%，但增加 3.67 GB 投影副本；它没有直接解决主干大矩阵的读取。本轮保持原权重、数值语义和 MTP 关闭，重新核算模块权重，并接入当前 MLX 自己的 Metal 命令缓冲计时。

## 每 token 的主要权重

[审计脚本](scripts/audit_decode_weights.py)读取实际 safetensors headers，核对当前 48 层文本路径的 1,499 个张量。加载源权重为 **77,843,121,920 B**，静态逻辑 decode 权重为 **9,951,107,840 B/token**。[完整账本](results/gpu-bottleneck-v1/weight-audit.json)

| 模块 | 逻辑权重 GB/token，十进制 | 比例 |
|---|---:|---:|
| GatedDeltaNet，36 层，BF16 | 4.173 | 41.94% |
| 每层选中的 10 个 Q4 专家及 scale/bias | 1.327 | 13.34% |
| Hyper Connection，BF16 | 1.281 | 12.88% |
| 输出 head，BF16 | 1.271 | 12.78% |
| Attention，12 层，含 indexer | 1.235 | 12.41% |
| Shared experts，BF16 | 0.472 | 4.74% |
| Router、shared gate、PLE 等 | 0.192 | 约 1.93% |

这里没有按 512 个专家全部计费，也没有把 51.2 GB n-gram 表当作逐 token 读取：表的逻辑请求另计 **2,560 B/token**。HC 缩放副本和上一轮投影拼接副本属于常驻容量，单独记账，没有重复算入本表。短上下文中未执行的两个 QSA indexer norm 造成静态账本多计 6,144 B/token，审计明确保留这项边界。

以上是逻辑权重，不是实测 DRAM 读写。相同权重可能被缓存或重复获取，状态和临时数据另有流量。

## 首次完整的命令缓冲时间记录

诊断副本在当前 MLX `CommandEncoder::commit` 的已有 completion handler 开头记录 [GPUStartTime / GPUEndTime](https://developer.apple.com/documentation/metal/mtlcommandbuffer/gpustarttime)。这些是 GPU 执行整条命令缓冲的 host 起止时间；`kernelStartTime / kernelEndTime` 则是 CPU 驱动调度时间，不能当作 GPU shader 内核时间。

Swift 同时保存每个 token 的开始、forward 返回、evaluate/readback 返回时间，使用同一 boot 时钟。原生记录验证 GPU 时间位于 CPU commit/completion 包络内。按实际区间求并集，避免重叠相加。

[本轮原始记录](results/gpu-bottleneck-v1/commands.json)含 **197,678 条已完成记录**，dropped/pending/failed status/missing timestamp/clock mismatch 均为 0。[同进程对齐结果](results/gpu-bottleneck-v1/command-analysis.json)中，排除第一轮预热后，510 个 decode 步：

| 时间口径 | 结果 |
|---|---:|
| 单步墙钟耗时合计 | 14.850781 s |
| 步内 GPU 命令缓冲跨度并集 | 12.845355 s |
| 命令缓冲跨度覆盖比例 | 86.50% |
| 每步 GPU 命令缓冲跨度并集，平均 | 25.187 ms |
| 每步位于这些跨度之外的时间，平均 | 3.932 ms |
| 两轮 forward 墙钟中位数 | 3.047 / 2.480 ms |
| 两轮 evaluate/readback 墙钟中位数 | 25.401 / 26.605 ms |
| 每步重叠命令缓冲数量，中位数 | 239 / 236 |

**86.50% 不是 GPU shader 活跃率，更不是物理带宽利用率。** 命令缓冲区间内可能包含访存停顿、调度和其他空隙。区间外时间也不能全部归为可以删除的 Swift 开销；forward、evaluate 与 GPU 工作会重叠，不能把它们相加。这些数据支持优先调查 GPU 内部的大矩阵执行，但还不能独立区分带宽、计算指令和 GPU 调度瓶颈。

同一诊断库、相同 1217-token 项目文档输入、每轮 256 个输出 token，依次运行计时关闭、开启、再关闭，每个进程重复三轮。共九轮生成 ID 全部匹配保存的 reference。第二、三轮聚合分别为 **36.286、34.341、35.065 token/s**。[交接与命令](results/gpu-bottleneck-v1/run-ledger.json)

计时运行相对前后两组控制均更慢，控制本身也有漂移；不能声称计时开销为零，或把差值全部当作纯采集开销。这里的硬件时间用于诊断，正常性能对比工具会拒绝把 `gpu_command_timing.enabled=true` 报告当作无采样对照。72 项 Swift 测试在这套独立库下通过，三组完整生成也保持数值一致。

## 真实 BF16 大矩阵微基准

为了直接检查主要矩阵，新增 `probe-gpu-matvec`，仅载入 13 个真实矩阵，合计 **1,732,771,840 B**。GDN 每种形状轮换层 0/4/8/12 的四个矩阵，head 重复一个较大矩阵；使用固定种子合成的非零有限 BF16 输入，每类预热 8 次、测量 64 次。每次重新构造 matmul 并求值，避免测到已计算结果的空操作。读回和重复性比较放在计时窗口外。

[原始结果](results/gpu-bottleneck-v1/matvec/report.json)全部 finite / bitwise 重复性检查通过；[395 条原生记录](results/gpu-bottleneck-v1/matvec/commands.json)完整。[离线对齐](results/gpu-bottleneck-v1/matvec/analysis.json)得到：

| 矩阵 | 每次逻辑权重 | GPU 命令缓冲跨度中位数 | 权重字节／GPU 跨度，聚合 |
|---|---:|---:|---:|
| GDN qkv | 52.429 MB | 0.1983 ms | 269.6 GB/s |
| GDN z | 31.457 MB | 0.1261 ms | 239.9 GB/s |
| GDN out | 31.457 MB | 0.1223 ms | 252.9 GB/s |
| lm_head | 1,271.398 MB | 2.5879 ms | 495.4 GB/s |

**这是一处值得优先追查的效率差异，不是已证明的整模型提速幅度。** GDN 三组矩阵本身占每 token 逻辑权重约 42%，所以后续先检查这些固定形状的 GEMV 内核、权重布局和执行调度。微基准中的轮换工作集、重复 head、GPU 频率和逐次同步条件不同于完整 decode；表中的 GB/s 不是 DRAM 硬件计数，也不能把 head 的速率直接套用到 GDN 或相加预测 token/s。

复现前完成模型资源交接，使用同一个独立诊断库：

```bash
env DYLD_LIBRARY_PATH="$PWD/results/manual-command-timing-native/lib" \
  .build/release/ane-runner probe-gpu-matvec \
  --model-dir ../qwen38-ssd/models/Qwen3.8-Flash-Next-MLX-SSD-Stream \
  --repeats 64 \
  --gpu-command-timing-output results/manual-matvec-commands.json \
  --output results/manual-matvec.json

python3 scripts/analyze_gpu_matvec.py \
  --report results/manual-matvec.json --commands results/manual-matvec-commands.json \
  --output results/manual-matvec-analysis.json
```

最后的 [微基准交接记录](results/gpu-bottleneck-v1/matvec/run-ledger.json)保存恢复的作者服务，MTP / drafter / PLD 保持关闭。原服务和原模型文件未被诊断库替换。

## 大幅减少数据量的独立路线

GDN 的 qkv、z、out 共 108 个大矩阵，合计 **4,152,360,960 B**。仅从存储格式估算，若另外量化为 group64、BF16 scale/bias：

- Q8，实际 8.5 bit/parameter：2,205,941,760 B，减少 **1.946 GB/token**。
- Q4，实际 4.5 bit/parameter：1,167,851,520 B，减少 **2.985 GB/token**。

这些是假设字节量，**没有进行权重转换，也没有验证精度或获得相应速度提升**。保持原权重逐位一致时，不能把这部分压缩空间计作已实现的 runtime 收益。原始 checkpoint 保持不变。

## 复现诊断

在本目录创建独立库：

```bash
python3 scripts/build_mlx_command_timing.py \
  --output-root results/manual-command-timing-native
```

脚本只重编复制后的 device.cpp，其他固定构建对象只读复用，重新链接到新的目录；不会覆盖作者源码、对象或库。保存的 [构建 provenance](results/gpu-bottleneck-v1/native/build-provenance.json)验证原始输入哈希未改变。collector 使用固定 262,144 个槽位，回调只写记录和原子状态；写文件在停止采集后进行。超限、超时或时间戳缺失都明确报告 incomplete；普通生成不依赖这些附加 ABI。

先做小矩阵能力检查：

```bash
env DYLD_LIBRARY_PATH="$PWD/results/manual-command-timing-native/lib" \
  .build/release/ane-runner probe-gpu-command-timing \
  --gpu-command-timing-output results/manual-command-probe-native.json \
  --output results/manual-command-probe.json
```

完整模型必须先完成与作者服务的资源交接；按上一轮实验规模采集三轮，不应任意增大次数超出固定记录容量：

```bash
env DYLD_LIBRARY_PATH="$PWD/results/manual-command-timing-native/lib" \
  .build/release/ane-runner generate-gpu \
  --model-dir ../qwen38-ssd/models/Qwen3.8-Flash-Next-MLX-SSD-Stream \
  --tokens-file fixtures/gpu-telemetry/prompt-token-ids.json \
  --max-tokens 256 --repeat 3 \
  --gpu-command-timing-output results/manual-commands.json \
  --output results/manual-command-generation.json

python3 scripts/analyze_gpu_command_timing.py \
  --generation results/manual-command-generation.json \
  --commands results/manual-commands.json \
  --output results/manual-command-analysis.json
```

输出路径必须为新文件。分析器检查 PID、hook 版本、完成状态和时钟包络；不完整时保留已观察区间，抑制完整覆盖和间隙推断。普通库缺少诊断符号时，CLI 会在载入模型前明确拒绝，而不会生成虚假的硬件时间。
