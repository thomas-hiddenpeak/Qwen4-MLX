# MTP 验证阶段的实际专家重叠

2026-09-07。原始 11,057 token 提示词、D2、128 token 输出中，55 轮验证全部为 S3。每层三个 token 共选择 30 次专家，平均涉及 21.9845 个不同专家；26.7184% 的选择与同轮同层其他 token 重复。这个结果支持试做共享 gate/up 权重读取的局部算子，尚未证明实际带宽节省或生成加速。

## 采集方式与正确性

新增默认关闭的 `generate-gpu --capture-verification-routing true`。入口限制单请求、D2、`batchedScalarLinear`、最多 128 个输出 token，禁止与同步 profiler 或原生命令跟踪组合。回调只保留验证阶段路由器产生的原始 `[1,S,10]` IDs 张量，不排序、不重算路由、不在层内求值或读回。请求计时结束后统一转换、读回并校验，每行保持原 top-k 槽位顺序。

采集对象最多保留 8192 个张量；越界、非法 ID 或重复的单行 ID 不会产生可接受的完整分析。保留张量仍可能影响图和缓冲区寿命，因此采集运行只作诊断。`finish()` 的转换、求值、读回与校验耗时单列，本次为 402.4365 ms；它不是纯 CPU 或纯 GPU 时间。

普通与采集两次运行都与同一 AR golden 的完整 128 IDs 一致，最终 offset 为 11184，length 终态一致。两者均为 55 轮、110 草稿、72 接受、165 验证 token、127 decode 输出、0 replay，接受直方图 `[12,14,29,0,0]`。采集覆盖 55×48=2640 条记录，每轮层序完整，0 丢弃。所有实际执行过的草稿行均纳入，包括未接受的草稿；没有跨层或跨轮合并。此次没有 S2 样本，不能据此判断 S2 重叠率。

## 实测分布与逻辑上限

| 指标 | 结果 |
| --- | ---: |
| 专家选择总次数 | 79,200 |
| 同轮同层去重后的专家次数 | 58,039 |
| 重复选择次数 | 21,161（26.7184%） |
| 每层不同专家数 U≤24 的记录 | 1,916 / 2,640（72.58%） |
| U≥28 的记录 | 193 / 2,640（7.31%） |
| 被一个 / 两个 / 三个 token 使用的专家次数 | 41,434 / 12,049 / 4,556 |

重复选择数是 `n2 + 2*n3`。三行两两交集会把三重共享计算三次，不能直接充当可消除的选择次数。各层差别明显：第 45 层重复比例为 38.67%，第 0 层为 7.27%；不能只用最高重叠层代表整模型。

已检查原始 safetensors 的 432 张 routed tensor 头部（48 层、三投影、weight/scale/bias）。每个专家每个投影包含 819,200 B Q4 数据、51,200 B BF16 scale 和 51,200 B BF16 bias，共 921,600 B。三个投影共 2,764,800 B，gate/up 为 1,843,200 B。

按逐次选择重复读权重计算，本次全部路由专家投影的逻辑量为 218.97216 GB。若仅 gate/up 对同一专家共享一次读取，理想可少读 39.0039552 GB，即全部路由专家权重逻辑量的 **17.8123%**。若三个投影都能共享，逻辑上限才是 26.7184%。这些量不含常驻共享专家、router、激活、GDN、attention、head 或其他状态；也不等于实际 DRAM 流量。硬件缓存可能已复用部分权重，新规划、分支和寄存器开销也可能抵消收益。

## 后续算子筛选

先尝试 GPU 上的小型专家成员表，让同一专家的 gate/up 权重服务一至三个 token，并将结果写回原 `[token,slot,640]` 位置；保持各 token 的累加、BF16 舍入、激活及槽位归约顺序。down 先沿用既有实现。局部测试必须包含规划成本，对照原逐 token S1 路径及已有 token-axis 路径，并包含低重叠、不重叠和不同槽位的共享情况。约 5% 且可重复的完整局部路径收益才值得进入整模型验证，路由统计本身不改变默认或 MTP 发布状态。

## 记录与复算

Release 构建 49.23 秒，19 项选定 Swift CPU 检查通过；离线分析器 22 项 CPU 控制通过，7 种非法 CLI 组合均在加载模型前拒绝。独立分析未导入主分析器，用 512 专家 bitmask 重算全部 2640 条记录、交集、fanout、字节量及 48 层排名，12 项检查通过。原始运行记录在 `results/mtp-expert-overlap-v1/run/`，权重头部记录在同级 `weight-geometry.json`。以下命令按固定 11k/128/D2 合同检查完整输出、计数、覆盖与逻辑量，输出须使用新路径：

```sh
python3 scripts/analyze_mtp_expert_overlap.py \
  --normal results/mtp-expert-overlap-v1/run/normal.json \
  --diagnostic results/mtp-expert-overlap-v1/run/captured.json \
  --golden results/prefill-agent-11k-default-check/default.json \
  --weight-geometry results/mtp-expert-overlap-v1/weight-geometry.json \
  --output NEW_ANALYSIS.json
```

实测二进制 SHA `f6f6fb63321e96899334d42e1bb25d500ef40c71cd66e02bbde12cfed6c7800a`；普通原始报告 SHA `691eb697d7c7a316b12ecf88d4beb4488b7661e33428e9872f58287df4626c97`，采集报告 SHA `978b160ba4c953cf919af4dc98d29fb9195af46da2d4163130517257719646b9`。141 个文件 SHA、102 个模型 payload 大小及 mtime 运行后复核通过。参考服务 PID 49817 的原参数、监听、空闲和 MTP/drafter 关闭已核对。

普通运行 prefill target 549.785 token/s、decode 40.314 token/s；采集运行分别为 515.813 / 34.576 token/s。二者只用于记录运行状态，没有交错性能对照，且采集会影响张量寿命，不将该差值解释为采集开销或优化效果。此次完整输出一致也不单独证明全部中间状态逐位相同。
