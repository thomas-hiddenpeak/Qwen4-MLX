# HTTP 共享 KV 页实验

2026-09-14。独立 Swift runner 的 `serve-gpu` 接入可选物理页池，沿用[跨请求页附件](KV_PAGED_PREFIX_RESULTS.md)的完整混合状态、前缀身份和整游标容量回退合同。默认配置保持 reference；单请求和热缓存测量尚未证明分页的稳定吞吐优势。

## 服务配置与生命周期

两个参数必须一起提供：

```text
--paged-kv-pool-library /absolute/path/build/lib/paged_kv_pool.dylib
--paged-kv-pages-per-layer 512
```

动态库按[原生模块构建说明](../../native/paged-attention/README.md)独立构建。参数在模型加载前验证；页数范围为1…4096，禁止与 `--kv-append-mode capacity256` 组合。启用页池后，HTTP 请求必须使用 `mtp_depth=0`，不兼容请求在入队前返回400。

模型的固定推理线程创建一个 context，共享给生成器与前缀缓存。网络线程只读取该线程复制的统计 JSON。模型仍以 dense 状态执行 prefill，缓存仍保留完整 dense 快照；可选附件共享 Attention KV 页，GDN/QSA/PLE 等状态按原合同恢复。SSD 恢复后的附件属于本次进程，不是跨重启的物理页映射。

页池尺寸独立于服务上下文。物理页不足时，该请求整段 decode 使用 dense；以后出现空闲页也不会中途切换。物理池本身最多覆盖131072个逻辑 token，不能据此宣称已支持模型的全部262144上下文。

## 可观测性

`/health.paged_kv_pool` 发布配置、容量策略和已复制的统计；`/metrics` 提供对应的 `qwen_paged_kv_*` 指标。终态日志分别记录 decode 页步骤、复用前缀、导入行数、导入时间、一次性容量回退和页预留。取消或失败若没有完整 phase 统计则保持未知，由独立终态计数记录，不补造零值。

512页×12层的固定 raw arena 是402653184字节；本机包括分配余量和静态元数据的 workspace 预留为405012480字节。空闲页仍在 arena 中，因此空闲服务不应要求这部分 workspace 归零。检查脚本根据公开 VM 页大小与页数独立复算预留；请求和 I/O 排空后要求 request=0、workspace精确等于固定预留、claims=0、in-flight=0。缓存可保留自己的页；缓存和请求都为空时，活动页必须为0。

运行中完成计数和在途计数来自不同 atomic load，允许跨越完成回调；排空后再要求精确守恒。已编码字节和操作次数均不等于实测 DRAM 带宽。

## 验证配置

本轮使用原有 `counter-witness` 混合负载，RAM768MiB、SSD1GiB、联合状态额度8GiB、每层512页。10个可区分的冷输出作为本轮 oracle，之后持续执行恢复、淘汰、分支和取消；实测不同归档工作集保持SSD容量的2–4倍。该负载是短期接入检查，不替代长期稳定性和真实系统压力验收。

```sh
python3 -B scripts/run_http_cache_churn.py \
  --output-directory results/NEW-http-paged-run \
  --duration-seconds 600 --port 11248 --fixture-mode counter-witness \
  --paged-kv-pool-library /absolute/path/build/lib/paged_kv_pool.dylib \
  --paged-kv-pages-per-layer 512 \
  --prefix-cache-bytes 805306368 --state-budget-bytes 8589934592
```

通过[实验控制器](../EXPERIMENT_CONTROLLER.md)串行运行；重新生成冻结清单，不能复用旧 plan。独立分析对齐客户端 request ID、模型终态和健康累计值，并要求至少一次 memory 命中满足 cached tokens=物理复用前缀>0。单个冷请求刚发布附件再使用它，不算跨请求命中。

## 本轮结果

`results/paged-http-churn-v1` 已完成独立审计。实际混合工作段639.468秒，包含23次成功、2次取消，另有10次可区分冷参考；35条客户端记录与模型终态完整唯一匹配。全部10种前缀均被覆盖，两段300秒淘汰窗口通过。178项相关Swift CPU、39项Python控制及8项服务参数负控通过。

- 23次完成请求实际使用paged32，共345个decode步骤；10次完成请求仅发生一次dense容量回退。
- 20次完成请求复用附件，其中4次严格满足RAM跨请求命中条件。另有两次成功的SSD恢复后附件晋升并执行分页decode，分别命中10816和1248 tokens；不能把它们称为跨重启共享物理页。
- 原生统计为4656次write、4152次read、8808次完成，materialization/failed/in-flight均为0。完成请求贡献4140次layer read，其余包含取消请求的已执行工作，不用完成请求统计补造取消phase。
- 最终request=0、claims=0，workspace精确为405012480字节。热缓存保留2条、734566787字节，4056个活动页属于保留前缀（每层338页，共12层）；热缓存保留不是泄漏。
- 混合阶段归档读553553920字节、写4311539311字节、15次淘汰；不同归档工作集3118655963字节，为SSD额度的2.904倍。零归档损坏/写失败，508个health观察无错误。
- 33组RSS/FD采样、有限关闭日志、server/client退出0均通过。MLX累计allocator peak为81140547326字节，不等于进程RSS或物理内存硬上限。

二进制SHA256为 `97e54c742ef9fe2bacadc26932881d5f6953dcc1239bfd8683c9836db7719f39`。独立记录为 `independent-agent-analysis-v1.json`，终态详细对账为 `http/paged-terminal-validation.json`。561个冻结文件和102个模型payload stat复核无变化；参考服务以PID59001恢复，精确原argv、idle、MTP/drafter关闭均核实。

本轮未发生周期性中途强制drain（600秒规划与300秒间隔的边界条件），只报告实际oracle/最终drain及取消后的清理。它不构成长时间耐久、真实OS压力、262K上下文或吞吐提升验收。
