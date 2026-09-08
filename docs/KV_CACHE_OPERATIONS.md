# KV cache 本机运行与维护

适用：`1ca481b` 的 Swift/MLX HTTP AR 服务。恢复单位是 Attention KV/QSA、GDN、PLE 与历史的完整状态。详细合同及分版本证据见[缓存可靠性](KV_CACHE_RELIABILITY.md)；接口与固定限额见[HTTP 服务](HTTP_SERVER_EXPERIMENT.md)。

## 启动

从仓库目录运行；先创建缓存父目录，取物理路径，最后一级由 store 创建为 0700。选择专用目录，一次只由一个服务持有。

```sh
mkdir -p "$HOME/Library/Caches/Qwen4-MLX"
KV_CACHE_PARENT="$(cd "$HOME/Library/Caches/Qwen4-MLX" && pwd -P)"
.build/release/ane-runner serve-gpu \
  --model-dir ../qwen38-ssd/models/Qwen3.8-Flash-Next-MLX-SSD-Stream \
  --port 11236 \
  --prefix-cache-directory "$KV_CACHE_PARENT/qwen38" \
  --prefix-cache-bytes 536870912 --prefix-cache-entries 8 \
  --prefix-cache-disk-bytes 8589934592 --prefix-cache-disk-entries 32 \
  --prefix-cache-ttl-seconds 86400 --prefix-cache-min-free-bytes 1073741824 \
  --prefix-cache-restore-timeout-seconds 5 --prefix-cache-shutdown-timeout-seconds 30 \
  --state-budget-bytes 4294967296
```

这是 512 MiB RAM / 8 GiB SSD / 4 GiB 联合逻辑状态额度的示例，SSD另保留至少1 GiB可用空间。已有缓存末级目录必须属于服务UID且group/other无权限，推荐0700；CLI先规范化路径、解析符号链接，store再逐组件以O_NOFOLLOW打开物理路径。省略SSD目录及SSD专属参数就是RAM-only；`--prefix-cache-bytes 0`关闭RAM，同时不能启用SSD。服务仅监听127.0.0.1，加载期间`/health`返回503，等`status=ready`再送请求。

## 看是否命中、是否排空

```sh
curl --max-time 5 -sS http://127.0.0.1:11236/health
curl --max-time 5 -fsS http://127.0.0.1:11236/metrics
```

| 关注点 | 看什么 |
|---|---|
| 本次真实复用 | 响应`usage.prompt_tokens_details.cached_tokens`；JSON模型终态的`cache_source`与cached/computed tokens。`prefix_cache.restoredHits`包括RAM/SSD，`diskHits`只计实际SSD恢复；索引hits不等于成功复用。 |
| 请求与临时owner结束 | 停止送新请求后，`idle=true`，active、active_jobs、pending_requests、queued_prefills、ready_decodes、resident_sequences、reserved_tokens、waiting_prefix_sequences均为0；再看`state_budget.requestBytes/workspaceBytes`、SSD`pendingJobs/pendingBytes/foregroundReadIntents`回到0。 |
| 正常保留 | 空闲时RAM entries、`cacheBytes`和对应lease可以大于0。MLX allocator保留也不会随请求结束必然清空。 |
| 故障线索 | `restoreFailures`、SSD`corruptions/writeFailures/storageUnavailable`、`logging.dropped_events/write_failures`，结合本次请求ID与时间查看；有日志丢失时不能声称终态对账完整。 |

health是执行器边界快照，空闲也约每100ms刷新；不要用单次采样或`running_job=null`判定已释放。结构化日志只统计`qwen-http-lifecycle-v1`的`model_terminal`，不再加一次兼容文本行；模型完成与客户端实际收到完整响应分别核对。

## 跳过、等待和恢复

- 首次请求、历史编辑、TTL淘汰、超出RAM/归档额度、保留共享系统锚点、可选workspace不足，都可能正常miss或跳过保存。它们不表示推理失败；完整prompt仍按原长度预留。提交前或非流式的请求额度不足为HTTP429/resource_limit，调度token/队列不足为429/queue_full；已开始的SSE遇额度不足会发送error(code=resource_limit)和DONE，HTTP头保持200，客户端也须检查事件。
- 默认最多两个resident sequences、32768个调度预留token，含排队请求；`--max-connections`不是长请求并发额度。客户端RST返回时，服务端旧预留可能尚未释放，重试需给取消清理留余量。
- `foregroundReadIntents`是元数据优先权，暂时挡住新的可选写，不是已开始读或占用payload。默认可配置的5秒期限限制未完成的准入与读取等待；未提交的过期意图在后续store访问时撤销。超时后停止等待SSD，改用可用RAM前缀或冷算；已ready结果仍可用，期限不是同步导入或完整TTFT的硬截止。后台已接收IO保留workspace至真实完成，不能提前把pending/lease记成0。
- `diskReadTimeouts`和`diskPublicationTimeouts`分别看。相同key的旧发布未完成时，不因等待者超时而再写一份。持续增长时对照磁盘耗时和workspace，不靠反复重发大请求掩盖原因。
- 可用空间不足或采样失败会跳过写入，已有归档仍可读；释放文件系统空间后，后续写入自动重查，观察`spaceConstrained/spaceRecoveries`。水位保护不能阻止其他进程随后占用空间。
- `memory_pressure.effectiveLevel=warning`暂停可选保存/提升并渐进trim RAM；critical还拒绝新请求。收到更低级OS通知并稳定5秒后降到该目标级别：critical降为warning恢复新请求，但缓存仍暂停；normal稳定后才全部恢复。首次通知前是unknown，不等同normal；`memory_pressure_monitor_running=true`只表示监听已运行。无需用系统级压力仿真命令维护缓存。

## 停服与同目录恢复

1. 先让客户端停止新增请求，按上面的空闲与owner条件检查。前台服务用Ctrl-C；进程管理器向自己启动的服务发SIGTERM。信号会停止监听、关闭连接并取消剩余请求，不等待所有业务请求成功返回。
2. 服务在推理线程清理状态，再按同一个默认30秒期限等待SSD IO和callback排空。若日志记录`HTTP prefix cache shutdown completed=true io_completed=true callbacks_completed=true`，该次SSD关闭已完成；超期只停止等待，不能据退出码0声称IO全部完成。GPU同步或正在运行的系统调用不受这个30秒期限强制中断。
3. 等旧进程真正退出，再用相同缓存目录启动。store自动清理自有临时文件、校验并恢复索引；无需手动删除有效归档。复用相同模型、二进制/原生库及数值配置，才有机会命中原namespace。升级或移动checkpoint可能自然miss，按实际cached_tokens与diskHits确认。

在线clear/flush只有库接口，HTTP没有管理员端点。归档损坏会冷退；目录不可用先修复路径/权限/磁盘问题，再有序重启。不要在服务仍持有目录时手动搬移或清空归档。

## 指标与验收口径

Prometheus示例：`qwen_prefix_restores_total`、`qwen_prefix_ssd_restores_total`、`qwen_state_workspace_bytes`、`qwen_ssd_pending_jobs`、`qwen_ssd_foreground_read_intents`。counter按进程/统计reset生命周期比较；未采到的字段可能缺省。SSD read/write bytes是应用层成功归档字节，不是物理SSD流量，也不包括n-gram表读取。

prefill计算、decode round、active service、suspension和网络端到端耗时分开看；AR实际decoded tokens为completion减一。每请求平均decode时间的分位数不是逐token TPOT。联合账本、MLX内存和外部RSS/FD分别观察，不能相加或互相替代。

只读进程观测入口：`.build/release/ane-telemetry --pid PID --output NEW_PATH --interval-ms 30000 --max-samples 240`；填本次服务PID，输出父目录须存在且文件不能已存在。采样以PID和启动时刻校验身份；`process.rusage.physical_footprint_bytes`来自`proc_pid_rusage`的`ri_phys_footprint`，与RSS、MLX内存、逻辑账本分别记录，RSS单独不能代表全部Metal内存。IOReport bin/residency尚未校准，不能倒算物理DRAM GB/s。

截至2026-09-09 06:00，600秒持续淘汰预检已完成；C3的2小时窗口正在单独验收，未有最终完整对账前不记通过。2小时候选门槛与24小时发布配置门槛分开，真实OS压力与24小时混合负载仍未关闭；最新状态以[关键能力计划](KV_CACHE_CAPABILITIES.md)与对应冻结运行证据为准。
