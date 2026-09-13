# HTTP 请求进度

`GET /health` 现在包含 `request_progress`：每个已进入调度器的请求对应一行，字段为 `request_id`、`stage`、`last_event`、`prompt_total`、`processed_prompt_tokens`、`generated_tokens`。它与 `active_jobs` 使用同一份执行器快照，终态处理后移除，空闲时为 `[]`。

`request_progress_sample: completed_scheduler_slice` 表示最后一个已完成的调度片段。长 GPU 调用仍在执行时，数字可能暂时不变；这不是卡死检测。`processed_prompt_tokens_includes_cache: true` 表示逻辑前缀包含恢复的缓存，不能把它当作实际 prefill 工作量或带宽计算分母。实际计算与 decode 时长继续使用结构化模型终态中的分阶段指标。

进度只从已有调度事件复制整数、请求ID和枚举。健康接口不读取 GPU tensor，也不为进度额外同步设备；网络线程读取受锁保护的值快照。当前调度队列及ready上限使列表最多十项，不包含提示词正文或token IDs。

2026-09-14 的 `results/live-progress-http-v1` 使用 binary `9eb775401b96350a8265dcbbba9d9b13a3c4a1cf96d23ee7cc2a6bc749e0b90c`，36项相关CPU测试通过。真实P11053/O32请求、RAM命中B10816、独立P11058的部分prefill取消以及再次命中原缓存全部通过；226次健康采样检查字段、请求ID、阶段、单调计数、缓存口径及最终清理。三次完整输出一致，结束后列表为空、request/workspace归零。服务正常退出，573个冻结文件和102个模型payload stat复核无变化，参考服务恢复原argv。该短测试不证明最大队列压力或耐久能力，也不作为吞吐基准。

独立审计重新解析原始JSON/SSE、226条health、四个唯一模型终态及输入分词；三次完成各有31次decode，final offset11084，热请求实际计算237 token。取消发生在416/11058的部分prefill，输出计数为0，之后对应ID消失。最终保留366366728B有效RAM及一个cache lease，日志排空。审计 `independent-progress-audit.json` 的SHA256为 `2f19b7c53f2f85e72fcc3bee2c00f40e36b7b6d74030d64a8b61deed5c7b326d`。217个busy样本的旧`running_job`仍为null且known=false；本改动提供新字段，未修复或重新定义旧字段。

可对一个新启动、空RAM、reference attention、关闭SSD/paging的自有测试服务运行：

```sh
python3 -B scripts/probe_http_live_progress.py \
  --runner "$PWD/.build/release/ane-runner" --model-dir /ABS/MODEL \
  --server-pid OWNED_PID --server-log /ABS/SERVER.log \
  --port 11250 --output /ABS/NEW_RESULT_DIRECTORY
```

服务需要至少16384上下文、805306368字节RAM缓存及两个条目；本次使用8GiB联合状态预算和300秒连接期限。客户端仅拥有自己的socket，拒绝参考端口11235，不负责启动或终止服务。它有180秒工作期限；调用者仍须管理服务退出和资源恢复。实际运行的客户端保留在上述结果来源目录，公开脚本仅调整仓库根路径解析以适应 `scripts/` 位置。
