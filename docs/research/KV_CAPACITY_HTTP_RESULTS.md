# 可区分前缀的 HTTP 缓存预检

2026-09-09，`counter-witness` + 显式 `capacity256` 完成独立 608.070 秒工作段。新 fixture、缓存复用、SSD 替换、取消、有限关闭和容量步骤对账通过。库与 HTTP 默认仍为 `reference`；本轮没有 Swift、模型或 kernel 变更，也未启用 MTP。

这是一轮 10 分钟预检。旧 C3 两小时结果继续保留原版本和 legacy 边界；此前 105 分钟计划因用户暂停只完成约 64.823 分钟，仍为未完成，不与本轮拼接。新负载和当前二进制的两小时、24 小时及真实 OS 内存压力验收仍待完成。

## 修复的测试盲区

旧 10 个 fixture 的用户问题都要求从 1 开始计数，输出全部相同，即使错误恢复另一个前缀也可能通过文本比较。新模式让 8 个长前缀分别从 1000 至 8000 起步，2 个短前缀分别从 65000、66000 起步。起始值只出现在系统消息最开头、首个 416-token checkpoint 之前，所有用户问题保持相同。

本轮长提示 11116–11121 tokens，短提示 1346–1348 tokens，输出额度保持 16。10 个真实冷 oracle 首整数全部正确、文本 SHA 全部互异，随后才进入工作负载；末尾被截断的下一个整数按原始正文参与比较，不扩展输出预算。冷/热/SSD 请求继续比较各自的完整正文、finish 和 usage。独立 base analyzer 从事件原文重算判别条件并核对保存的声明，不能只信 summary 的 passed。

这提高跨前缀误恢复的检出能力，不等于验证全部内部状态；完整混合张量对照仍见[模型容量追加结果](KV_CAPACITY_MODEL_RESULTS.md)。HTTP 不提供原始生成 token IDs。

## 本轮结果

| 检查 | 结果 |
| --- | --- |
| 工作段 | 608.070444292 秒，58 成功 + 6 主动取消 |
| 全量唯一终态 | 10 冷 oracle + 58 成功 + 6 取消 = 74；客户端和服务器逐 ID 对齐，无缺失、重复或孤立终态 |
| 冷热覆盖 | 33 次冷完成、25 次缓存完成，全部 10 个 profile 被再次使用 |
| SSD 工作集 | 冷填充累计归档 3,118,655,937 B，为 1 GiB SSD 额度的约 2.90 倍 |
| 工作段 SSD | 读回 6,323,179,520 B，写入 10,915,295,776 B，18 次读命中、36 次淘汰 |
| 持续替换 | 2 个完整 300 秒窗口均有冷/热完成和持续 SSD 读写淘汰 |
| 容量执行 | 68 个完成请求均使用 capacity256；共 1020 个实际 decode 步骤，workspace fallback 0，workspace 峰值预留 549,715,968 B |
| 错误 | 请求、健康检查、归档损坏、写失败均为 0 |
| 最终所有权 | request/workspace、SSD pending jobs/bytes、read intents、live flights 全为 0；保留有效 RAM cache lease 93,523,976 B |
| 退出 | 客户端与服务器 exit 0，采样线程结束，IO/callback 有限关闭完成，controller 无需 TERM/KILL 清理 |
| 独立审计 | base、prefill/decode 阶段分析、capacity terminal 审计全部 complete/passed，issues/errors 为空 |

4 个客户端 worker 不等于 4 个长请求同时驻留。联合逻辑状态额度峰值 1,957,769,172 B，低于 4 GiB 上限。25 次进程采样的 RSS 范围为 36,542,431,232–38,805,553,152 B，数字 FD 为 11–16；RSS 不是唯一物理 Metal 内存，采样范围不能证明长期无泄漏。378 次健康观测的 OS 压力级别均为 unknown，不能算真实压力验收。

600 秒配置的前半段截止与第一个 300 秒周期排空时刻重合，本轮没有中途强制排空；完整两小时配置的定期排空覆盖不能由本次代替。工作段结束后执行最终排空。归档字节是缓存后端计数，不代表物理 NAND 写入或 DRAM 带宽；本轮也不宣称性能提升，prefill/decode 分阶段结果分别保存。

## 复跑与来源

使用[公开 churn 入口](../HTTP_CACHE_CHURN_REPRODUCIBILITY.md)，在唯一 controller 的 case 中给 `scripts/run_http_cache_churn.py` 加 `--duration-seconds 600 --fixture-mode counter-witness --kv-append-mode capacity256`。默认 legacy 继续用于历史复跑；后续发布长测应显式使用可区分模式。统一 wrapper 已包含容量检查，无需维护第二套容量 wrapper。

相关 CPU：parser 14、base oracle audit 10、capacity 10 项通过；原 phase 16、telemetry 6 项控制也通过。新版公开 CLI 离线重审旧 C3 的 755 个终态通过，同时明确其 10 个 oracle 仅 1 个文本 SHA。

本机原始目录为 `results/kv-counter-capacity-smoke-v1/`，含计划、冻结清单、原始事件与服务日志、三份审计及 postflight。运行时二进制 SHA256 为 `31daa9b372634b5a7163e3a99a16957c316d6efe4a0dd589ae7b0ba0eaa9db44`；414 文件 SHA 与 102 模型 payload size/mtime 核对无变化。参考服务恢复 PID98393，精确 argv、idle、MTP/drafter 关闭均在本轮 postflight 重新确认；后续操作先核对最新实际身份。

原始事件 SHA256：`786ed61ea7b66a8761b908fcd89c8d3f9db6c969d89dfe4698c07c7b157bfac7`。服务日志 SHA256：`6f787b968a813e34ee59ec507b4913dc58e3294bcbc1a951fe001e0855bb9c0c`。base 与 phase 报告绑定各自消费的事件/日志字节长度和 SHA；这些原始大文件保留于本机 results，不随 Git 提交。
