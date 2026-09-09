# 本机实验控制器的所属进程与退出合同

[run_specialization_experiment.py](../scripts/run_specialization_experiment.py) 负责核对并暂停已登记的空闲参考服务，串行执行本机实验，最后恢复原参数。新增 [owned_process_group.py](../scripts/owned_process_group.py) 只管理本次创建的一组进程，不是通用作业系统。

```bash
python3 scripts/run_specialization_experiment.py /absolute/path/controller-plan.json
```

执行计划的 `cases` 必须是含 `name`、`command` 的平铺列表，`reference_ledger` 指向最近一次已核验恢复记录，输出目录不能已有 run-ledger。MTP 的嵌套窗口分析计划含 task/budget/processes，不能直接执行；控制器会在暂停参考前拒绝这种结构。运行前还核对参考 PID/精确 argv、状态文件、空闲 metrics 和关闭 MTP/drafter/PLD 的参数。不要绕过单 GPU 所有者的约定启动第二个控制器。

2026-09-10 起，常驻 mlx-serve 采用[业务配置](MLX_SERVE_SERVICE.md)：原生 262144-token 上限和热前缀缓存。新计划从 `../qwen38-ssd/results/experiment-status.json` 的 `reference_ledger` 读取当前记录，再核验其 ready/PID/argv；每次实验恢复都会更新此指针。历史 4096-token、关闭缓存的 argv 仅是旧基线，不得用来覆盖当前服务。实验 case 如需禁用缓存，应在该 case 的独立进程中配置，结束后仍恢复业务服务原参数。

## 所有权与中断顺序

每个 case 用 `Popen(start_new_session=True)` 创建独立 session/process group，立即记录 Popen PID 与相同的 PGID。清理只向这个新组发送信号，不使用 controller 的父进程组，不按进程名查杀，也不从历史报告取 PID 发信号。case 的模型后代必须保留该组，不能自行 setsid/daemonize；当前 Python HTTP harness 的模型子进程遵守这一约定。

SIGTERM/SIGINT handler 仅记录停止 flag。case 的等待每次最多 1 秒，在 Popen 与 PGID 已记录后的安全点检查 flag 并转入 finally，避免在进程已创建、句柄尚未赋值时抛异常。重复信号不会打断清理或参考服务恢复；实际 GPU/SSD 操作是否及时停止仍由子进程自身决定。

清理先给自有整组 SIGTERM，最多等待 **45 秒**，覆盖 HTTP harness 自身的 30+10 秒退出过程。仍有残留时向同一组 SIGKILL，再最多等 **10 秒**。leader 退出不等于整组退出：控制器会 reap 自己的直接子进程，并继续检查同组后代。正常完成的 case 也要完成这项检查。`finish` 成功或失败的结果都会保留，重复调用不会再次向可能已失效的 PGID 发信号。

**只有确认整组已不存在，才释放该组所有权并恢复参考服务。** 无法确认时，ledger 记录 `restoration.ready=false` 与 `blocked_by_owned_group`，状态标记 `reference_restore_blocked_by_owned_group`，拒绝加载第二份模型。超时不会因为子进程最终以 0 退出而被算作成功实验。

恢复参考使用另一个 `Popen(..., start_new_session=True)`，发生在实验组清理之后；它不属于已清理的 case 组。仍存在的原参考须重新匹配精确 argv 并核对 `/v1/models` 的 ready 条件，不能只凭 PID 存在写 ready。新参考最多等待 90 秒核对 MTP/drafter 未加载。后续实验仍须重新核对最新 ledger、状态文件与实际进程。

## 已验证的范围

2026-09-07，[4 项 CPU 进程测试](../scripts/test_owned_process_group.py)全部通过：leader 先退出但后代仍在、TERM 清理宽限、强杀自有组时另一独立对照组仍存活、重复 finish 不再发送信号。

同日真实控制器中断 smoke 使用 CPU-only case，确实暂停并恢复已有参考，但没有运行实验模型推理。controller13573 收到 SIGTERM；case PGID13605 的 leader 先退出，child13606 收到 TERM 后延迟 6 秒完成，整组 **6.0846 秒**清空，未使用 KILL。ledger 的 case 结束在北京时间07:35:39.347468，参考13611 随后于07:35:39.357111启动，07:35:54.598390核对 ready；启动方额外核对精确 argv、独立进程组及 `127.0.0.1:11235` 监听 PID。总中断到控制器退出 **22.333 秒**。controller exit1 / InterruptedError 是预期的中断标志，不是恢复失败。

本机证据为 `results/controller-interrupt-v1/smoke.json`、`run-ledger.json`、`controller.log`、case/child ready 与 stopped 记录；这些运行产物不随源码提交。当前阶段的最新已验证参考 PID/ledger 见 [接续进度](AUTONOMOUS_PROGRESS.md)，文档中的旧 PID 不能当作后续信号目标。

## 实际边界

SIGKILL 无法被 Python 捕获；控制器被强杀、宿主崩溃或断电时，finally 不能保证执行。这种情况须先核对已登记实验进程是否仍在、参考是否已恢复，再继续工作，不能直接载入另一模型。外层启动方应发送 TERM/INT 并留足清理与恢复时间，不能仍在 5 秒后强杀 controller。

一次 smoke 不证明所有卡死或设备故障都能恢复。整组残留、无法核验身份、恢复未 ready 都应明确保留失败状态；本轮未构造不可杀死的进程或模拟断电。该控制器不支持模型子进程主动脱离所属组，也不负责停止无关训练或服务。
