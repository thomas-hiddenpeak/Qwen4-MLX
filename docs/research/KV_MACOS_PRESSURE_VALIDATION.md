# macOS 压力通知验证边界

核对日期：2026-09-09。只读本机手册、二进制字符串/导入符号与 Apple 开源代码；未执行 `memory_pressure`，未写入任何 sysctl，未制造实际内存压力。本文不包含新增运行框架或上游代码复制。

## 当前决定

**不在当前共享开发机运行 `memory_pressure -S`。真实 OS 通知链路和真实内存压力验收均仍未关闭。** 已有 policy 注入可验证缓存驱逐、可选写入拒绝、HTTP 429 和恢复行为，但不能记为真实 `DispatchSourceMemoryPressure` 事件或实压验收。

`-S` 不通过分配大量内存制造压力，但它修改全系统状态，并非仅向 runner 发送事件。`-s` 也不是整个命令的硬期限。因此将持续时间缩短到 1–2 秒，仍不能把影响限制在当前服务内。

## 本机证据与版本边界

- `sw_vers`：macOS **26.6.2 / 25G83**。
- `uname -v`：Darwin Kernel Version **25.6.0**，`xnu-12377.161.14~5/RELEASE_ARM64_T6050`。
- 本机 `/usr/share/man/man1/memory_pressure.1`：`-S` 模拟系统的 warn/critical；`-s` 指人工压力维持时间。
- `/usr/bin/memory_pressure` 为 `root:wheel`、普通 `0755`，没有 setuid 位。只读字符串包含 `kern.memorypressure_manual_trigger`、等待和复位提示；导入符号包含 `sleep`、`sysctlbyname`、`exit`，未见 `signal`、`sigaction`、`atexit`。

下列固定 Apple 开源版本提供实现依据，不声称它们逐字对应本机封闭发布构建；本地手册和二进制符号只用于交叉核对可见路径。

| 上游 | 固定提交 | 核对文件 |
| --- | --- | --- |
| Apple `system_cmds` | `408bba7453608006b89772db185defbac8fe2fd0` | [memory_pressure.c](https://github.com/apple-oss-distributions/system_cmds/blob/408bba7453608006b89772db185defbac8fe2fd0/memory_pressure/memory_pressure.c) |
| Apple `xnu` | `f6217f891ac0bb64f3d375211650a4c1ff8ca1ea` | [kern_memorystatus_notify.c](https://github.com/apple-oss-distributions/xnu/blob/f6217f891ac0bb64f3d375211650a4c1ff8ca1ea/bsd/kern/kern_memorystatus_notify.c)、[kern_newsysctl.c](https://github.com/apple-oss-distributions/xnu/blob/f6217f891ac0bb64f3d375211650a4c1ff8ca1ea/bsd/kern/kern_newsysctl.c) |

## 调用路径、权限与恢复

1. 工具的模拟分支固定选择 `TEST_LOW_MEMORY_PURGEABLE_TRIGGER_ALL`，写入全局 `kern.memorypressure_manual_trigger`；大范围 `mmap` 在另一个真实压力分支中。因此“不分配压力内存”成立，“不影响其他应用”不成立。[工具源码 L692–747](https://github.com/apple-oss-distributions/system_cmds/blob/408bba7453608006b89772db185defbac8fe2fd0/memory_pressure/memory_pressure.c#L692-L747)
2. 内核改写全局压力级别，按配置清除 purgeable 对象，并循环通知符合条件的进程。macOS 通知路径还包含清退 idle-exitable 进程的逻辑，模拟模式会略过相应延迟；这里不是任意杀死应用，但已超出本 runner 的边界。[内核触发路径 L1925–2019](https://github.com/apple-oss-distributions/xnu/blob/f6217f891ac0bb64f3d375211650a4c1ff8ca1ea/bsd/kern/kern_memorystatus_notify.c#L1925-L2019)、[macOS 通知路径 L1484–1514](https://github.com/apple-oss-distributions/xnu/blob/f6217f891ac0bb64f3d375211650a4c1ff8ca1ea/bsd/kern/kern_memorystatus_notify.c#L1484-L1514)
3. 该 sysctl 没有 `CTLFLAG_ANYBODY`；通用写权限检查要求 root。不能因工具文件可执行，就把普通用户执行权限等同于模拟权限。本次未尝试提权或写入。[权限检查 L1874–1885](https://github.com/apple-oss-distributions/xnu/blob/f6217f891ac0bb64f3d375211650a4c1ff8ca1ea/bsd/kern/kern_newsysctl.c#L1874-L1885)
4. 首次 sysctl 完成后，工具才 `sleep(-s)`，再显式写 NORMAL；成功才打印复位提示。源码没有信号退出清理。`-s` 不覆盖前后 sysctl 的执行时间；进程中断或复位调用失败时，不能依赖这个参数自动恢复系统。内核人工模式没有与工具进程绑定的租约或到期参数。[工具恢复 L729–744](https://github.com/apple-oss-distributions/system_cmds/blob/408bba7453608006b89772db185defbac8fe2fd0/memory_pressure/memory_pressure.c#L729-L744)、[人工模式复位 L2011–2019](https://github.com/apple-oss-distributions/xnu/blob/f6217f891ac0bb64f3d375211650a4c1ff8ca1ea/bsd/kern/kern_memorystatus_notify.c#L2011-L2019)

## 是否有当前进程专用路径

该工具没有 PID 参数。内核的 `TRIGGER_ONE` 是按内部策略选一个进程，仍修改全局级别，不能当作 runner 专用方式。

Apple 源码另有按 PID 投递的 `kern.memorystatus_vm_pressure_send`，但整个入口受 `DEBUG || DEVELOPMENT` 编译条件限制，且要求 root 或私有 memorystatus entitlement。本机为 RELEASE 内核，不能把它当作当前发布系统可依赖的公开测试 API；本次也未探测写入。[定向入口及编译边界 L2029–2150](https://github.com/apple-oss-distributions/xnu/blob/f6217f891ac0bb64f3d375211650a4c1ff8ca1ea/bsd/kern/kern_memorystatus_notify.c#L2029-L2150)

因此本轮保留受控 policy 注入的验证结论，明确记录真实 OS monitor 覆盖缺口。后续若要补齐，需先具备能够承受全系统通知与可清除缓存变化的隔离环境，并独立验证恢复；不在当前并行工作和模型回归窗口追加全局仿真。
