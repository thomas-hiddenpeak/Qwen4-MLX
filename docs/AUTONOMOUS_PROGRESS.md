# 自主研究与开发接续

用户在2026-09-07凌晨授权至少八小时自行推进，研究vLLM、SGLang和Redis作者的runner，吸收合适特性，并阶段性commit、推送GitHub。首轮工作窗口截至北京时间2026-09-07 13:30（UTC05:30）；到期完成在途实验的收尾、恢复参考服务并整理成果，不再自动启动新实验。

## 当前执行约定

- 独立仓库：`experiments/ane-runner`，远程`thomas-hiddenpeak/Qwen4-MLX`；外层`coreai-models`是另一个仓库。仅提交本项目源码、测试、文档和小型样本，权重、构建与大实验结果保持忽略。
- 用当前工作分支做阶段提交并推送，不强推、不替换已有历史。每次接续先看Git状态及本文件，接手已有工作，不重复开同一项。
- 单个GPU实验所有者；研究与CPU工作可并行，模型加载及GPU测试串行。使用既有控制器核对参考服务身份和空闲请求，测试结束恢复原参数；不停止无关训练或其他项目。
- Prefill/decode分别计时；普通AR与MTP分别比较。现有默认不因局部微测收益自动提升。先跑小数值门槛，再做真实11k完整生成；保存没有收益的结果。
- MTP稳定性、取消、状态一致性和服务接口是缓存开发的前置条件。前缀缓存必须包含Attention KV、QSA、GDN、PLE/n-gram及MTP适用状态，不能仅缓存KV就宣布可复用。
- 优先采取局部可验证改动。引用原始项目文档、代码版本与许可；借鉴设计和复制实现分别说明。

## 当前状态

已推送`9a60fbd`到`codex/moe-composition`。上一轮专家+归约组合330项局部比较、6轮11k生成和5组边界回归通过；单层+5.21%，完整prefill828→824 token/s，保持可选。最新完整记录见[组合回归](MOE_PREFILL_COMPOSITION.md)。

初始参考服务：PID97647，`http://127.0.0.1:11235`，MTP/drafter关闭。最新恢复ledger为`results/moe-prefill-composed-boundaries-v1/run-ledger.json`。PID只作为本次起点；执行前必须与`../qwen38-ssd/results/experiment-status.json`及实际进程重新核对。

## 进行中的工作

1. 研究vLLM/SGLang：分阶段调度、混合状态管理、MTP验证/回滚、前缀缓存、取消与背压，产物`docs/research/VLLM_SGLANG.md`。
2. 核实Redis作者runner并读代码，产物`docs/research/REDIS_AUTHOR_RUNNER.md`。
3. GDN decode qkv的载入流水实验：保持BF16逐标量累加和归约顺序，只改K块载入；独立构建，不替换原运行库。先测真实四层矩阵，再决定是否进入完整decode。
4. 汇总为有依赖顺序的吸收计划，明确现在可做、需要改造和暂缓项；选择最小高价值项实施并回归。

## 接续记录

- 05:20左右：分派三路研究/实现；参考服务保持运行；GDN agent只允许独立编译，尚未获得GPU运行权。
