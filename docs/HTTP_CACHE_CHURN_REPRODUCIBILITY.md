# 复跑本机 HTTP cache churn

这是已验证C3配置的运行与离线审计入口：160 MiB RAM缓存、1 GiB SSD归档、4 GiB联合状态额度，8个长前缀、2个短前缀、4个客户端worker；AR、416-token checkpoint网格，客户端token额度30000。它不能证明4个长请求并行驻留，也不包含24小时模式。已有7205.783秒结果及剩余门槛见[缓存可靠性](KV_CACHE_RELIABILITY.md)。

历史 C3 使用的 `legacy` fixture 的 10 个冷 oracle 虽然输入不同，16-token 输出文本 SHA 却相同。该结果主要验证持续生命周期、资源与既定文本一致性，对跨前缀误恢复的辨识有限；需结合完整混合状态对照，不能独立称为隔离正确性验收。`legacy` 仍是默认值，用于历史复跑，不追溯改变已冻结版本的通过口径。

新测试可显式选择 `--fixture-mode counter-witness`：每个前缀独有的起始整数只放在系统头部、首个 416-token checkpoint 之前，用户问题保持相同，输出额度仍是 16 tokens。进入 soak 前必须取得完整一组冷 oracle，实际正文以对应整数开头且所有正文 SHA 互异；缺失、重复或错误均停止整组测试。离线 base analyzer 从事件原文重新计算这些条件。它提高跨前缀误恢复的识别能力，仍不能代替全部混合张量的逐位对照。

2026-09-09，该模式与显式 capacity256 完成[608 秒真机预检](research/KV_CAPACITY_HTTP_RESULTS.md)：74 个唯一终态、两段持续 SSD 替换窗口及有限关闭全部通过。新配置的两小时、24小时结果尚未完成。

所有命令从仓库根目录执行。需要已构建的`.build/release/ane-runner`、现有`scripts/probe_http_cache_churn.py`及其reliability helper，模型位于仓库的`../qwen38-ssd/models/Qwen3.8-Flash-Next-MLX-SSD-Stream`。wrapper保留这一已测布局，尚无`--model-dir`参数；不要将它当作任意目录的通用安装器。

## 由既有controller运行

仅在本机没有其他实验controller运行时安排此case。`run_http_cache_churn.py`只管理自己启动的服务/客户端，由`scripts/run_specialization_experiment.py`提供外层独占模型与参考服务恢复；不要在参考模型仍占用资源时单独启动wrapper。controller需要当前ready参考服务的ledger和`../qwen38-ssd/results/experiment-status.json`，会核对11235端口的参考身份、空闲状态及关闭MTP/drafter的参数。

创建一个新的结果目录，保存`plan.json`。将下列`reference_ledger`填为当前有效ledger绝对路径；端口选择新的空闲端口。`cases[].command`里的相对路径由controller在仓库根目录解释。

```json
{
  "schema": "qwen-specialization-experiment-v1",
  "reference_ledger": "/absolute/path/to/current-ready-run-ledger.json",
  "cases": [{
    "name": "churn",
    "command": ["python3", "-B", "scripts/run_http_cache_churn.py",
                "--output-directory", "results/kv-churn-repro",
                "--duration-seconds", "7200", "--port", "11248"],
    "timeout": 8100
  }]
}
```

```sh
python3 -B scripts/run_specialization_experiment.py results/kv-churn-repro/plan.json
```

先做600秒预检时只改本case的duration和结果目录/端口，确认测得的distinct archive工作集为SSD额度的2–4倍，再另建7200秒case；不要往同一目录续跑。wrapper最多接受7200秒，收到取消后自身清理共用40秒期限，外层controller有45秒TERM宽限。controller每case检查二进制SHA；源码/native/模型payload冻结与事后核对仍需保存独立manifest，不能把plan中的描述字段当作controller已自动验证。

测试新 fixture 时给 wrapper case 增加 `--fixture-mode counter-witness`。测试容量追加候选时再增加 `--kv-append-mode capacity256`；默认 `reference` 保留原服务参数，未切换运行时默认。容量模式在 health 中核对 AR 专属策略，并在服务正常关闭后生成 `capacity-terminal-validation.json`，逐请求检查实际 capacity 步数、workspace/fallback 及取消终态字段；这些检查补充原输出、缓存与生命周期审计。不要绕过外层 controller 单独加载第二个模型。新模式的真机验收结果须以对应独立运行目录为准。

## 完成后顺序分析

先确认`churn.json`与`lifecycle.json`的complete/passed均为true、日志停止追加，controller的run-ledger记录参考恢复ready。所有分析输出只能新建，已有同名文件时选择新版本名。

```sh
python3 -B scripts/analyze_http_cache_churn.py \
  --events results/kv-churn-repro/churn.events.ndjson \
  --server-log results/kv-churn-repro/server.log \
  --process results/kv-churn-repro/process.ndjson \
  --summary results/kv-churn-repro/churn.json \
  --lifecycle results/kv-churn-repro/lifecycle.json \
  --output results/kv-churn-repro/analysis.json
python3 -B scripts/analyze_http_cache_churn_phases.py \
  --audit results/kv-churn-repro/analysis.json \
  --events results/kv-churn-repro/churn.events.ndjson \
  --server-log results/kv-churn-repro/server.log \
  --output results/kv-churn-repro/phase-analysis.json
```

第二条仅在第一份报告complete/passed=true、issues={}、两项unmatched计数为0后执行。公开base CLI默认对未完成证据返回非零；仅需预览时显式加`--allow-incomplete`，无issues的预览可以exit 0，但报告仍`complete=false, passed=false, cli_preview_only=true`，不能进入phase验收。phase analyzer重新校验events/server路径、字节长度和SHA，所以两次执行间不能继续追加日志。

验收必须同时包括：唯一JSON模型终态与success/取消逐请求对应、usage/content SHA/finish匹配每个profile的oracle、持续SSD读写淘汰、所有profile及冷热请求覆盖、最终request/workspace/SSD pending归零、有限close完成和采样线程退出。公开base CLI完整证据还检查final health的foregroundReadIntents/liveFlights=0；已提供lifecycle时检查同PID服务、close_completed_logged、client/server退出0及sampler完成/无错/样本数相符。RAM保留cache lease正常；不能从某个中途drain推断全部owner结束。summary/lifecycle先读成有界快照，实际消费的字节长度/SHA与events/server/process一起写入inputs；ledger仍需归档时另作指纹绑定。

phase按oracle/soak、profile、cache_source分组。AR decoded_tokens=completion−1；prefill、decode round、service、suspension与handoff分别看。请求mean decode时间的p50/p95不是逐token TPOT；模型first-ready不是socket TTFT。HTTP只暴露文本及usage，不能宣称原始token IDs逐个比对。

## 可选的partial进程遥测

wrapper不会自动启动`ane-telemetry`。需要采样时，由本次实验拥有者用已确认的服务PID单独安排只读sidecar；父目录需存在，输出文件不能已存在：

```sh
.build/release/ane-telemetry --pid PID --output results/kv-churn-repro/telemetry.ndjson --interval-ms 30000 --max-samples 180
python3 -B scripts/summarize_http_cache_churn_telemetry.py \
  --telemetry results/kv-churn-repro/telemetry.ndjson \
  --process results/kv-churn-repro/process.ndjson \
  --lifecycle results/kv-churn-repro/lifecycle.json \
  --expected-pid PID --output results/kv-churn-repro/partial-telemetry-summary.json
```

PID为占位符，填写本次服务真实PID。180个30秒采样是最多90分钟，从运行中途启动时只覆盖partial窗口；摘要只使用同PID/启动时刻且仍存活的libproc样本。footprint与resident RSS独立于逻辑账本；原wrapper的FD/ps RSS使用自己的时间窗，两种时钟没有对时锚点，不强行拼成同一partial窗口。采样最大值不是瞬时真实峰值，端点变化不证明无泄漏；IOReport未校准bin不能倒算DRAM GB/s。

## 仅CPU控制

```sh
python3 -B scripts/analyze_http_cache_churn_phases.py --self-test
python3 -B scripts/summarize_http_cache_churn_telemetry.py --self-test
python3 -B -m unittest discover -s scripts -p test_http_cache_churn_parser.py
python3 -B -m unittest discover -s scripts -p test_http_cache_churn_oracle_audit.py
python3 -B -m unittest discover -s scripts -p test_capacity_http_validation.py
```

前两项分别是 16 项和 6 项保存文件的虚拟控制；parser 为 14 项，新增 base oracle audit 为 10 项，capacity 为 10 项，全部不加载模型。base analyzer 没有内置 self-test，wrapper 没有可安全导入的测试入口。base 的原逐请求对账与阶段计时口径保留，新增从原文重算的 oracle 判别检查及声明核对；当前公开 CLI 已对历史 C3 的 755 条唯一终态重审通过，并如实记录 legacy 的 10 个 oracle 只有 1 个文本 SHA。CPU 通过不能代替真实 GPU、服务生命周期或新二进制上的长时验收。
