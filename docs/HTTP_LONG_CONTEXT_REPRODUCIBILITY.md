# HTTP 长上下文与 RAM cache 复跑

这些入口提供显式容量配置和有界验证流程，不会修改普通 `serve-gpu` 的默认16384上下文，也不代表目标机器已经通过验收。先完成相同模型、二进制和 attention 策略的 CLI 数值/容量检查，再运行 HTTP；不要同时常驻两个完整模型。具体实验结果应与对应原始报告一起记录。

## 前台262K配置

模型目录为必填参数，端口默认为11236：

```sh
python3 -B scripts/run_long_context.py --model-dir /ABS/MODEL --port 11236
```

脚本按自身位置寻找本仓库的 `.build/release/ane-runner`，不执行构建。它通过 `os.execv` 替换当前进程；进入推理后不再保留 Python 进程，前台信号直接交给 Swift runner。仅接受模型目录和端口，没有覆盖固定配置的额外参数尾部，也不会停止或重启其他服务；11235被排除。

固定配置为 reference prefill/reference KV append、总上下文262144、调度 token 配额262144、1个 resident、8MiB body、3600秒连接总期限、24GiB联合状态额度、8GiB RAM prefix cache/最多2条。SSD和物理页池关闭；长上下文服务策略拒绝 MTP。原有15秒未完成请求接收和在途发送期限仍生效。

262144是 prompt+请求输出预算之和。满长请求会占满调度 token 配额，新短请求可能收到429；24GiB只是状态记账，不含权重、一般 attention 激活、MLX allocator 常驻和 HTTP/分词对象，不能作为RSS硬上限。3600秒是期限，不保证冷 prefill 一定完成。完整262K状态不能通过当前2GiB归档合同持久化；本配置只验证RAM cache。

## 准备真实 chat fixture

从仓库目录执行，以下所有输出目录必须尚不存在。原生 `tokenize` 只运行CPU分词；固定同一个二进制用于准备和随后服务启动。

```sh
python3 -B scripts/probe_http_long_context.py prepare \
  --runner /ABS/ane-runner/.build/release/ane-runner --model-dir /ABS/MODEL \
  --context-limit 262144 --output /ABS/NEW_BASE_FIXTURE
```

也支持显式32768/65536/131072初步检查。每个 context 的合法输入为 P=context−2/O2，越界输入为 P=context−1/O2。262144情况下即P262142/O2与P262143/O2，最终RAM检查点262080、热请求计算62个prompt tokens。三个很小的全chat分词样本先验证重复单位，然后只执行两次完整长度分词；另有独立约32K取消输入和小于416tokens的短恢复输入。

每份 fixture 保存真实单条user消息、完整原生渲染/解码、原始token IDs、请求和哈希、模型config/tokenizer/chat template与二进制哈希。实际HTTP body大于旧1MiB ceiling，但仍必须通过8MiB配置。客户端重新读取原始IDs和输入内容核对计数及绑定，不能用单独声明的token数量代替分词证据。

可选增加更长的decode检查，仅适用于上述262144 base：

```sh
python3 -B scripts/long_context_decode_fixture.py \
  --runner /ABS/ane-runner/.build/release/ane-runner --model-dir /ABS/MODEL \
  --base-fixture /ABS/NEW_BASE_FIXTURE --output /ABS/NEW_DECODE_FIXTURE
```

该转换减少22个重复单位，并缩短末尾数数指令；一次完整Swift分词必须得到P262112，真实共同前缀至少262080。结果保存为独立目录，不修改base。读取时再次验证整份新prompt/原始IDs/HTTP消息和模型/二进制绑定。不能直接删除30个重复单位：尾部会提前进入前缀，未必还匹配缓存点。

## 独立测试端口上的完整HTTP case

外层实验控制器负责安排独占资源及参考服务生命周期。下面的wrapper只管理它自己创建的测试服务和客户端，默认端口11249；不触碰11235。

```sh
python3 -B scripts/run_http_long_context.py \
  --runner /ABS/ane-runner/.build/release/ane-runner --model-dir /ABS/MODEL \
  --fixture /ABS/NEW_BASE_FIXTURE --decode-fixture /ABS/NEW_DECODE_FIXTURE \
  --prefill-attention reference --port 11249 --output /ABS/NEW_HTTP_RUN
```

不需要长decode扩展时省略 `--decode-fixture`。`--prefill-attention` 必填；`fusedQSA`只适用于另行通过所需数值检查的实验，不因本入口而成为默认。可选 `--paged-kv-pool-library /ABS/VALIDATED.dylib` 显式配置每层512页；长请求必须整游标dense回退，不能把已配置页池当成真正使用物理页的证据。

wrapper根据fixture选择context和相等的调度token配额，其他资源配置与前台profile一致。它先核验真实模型文件/runner hash，再启动单个服务并等待相同PID的ready health（最多240秒）。单请求客户端和连接期限均为3600秒；整个owned case最多4800秒，并预留40秒统一清理。因此整体期限可能截断多个请求分别耗尽3600秒的组合。保存完整argv、PIDs、初始/最终health、Prometheus、日志/输入/脚本哈希、子进程退出状态；清理仅TERM自己拥有的子进程，必要时在期限内强杀并回收。强杀、异常退出或缺失最终监控都不能通过。

如果外层已拥有一个全新、配置相符的测试服务，可只运行客户端：

```sh
python3 -B scripts/probe_http_long_context.py run \
  --runner /ABS/ane-runner/.build/release/ane-runner --fixture /ABS/NEW_BASE_FIXTURE \
  --decode-fixture /ABS/NEW_DECODE_FIXTURE \
  --server-pid PID --server-log /ABS/SERVER.log --port 11249 \
  --prefill-attention reference --paged-pages 0 --output /ABS/NEW_CLIENT_OUTPUT
```

只有服务显式配置512页时才传 `--paged-pages 512`。客户端验证health PID但不会启动或停止服务；直接模式的实际服务启动/模型文件出处仍由外层控制器负责。Python必须关闭优化（不使用`-O`/`PYTHONOPTIMIZE`），因为复用的HTTP edge helper含断言。

## 验证范围

1. 在全新RAM-only服务上验证精确越界400、MTP400和声明body超限413。越界400必须包含真实context校验原因；等待日志排空后要求没有新增model terminal，状态/cache admission计数和可选native操作未变化。这支持模型admission前拒绝，不代表过长输入避免CPU分词，也不是所有GPU操作的全局追踪。
2. 保存短AR基线，再运行长输入cold JSON和warm完整SSE。严格核对MIME/身份/choices/结束帧/usage、完整wire body及response hash；两次都必须输出2tokens并产生1次实际decode，实际final state offset为262143，热请求必须从RAM恢复262080并只计算62tokens。输出正文和finish一致；各prefill/decode/handoff成本分开保存。
3. 启用可选扩展时，先执行P262112/O32的JSON请求，必须从RAM恢复262080、计算32个prompt tokens、实际decode31次，实际offset262143。自然提前EOS会失败，不能缩小输出预算通过。
4. Admit独立约32K prefill并观察resident ownership，读取SSE role后RST该client，要求唯一对应的cancelled/prefill终态和资源回收，然后同一短请求恢复原正文、usage与finish。
5. 启用扩展时，在取消和短恢复之后再次执行P262112/O32完整SSE，要求同样的RAM恢复/计数/offset。两次32输出的完整正文、finish与usage必须一致，从而覆盖大RAM快照在其他请求取消后继续可用。HTTP这里不提供输出token IDs，没有新prompt的独立cold oracle，不据此声明逐token数值或质量验证。

每次settled检查要求jobs/token reservation/request leases/cache flights归零，RAM缓存字节与ledger一致。关闭paging时workspace为0；配置paging时只能保留由VM页大小独立重算的12层固定arena预留，并要求claims/in-flight/failed operations为0、完成操作守恒。本fixture的缓存均为dense，native live pages也必须为0。RAM快照允许在idle保留；MLX峰值是累计值，不是各阶段独立峰值。

`summary.json`、实际请求/响应、唯一对应的终态日志、health快照和`owned-server.json`共同构成这一次有界证据。任一字段缺失、错误checkpoint、输出不足、取消竞态或资源不归还都应保持失败。约32K取消不等于完整262K取消；一次冷热与31次decode也不等于长期稳定性、速度收益、完整质量或262K SSD支持。CLI的121状态/数值检查与这里的HTTP验收应分别记录。
