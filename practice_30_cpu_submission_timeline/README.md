# Practice 30：单请求的 CPU 下发时间线

**这一轮把“CPU 一直在提交工作”和“CPU 等设备结果”分开观察。** 使用完整原生请求，保留调度、KV 增长和结果回收；没有起跑门、第二个请求或人工改 stream。

入口：[离线交互报告](report/index.html) · [发现与证据](RESULTS.md) · [完整请求 SVG](report/request.svg) · [decode 32 SVG](report/decode32.svg)。

现已补充可切换的 **eager / PIECEWISE graph**：[对照报告](report/comparison/index.html) · [graph 发现](GRAPH_RESULTS.md) · [graph 时间线](report/graph-01/index.html) · [新 eager 对照](report/eager-02/index.html)。无 profiler 参考请求中位数分别为 725.879 / 291.454 ms；graph 每个 decode 有 25 次 replay，26 条物理 stream 上未观察到计算重叠。以下首轮链接保留原始 eager 结果。

继续细分：[forward 报告](report/forward/index.html) · [细分结论](FORWARD_RESULTS.md) · [单次 RoPE 时间线](report/forward/rope.svg)。已有 trace 显示 graph forward 的图外 attention Host 范围约占 66.6%；新增观测只放大一次 eager RoPE，明确记录插桩扰动。

下一轮已完成：[单次 graph 图外 attention 报告](report/attention/index.html) · [结论](ATTENTION_RESULTS.md) · [精确时间线](report/attention/attention.svg)。只在 decode 32 第一层记录上下文、KV 准备/调用、FIA 参数与调用，保持原生请求和同步逻辑。

FIA 继续细读：[提交时间、参数与 binary 选择](fia_probe/README.md)。原模型 trace 确认入队到设备开始为 37.511 µs；独立最小探针核验 `.o` 字节哈希 → binary/入口/function handle → launch。KV 长度 42→43 保持同一入口，BF16→FP16 更换文件与入口，两组证据分开解释。

## 怎么读

1. 先看完整请求的 64 个步骤：prefill 一次，decode 63 次。
2. 选择 decode 32，横向阅读 CPU 阶段、PyTorch host、Enqueue、下发线程 Dequeue、CANN launch、NPU。
3. 点击“输入 H2D / 线性层 / RoPE / token D2H 与等待”，查看精确关联和实际源码入口。
4. 看阶段表中的墙钟与线程 CPU 时间，再看三个最大设备任务空隙。不要把父子阶段时长相加。

报告保留所有步骤，CPU 非设备算子表和四个细读实例集中在 decode 32。图中灰白空白不是 CPU/硬件 idle 的证明；CPU 调用范围也不代表线程始终运行。

## 工作负载与代码主流程

Qwen2.5-0.5B-Instruct，BF16，单 Ascend910B2C，eager；固定 10-token 输入、64-token 贪心输出，`ignore_eos=True`、`logprobs=1`。TP=1、batch=1、max model len=256、block size=128、KV budget=16 MiB。prefix cache、chunked prefill、async scheduling 关闭，`distributed_executor_backend=uni`、`VLLM_ENABLE_V1_MULTIPROCESSING=0`。

这是 `LLM.generate()` 的 in-process 引擎路径，不含 HTTP、跨进程服务通信或并发调度。分词在正式计时前完成，输出处理属于请求范围。当前配置仍存在 torch-npu 自己的下发线程。

| 文件 | 作用 |
|---|---|
| [run.py](run.py) | 冻结脚本、记录源码/版本、按顺序管理三个自有子进程，执行恢复检查 |
| [child.py](child.py) | 原生初始化与预热，运行 reference / diagnostic / recovery |
| [observer.py](observer.py) | 在实际对象上临时包装 17 类阶段，记录嵌套、线程与双时钟；退出恢复 |
| [graph_observer.py](graph_observer.py) | 仅 graph diagnostic 启用：初始化期间记录 capture/dump，正式请求逐次记录 replay 与 task update 入口；退出恢复 |
| [analyze.py](analyze.py) | 阶段自身时间、队列关联、同步边界、decode 32 三个最大空隙 |
| [exact_join.py](exact_join.py) | 从 Practice 26 复用精确 flow/CSV 关联，适配本轮输入并导出队列证据 |
| [render.py](render.py) / [viewer.html](viewer.html) | 生成两张 SVG 和离线可点击报告 |
| [compare.py](compare.py) | 核对跨模式的版本、源码、配置、输入和输出，生成同口径对照 |
| [forward_analysis.py](forward_analysis.py) | 按同线程包含关系整理 forward 的非重叠范围、保留余量，再关联单次 RoPE |
| [forward_observer.py](forward_observer.py) | 只选中 eager decode 32 的第一次 RoPE，临时包装 Python/JIT/binder/native 四段并恢复 |

`child.py` 只复用 Practice 28 的 `make_engine`、配置、prompt 和 `generate`，不调用其 capsule、快照恢复或双流提交。模型计算由原生引擎驱动。

```mermaid
sequenceDiagram
    participant M as CPU 引擎线程
    participant Q as torch-npu 主机队列
    participant W as CPU 下发线程
    participant N as NPU 当前流
    M->>M: schedule → 更新状态 → 准备输入
    M->>Q: 输入 H2D、forward 算子入队
    Q->>W: Dequeue
    W->>N: CANN launch
    par CPU 继续
        M->>M: 后续层、参数/形状处理、logits、采样提交
    and NPU 执行
        N->>N: 执行已提交任务
    end
    M->>Q: sampled token D2H、Event record
    Q->>W: Dequeue
    W->>N: 提交回传和完成标记
    M->>M: 原生 Event synchronize
    N-->>M: 达到完成边界
    M->>M: logprob 回传 → scheduler update → output processing
```

这是调用关系示意。实际下发、完成和等待时刻以报告为准。

## 计时与干预边界

每个阶段保存 `perf_counter_ns()` 和 `thread_time_ns()` 增量，所有记录在 profiler 停止后统一写出。线程 CPU 时间包含 Python/C++ 及可能的自旋，不能理解为“有效业务计算”；墙钟减线程 CPU 不能直接归因为 GIL、锁、设备等待或 OS 抢占。

同线程直接子阶段从父阶段扣除，得到自身墙钟和自身线程 CPU 时间；跨线程不相减、不相加成请求总时长。阶段计时位于 `record_function` 边界内，因而与图上的 profiler 标记时长略有差异。标记、包装及 profiler 自身也有开销。

阶段观察器只在预热后安装，保存原 callable，并在异常和正常退出时都恢复。`EngineCore.step_fn` 缓存了原 bound method，因此包装这个实际被调用的属性；没有仅替换未被调用的 `step`。eager 的 `LogprobsTensors.tolists` 是唯一类级包装，其余是本实验引擎实例属性。graph diagnostic 另在初始化前临时包装 `NPUGraph.capture_begin/end/replay` 与两个 task update Python 入口；capture dump 在初始化完成捕获后写出，请求内只缓冲记录。未删除或新增设备等待，不在热路径 `.cpu()`、复制 KV 或打印日志。

`exact_join.py` 复用 P26 关联检查，新增本轮格式适配、原始 queue 范围导出、可测试的步骤检查；对已经验证为嵌套的阶段使用逆序查找最内层范围。优化前后完整分析证据逐项相同。CPU/NPU 实际时间均来自同一远端 profiler 时钟；不会拿本地时间与远端时间相减。

## 远端复现

保持仓库同级 Practice 17、28、29 的目录布局。P28 提供原生初始化及配置，P29 提供有界子进程控制，P17 提供离线区间辅助函数。

```bash
ssh ascend910
cd /data/tianchi
python -B practice_30_cpu_submission_timeline/run.py \
  --output practice_30_cpu_submission_timeline/results/new-run
python -B practice_30_cpu_submission_timeline/analyze.py \
  practice_30_cpu_submission_timeline/results/new-run
python -B practice_30_cpu_submission_timeline/render.py \
  practice_30_cpu_submission_timeline/results/new-run \
  --output practice_30_cpu_submission_timeline/report/new-run
```

输出目录必须不存在；默认 `--mode eager`。新增 graph 模式使用 Practice 28 的原生 `settings('graph')`：`mode=3`、`cudagraph_mode=PIECEWISE`、`cudagraph_capture_sizes=[1]`、`custom_ops=['all']`。不开放额外 workload 矩阵。三个隔离进程分别为：

- reference：预热 2 次，3 次无 profiler 请求。
- diagnostic：预热 2 次，1 次带阶段观察器的 CPU/NPU profiler 请求。
- recovery：新进程预热 2 次，1 次原生请求。

设备已有进程时拒绝开始。eager 每个子进程上限 300 秒，graph 为编译/捕获留出 900 秒；超时仅终止该自有进程组，先 TERM、再有界 KILL，不 reset NPU。控制器在异常路径也尝试对应模式的原生恢复。安装源码、配置和已有服务不修改。

最小双模式复现（顺序执行，不同时争抢 CPU/NPU；输出使用新目录）：

```bash
cd /data/tianchi
for mode in eager graph; do
  python -B practice_30_cpu_submission_timeline/run.py \
    --mode "$mode" --output "practice_30_cpu_submission_timeline/results/repro-$mode" || break
  python -B practice_30_cpu_submission_timeline/analyze.py \
    "practice_30_cpu_submission_timeline/results/repro-$mode" || break
  python -B practice_30_cpu_submission_timeline/render.py \
    "practice_30_cpu_submission_timeline/results/repro-$mode" \
    --output "practice_30_cpu_submission_timeline/report/repro-$mode" || break
done
python -B practice_30_cpu_submission_timeline/compare.py \
  practice_30_cpu_submission_timeline/results/repro-eager \
  practice_30_cpu_submission_timeline/results/repro-graph \
  --output practice_30_cpu_submission_timeline/report/repro-comparison \
  --eager-report ../repro-eager/index.html --graph-report ../repro-graph/index.html
```

graph 的启动编译/捕获记录与稳态请求分开保存。实际启用依据是 dump、逐次 replay scope、CANN connection、设备 model/stream/task 序列与完成边界，不仅看配置。prefill 仍无 replay；图内 kernel 关联到所属 replay，图外任务保留自己的提交链。图模式 SVG 按物理 stream 分行，跨流连线表示所属 replay 的关联，不伪造每个 kernel 的独立 launch 或完整 notify ID。

本轮远端原始目录为 `/data/tianchi/practice_30_cpu_submission_timeline/results/round-01`。源码从远端实际安装路径读取；本地旧快照未用于代替远端检查。运行前后审计 25 个关键文件。运行时 introspection 另发现 `BalanceScheduler.schedule`，随后补采其源码并核对与记录 revision 的 Git 内容完全一致；该补充文件不冒充运行前后双次审计。未来复现的控制器已把它加入初始审计。

## 离线复核与恢复证据

[便携证据包](results/round-01-archive/)保留冻结采集代码、trace/CSV、阶段记录、源码快照、响应、分析工具和哈希清单；编译缓存与大型 CANN 二进制缓冲留在远端。

```bash
mkdir -p /tmp/p30-review
tar -xzf practice_30_cpu_submission_timeline/results/round-01-archive/evidence.tgz -C /tmp/p30-review
/usr/bin/python3 -B /tmp/p30-review/round-01/analysis_tools/practice_30_cpu_submission_timeline/analyze.py \
  /tmp/p30-review/round-01
/usr/bin/python3 -B -m unittest discover -s practice_30_cpu_submission_timeline -p 'test_*.py' -v
```

本地离线工具使用 Python 3.10+；远端采集使用安装的 Python 3.12。归档内离线分析不需要 NPU。原始 trace 的 profiler 停止提示保留在日志中，本轮通过完整 kernel CSV 覆盖和身份关联检查核验用于结论的数据。

本轮没有采集完整 Python/C++ 栈、OS `sched_switch` 或硬件计数器。尚未覆盖的时间保留“未归因”，下一轮再针对具体空隙加细粒度观察。

## Forward 内部细分复现

第一部分使用已有 `eager-02` / `graph-01` 证据整理占比；第二部分只增加一次选中的 RoPE 调用，不扩展 workload 矩阵。两部分使用不同轮次，不把耗时相减归因。

```bash
cd /data/tianchi
python -B practice_30_cpu_submission_timeline/run.py \
  --mode eager --forward-detail \
  --output practice_30_cpu_submission_timeline/results/new-forward
python -B practice_30_cpu_submission_timeline/analyze.py \
  practice_30_cpu_submission_timeline/results/new-forward
python -B practice_30_cpu_submission_timeline/forward_analysis.py \
  --eager practice_30_cpu_submission_timeline/results/eager-02 \
  --graph practice_30_cpu_submission_timeline/results/graph-01 \
  --detail practice_30_cpu_submission_timeline/results/new-forward \
  --output practice_30_cpu_submission_timeline/report/new-forward
```

`--forward-detail` 只允许 eager；reference/recovery 不安装细分包装，diagnostic 在两次预热后安装，只有 step 32 的第一次 RoPE 记录四个 scope。包装全在实验子进程内，缓存须已加载，结束检查绑定和缓存指纹恢复。其余 RoPE 只经过选择条件检查，不新增计时 scope。已有设备等待、stream 和算子参数不改；本轮增加三个 Triton 源码文件的审计，总计 30 个。

离线复核可解压 `results/{eager-02,graph-01,forward-01}-archive/evidence.tgz` 到同一临时目录，再使用 `forward-01/analysis_tools/practice_30_cpu_submission_timeline/forward_analysis.py`，将三个参数指向各自解压目录。新归档包含冻结的采集代码和最终离线分析脚本，无需 NPU。

占比使用 profiler 的 forward 范围；调用内部双时钟另列。微秒级多层 scope 的扰动已显式报告，不能用新插桩调用的百分比外推生产耗时。

## 单次 graph 图外 attention 复现

```bash
cd /data/tianchi
python -B practice_30_cpu_submission_timeline/run.py \
  --mode graph --attention-detail \
  --output practice_30_cpu_submission_timeline/results/new-attention
python -B practice_30_cpu_submission_timeline/analyze.py \
  practice_30_cpu_submission_timeline/results/new-attention
python -B practice_30_cpu_submission_timeline/attention_analysis.py \
  practice_30_cpu_submission_timeline/results/new-attention \
  --output practice_30_cpu_submission_timeline/report/new-attention
```

`--attention-detail` 只允许 graph；与 eager 的 `--forward-detail` 不能同时启用。reference / recovery 保持原生调用；diagnostic 在初始化、捕获、两次完整预热之后才安装包装。完整请求仍为 10 输入 / 64 输出，只有 decode 32 第一层新增 8 个方法范围。其他调用经过选择条件后直接调用原函数；不新增设备等待、Tensor 内容读取或 stream 切换。审计新增 attention 入口及 Ascend forward context，合计 29 个源文件。

入口代码：[attention_observer.py](attention_observer.py)；离线分析和 HTML/SVG 生成：[attention_analysis.py](attention_analysis.py)。记录的是 Python/torch-npu callable 边界，不是完整 C++ 栈。保留没有设备任务的队列记录，区分 profiler 外层时间、范围内部双时钟与设备时间。

离线复核无需 NPU：

```bash
mkdir -p /tmp/p30-attention-review
tar -xzf practice_30_cpu_submission_timeline/results/attention-01-archive/evidence.tgz \
  -C /tmp/p30-attention-review
/usr/bin/python3 -B /tmp/p30-attention-review/attention-01/analysis_tools/practice_30_cpu_submission_timeline/analyze.py \
  /tmp/p30-attention-review/attention-01
/usr/bin/python3 -B /tmp/p30-attention-review/attention-01/analysis_tools/practice_30_cpu_submission_timeline/attention_analysis.py \
  /tmp/p30-attention-review/attention-01 --output /tmp/p30-attention-review/report
/usr/bin/python3 -B -m unittest discover \
  -s /tmp/p30-attention-review/attention-01/analysis_tools/practice_30_cpu_submission_timeline \
  -p 'test_*.py' -v
```

完整原始结果在远端 `results/attention-01`；本地便携归档冻结 trace/CSV、源码审计、采集代码、离线分析及哈希，不包含编译缓存。关于新增 scope 的显著扰动及尚未解释的余量，见 [ATTENTION_RESULTS.md](ATTENTION_RESULTS.md)。
