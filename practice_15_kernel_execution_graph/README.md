# Practice 15：从主机下发构建 kernel execution graph

这次把真实推理中的 **CPU 算子 → 异步队列 → CANN 下发 → NPU task / stream → 同步与数据关系** 组织成一张有类型的执行图。

已在 Ascend 910B2C 上重新采集单请求 eager 模型运行，并完成独立的双 stream 验证。
先看 [RESULTS.md](RESULTS.md)，再打开以下任一入口：

- [模型执行图浏览器](results/2026-09-24-model-run01/analysis/index.html)：1,444 个设备任务，按 prefill / decode、名称和分页查看；点击任务看提交链、队列、参数、依赖证据。
- [第一层 prefill 图](results/2026-09-24-model-run01/analysis/first_attention_prefill.svg) / [第一层 decode 图](results/2026-09-24-model-run01/analysis/first_attention_decode-1.svg)：并排理解 CPU、CANN 和设备任务。
- [双 stream 执行图](results/2026-09-24-stream-run02/analysis/execution_graph.svg) / [交互浏览器](results/2026-09-24-stream-run02/analysis/index.html)：真实 event record/wait，以及同一个 event 的复用。

HTML 完全离线，在本地浏览器打开。SVG 可以直接阅读。

## 与已有实验的关系

| Practice | 图的节点 | 图的边 / 主要证据 |
|---|---|---|
| 09 | 主机范围、算子、设备 task | profiler flow、时间线、kernel CSV |
| 10 | FX 节点 | 编译期值引用与模型分图 |
| 13 | 下发与执行记录 | 实际参数、队列、CANN、设备关联 |
| 15 | 具体发生的一次 host 调用、CANN 下发、kernel/copy/event task、CPU 等待完成 | 类型化执行边、同 stream 顺序、event 同步、局部数据契约 |

这里的 kernel execution graph 是**运行后重建的证据图**，包含 memcpy 和 event 任务。
它不是编译器生成的可执行调度计划，也不是 `ACL Graph` 对象；没有把 Practice 10 的 FX 图强行映射到本次 eager 运行。

## 怎样解释一条边

| 类型 | 含义 | 不能由它推出什么 |
|---|---|---|
| `dispatch` | host 算子与 CANN 下发通过同一设备任务及队列关联 | 主机函数已返回、设备已经完成 |
| `launch` | CANN launch entry 与设备任务的真实 flow | launch return 必须早于设备开始 |
| `host_issue_order` / `submission_order` | 同一 PID/TID 上观测到的入口顺序 | 不同设备 stream 之间自动同步 |
| `stream_order` | 同一物理 stream 上相邻的已观测任务 | 两任务间存在 tensor 数据依赖，或不存在未观测任务 |
| `event_wait` | 某次 record 完成后，另一 stream 的对应 wait 才能完成 | CPU 提交 wait 时发生阻塞 |
| `event_sync` / `host_after_wait` | event 完成 → CPU 同步返回 → 同线程后续调用 | 未被该 event 捕获的其他 stream 也完成 |
| `data_contract` | 实际参数匹配，加上特定算子的已知读写语义 | 自动增加 stream 同步；完整硬件内存访问记录 |
| `storage_candidate` | KV 写入和 attention 使用同一缓存池 | 在未采集 slot 值时断言精确读写字节范围 |

**这张图不能把所有类型的边统一当作“源节点执行结束后目标节点才能开始”。**
host/launch 节点的关联锚点是调用入口；`event_wait` 描述完成条件。
JSON 每条边都保留 `kind`、`evidence`、`semantics`。DAG 校验用于发现连边错误，不把混合图当作性能 critical path。

## 三种 stream 标识

1. Python/torch-npu 的 `stream_id`：框架自身的流标识。
2. launcher 的 `npu_stream` / `runtime_stream`：传到运行时的 host 句柄。
3. profiler 的 `Physic Stream Id`：设备任务上的物理 stream 编号。

它们是不同命名空间。模型中，通过 111 次 Triton launcher 的唯一任务关联建立句柄到物理 stream 的观测映射。
全部 1,444 个任务都有物理 stream 和 host 下发关联；其他原生算子的原始 stream 参数没有逐项转储。
双 stream 实验同时记录三个标识，用真实 flow 验证对应关系。映射只在本次进程/采集期间有效。

## 数据边为何只做局部

本次模型的 24 层 × 4 步中，RoPE 的真实 Q/K view 与随后 attention / KV 写入的参数一致。
结合 RoPE 的原地写入语义，建立 192 条 RAW 数据契约边。96 次 attention 与 96 次 RoPE 一一配对。

decode 中还保留 144 条 K/V 池候选关系：24 层 × 3 步 × K/V 两份存储。
prefill 的无缓存 attention 读取当前 K/V，因此不会仅凭“KV 写入先执行”给它补一条 cache→attention 数据边。

723 个参数范围保留线性层、Triton、KV/FIA 和指定拷贝的 tensor 元数据；多 kernel 范围不会被误拆成每个 kernel 的精确读写集合。
没有完整的 allocator lifetime、全部原生算子读写参数和隐藏 workspace，不能用“同地址的最后一次写入”生成全模型数据图。
因此 `complete_data_graph=false`；没有画数据边表示**未知或未覆盖**，不表示两个 kernel 相互独立。

## 两项实验

**模型运行**复用 Practice 13 的采集器，重新运行 Qwen2.5-0.5B-Instruct、BF16、TP=1、eager。
一个正式请求，10 个输入 token、4 个输出 token，包含一次 prefill、三次 decode。
关闭 prefix caching、chunked prefill、async scheduling；保留原生 KV 池容量。新端口为 8015。
实际命令、环境、安装源码、instrumentation、编译产物和原始 profiler 全部归档。

**双 stream 验证**单独执行 `y=x+1 → z=y*2 → out=z+x`，输入 x 全为 1，结果核对为 5。
生产 y 和最终加法在 stream A，消费 y 在 stream B，中间使用两次 event wait。
同一 event 共 record 三次，最终 CPU 等待第三次；四张 tensor 一直持有到最终完成后。
这个控制实验验证连边方法，不表示 vLLM 本次也使用了两条 stream。

## 远端复现

将本目录及 Practice 03、07、09、13 的脚本同步到已有服务器；无需安装新包或下载模型。
输出目录必须不存在。先确认设备可用于实验，再执行：

```bash
cd /data/tianchi
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export PATH=/usr/local/python3.12.13/bin:$PATH

python practice_13_operator_submission/run_submission.py \
  --port 8015 --output practice_15_kernel_execution_graph/results/my-model-run

python practice_15_kernel_execution_graph/run_stream_probe.py \
  --output practice_15_kernel_execution_graph/results/my-stream-run
```

模型 runner 只关闭自己创建的服务进程组；双 stream 程序结束即释放自身资源。
不修改已安装库。模型观察器不添加设备同步；控制实验的 record/wait 是有意设计的工作负载。
采集有额外开销，耗时不用于性能比较。

## 本地重建与检查

Python 3.7+ 标准库即可分析，不需要 NPU、torch 或网络；安装 Graphviz 时额外输出模型 SVG。

```bash
python3 practice_15_kernel_execution_graph/build_model_graph.py \
  practice_15_kernel_execution_graph/results/2026-09-24-model-run01 \
  --output practice_15_kernel_execution_graph/results/2026-09-24-model-run01/analysis

python3 practice_15_kernel_execution_graph/build_probe_graph.py \
  practice_15_kernel_execution_graph/results/2026-09-24-stream-run02

dot -Tsvg practice_15_kernel_execution_graph/results/2026-09-24-stream-run02/analysis/execution_graph.dot \
  -o practice_15_kernel_execution_graph/results/2026-09-24-stream-run02/analysis/execution_graph.svg

python3 -m unittest discover -s practice_15_kernel_execution_graph -p 'test_*.py' -v
cd practice_15_kernel_execution_graph
sha256sum -c SHA256SUMS
```

模型分析器每次重新核验原始 flow、kernel CSV、队列、参数与编译产物；不直接信任已有分析 JSON。
14 项测试覆盖真实模型/双 stream 图，以及漏 flow、错指针、别名误判、同 stream 重叠、event 代次错误、环和悬空边。
JSON 时间戳保留十进制字符串，分析使用 Decimal；浏览器用 BigInt 纳秒差计算耗时，避免 epoch 微秒被 float 舍入。

## 代码与证据

- `build_model_graph.py`：调用 Practice 13 的原始证据检查器，构建分类图与参数范围。
- `run_stream_probe.py` / `build_probe_graph.py`：真实双 stream 采集与独立检查。
- `render_graph.py` / `viewer.html`：离线浏览器；`export_graph.py` 导出第一层局部图。
- `results/*/analysis_tools/`：本次生成工具及依赖的快照。
- `results/*/analysis/execution_graph.json`：完整节点、边、stream 映射、限制和输入指纹。
- `contract_sources/rope.py`：模型采集结束后的只读源码补充；当次实际编译的 TTIR 和 NPU 二进制保留在 `compiler_cache/`。

与 Practice 13 相同，repo 省略约 72.5MB 的 host `precompiled.h.gch`，原文件保留远端；其原始指纹见
`unarchived_build_intermediates.json`。未省略 NPU kernel。第一次双 stream 试采因 profiler 关闭调度的警告被排除，见 `excluded_pilots.json`。

接口语义参考：[Ascend aclrtRecordEvent](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/910beta2/API/runtimeapi/aclcppdevg_03_0083.html)。
实际 `wait_event` / `Event.synchronize` 行为以双 stream 归档的已安装 `streams.py` 和本次 trace 为准。
