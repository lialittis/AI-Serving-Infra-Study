# Practice 22：stream 的创建、任务分配与依赖

目标是解释真实模型为什么使用这些 stream：谁创建或取用了它，算子由哪里提交，在哪里等待。直接打开 [离线报告](report/index.html)、[实测结论](RESULTS.md) 或 [实际 replay 序列 SVG](report/graph_replay_sequence.svg)。

## 本次完成了什么

在同一 Ascend910B2C 上执行 Qwen2.5-0.5B-Instruct 的四种场景，每种有独立的诊断进程和无观察器基线。每次启动先预热，再发送两个正式 HTTP 请求。

| 场景 | 执行配置 | 正式输入 |
|---|---|---|
| eager | BF16、TP=1、贪心 | 单序列，10 输入 / 4 输出 |
| graph | 同上，PIECEWISE，capture sizes=[1] | 同一输入 |
| sampling-off | eager、提前生成关闭 | batch 32，temperature=0.8、top_p=0.9 |
| sampling-on | eager、提前生成开启 | 与关闭组相同 |

关闭 prefix caching、chunked prefill、async scheduling；服务 seed=123，随机请求不设独立 seed。为了比较确切输出 token，所有场景请求 `logprobs=1` 和 token ID 格式；因此任务总数不能直接与未请求 logprobs 的 Practice 15 相等比较。

采样的 32 个序列并非同一步全部进入：两次请求各实际出现 5 个调度步，包括一个 mixed prefill/decode 步。报告按真实调度记录标注，不把 HTTP batch 当成恒定的活动 batch。

## 代码的主流程

1. [run_experiment.py](run_experiment.py) 创建独立目录，保存环境和源码，构建进程级观察器，启动服务。
2. [stream_trace.py](stream_trace.py) 从 Python 启动起记录 stream 对象获取、上下文切换、event、graph 捕获与 replay 的调用栈；组合已有 Practice 13/15/17 的请求观察器。
3. [build_native.py](build_native.py) 根据安装的 CANN 头文件生成并编译原生包装；[native_template.c](native_template.c) 通过 `LD_AUDIT` 保留实际解析函数并记录原调用。来自 torch-npu、ATB、op-api 的目标接口会被观察。
4. 捕获结束后用 `NPUGraph.debug_dump()` 导出内部 Model ID、Stream ID、Task ID；对已经存在的 Python stream，用 `aclrtStreamGetId` 查询运行时 ID。不调用会惰性创建 stream 的 accessor 来“发现”资源。
5. 预热完成后开启 profiler，执行两次请求，正常停止 profiler 和本次服务。
6. [analyze_streams.py](analyze_streams.py) 离线核验生命周期与关联；[render_report.py](render_report.py) 生成交互报告；[validate_results.py](validate_results.py) 与无观察器基线比较。

本次不修改安装库，不改变算子的 stream 或核数，不插入设备等待或读取模型中间 tensor。ID 查询和调试导出会增加 CPU 工作；所有耗时仅作为诊断时间线。

## 身份与关联规则

- Python `id()`、Python stream ID、CANN stream 句柄、运行时 stream ID、profiler 物理流分开保存。
- 原生创建／销毁界定句柄代次；Python 对已有 stream 的包装不算再次创建资源。未被取用的池资源不强行指定物理流 ID。
- 捕获调用关联原生 `aclmdlRI` 句柄，再通过调试导出取得 Model ID 与内部 stream/task。销毁后复用同一 ID 不能串到旧 graph。
- 直接 kernel 按原生 flow 与 CSV 名称、stream、task、开始时间关联。graph 内任务按有效 graph 生命周期和 dump 的 stream/task ID 关联；dump 的编译符号名与 profiler 的 aclnn 别名可不同，保留两者。
- 每次 replay 通过原生实例句柄、同 API／线程完整调用序列、connection ID、内部重复任务序列和完成区间共同核验。不是按最近时间猜测。
- event 按 PID、句柄与 record 代次匹配。设备等待、CPU 等待、CPU 等待后提交和同流顺序是不同类型的边。数据契约不能代替同步边。

graph 的内部归属现已可验证，但内部流的底层分配调用和具体 NOTIFY 标识配对仍未直接记录。报告保留这些缺口，不能称为所有 CANN 内部实现已完全恢复。

## 远端复现

在 `/data/tianchi` 使用现有 Python/CANN 环境。先确认 NPU 没有其他实验负载，使用尚不存在的输出目录：

```bash
cd /data/tianchi
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export PATH=/usr/local/python3.12.13/bin:$PATH
npu-smi info

python practice_22_stream_lifecycle/run_experiment.py \
  --case graph --port 8022 \
  --output practice_22_stream_lifecycle/results/my-graph

python practice_22_stream_lifecycle/run_experiment.py \
  --case graph --baseline --port 8023 \
  --output practice_22_stream_lifecycle/results/my-graph-baseline

python practice_22_stream_lifecycle/analyze_streams.py \
  practice_22_stream_lifecycle/results/my-graph
```

其他场景替换 `--case` 为 `eager`、`sampling-off`、`sampling-on`，依次执行，避免设备争用。默认模型路径为 `/data/huggingface_home/hub/Qwen2.5-0.5B-Instruct`。观察器仅注入本次服务及子进程，原有 jemalloc 保留；子进程增加 glibc 静态 TLS 预留以兼容 `LD_AUDIT`。

独立采样正确性检查可运行：

```bash
python practice_17_vllm_multistream/verify_sampling_numerics.py \
  --output practice_22_stream_lifecycle/results/my-sampling-numerics
```

它在独立进程中用固定 q 验证已安装采样消费和同步路径，并另验真实随机数有限且为正；不评估模型精度或随机分布。

## 本地离线复现

分析和报告仅需 Python 标准库，不需要 NPU 或网络：

```bash
for case in eager graph sampling-off sampling-on; do
  python practice_22_stream_lifecycle/analyze_streams.py \
    "practice_22_stream_lifecycle/results/2026-09-28-${case}-run01"
done
python practice_22_stream_lifecycle/validate_results.py practice_22_stream_lifecycle/results
python practice_22_stream_lifecycle/render_report.py
python -m unittest discover -s practice_22_stream_lifecycle -p 'test_*.py' -v
cd practice_22_stream_lifecycle
sha256sum -c SHA256SUMS
```

汇总器默认读取本次固定四组目录；单个新目录可独立分析，汇总新套件时按代码中的四组目录约定命名。浏览器直接打开 `report/index.html`，不需要本地 HTTP 服务。

结果保留原始 profiler、Python/native JSONL、graph dumps、请求响应、源码与观察器快照、缓存清单和分析 JSON。每组约 69 MiB 的可再生 `precompiled.h.gch` 不纳入 Git，但保留大小和 SHA256；其余采集证据按清单归档。原生 `.so` 是本机架构产物，重新运行时会重新编译。

## 资料与证据边界

- 首次批量创建的 Python 调用栈见报告的“来源”；完整原生调用记录包含调用库和偏移。默认初始化的两个资源没有完整 Python 创建栈，不补造源码行。
- 安装的 `torch_npu.npu.graphs.py` 明确在未传入 stream 时懒创建 `default_capture_stream`；实际 `model_runner_v1.py` 还有外层 graph capture stream。这些与 graph 内部任务流分开记录。
- [CANN 捕获说明](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/910beta2/API/runtimeapi/aclcppdevg_03_1782.html) 描述内部运行实例及流资源消耗；具体行为以本次 CANN 9.0 头文件和运行证据为准。
- [CANN graph 调试接口](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/910beta2/API/runtimeapi/aclcppdevg_03_1786.html) 说明调试信息中的设备、流和任务 ID。导出的示意 `ts/dur` 不是真实设备时间，报告时间线只使用 profiler 时间。
