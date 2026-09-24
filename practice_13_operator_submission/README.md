# Practice 13：CPU怎样把一次算子调用交给NPU？

先记住这条链：**CPU准备地址和参数 → 任务入队 → 下发线程调用运行时 → NPU按stream执行 → CPU在需要结果时等待。** 编译和加载是另一条生命周期，不需要每执行一次就重新进行。

打开[离线交互报告](results/2026-09-24-run03/analysis/index.html)，先选 `prefill · _triton_rope` 看提交，再选 `FusedInferAttentionScore` 看CPU与NPU的时间重叠。`MEMCPY_ASYNC / buffer_copy` 和 `/ to_list` 分别展示输入拷贝与采样结果回传。报告还提供全部设备任务的执行顺序、搜索和每步完成边界。

详细结论见 [RESULTS.md](RESULTS.md)。报告需要在本地浏览器打开；GitHub文件页不会直接运行HTML。

## 本次实验做什么

真实环境是Ascend 910B2C、Qwen2.5-0.5B-Instruct、vLLM / vLLM-Ascend。主函数启动一个单卡BF16 eager服务，发送两次相同自然语言prompt：一次预热，一次正式观测。正式请求是10个输入token、4个输出token，对应1次prefill和3次decode。

与Practice 12不同，本次不人为限制为两个KV block；使用原生KV池容量。关闭prefix caching、chunked prefill和异步调度来简化因果关系。**关闭async scheduling并不关闭CPU→NPU异步任务队列。** 本次不研究graph replay。

| 记录层面 | 证据 |
|---|---|
| 编译 | 使用全新独立Triton缓存目录，记录真实compiler入口/返回、编译阶段、特化参数、IR和二进制SHA256 |
| 加载/注册 | 记录`_init_handles`、`load_binary`、模块/函数句柄及实际生成的C++ launcher |
| CPU调用参数 | tensor形状、dtype、device、地址、stride、offset；Triton grid/stream/句柄/标量/constexpr；FIA与KV写入的实际API参数 |
| 队列与下发 | PyTorch profiler的入队/出队、correlation ID、CANN调用与HostToDevice flow |
| NPU执行 | kernel/task ID、物理stream、开始/结束时间；与kernel CSV交叉验证 |
| CPU同时做什么 | 同一时间轴上的主线程算子、Python范围、下发线程范围；采样后原生event等待 |

所有设备任务都保留关联。参数观测重点覆盖Triton、线性层、KV写入、attention和指定拷贝边界；**不是所有原生算子的隐藏参数全量转储**。不读取tensor内容，不添加NPU同步，不改动已安装库。观测本身有CPU开销，结果不能当作性能基准。

## 远端复现

在已有库和模型的`ascend910`服务器执行。将本目录和依赖的Practice 03、07、09同步到`/data/tianchi`；无需重新下载模型。连接凭据不保存在repo中。

```bash
ssh ascend910
cd /data/tianchi
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export PATH=/usr/local/python3.12.13/bin:$PATH

# 每次使用一个尚不存在的目录；不会清空共享Triton缓存。
python practice_13_operator_submission/run_submission.py \
  --output practice_13_operator_submission/results/my-run

python practice_13_operator_submission/analyze_submission.py \
  practice_13_operator_submission/results/my-run
python practice_13_operator_submission/render_submission.py \
  practice_13_operator_submission/results/my-run
```

默认模型路径为`/data/huggingface_home/hub/Qwen2.5-0.5B-Instruct`，端口8013；可用`--model`和`--port`修改。程序检查端口占用，保存实际启动命令，结束后关闭它自己启动的服务。`server.log`保存启动及profiler导出日志。

新运行的PID、地址、句柄、task ID、耗时与KV池容量可能不同；验证的是关联关系。冷编译次数也可能随版本、特化和预热路径变化。本次记录为8次，不能当成任意模型的固定数量。

## 本地重建与核验，不需要NPU

使用Python 3.7或以上，仅标准库：

```bash
python3 practice_13_operator_submission/analyze_submission.py \
  practice_13_operator_submission/results/2026-09-24-run03
python3 practice_13_operator_submission/render_submission.py \
  practice_13_operator_submission/results/2026-09-24-run03
python3 -m unittest discover -s practice_13_operator_submission -p 'test_*.py' -v

cd practice_13_operator_submission
sha256sum -c SHA256SUMS
```

分析器拒绝丢失flow、错误队列correlation、错配函数句柄、损坏编译产物、缺失拷贝元数据及不成立的结果等待边界。HTML纯离线，不依赖CDN。

## 按什么顺序读代码

1. [run_submission.py](run_submission.py) 的`main()`：建目录、记录环境、启动服务、预热、开启profiler、发正式请求、关闭服务、归档。
2. [submission_trace.py](submission_trace.py) 的`TARGETS`、`handle()`：在哪些真实函数入口/返回读取哪些元数据。`source`是源码位置，`source_tensor`才是拷贝源tensor。
3. [analyze_submission.py](analyze_submission.py) 的`analyze()`：先验证编译/加载和参数，再使用flow关联CPU/NPU，最后检查执行顺序与结果回传。
4. [render_submission.py](render_submission.py)：把已经验证的记录转换成HTML，不重新执行模型。

复用Practice 07的日志基础设施和Practice 09的flow/CSV核验。`sources/`保存远端实际安装源码的只读快照；它们是证据，不参与替换库。`instrumentation/`是本次采集时的脚本快照。

## 归档范围

正式证据是`results/2026-09-24-run03/`；两个排查轮次及排除原因记录在[excluded_pilots.json](excluded_pilots.json)，不混入最终统计。

- `events/`：主机函数调用、参数、编译、加载日志。
- `profiler/`：原始设备/主机记录及CANN导出的trace、CSV。
- `compiler_cache/`、`launcher_sources/`：实际Triton IR、NPU二进制、host launcher及其生成源码。
- `source_manifest.json`、`compiler_artifacts.json`：原始来源、大小与SHA256。
- `analysis/`：离线报告、全部关联证据、统计摘要。
- `analysis_tools/`：生成这些派生结果的工具快照及依赖。

repo省略约72.5MB的host预编译头`precompiled.h.gch`，原文件仍在远端运行目录，原始hash/大小保留于`unarchived_build_intermediates.json`。它不是NPU kernel；NPU二进制、IR及launcher均保留。分析器只允许这一类声明的省略，不允许缺失kernel。`SHA256SUMS`核验repo实际归档文件。
