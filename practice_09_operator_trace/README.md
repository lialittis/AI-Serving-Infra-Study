# Practice 09：追踪真实算子的调用与执行

本实验回答：**一次真实 vLLM forward 怎样经过主机算子调用，最终在 NPU 上执行？**

沿用 Ascend 910B2C、Qwen2.5-0.5B-Instruct、单卡 BF16、eager。
根据新的学习顺序，将原路线图中的算子与时间线观测提前；block 释放与复用顺延。

已于 2026-09-22 在真实服务器运行通过。先看 [RESULTS.md](RESULTS.md)，再逐段阅读代码。

## 先区分三种事件

```text
Python 函数范围：vLLM-Ascend 的 attention / KV 写入适配器
    ↓ 发起调用
PyTorch 算子：torch-npu 注册的算子、aten 算子等
    ↓ 经运行时下发（可能异步排队）
NPU task / kernel：设备上实际开始和结束的执行
```

Python 函数返回不意味着对应设备任务已结束。一个 PyTorch 算子也不一定只有一个 kernel。
不能只根据名称相似或时间相近就宣称两者对应；需要 profiler 导出的关联信息。

## 实验设计

1. 启动一个独立的本地服务，端口 8009，关闭 prefix caching、chunked prefill、async scheduling。
2. 先发送一次预热请求，不开启采集。
3. 调用 `/start_profile`。
4. 发送唯一的采集请求：126 个输入 token，生成 2 个 token，固定温度和 seed。
5. 调用 `/stop_profile`，等待导出，再关闭本实验启动的服务。
6. 对照主机事件、PyTorch 算子与设备 trace，分析第一层 KV 写入和 attention。

两个生成步足够看到一次 prefill 和一次 decode。输入沿用 Practice 08 的 `hello` token ID 重复序列，
目的是控制 shape，不评价模型回答质量。

## 代码阅读顺序

- `run_profile.py`：管理服务、预热请求和 profiler 的采集窗口。
- `sitecustomize.py`：仅在设置 `P09_TRACE_DIR` 的子进程中启用注解。
- `operator_trace.py`：复用 Practice 07 的请求事件；用 `torch.profiler.record_function`
  将真实 Python 函数范围写入同一条 profiler 时间线。
- `summarize_profile.py`：读取导出文件，核对事件并生成可阅读的结果。

`P09/...` 范围是我们添加的主机注解；算子事件、关联线和设备任务来自安装的 Ascend profiler。
不修改已安装的 vLLM 源码，不用自己的算子替换模型执行，不做 Practice 08 的 KV 数据 `.cpu()` 校验。
记录的 shape 仅来自 tensor 的主机元数据。

## 远端复现

同级目录需保留 Practice 03 的 `collect_environment.py` 和 Practice 07 的 `trace_hooks.py`。
确认设备没有其他模型服务占用后执行：

```bash
cd /data/tianchi
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export PATH=/usr/local/python3.12.13/bin:$PATH

python practice_09_operator_trace/run_profile.py \
  --output /data/tianchi/practice_09_operator_trace/results/my-new-run
```

输出目录必须不存在。模型使用本地路径，不下载模型或安装依赖。
命令和 profiler 配置存入 `command.json`；HTTP 请求、响应、原始日志及注解代码快照同时归档。

## 本地查看与校验（不需要 NPU）

```bash
python3 practice_09_operator_trace/summarize_profile.py \
  practice_09_operator_trace/results/2026-09-22-run01
python3 -m unittest discover -s practice_09_operator_trace -p 'test_*.py' -v
cd practice_09_operator_trace
sha256sum -c SHA256SUMS
```

检查器将 `async_npu` 的起止 flow ID 连接到真实设备事件，再通过 `HostToDevice` flow
查找 CANN 下发范围，并与 kernel CSV 的 task、stream、开始时间和执行时间核对。
时间戳使用 Decimal，避免 epoch 微秒转换为 float 后损失精度。
7 项测试覆盖正常证据和缺失设备事件、断开的 flow、错误 task/时间及缺失 Python 范围。
当前校验器针对本次版本与形状；其他版本改变事件名或后端路径时会明确失败，需要重新检查真实 trace。

## 观测边界

- profiler 和 Python 注解会引入开销，时间数字仅用于理解这次诊断采集，不作性能基准。
- 只给第一层 attention 的子调用添加详细 Python 注解；profiler 仍然采集整个请求的算子。
- eager 模式的结果不代表 graph 模式、多请求或多卡执行。
- kernel 名称和关联形式以当前实际导出为准，不能仅凭安装了 Triton 就认定某个算子经过 Triton。

参考：[vLLM-Ascend Service Profiling Guide](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/performance_and_debug/service_profiling_guide.html)。
远端当前 `TorchNPUProfilerWrapper` 实际开启 CPU + NPU、Level1 和 Text 导出；源码核验优先于 latest 文档。
