# Practice 07：追踪一次真实 vLLM-Ascend 请求

问题：**一个真实 prompt 从 HTTP 入口进入后，怎样经过 scheduler、Ascend worker/runner，最终返回 token？**

后续路线见 [REAL_SYSTEM_ROADMAP.md](../REAL_SYSTEM_ROADMAP.md)。
本实验直接运行远端安装的 vLLM / vLLM-Ascend 和 Qwen 模型，不模拟调度器。

已完成 2026-09-20 的真实实验：输入 5 个 token、输出 8 个 token，8 个调度步。
先看 [RESULTS.md](RESULTS.md) 的实际架构图和结论，再看
[完整摘要](results/2026-09-20-run02/summary.md)。首次启动失败和修正过程也已保留。

## 复现

在远端 `/data/tianchi`，沿用 Practice 03 的模型与依赖：

```bash
cd /data/tianchi
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export PATH=/usr/local/python3.12.13/bin:$PATH
python practice_07_real_request_trace/run_trace.py \
  --output /data/tianchi/practice_07_real_request_trace/results/my-new-run
```

输出目录必须不存在。运行前确认当前卡没有其他模型服务占用资源。
默认只监听 `127.0.0.1:8007`，端口占用时退出，不会向已有服务发送实验请求。
脚本会启动自己的服务、等待健康检查、发送唯一一次推理请求，然后停止自己的服务进程组。
模型不会下载，依赖不会升级。运行用远端 Python 3.12；trace 使用 Python 3.11+ 的 code 元数据。

固定配置：单卡 TP=1、BF16、eager、max_num_seqs=1、max_model_len=2048、
max_num_batched_tokens=2048、memory utilization=0.3、block_size=128。
关闭 prefix caching、chunked prefill 和 async scheduling，以便逐步关联。
这些设置是本实验基线，不是吞吐优化建议。实际生效配置还会从 scheduler 实例采集。

请求：`The capital of France is`，`temperature=0`、`seed=0`、`ignore_eos=true`、
`max_tokens=8`、非流式，便于将最终 HTTP token 计数与逐步 engine 输出核对。

## 观察方式

`run_trace.py` 只给实验子进程设置 `PYTHONPATH` 和 `P07_TRACE_DIR`。
Python 启动时加载本目录的 `sitecustomize.py`，使用 `sys.setprofile` 和
`threading.setprofile` 观察选定函数的 call/return。
没有修改已安装源码、替换函数或更改 allocator 行为。

记录范围：

- API：请求参数、最终 CompletionResponse。
- Engine：实际 executor 类型。
- Scheduler：入队、每轮请求及 token 数、该轮前的 computed token 数、输出与结束清理。
- Ascend Worker / Runner：实际类型及调用顺序。
- Model forward：实际模型类、input IDs 和 positions 的 shape/dtype/device。
- Attention：每轮第一层的 backend 类和 Q/K shape。

用 PID/TID 确认进程与线程关系。每条事件保存时间戳、实际源码文件、函数和入口行号。
每线程独立 JSONL，避免多进程混写；汇总按同一主机的 monotonic 时钟排序。
协程恢复会触发多次 call，API 接收事件按 request ID 去重；只在真正产生 CompletionResponse 时记录 API 响应。
启动期的模型预热不在实际 runner.execute_model 请求上下文中，不计入用户请求的 forward/attention 事件。

## 文件

| 文件 | 作用 |
|---|---|
| `run_trace.py` | 启动、发请求、停止、采集环境与源码指纹 |
| `sitecustomize.py` / `trace_hooks.py` | 按环境变量启用调用观察 |
| `summarize_trace.py` | 无需 vLLM/NPU，校验证据完整性并生成摘要 |
| `results/<run>/command.json` | 实际命令及本实验设置的环境变量 |
| `results/<run>/environment.json` | 软件版本、源码 commit/状态、NPU 信息、模型指纹 |
| `results/<run>/events/*.jsonl` | 原始事件 |
| `results/<run>/source_hashes.json` | 实际观测源码文件 SHA256 |
| `results/<run>/instrumentation/` | 该次运行的追踪脚本快照及独立 SHA256 清单 |
| `results/<run>/server.log` | 服务完整日志，含真实配置和启动预热 |
| `results/<run>/request.json` / `response.json` | 真实 HTTP 请求体和响应体 |
| `results/<run>/processes.txt` | 实验服务进程组快照 |
| `results/<run>/summary.md` | 校验后的逐步摘要和源码导航 |

环境采集复用 `../practice_03_ascend_start/collect_environment.py`，复制到新机器时保留这个相对位置。

本地查看归档时执行：

```bash
python3 practice_07_real_request_trace/summarize_trace.py \
  practice_07_real_request_trace/results/2026-09-20-run02
python3 -m unittest discover -s practice_07_real_request_trace -p 'test_*.py' -v
```

成功运行的 `instrumentation/` 保留当时脚本；当前入口额外自动保存脚本快照，汇总器也增加了
step 关联和 token 递进校验。追踪回调与成功运行时一致，原始 JSONL 未修改。

本地和远端工作目录副本均包含本实验代码、文档与结果。在实验目录内可执行
`sha256sum -c SHA256SUMS` 核对归档完整性。

## 证据边界

- **call/return 是 Python 主机事件，不是 NPU kernel 时间线。** 不把函数返回解释为设备完成。
- 只读取 shape/dtype/device 等主机元数据，不复制 NPU tensor 内容，也不主动同步设备。
- profile 回调和写日志有额外开销，本实验不能作为性能基准。
- 基于所记录的 0.21 源码路径；换版本后必须核对观察点。缺少关键事件或发生 trace_error 时汇总失败。
- step 的跨层关联针对当前同进程 worker 和同步调度。多进程 executor 需要显式传递关联信息。
- request_cleanup_return 说明 scheduler 清理入口已返回，不代表归还 NPU 大块内存，也不证明数据清零。
- 本轮还不解析 block table/slot mapping；这是 Practice 08 的工作。
