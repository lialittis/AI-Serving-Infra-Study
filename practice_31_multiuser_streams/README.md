# Practice 31：单个 vLLM 服务的多用户请求与 NPU 执行并发

本实验通过**独立 HTTP 请求**模拟多个用户。先观察 vLLM-Ascend 原生 eager 服务，关联四个不同层次：

```text
用户/请求 → 调度 step 的请求集合 → host 算子与运行时提交 → NPU stream 上的任务
```

同一个批次中的 kernel 通常共同处理多个请求，不能拆成“每个用户独占一个 kernel/stream”。多用户请求时间重叠、合批、多 stream 和跨 stream 计算重叠分别计量。观察到单 stream 串行计算也是有效结果。

报告中 kernel 的 `requests` 表示其**所属调度批次的请求集合**，不证明该 kernel 逐一读写了集合中所有请求的数据；混合 prefill/decode 批次中的部分任务可能只服务一个子集。

## 从哪里开始

- [P31a 实验结果](RESULTS.md)：greedy 多用户基线。
- [P31b 提前生成开／关](SAMPLING_RESULTS.md)：固定 eager 随机采样，对比提交时机、同步和真实重叠；入口为 `run_sampling.py`。
- 完整交互报告：本地 `results/formal-01/index.html`，选择并发档位、请求和调度 step；Git 不包含这些大文件，[恢复方法见下](#完整报告与大文件)。
- `run.py`：服务生命周期、独立 HTTP/SSE 客户端和固定输入。
- `observer.py`：仅在 diagnostic 服务中启用的 host 元数据观测。
- `analyze.py`：精确请求映射、flow/CSV 关联、区间并集和覆盖率校验。
- `audit_raw.py` / `archive.py`：抽查原始 prefill/decode MatMul，导出带哈希的紧凑证据。
- [任务清单](../tasks/2026-10-02-practice31-multiuser-streams.md)：后续阶段。

## P31a 固定配置

| 项目 | 设置 |
|---|---|
| 模型 | `/data/huggingface_home/hub/Qwen2.5-0.5B-Instruct` |
| 服务 | 单实例 HTTP 服务，保留正常 frontend/engine 进程结构 |
| 设备 | 当前主机物理 NPU 5，容器逻辑 NPU 0，TP=1、BF16、eager |
| 用户并发数 | 1、2、4、8；不是一条带多个 prompts 的 HTTP 请求 |
| 用户行为 | 初始屏障同时开始；各自收到完整响应后立即发下一请求；每用户 3 个独立请求 |
| 输入/输出 | 128 / 64 tokens；greedy、ignore_eos、n=1，服务 seed=123 |
| 服务容量 | 所有并发档位固定 max_num_seqs=8、max_num_batched_tokens=1024、max_model_len=512 |
| 其他设置 | prefix caching、chunked prefill、async scheduling、异步采样预计算关闭；无 offload |
| 预热 | 每个用户 2 个请求；不计入正式窗口 |
| benchmark | 每档 3 次工作负载重复；无 observer、无 profiler |
| diagnostic | 每档一次工作负载；独立服务、一次 start/stop_profile 包住整个正式窗口 |

“轮次”是同一用户连续发送的第几个请求；第一阶段不携带聊天历史。“重复”是同一工作负载重新执行一次，和轮次不同。客户端在同一 Ascend 主机访问 localhost。实验重复输入与配置，不要求线程到达顺序或调度批次完全相同；这些变化正是实际观测内容。

## 运行与复现

在已配置 CANN 和 Python 3.12 的 Ascend 主机仓库根目录执行。保留原有 jemalloc，脚本会清除其他 practice 的 Python 插桩，并在每个案例后关闭自己创建的服务进程组。默认端口 8031；输出目录必须不存在。

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
PY=/usr/local/python3.12.13/bin/python

# 首先确认设备空闲。脚本也会检查无 NPU 运行进程、端口和剩余磁盘。
npu-smi info

# 小规模：保持 128-token 输入，缩短输出和轮次。
$PY practice_31_multiuser_streams/run.py \
  --output /data/tianchi/practice_31_multiuser_streams/results/smoke-new \
  --concurrency 1 2 --rounds 1 --repeats 1 --warmup 1 --output-tokens 4
$PY practice_31_multiuser_streams/analyze.py \
  /data/tianchi/practice_31_multiuser_streams/results/smoke-new

# 正式矩阵：默认 1/2/4/8 用户，先 benchmark，再 diagnostic。
$PY practice_31_multiuser_streams/run.py \
  --output /data/tianchi/practice_31_multiuser_streams/results/formal-new
$PY practice_31_multiuser_streams/analyze.py \
  /data/tianchi/practice_31_multiuser_streams/results/formal-new
```

支持 `--phase benchmark|diagnostic|both` 和 `--concurrency` 选择子集。P31a 限制最多 8 用户、128 输入和 64 输出；后续实验改变这些范围时需要明确记录配置，不能混为本轮基线。

运行输出包括 `plan.json`、实际 token IDs、源码快照与 SHA256、版本、设备健康/进程状态，以及每档位的命令、日志、请求记录、观测 JSONL、原始 profiler。Git 仅保留代码、文档、摘要、抽查证据、压缩请求记录和哈希清单；远端原始 trace 的路径与 SHA256 在每个案例的 `analysis/summary.json` 中。

## 完整报告与大文件

完整 HTML、`analysis.json.gz`、`tasks.csv.gz`、observer JSONL、日志和重复源码快照由本 practice 的 `.gitignore` 排除，已有本地文件不删除。Git 中的 `archive_manifest.json` 描述完整归档，文件列表不是 Git 上传列表；`requests.json.gz` 解压后与清单中的 `requests.json` 字节一致。

在新 checkout 中，结果结论、各案例 `analysis/summary.json` 和 `raw_spot_checks.json` 可直接阅读。要查看完整交互报告，可从已有 Ascend 主机恢复约66 MB的压缩归档。在仓库根目录执行：

```bash
ssh -F ~/.ssh/config ascend910 'cat /data/tianchi/p31-formal-evidence.tgz' > /tmp/p31-formal-evidence.tgz
tar -xzf /tmp/p31-formal-evidence.tgz -C practice_31_multiuser_streams/results
# 然后用浏览器打开 practice_31_multiuser_streams/results/formal-01/index.html
```

该归档不含约2.51 GB原始 profiler；原始文件仍在远端 `results/formal-01/` 下，按清单取回后才能重做 raw flow join。完整报告无需原始 profiler 即可离线查看。

## 关联与计量方法

1. **客户端 → 服务内部请求**：观测 `create_completion/_create_completion` 的客户端 ID、response ID 和 prompt item ID；在 `InputProcessor.assign_request_id` 返回时记录 external/internal ID。内部随机后缀保持原样。`p31-measure-` 仅用于选择正式请求，不用于猜测身份关联。
2. **请求 → 调度**：记录 scheduler admission；`schedule()` 调用前读取 host 上的 `num_computed_tokens/num_prompt_tokens`，返回时记录 scheduled tokens。prefill/decode 由调度前进度与 prompt 长度确定。
3. **调度 → 执行**：在关闭 async scheduling 的 TP1 eager 配置下，用完整 scheduled-token 映射及该映射的出现次数匹配 scheduler 与 worker。两个序列都采集并校验。扩展异步执行时必须重新核对该契约。
4. **执行 → kernel**：`record_function` 标注 execute/forward/sample/sampler；使用 `async_npu`、`HostToDevice` 和 CANN connection ID 关联。重复 flow ID 使用 task-queue enqueue/dequeue 证据消歧。设备任务再用名称、stream、Task ID、开始时间、时长校验 `kernel_details.csv`。不按最近时间猜测归属。
5. **Stream 身份**：对框架已返回的 stream 对象读取 Python ID、handle，并查询 `aclrtStreamGetId`。报告分别保留这些身份和 profiler Stream ID。本版没有 native create/destroy 全生命周期审计，不声称已证明所有 stream 的创建位置或 handle generation。
6. **重叠**：仅核对 CSV 中的计算任务；copy、event、notify 不纳入计算重叠。P31b 的 `DSA_SQE` 随机数任务列入计算并单独区分于 AI Core／Vector。区间扫描统计至少两条不同计算 stream 同时活跃的时间并集，三条 stream 同时执行不会重复累加。copy/compute 重叠另算。NPU 上报区间重叠不等于 SM/AICore 资源占用或性能收益。
7. **计时**：客户端同时保存 wall 和 monotonic 纳秒时钟；延迟使用 monotonic。trace 大时间戳使用 Decimal，HTML 在转换为浮点前减去 epoch。SSE chunk 可能携带多个 token，报告不把 chunk 间隔伪装成逐 token 延迟。

观测器只读取 host 元数据，不取设备张量值、不创建 stream、不插入设备等待；元数据先缓存，在 profiler 停止后写出。Python hook 和 profiler 会改变 CPU 提交速度、批次形态和时序。因此性能只看 benchmark，诊断结果只证明该诊断运行内的实际行为。

## 报告使用

`index.html` 汇总每个案例。单案例 `analysis/report.html` 包含 gzip 压缩的完整数据，不依赖网络或 HTTP 文件服务器，用支持 `DecompressionStream` 的现代浏览器直接打开。

- 选择用户请求：看到其参与的所有 step 和共享 kernel。
- 再选择一个 step：自动放大到设备执行窗口，查看请求集合和 prefill/decode。
- 点击 kernel：查看物理 Stream ID、Task ID、trace index、CSV 行、host 算子、native 提交和 flow ID。
- 重叠列表：定位一段精确重叠窗口及同时执行的 kernel；无重叠时只有覆盖率通过后才显示零重叠结论。
- `analysis.json.gz` 保留结构化数据；`tasks.csv.gz` 可离线二次分析。

这是请求、批次和执行顺序的关联实验，不是完整的逐 kernel 张量读写依赖 DAG。

## 验证

```bash
python -m unittest discover -s practice_31_multiuser_streams -v
node practice_31_multiuser_streams/check_report.cjs \
  practice_31_multiuser_streams/results/formal-01/c8-diagnostic/analysis/report.html
```

测试覆盖边界相接、三 stream 并发去重、copy 排除、时间戳精度、缺失身份不猜测、共享批次 flow/CSV 关联，以及本机 HTTP 闭环多用户。浏览器检查使用已有 `/tmp/inference-report-check/node_modules/playwright`，也可通过 `P31_BROWSER_PACKAGE` 指定安装位置。

在保留原始 profiler 的远端可重新抽查、归档（目标目录必须不存在）：

```bash
$PY practice_31_multiuser_streams/audit_raw.py /data/tianchi/practice_31_multiuser_streams/results/formal-new
$PY practice_31_multiuser_streams/archive.py \
  /data/tianchi/practice_31_multiuser_streams/results/formal-new \
  /data/tianchi/p31-evidence/formal-new
```
