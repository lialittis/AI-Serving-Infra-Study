# P31b：eager 随机采样的提前生成开／关对照

本轮固定 eager 和随机采样，只切换 `enable_async_exponential`。问题分为三层：随机数任务何时提交、采样如何等待随机数、设备上是否真的与模型计算重叠。两种模式都可能使用框架的随机采样辅助 stream；开关并不代表“单 stream／多 stream”。

## 实测结论（sampling-01）

2026-10-04（Europe/Berlin；主机日志仍为10月3日 UTC）完成12个案例，**480个 benchmark 请求 + 240个 diagnostic 请求 = 720个测量请求**，均成功输出64 tokens。四组诊断的787个 step、302,069个计算任务全部完成请求、host、CANN、设备 trace/CSV 关联。

| 用户数 | 提前生成 | steps | 计算任务 | 计算 streams | 全窗口计算重叠 | 同 step q/forward 重叠 | q/forward 有重叠的 steps |
|---:|---|---:|---:|---|---:|---:|---:|
| 8 | off | 195 | 74,892 | 44、46 | 24.840 ms | 0 | 0 |
| 8 | on | 195 | 74,892 | 44、46 | 0 | 0 | 0 |
| 32 | off | 198 | 75,927 | 44、46 | 93.699 ms | 0 | 0 |
| 32 | on | 199 | 76,358 | 44、46 | 10.351 ms | 10.351 ms | 164 |

**两组都使用两条计算 stream：模型在46，随机数生成在44。提前生成改变了重叠对象。** 关闭组的重叠发生在随机数生成与排序、top-p、softmax 等采样任务之间；没有与同一步模型 forward 重叠。开启组8用户时，随机数结束得太早，模型首 kernel 开始前已经完成；32用户时随机分支更长，尾部与模型前部计算出现真实重叠。

因此“总重叠变少”和“成功让 q 与模型重叠”可以同时成立。重叠量不能单独作为优化收益指标。

### 匹配实际满批 decode 后的提交与同步

下表仅比较实际 batch=8／32、纯 decode 的 step，时间是中位数。C8 两组均186个 step；C32 off179个、on178个，未假定两次服务具有完全相同的合批顺序。

| 用户数 | 提前生成 | q 首次 native 提交 − forward 首次 native 提交 | q 最后计算结束 − forward 首个 kernel 开始 | q 消费同步 |
|---:|---|---:|---:|---|
| 8 | off | +17,266.611 µs | +17,455.278 µs | 主 stream 设备 EVENT_WAIT，0.020 µs |
| 8 | on | −393.234 µs | −193.948 µs | host aclrtSynchronizeEvent，1.649 µs |
| 32 | off | +18,207.265 µs | +18,793.992 µs | 主 stream 设备 EVENT_WAIT，78.403 µs |
| 32 | on | −399.119 µs | +191.988 µs | host aclrtSynchronizeEvent，1.746 µs |

这些数值来自插桩诊断运行，尤其约17–18 ms的提交间隔不能作为无插桩模型耗时。开启后 q 的首次 native 提交约提前0.4 ms；C8 的 q 提前完成约0.194 ms，C32 的 q 则延伸至 forward 开始后约0.192 ms。消费时 host 同步调用约1–2 µs，说明此次 API 很快返回；不能据此断言其他负载也不会等待。

C32 on 的满批 decode 中155/178个 step有 q/forward 重叠，合计10.020 ms；其中 DSA/forward 为1.476 ms，其他随机数计算/forward 为8.544 ms。具体相交任务包括 `MaskedFill` 与模型 `Cast`／`MatMul`、`Log` 与 `MatMul`，以及 DSA uniform 与 `Embedding`。这既包含 DSA 任务重叠，也包含两个计算分支上的 AI Core／Vector 任务重叠；仍不代表已经测得硬件资源利用率。

### 无插桩性能：只作本轮描述

| 用户数 | off 两个服务区块 tokens/s | on 两个服务区块 tokens/s | 均值变化 |
|---:|---|---|---:|
| 8 | 664.58、662.50 | 595.94、654.69 | −5.76% |
| 32 | 2222.74、2024.48 | 2300.38、2232.02 | +6.71% |

每种设置只有两个独立服务区块，组内波动明显；早期 C8 trace 后处理还与部分 C32 benchmark 服务生命周期并行，因此 C32 的 host 负载并未严格隔离。这些数据支持“本轮观测值”，**不能证明稳定加速，也不能将差异全部归因于 kernel 重叠**。若下一步要做性能结论，应先在没有采集和后处理后台任务的窗口增加独立重复，再采用更低扰动的设备诊断验证重叠是否仍然存在。

## 固定配置与复现

沿用单个 vLLM HTTP 服务、Qwen2.5-0.5B-Instruct、TP1、BF16、物理 NPU 5／逻辑 NPU 0。8 和 32 个独立 HTTP 用户，每人闭环发送3个请求，每个128输入／64输出 tokens，预热每人2个请求。`temperature=0.8, top_p=0.9`，服务 seed=123，不设置 per-request seed；后者可能进入另一条逐请求 generator 分支。

两档服务容量均固定 `max_num_seqs=32, max_num_batched_tokens=4096, max_model_len=512`。prefix caching、chunked prefill、async scheduling 关闭。P31a 使用 greedy 和不同容量，因此不把两轮吞吐差异归因于单一因素。

每档 benchmark 按 **off → on → on → off** 启动4个独立服务，均无 observer／profiler；每种设置有两个服务区块。diagnostic 每种设置各启动一个服务，C8 顺序 off/on，C32 顺序 on/off。随机输出不要求逐 token 相等；实际到达、合批形态和每个 step 的 batch/phase 均保存。

在 Ascend 主机仓库根目录运行，输出目录必须不存在：

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
PY=/usr/local/python3.12.13/bin/python
$PY practice_31_multiuser_streams/run_sampling.py \
  --output /data/tianchi/practice_31_multiuser_streams/results/sampling-new
$PY practice_31_multiuser_streams/analyze.py \
  /data/tianchi/practice_31_multiuser_streams/results/sampling-new
```

资格检查可指定 `--phase diagnostic --concurrency 8 --rounds 1 --warmup 1 --output-tokens 4`；这些短运行不计入正式结果。

## 提交与同步的源码关系

本轮核对安装版本的 `vllm_ascend/worker/model_runner_v1.py`、`vllm_ascend/sample/sampler.py` 和 `vllm_ascend/utils.py`，源码与哈希在采集目录保存。

**关闭提前生成**：host 先执行模型 forward 的提交路径；到随机采样 `random_sample()` 时，才在 `global_stream()` 上提交指数随机数 `q` 的生成。主 stream 调用 `wait_stream(global_stream)`，建立设备 event record/wait 依赖，再执行 `probs.div_(q)` 和 `argmax`。这个设备等待不等于 CPU 在该处等待随机数执行完毕。

**开启提前生成**：`do_async_exponential()` 在 host 模型 forward 之前提交 `q`，并记录对应 event；到 `forward_native()` 消费 `q` 时调用 `self.async_event.synchronize()`。实测通过同一 event handle 将它关联到 CANN 的 `aclrtSynchronizeEvent`，返回后才提交 `probs.div_(q)`。这是 host 同步 API；其调用时长包括 API 开销，不能全部解释为等设备的净阻塞时间。即使 `q` 早已完成，该调用仍然存在。

另外，安装源码在进入 `torch.npu.stream(global_stream())` 后调用 `global_stream().wait_stream(torch.npu.current_stream())`。观测到两端均为同一条辅助 stream44；不能只凭这行 `wait_stream` 的名字，把它解释为“辅助 stream 等待模型 stream”。消费端的 event 同步另行核对。

两条路径共享的算法含义是用指数随机变量参与随机采样。辅助 stream 由框架创建和选择；并不是给每个 HTTP 用户创建一条 stream。开关改变的是随机数生成的提交位置和消费同步方式，模型 forward 本身仍需根据设备 trace 单独判断使用了哪些 stream。

## 如何证明

- 沿用 P31a 的请求身份、scheduler/worker step、host flow、CANN connection 与设备 CSV 精确关联；缺项标为 incomplete。
- 每 step 核对 `q` 的 shape/dtype；开启组还核对 producer/consumer 的设备地址、storage 地址和 event handle 一致。仅读取 host 元数据，不读回随机数内容。
- `q_ready` 记录名表示 Python 生产函数返回时读到了元数据，**不表示设备计算完成**。Python wall clock 与 profiler 对齐后的时间戳不用于微秒级跨域筛选；消费端由同一 profiler 时间域内的 scope 和 CANN 同步边界定位。设备完成关系由随机分支末尾 `EVENT_RECORD` 和消费端同步证据核对。
- `native_submission_delta_us = q 首次 CANN 调用开始 − forward 首次 CANN 调用开始`，负值表示 q 先提交。这是观测到的 CANN 调用入口时刻，并非设备执行时刻或完整 runtime 队列解码。
- `q_end_minus_forward_start_us = q 最后一个计算任务结束 − forward 首个计算任务开始`，负值表示模型首 kernel 开始前 q 已完成。
- 同一步 q/forward 重叠对两组设备计算区间取交集并集；全窗口跨 stream 重叠另算，可能来自 q 与 softmax 等采样任务，不能当作 q 与模型 forward 的重叠。
- `DSARandomUniform` 在本版本 profiler 中为 `DSA_SQE`，出现在 kernel CSV 中，计入设备计算，同时单独报告 DSA 与其他随机数计算的 forward 重叠。它不等于 AI Core／Vector kernel；copy/event 不计入计算重叠。
- 关闭组的 `EVENT_WAIT` 是设备区间；开启组 `aclrtSynchronizeEvent` 是 CPU API 区间，两者时长不可直接作为“CPU 阻塞增减”相减。

诊断插桩会改变 host 提交速度、批次和间隔。设备区间相交证明该次诊断运行的时间重叠，不证明硬件单元占用率，也不能直接解释无插桩运行的吞吐。

## 阅读报告

正式报告位于 `results/sampling-01/index.html`。进入 diagnostic 案例，点击“随机分支提交与同步”中的“定位代表性 step”，查看 host 的 `async_exponential`／`inline_exponential`、forward、`q_wait` 和两条设备 stream；选中 step 时范围同时覆盖 host 提交与设备执行。`samplingDetail` 给出提交差、同步类型、q 的身份和同一步重叠。

“跨 stream 计算重叠证据”的下拉框可定位具体相交任务。任务详情保留原始 trace index、CSV 行、CPU 算子和 CANN 提交。`sampling_examples.json` 提供少量可直接审阅的完整任务证据；逐 step 全量数据保留在忽略的 `sampling_steps.json.gz` 和离线 HTML 中。

## 限制、验证和归档

- 安装源码 revision 与 P31a 一致：vLLM `ad7125a431e176d4161099480a66f0169609a690`，vllm-ascend `80610e4438dba05011b05f89fc45d91e96992671`。该同步路径的结论限定于本次记录的版本和设置。
- 设备仍报告 `Alarm / 80C98001`；本轮没有 reset 或修改安装源码。采集成功不等于硬件健康检查通过，不作为健康设备的通用性能基准。
- 四个真实诊断 HTML 已通过离线 Chromium 交互检查，221个归档文件的大小和 SHA256 全部核对。实验结束后 NPU 无遗留运行进程。
- 17项本地回归测试通过，覆盖精确时间、DSA 分类、event 身份、开关唯一性、共享批次、HTTP 闭环，以及“采样重叠不能当作模型重叠”。四组诊断均要求完整关联通过后才报告零／非零重叠。
- `sampling_examples.json` 保存 q、ready event、消费者和模型首任务的精确来源；`sampling_spot_checks.json` 独立重读原始 trace/CSV，并核对所有 step 的 stream wait 两端身份。完整 HTML、observer、全量逐 step 和任务表不加入 Git。
- 原始 profiler 留在 `ascend910:/data/tianchi/practice_31_multiuser_streams/results/sampling-01/`；恢复离线完整报告可在仓库根目录执行：

```bash
ssh -F ~/.ssh/config ascend910 'cat /data/tianchi/p31b-sampling-evidence.tgz' > /tmp/p31b-sampling-evidence.tgz
tar -xzf /tmp/p31b-sampling-evidence.tgz -C practice_31_multiuser_streams/results
```

完整分析归档约137 MB，压缩包约81 MiB；Git 可见的小型证据约1.85 MB，单文件均小于256 KiB。归档不含约2.71 GB原始 profiler；重做 raw flow join 需按 `archive_manifest.json` 恢复原始文件，浏览离线 HTML 无需恢复它们。本轮是批次、提交和执行区间的关联实验，尚未建立完整的逐 kernel 张量读写依赖 DAG。
