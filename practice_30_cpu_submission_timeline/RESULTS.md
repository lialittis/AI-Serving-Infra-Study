# 发现：主线程大部分时间在运行，结果回收有同步，但它不是本轮耗时大头

2026-09-30，单请求、BF16、eager、原生 `LLM.generate()`。先打开[报告](report/index.html)，默认展示 decode 32。

本文保留首轮 eager 结果。后续已完成同配置的 graph 分析与新 eager 对照，见 [graph 发现](GRAPH_RESULTS.md)及[双模式报告](report/comparison/index.html)；下文旧轮数值不与新轮诊断值混算。

## 1. “下发慢”与“被同步阻塞”已经能分开观察

| 请求 | 墙钟 | 提交线程 CPU 时间 |
|---|---:|---:|
| 无 profiler 参考 1 | 734.326 ms | 730.921 ms |
| 无 profiler 参考 2 | 789.811 ms | 785.901 ms |
| 无 profiler 参考 3 | 740.014 ms | 736.641 ms |
| 带观察器的诊断 | 935.978 ms | 935.032 ms |
| 新进程恢复请求 | 760.082 ms | 759.498 ms |

在这组请求中，提交线程 CPU 时间接近墙钟时间。**它大部分时间获得 CPU 执行，而非长时间没有运行。** 这比单凭 Python scope 长度推断“CPU 忙”更具体。

但线程 CPU 时间包含 Python、C++、运行时工作和可能的自旋，不能区分有效工作、锁自旋或 profiler 成本，也不是整机 CPU 利用率。诊断请求比三次参考的中位数长约 **26.5%**；这个差值混合观测开销和独立运行差异，不是精确的 profiler 成本估计。

## 2. 一个 decode 步，时间主要落在哪里

decode 32 的观测值（以下为嵌套计时，不能全部相加）：

| 阶段 | 墙钟 µs | 线程 CPU µs |
|---|---:|---:|
| 引擎整个 step | 14,767.383 | 14,757.629 |
| 调度 | 42.417 | 42.070 |
| 请求状态更新 | 74.001 | 73.248 |
| 输入准备 | 560.582 | 559.671 |
| 模型 forward | **11,947.488** | **11,945.821** |
| logits | 35.846 | 35.264 |
| 采样提交 | 244.569 | 243.830 |
| token 回传与列表转换 | 104.568 | 104.003 |
| logprob 回传 | 89.854 | 79.781 |
| scheduler 输出更新 | 22.440 | 22.061 |
| 输出处理 | 72.161 | 71.781 |

forward 包含逐层框架执行、算子准备和提交，约占这个 step 墙钟的 **80.9%**。它的 CPU 时间几乎等于墙钟时间。本轮没有继续把 forward 的全部 CPU 时间拆成 Python、C++、allocator、运行时自旋等，因此不能宣称已定位到某个具体函数的性能缺陷。

报告另列每个父阶段扣除直接子阶段后的自身时间。例如 executor 提交自身约 543.948 µs，runner execute 自身约 670.571 µs；这些剩余范围含尚未单独包装的框架工作及观察开销，未统一归为 Python 开销。

## 3. 真实主线程、队列和下发线程

本轮主线程 OS TID **1419007**，torch-npu 下发线程 **1419686**。全部相关 NPU 任务使用物理 stream **46**。

```text
主线程：调度 → 输入准备 → PyTorch 算子 → Enqueue → 继续下一项工作
                                                │
下发线程：                                 Dequeue → CANN launch
                                                         │
NPU stream 46：                                      执行已提交任务
```

共验证 **21,512 对 Enqueue/Dequeue**；它们通过实际 flow 和 correlation ID 配对，不能用名称或“时间最近”替代。**19,298 个计算任务**全部通过 kernel CSV 身份核验并归属到 64 个步骤。

设备记录共 21,793 项，其中 21,792 项有队列关联；另一个 `PLACE_HOLDER_SQE` 没有 Host/CANN 来源，保留为未归属运行时记录，未强行分配给请求。队列数不等于 kernel 数，一个队列任务可能产生多项设备工作。

四个按钮分别展示已验证的 H2D、第一项线性层计算、RoPE 和 token D2H。H2D/D2H 方向还核对了队列操作名 `acl_memcpy_host_to_device` / `acl_memcpy_device_to_host`。

## 4. 同步确实存在，主要位于结果回收

整条请求共观察到 **256 次**原生 CANN 同步：

| 来源 | 次数 | CANN 调用范围总计 |
|---|---:|---:|
| token 回收 `aclrtSynchronizeEvent` | 64 | 612.869 µs |
| logprob 回收 `aclrtSynchronizeStream` | 192 | 223.236 µs |
| 合计 | 256 | **836.105 µs** |

token 的来源是 `/vllm-workspace/vllm/vllm/v1/worker/gpu_model_runner.py:7099`：

```python
pinned.copy_(sampled_token_ids, non_blocking=True)
self.transfer_event.record()
self.transfer_event.synchronize()
return pinned.tolist()
```

logprob 的来源是 `/vllm-workspace/vllm/vllm/v1/outputs.py:61` 的 `LogprobsTensors.tolists()`，依次对 token IDs、logprobs、selected ranks 调用 `.cpu().numpy()`。本配置每步对应三个 Stream synchronize。

在 decode 32，token Event synchronize 持续 8.642 µs；三个 logprob Stream synchronize 分别为 2.475、0.586、0.373 µs。**这些已观测的同步调用不是这一步约 14.77 ms 的主要组成部分。** 同步函数总时长不是全部潜在等待的上界；未记录的等待、自旋或队列压力仍可能存在。

## 5. 三个最大设备任务空隙，后续任务当时都尚未开始入队

| decode 32 的空隙 | 入队开始前的空隙部分 | 所处阶段与具体记录 |
|---|---:|---|
| 135.186 µs | 103.765 µs | runner execute 中未单独包装的输入/元数据准备范围；能看到 `pin_memory`、`aten::to`、`_to_copy` 等 |
| 134.125 µs | 103.759 µs | `_prepare_inputs`；能看到 slot mapping 提交、slice、sub 等，但没有覆盖全部区间 |
| 127.285 µs | 97.233 µs | forward；`vllm::npu_rotary_embedding` CPU 范围覆盖其中 118.860 µs，随后出现 `_triton_rope` 入队与下发 |

这些是“当时还没有交出下一份工作”的直接时序证据。第一、二个空隙并未被列出的 CPU 算子完全解释；第三个虽然定位到 RoPE 调用范围，也没有继续区分 Triton launcher 内部的具体工作。**空隙与某调用重叠不等于已经证明全部因果。**

对应源码 `/vllm-workspace/vllm-ascend/vllm_ascend/ops/rotary_embedding.py:153` 的 `rope_forward_oot`，在本轮已观察到的 Triton 路径中调用 `rope_forward_triton`（168 行）。线性层入口在同目录 `linear.py:51` 的 `unquantized_gemm`，调用 `torch.nn.functional.linear`（57 行）。这些是静态源码入口与已观测任务的解释，不伪造未采集的完整 Python 栈。

## 6. 两个实际代码细节

**Future 不等于后台执行模型。** `/vllm-workspace/vllm/vllm/v1/executor/uniproc_executor.py:96–105` 的当前路径先调用 worker 方法，再创建并填入 Future。本轮 64 次 executor 返回时，Future 全部已经 done。设备计算仍可异步进行；CPU 模型函数的执行位置与设备完成是两件事。

**实际 schedule 是 Ascend 包装类。** introspection 指向 `/vllm-workspace/vllm-ascend/vllm_ascend/patch/platform/patch_balance_schedule.py:71` 的 `BalanceScheduler.schedule`。源码在未启用 balance scheduling 时调用父类；本轮未配置该可选功能。该源码运行后补采并与记录 revision 的 Git 内容核对一致，单独标注，不冒充初始 25 文件审计。

## 验证、边界和下一小步

vLLM revision `ad7125a431e176d4161099480a66f0169609a690`，vLLM-Ascend revision `80610e4438dba05011b05f89fc45d91e96992671`；实际导入仍来自 `/vllm-workspace/`。torch `2.10.0+cpu`、torch-npu `2.10.0`、CANN `9.0.0`。

- 三个隔离进程，共 11 次请求（含预热）的 64 个 token、logprob 和结束原因精确相同。
- 1,089 个阶段标记，64 个完整步骤；同线程嵌套自身时间守恒。包装恢复、25 文件前后指纹、NPU 空闲和新进程原生推理复测通过。
- 七项反例测试覆盖嵌套重复计数、跨线程误合并、重叠子阶段、负的自身 CPU 时间、缺失/错位步骤、错误 queue ID 和错误 kernel 身份。离线报告检查覆盖 64 步切换、四条链路、源码和窄屏。
- 不含 HTTP、graph、多请求、OS 调度跟踪或整芯片利用率结论；profiler 观测值不当作无观测性能。

下一轮建议只放大一个已定位的范围，例如 **RoPE 的 host 调用到 Enqueue 之间**，进一步区分 Python 参数处理、Triton launcher 和 C++ 入队边界；暂不增加模型或 benchmark 矩阵。
