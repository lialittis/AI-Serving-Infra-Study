# Forward 内部细分：先用已有 trace 分账，再放大一次 RoPE

2026-09-30。graph 对照已在 `d14b25a` 提交并推送，本轮继续 Practice 30 的第二个 subtask。入口：[forward 报告](report/forward/index.html) · [RoPE 精确 SVG](report/forward/rope.svg) · [完整分析数据](report/forward/forward.json)。

## 1. 现有 forward 时间可以进一步分到哪里？

复用上一轮 `eager-02` / `graph-01` 的 decode 32，不增加测量。先按同线程事件的包含关系建立树，只把互不重叠的顶层范围计入总表；子调用不能再加到父调用上。

**占比使用 profiler 的 forward 范围作为分母**：eager 13,127.186 µs，graph 4,263.469 µs。先前范围内部的双时钟墙钟是 13,098.944 / 4,249.342 µs；两者边界有标记开销差异，不混算。

| 顶层 Host 范围 | eager Host µs / 占比 | graph Host µs / 占比 |
|---|---:|---:|
| `unified_attention_with_output`，24 次 | 3,767.559 / **28.70%** | 2,839.592 / **66.60%** |
| RoPE，24 次 | 2,465.838 / **18.78%** | 本步无独立 Host 调用，相关 kernel 在图内 |
| GEMM，96 次 | 1,635.459 / **12.46%** | 本步无此独立 Host 范围 |
| add RMSNorm，48 次 | 1,185.222 / **9.03%** | 本步无此独立 Host 范围 |
| 其他顶层 CPU 操作 | 829.844 / **6.32%** | 无其他已记录顶层 CPU op |
| 25 次 replay profiler 范围 | 未发生 | 662.933 / **15.55%** |
| 未被上述范围覆盖 | 3,243.264 / **24.71%** | 760.944 / **17.85%** |

**graph 下主要剩余的已知 Host 范围是图外 attention。** 这些是调用时间，不是 NPU attention 的纯计算时间；RoPE/GEMM 的 Host 范围消失也不表示相应设备计算消失。

之前 graph 的 25 次 replay 内部双时钟墙钟合计为 316.317 µs，此处包含 profiler 标记边界的范围合计为 662.933 µs，差 346.616 µs。这部分不能误记为 native replay 工作。两种计时边界分别保留在报告中。

## 2. 图外 attention 包含什么？

对 graph 的 24 次 attention，只累加各自的直接子范围及父范围剩余，恰好得到 2,839.592 µs：

| attention 内部范围 | 次数 | Host 墙钟合计 µs |
|---|---:|---:|
| `npu_fused_infer_attention_score` | 24 | 742.934 |
| `aten::copy_` | 48 | 413.705 |
| `aten::slice` | 144 | 290.193 |
| `atb::_npu_reshape_and_cache` | 24 | 286.004 |
| `aten::view` | 72 | 93.471 |
| attention 自身，未被这些子范围解释 | 24 | **1,013.285** |

报告可展开第一处 attention 的完整已记录调用树。`self` 只是扣除直接子范围后的余量，可能包含参数准备、框架调度和观测成本，不等于“全是 Python”或“全在等待”。同理，不能只凭 `copy_` 名称认定是 H2D，或把其 Host 时间当成传输耗时。

## 3. 新实验只放大一次预热后的 RoPE

新增 `--forward-detail`，只在 eager diagnostic 中选中 **decode 32 的第一次 `rope_forward_triton`**。正常请求、64 个生成步骤、采样与 KV 增长不变；不增加设备同步，不修改安装源码。

沿实际远端入口读代码：

1. `/vllm-workspace/vllm-ascend/vllm_ascend/ops/rotary_embedding.py:153` 的 `rope_forward_oot`，在 `:168` 调用 Triton 分支。
2. `/vllm-workspace/vllm-ascend/vllm_ascend/ops/triton/rope.py:256` 的 `rope_forward_triton`，处理形状、stride、连续性条件和 grid，再调用 `_triton_rope[...]`。
3. `/usr/local/python3.12.13/lib/python3.12/site-packages/triton/runtime/jit.py:566` 的 `JITFunction.run`，读取设备/stream，绑定参数，查缓存，准备 launch metadata。
4. 同文件 `:650` 的 `kernel.run(...)`，本次实际 callable 类型为 **`ascend.NPULauncher`**。这是 native 边界；不能把安装目录里同名 Python launcher 类的源码直接当成本次 native 实现。

参数 binder 是生成的 `dynamic_func`，没有取到独立源码文本；保留实际调用范围，不伪造源码行号。native launcher 内部也未采集逐函数 C++ 栈，本轮仅将其外层调用与 profiler 的 Host/队列/CANN 记录关联。

新增四个嵌套范围的内部双时钟：

| 范围 | 墙钟 µs | 线程 CPU µs | 自身墙钟 µs |
|---|---:|---:|---:|
| `rope_forward_triton` Python 包装 | 201.253 | 200.749 | 34.294 |
| `JITFunction.run` | 166.959 | 166.554 | 113.621 |
| 参数 binder | 16.878 | 16.375 | 16.878 |
| native launcher | 36.460 | 35.910 | 36.460 |

JIT 自身余量仍包含 binder/native 子包装的边界成本，不能把 113.621 µs 全部归因于缓存查找或参数处理。初始化及预热留下的三个缓存 kernel 均已加载；选中调用的编译次数为 0，前后缓存指纹相同。这是使用已编译 kernel 的一次执行，不是首次编译成本。

## 4. 从 native launcher 到设备的精确关联

本次选中的记录是：

```text
CPU rope_native scope
    → Enqueue@_triton_rope / correlation 54355
    → 对应 Dequeue / correlation 54355
    → CANN connection 101148
    → NPU stream 46 / task 55088 / kernel _triton_rope
```

kernel CSV 的名称、stream、task、开始时刻和持续时间全部核验；NPU kernel 持续 **3.320 µs**。

native 的 profiler 范围为 **50.780 µs**，可按观测边界分为：Enqueue 开始前 31.088 µs、Enqueue 自身 3.590 µs、Enqueue 结束后 16.102 µs。这个分母包含 profiler 标记，**不能与上表内部计时的 36.460 µs 混算**。Dequeue 范围为 3.322 µs；它属于下发线程，与主线程范围不能串行相加。队列边界差也不等于纯排队等待。

## 5. 这次细分的观察成本必须保留

被选中的整个 `vllm::npu_rotary_embedding` Host 范围为 **268.323 µs**；同一步其余 23 次未加细分 scope 的范围中位数为 **108.489 µs**，范围 99.598–125.890 µs。细分插桩明显扰动了被观察的调用；不同层及调用顺序也会造成差异，不能把差值当成纯插桩成本的精确估计。

因此，这四段计时用于确认调用结构和边界，不作为生产环境下的稳定百分比。整体 forward 占比使用前一轮已有 trace，避免把这一次重度放大的调用推广到 24 层。

新一轮 reference 三次请求为 712.993 / 716.655 / 703.619 ms，diagnostic 为 1,022.094 ms，约为参考中位数的 1.434 倍；该差值混合 profiler、包装成本和进程间差异。恢复请求为 676.183 ms，不能只凭单次恢复时间宣称性能改善。

## 验证、复现及下一小步

- 本轮 11 次响应（含预热）token/logprob/结束原因精确相等；临时绑定、原生请求和设备空闲恢复通过。
- 30 个已审计文件前后未变，两个库 revision 与 graph 对照一致。新增审计包含 Triton JIT、compiler 和 Ascend driver 源码。
- 1,093 个阶段标记、64 个步骤、21,512 对队列和 19,298 个计算任务均通过关联检查；只有一次 RoPE 增加四个 scope。
- 反例测试覆盖嵌套重复计数、跨线程混算、部分交叠、错误范围，以及绝对微秒时间的精度；不会先转 epoch 时间为浮点数再相减。
- 保留 24.71% eager / 17.85% graph 的顶层未归因时间，以及 attention 内的自身余量和 native 内部未知行为。

复现命令与归档说明见 [README](README.md#forward-内部细分复现)。远端原始目录为 `/data/tianchi/practice_30_cpu_submission_timeline/results/forward-01`，便携证据见 [forward-01-archive](results/forward-01-archive/)。

下一小步建议转向 **graph 的图外 attention**：优先解释其约 1.013 ms 的 Host 自身余量，仍只选一个调用，不立即增加 benchmark 矩阵。当前没有展开此项，也没有优化或改动算子实现。
