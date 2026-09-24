# 结果：两个模型都能触发真实多 stream，触发阈值不同

## 六组对照

| 模型与条件 | batch | 请求级 generator | 随机分支 / 满批 step | 发生重叠的 step | 模型/随机 compute 交集 |
|---|---:|---:|---:|---:|---:|
| Qwen，async 开，请求级 seed 校准 | 32 | 32 | 224 tasks | 0/5 | 0μs |
| Qwen，async 开 | 32 | 0 | 7 tasks | 4/5 | **105.385μs** |
| Qwen，async 关 | 32 | 0 | 7 tasks | 0/5 | 0μs |
| Llama，async 开，阈值对照 | 32 | 0 | 7 tasks | 0/5 | 0μs |
| Llama，async 开 | 64 | 0 | 7 tasks | 4/5 | **1082.181μs** |
| Llama，async 关 | 64 | 0 | 7 tasks | 1/5 | **1052.263μs** |

所有 run 都使用物理 stream 44 生成随机张量，物理 stream 46 执行模型和采样主体。Qwen 正式开/关组各有 2,189 个设备任务、1,981 个计算任务；Llama batch 64 开/关组各有 3,785 个设备任务、3,507 个计算任务。所有计算任务都与 `kernel_details.csv` 唯一匹配，观测请求范围外任务为零。

总交集不能单独表示覆盖质量。Llama 关闭组的 1052.263μs 全部来自 step 2 的大 prefill/mixed forward 尾部；另外 4 个 step 为零。开启组在 step 2–5 分别重叠 289.612、283.910、241.229、267.430μs，说明提前分支稳定覆盖了所有满批 step。

## 实际 scheduler 不是理想化的四步

一次 HTTP completions 请求携带多个 prompt，frontend 逐项创建内部 request。scheduler 在全部 request 到齐前已经接收第一项，因此正式 trace 有 5 个 engine step：

```text
step 1：1 条请求 prefill
step 2：其余请求 prefill + 第一条请求 decode，形成满批 mixed step
step 3：满批 decode
step 4：满批 decode
step 5：第一条已完成，其余请求完成最后一次 decode
```

Qwen 满批为 32/31，Llama batch 64 为 64/63。分析器按实际 `scheduler_output.num_scheduled_tokens` 建图，没有把 HTTP 请求预设成固定四步。

## 开启路径的逐 step 图

以 Qwen async 正式组为例：

| step | 活跃请求 | 阶段 | stream 44 随机结束到 stream 46 首 kernel | compute 交集 |
|---:|---:|---|---:|---:|
| 1 | 1 | prefill | +554.822μs | 0μs |
| 2 | 32 | mixed | −55.762μs | 31.841μs |
| 3 | 32 | decode | −56.743μs | 28.202μs |
| 4 | 32 | decode | −46.162μs | 22.721μs |
| 5 | 31 | decode | −47.922μs | 22.621μs |

负值表示模型首个 compute task 开始时，随机分支仍未结束。交集只对两组 compute task 的区间求并集后再相交，所以小于简单的首尾外包络交集。

Llama batch 32 的对应间隔为 +43.201 至 +95.964μs，刚好没有越过模型起点；batch 64 变为 −449.718 至 −475.299μs，并在 4 个满批 step 形成交叠。这解释了为何同一个开关对不同 vocab 和 batch 不会自动产生相同结果。

## 七个随机计算任务

当请求不带独立 seed 时，每一步的 `[batch, vocab]` FP32 `q` 由同一条 stream 上七个计算 task 生成：

```text
DSARandomUniform → Neg → Add → GreaterEqual → MaskedFill → Log → Mul
```

Qwen 的 `q` 为 `[1|32|31, 151936]`，Llama 为 `[1|64|63, 128256]`。每一步生产端与 sampler 入口的 tensor shape、dtype、`data_ptr`、storage pointer 都相等，q-ready event handle 也相等。这条 `data_contract` 不是根据相近时间猜测。

请求级 seed 校准组的 generator 数等于活跃请求数。源码会跳过整批 `q.exponential_()`，逐行执行 `q[i].exponential_(generator=...)`；Qwen 满批 step 因而有 `32 × 7 = 224` 个随机计算 task。设备在 Host 循环投递期间逐行完成，最后一个随机 task 仍比模型首 kernel 早约 439–464μs，零重叠。

## 两种汇合方式

开启路径每一步观察到 q-ready `EVENT_RECORD`，sampler 内有真实 `AscendCL@aclrtSynchronizeEvent`。完整图增加：

```text
最后一个 q producer
  └─data_contract──────────────┐
q-ready EVENT_RECORD           │
  └─event_sync→ Host wait返回  │
                  └─host_after_wait→ div(probs, q) → argmax
```

`do_async_exponential` scope 中还会出现一次同在 stream 44 的前置 event record；trace 没有与它配对的跨 stream `EVENT_WAIT`，因此分析器不伪造跨流边。随机 `q` 不依赖模型输入，允许直接开始。

关闭路径每一步观察到一对 stream 44 `EVENT_RECORD` 和 stream 46 `EVENT_WAIT`：

```text
q producer → EVENT_RECORD ─event_wait→ EVENT_WAIT → div(probs, q) → argmax
      └────────────────data_contract───────────────────────┘
```

这是设备侧等待，Host 提交 wait 时不需要阻塞。模型与随机分支在 wait 之前没有数据依赖，所以当默认 stream 尚有模型任务积压时，两条 stream 仍可能执行重叠；Llama batch 64 的 mixed step 正是这种情况。

## 能推出和不能推出什么

本次已经证明：

- 两个模型在单卡 eager vLLM 请求中都能触发第二条真实计算 stream；
- 特定 batch 下存在物理设备 task 时间交集；
- stream 44 的 `q` 是同一步 sampler 实际消费的数据；
- 开关改变随机分支的 Host 投递位置和汇合机制；
- 每请求 seed、batch、vocab 大小和 Host 投递开销共同决定最终是否交叠。

本次不把交叠微秒数解释为端到端加速。Profiler 会改变 Host 与设备时序，而且两组模型、batch 和采样模式没有做无 profiler 性能重复。设备 task 区间相交也不说明两个 task 在同一时刻占用了哪些 AI Core/Vector Core 子资源。

完整模型内部的逐 tensor 依赖仍以 Practice 15 的局部契约边为边界。本练习完整描述的是这条跨 stream 随机采样分支、它与模型 stream 的执行顺序及汇合数据，不声称恢复所有原生算子的内存读写集合。

正式证据目录：

- [Qwen async batch 32](results/2026-09-24-qwen-enabled-b32-r03/analysis/summary.json)
- [Qwen inline batch 32](results/2026-09-24-qwen-disabled-b32-r02/analysis/summary.json)
- [Llama async batch 64](results/2026-09-24-llama-enabled-b64/analysis/summary.json)
- [Llama inline batch 64](results/2026-09-24-llama-disabled-b64/analysis/summary.json)
- [Qwen 请求级 seed 校准](results/2026-09-24-qwen-enabled-b32-r02/analysis/summary.json)
- [Llama batch 32 阈值对照](results/2026-09-24-llama-enabled-b32/analysis/summary.json)
