# 实测结果：2026-09-28 run01

设备查询得到 **24 个 Cube 核、48 个 Vector 核**。真实 KV 写入 kernel 为 `ReshapeAndCacheNdKernel`，核类型 `AI_VECTOR_CORE`，`Mix Block Num=0`。完整证据见 [报告](results/2026-09-28-run01/analysis/index.html) 和 [采集元数据](results/2026-09-28-run01/run.json)。

| 输入 tokens | 报告 Vector 核数 | 无 profiler 完成耗时 μs/次 | plain kernel 中位数 μs |
|---:|---:|---:|---:|
| 1 | 1 | 10.849 | 1.420 |
| 4 | 4 | 10.535 | 1.660 |
| 10 | 10 | 10.688 | 2.320 |
| 16 | 16 | 10.578 | 2.960 |
| 24 | 24 | 10.579 | 3.920 |
| 32 | 32 | 10.712 | 4.980 |
| 48 | 48 | 10.503 | 6.501 |
| 64 | 48 | 10.487 | 7.060 |
| 128 | 48 | 10.578 | 7.940 |
| 256 | 48 | 10.713 | 9.060 |

**这组输入下，核数增长到 48 后保持不变；更多 token 由同一核数配置处理。** 六次 profiler 观测（plain 和 pipe 各三次）在每个规模上报告一致。`min(tokens,48)` 是这些样本的经验规律，不能推广到所有 shape、dtype、slot 或 kernel 版本，也没有证明内部逐 token 分核方式。

## 耗时说明

无 profiler：7 轮 × 10 个规模，每个 trial 连续写入 100 次，末尾 event 完成后停止计时。表中为每轮完成时间除以 100，再取中位数；包含主机提交、运行时和队列间隙。约 10.5–10.8 μs/次的平台区间不能说明所有 kernel 都耗时一样。

plain 的 1.42–9.06 μs 是独立 profiler 中的设备 kernel 时间。其随规模增长，而完成批次的主机平均时间变化较小；这与短 kernel 场景下提交开销占比明显相容，但不足以精确分解开销或作出性能瓶颈结论。二者不能相减为纯 CPU 开销，也不能相除为加速比。

输入、KV 池反复使用且已经预热，工作集较小。没有做冷缓存或持续 HBM 带宽测量。

## 流水线指标

另一次 Level1 / PipeUtilization 采集提供 `aiv_vec_ratio`、`aiv_scalar_ratio`、`aiv_mte2_ratio`、`aiv_mte3_ratio` 等真实计数器。例如 48 token 的三次中位数分别为 **0.003、0.892、0.044、0.058**。可观察到 vector 算术流水线占比较低、scalar 指标较高；“运行在 Vector 核”不等于主要时间用于向量算术。

这些 ratio 可能重叠，不是全芯片利用率。没有每个物理核的逐时刻活动记录，无法证明 48 核在整个任务期间同时满载；也没有足够证据断言内存带宽瓶颈。

## 正确性与关联

- **100 次完整 KV 池校验全部通过**：10 次预热后、70 次定时后、20 次 profiler 分组后。每次检查 K/V 各 131,072 个元素及输入不变，覆盖已写和未写区域。
- **60 个 KV 设备任务精确关联**：独立的用户标记、PyTorch flow、CANN flow、connection ID 与 CSV 的名称/stream/task/开始时间一致。重置、拷贝、等待及 profiler 控制任务单列为非目标任务。
- 两份采集中 KV kernel 各在单一物理 stream 上有序执行，完成等待先于读取结果。无 profiler 定时使用末尾 event 等待。
- 源码快照与原始 trace/CSV 有 SHA256。离线负向测试验证错误结果、错误 slot、未等待、过大或未知核数、错误 task/connection、缺失 flow/指标会被拒绝。

## 回到模型图

Practice 15 的 eager / graph kernel 节点和八张精确关联 SVG 已追加 CSV 核数字段。可以对照：KV 写入的 prefill 10 token 报告 10 个 Vector 核，decode 1 token 报告 1 个；FIA 的 `MIX_AIC / Block 24 / Mix 48` 是混合核类型字段，需分别阅读；部分 replay 内 RoPE 报告零，显示为未知而非“没有核”。

本次 isolated eager KV 实验解释输入规模与核数的一个具体关系。它没有测量 graph 与 eager 的速度差、跨 stream 核资源竞争，也没有完成内核内部 tiling 的逐项证明。
