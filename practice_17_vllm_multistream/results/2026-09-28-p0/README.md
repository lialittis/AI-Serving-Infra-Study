# P0：无 profiler 性能与独立 trace

正的耗时降幅表示提前生成更快；负值表示更慢。所有耗时均为完整 HTTP 批量请求的接收时间。

| 模型 | 提交 batch | 输出 token | 开 / ms（中位数） | 关 / ms（中位数） | 耗时降幅 | AB / BA 降幅 | 判断 |
|---|---:|---:|---:|---:|---:|---|---|
| qwen | 1 | 4 | 50.219 | 47.697 | -5.29% | -5.41% / -1.40% | 不确定 |
| qwen | 1 | 64 | 741.753 | 716.950 | -3.46% | -5.60% / -0.82% | 不确定 |
| qwen | 32 | 4 | 72.986 | 69.792 | -4.58% | -6.05% / -1.08% | 本轮退化方向一致 |
| qwen | 32 | 64 | 884.136 | 841.644 | -5.05% | -6.22% / -2.98% | 本轮退化方向一致 |
| llama | 32 | 4 | 58.816 | 58.559 | -0.44% | -0.65% / -0.47% | 不确定 |
| llama | 32 | 64 | 656.902 | 662.397 | +0.83% | -0.63% / +1.55% | 不确定 |
| llama | 64 | 4 | 74.934 | 74.326 | -0.82% | -1.32% / +0.47% | 不确定 |
| llama | 64 | 64 | 830.683 | 814.720 | -1.96% | -3.00% / -1.27% | 本轮退化方向一致 |

开／关各 10 个正式样本；预热排除。判断依据为两组方向一致且合并样本的四分位区间不重叠，
仅作描述性判断，不是统计显著性检验。原始响应与测量位于 `benchmarks/`。

## 单独采集的诊断

这些 trace 有 profiler 和 Python 观测开销，不用于解释上述性能差值的大小。

| 运行 | 实际 step | 重叠 step | compute 交集 / us | 逐 kernel 对齐数 | 代表性 step 时间线 |
|---|---:|---:|---:|---:|---|
| llama-disabled-t4 | 5 | 1 | 1019.739 | 3507 | [step 3](diagnostics/llama-disabled-t4/timeline.svg) |
| llama-disabled-t64 | 65 | 0 | 0 | 44607 | [step 3](diagnostics/llama-disabled-t64/timeline.svg) |
| llama-enabled-t4 | 5 | 4 | 960.061 | 3507 | [step 3](diagnostics/llama-enabled-t4/timeline.svg) |
| llama-enabled-t64 | 65 | 64 | 12231.114 | 44607 | [step 3](diagnostics/llama-enabled-t64/timeline.svg) |
| qwen-disabled-t4 | 5 | 0 | 0 | 1980 | [step 3](diagnostics/qwen-disabled-t4/timeline.svg) |
| qwen-disabled-t64 | 65 | 0 | 0 | 24901 | [step 3](diagnostics/qwen-disabled-t64/timeline.svg) |
| qwen-enabled-t4 | 5 | 4 | 89.703 | 1981 | [step 3](diagnostics/qwen-enabled-t4/timeline.svg) |
| qwen-enabled-t64 | 65 | 62 | 1783.168 | 24901 | [step 3](diagnostics/qwen-enabled-t64/timeline.svg) |

代表时间线优先选择第一个满批 decode step，显示真实 kernel 区间与空隙，悬停可看任务名。
完整 execution graph、原始 trace 与 CSV 在证据包中；这里保留逐 step 摘要及证据清单。

独立数值验证见 [numerics.json](numerics.json)。当前设备仍报告历史硬件告警 `80C98001`；
结果仅描述该主机本轮条件，不代表健康设备上的普遍性能。
