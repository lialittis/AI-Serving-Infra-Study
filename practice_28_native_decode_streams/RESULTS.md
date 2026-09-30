# Practice 28 实测结果

2026-09-29，Ascend 910B2C，原生 Qwen2.5-0.5B-Instruct / BF16 / TP1。

**本次实验中，仅把第二个独立 decode 任务改到另一条 stream，没有形成实际计算重叠，也没有显示稳定加速。** 这不意味着 NPU 不支持多流并行，而是当前单 CPU 线程顺序提交的执行方式没有及时提供第二份可并行工作。

打开 [HTML 报告](report/index.html)，先看完成时间，再展开 `eager-plain-same32-parallel-AB` 和 `graph-plain-same32-parallel-AB`：蓝色 A 的设备计算结束后，橙色 B 才开始。图中有实际 CPU scope、物理 stream 行和 kernel 区间；悬停显示原始 trace/CSV 标识。另存 32 张独立 SVG。

## 无 profiler 性能

每后端三个独立进程，各场景、策略每进程 12 次，共 288 个 pair。表中加速比 = 同进程 serial wall-time 中位数 ÷ parallel 中位数，再汇总三个进程。大于 1 表示 parallel 更快；范围不是置信区间。

| 模式 | 两个固定 decode 状态 | 加速比中位数 | 三进程范围 |
|---|---|---:|---:|
| eager | A32 / B32 | 1.003× | 0.997–1.081× |
| eager | A16 / B47 | 0.989× | 0.961–0.998× |
| PIECEWISE graph | A32 / B32 | 0.989× | 0.985–0.991× |
| PIECEWISE graph | A16 / B47 | 1.003× | 0.999–1.020× |

eager 的 pair 通常约 20 ms，graph 约 6.6 ms。它们包含实验的状态绑定、forward、logits 和 sample；不是 HTTP 请求耗时，也不能把 graph/eager 差值归因为 stream 数。

## 设备证据

32 条独立诊断 trace（plain / Pipe 分开）全部完成精确关联，每条包含 576 个计算任务。共 18,432 个 CSV kernel 条目均映射到 A/B；同场景、提交顺序、角色在 serial/parallel 两组的 kernel 名称、核类型和核数字段库存一致。

| 观察 | eager | graph |
|---|---:|---:|
| serial 的实际计算 stream 数 | 1 | 51 |
| parallel 的实际计算 stream 数 | 2 | 52 |
| A/B 计算区间交集，全部诊断 trace | 0 μs | 0 μs |
| plain 双流：第二任务首 kernel/replay 下发晚于第一任务计算结束 | 325–347 μs | 645–774 μs |

graph 的两个实例各有 25 个捕获图，graph pool 分别为 `(0,1)` 和 `(0,2)`。serial 是一个调用流加 50 个图内部计算流；parallel 是两个调用流加 50 个图内部计算流。**内部流数量很多，仍然不等于 A/B 在并行。** 归属依据包括 Python graph 对象、Model ID、debug dump、CANN replay connection 与完成通知，未仅凭 stream ID 猜测。

两次 CPU forward scope 来自同一个线程；显式 CPU join 均在两次提交之后。诊断中第二任务的 CPU forward scope 已晚于第一任务计算结束，因而这一次更换 stream 并没有留下可重叠的 A/B 计算窗口。没有在 forward scope 内观察到主提交线程的 CANN Synchronize 调用；这不能排除所有未插桩的内部等待。

`wait_event` 也有需要区分的两层证据：调用存在，不保证 trace 必有一条设备 `EVENT_WAIT`。部分调用发生时 origin 已完成，trace 只记录运行时调用。分析器核对了该调用的队列关联及生产者已完成的时间关系，不把缺少设备等待条目当成缺少程序依赖。

## 核字段说明了什么

在本次对应 CSV 中，`RmsNorm`、`SwiGlu`、`ReshapeAndCacheNdKernel` 等出现 `Block Num=1`；`FusedInferAttentionScore` 出现 `Block Num=24`、`Mix Block Num=48`。graph 的部分 Triton kernel 核数字段为零或缺失，记录为 unknown。各 kernel 的 Pipe ratio 已保存在报告中。

这些信息说明不同算子的工作划分不同，不能据此断言整卡一直有多少“空闲核”。本次更直接的阻碍是：时间线上没有同时执行的 A/B 工作。没有测得整卡物理核占用率，也没有证明 HBM 带宽是否饱和。

## 正确性、共享资源与恢复

正式套件 `suite-01`：两个后端资格各 34 项通过；每个正式／诊断子进程又执行相同资格检查。288 个正式 pair 和 32 个诊断 pair 的 logits、sampled token、logprobs、完整 KV 池逐元素精确匹配各自原生参考，浮点值均有限。

参数、固定 RoPE 查找表是只读共享数据；KV、输入、输出、采样状态、workspace、可变 RoPE 缓冲、graph pool 独立。扩展 buffer 审计曾把共享 `cos_sin_cache` 也视为可写而拒绝运行；随后直接核查远端 `_ROPE_DICT`、`AscendRotaryEmbedding` 和 Triton kernel 的读取／写入位置，将这张固定表单独审计。该审计修正没有修改模型执行或 stream 提交逻辑，正式数据保持原样。

最终 `qualification-final-02` 两后端各 34 项检查通过，RoPE 表前后哈希一致，恢复通过。[最终审计摘要](results/final_audit.json) 还核对了最终代码与正式采集版本的关键执行函数 AST 一致、测量驱动字节一致。[完整增强资格证据](results/qualification-final-02-archive/archive.json) 单独归档。

所有运行均在独立实验子进程中，不修改已安装源码、shell 配置或已有服务。正式套件恢复检查：源码哈希一致、无遗留 NPU 进程、全新未安装实验适配器的进程输出 64 token 与实验前完全相同。早期失败试跑也保留日志和恢复结果；未将其纳入性能统计。

## 限制

- 结论限定于当前模型、两个固定 batch=1 decode 状态、单 CPU 提交线程、当前 eager/PIECEWISE 配置，不排除其他工作量或提交方式获得多流收益。
- profiler 会改变时间，不能拿其耗时估算无 profiler 加速比；在诊断中未观察到重叠，不等于证明未插桩执行中绝无极短重叠。
- 双 runner 的 Python 状态切换属于实验适配成本。上述提交滞后包含这一成本，不能全部归咎于原生 serving 框架。
- 可见 tensor、图池、源码路径和数值检查不能穷尽底层 native 库的所有隐式状态，也不是任意输入下的并发安全证明。

完整代码入口与复现命令见 [README](README.md)，数值统计见 [summary.json](report/summary.json)。正式原始证据见 [suite-01-archive](results/suite-01-archive/archive.json)，其中包含 trace、CSV、graph dump、冻结的采集与分析代码及逐文件 SHA256 manifest。
