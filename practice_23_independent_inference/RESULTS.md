# P3a：独立完整模型 forward 的双流并发与 batching

本轮已完成 Qwen2.5-0.5B-Instruct 完整预训练模型的单卡 eager harness。比较同样两条序列的串行、双 stream 和 batch=2：双流在 prefill 上缩短总完成时间，decode 基本持平；数值通过的三种形状中，batch=2 更快。1024-token prefill 的 batch 未通过数值校验，不能采用其收益结论。

查看[交互执行图](report/index.html)、[原始性能样本](results/published/measurements.json)、[数值检查](results/published/correctness.json)与[验证记录](results/published/validation.json)。

## 环境和工作量

2026-09-29，Ascend 910B2C 单卡，物理 NPU 5 / 逻辑 0；torch 2.10.0、torch-npu 2.10.0、Transformers 5.5.4。复用已有权重，不修改系统安装。24 层、BF16、eager attention、默认 RoPE、无 sliding window。一个 CPU 线程提交；这不是 vLLM 的 native runner、continuous batching、HTTP serving 或 graph replay。

每任务一次完整 forward，返回最后一个位置 logits 与更新后的全部 KV；prefill 长度 128 / 1024，decode 上下文长度 128 / 1024。decode 的前缀预先用相同基线生成；cache clone、batch 拼接、输入准备、同步就绪均在计时外。因此不包含 cache 合批成本、排队、权重加载或完整文本生成。

两任务共享一份只读权重，权重及模型 buffer 的 SHA256 前后相同。A/B 持有独立可变 KV，所有活跃缓存字节范围无跨任务重叠；输出和输入一直保留到两个末尾 event 完成。mask 和位置张量可只读共享。原生算子 workspace 由框架管理，未恢复其精确访存。

## 无 profiler 性能

每形状／模式预热 3 次，正式 12 轮；六种模式顺序轮换，AB/BA 交替，共 144 个 pair 样本。以共同输入已就绪为起点，以两个任务全部完成为终点，包含主机提交和 event 控制。下表为 ms/pair 中位数；配对变化先逐轮除以 serial 再取中位数。

| 形状 | serial | 双流 | batch=2 | 双流配对变化 | 双流更快的轮次 |
|---|---:|---:|---:|---:|---:|
| prefill-128 | 25.263 | 23.322 | 13.685 | -7.73% | 12/12 |
| prefill-1024 | 46.952 | 40.554 | 34.994（数值未通过） | -13.66% | 12/12 |
| decode-128 | 16.954 | 16.931 | 8.711 | -0.24% | 8/12 |
| decode-1024 | 17.326 | 17.316 | 9.162 | -0.19% | 6/12 |

这是本轮交替测量的描述性结果，完整 IQR、极值、每任务 NPU 就绪时间、主机观察时间和 allocated/reserved 峰值见[汇总](results/published/summary.json)和原始样本。没有做跨进程／跨日统计显著性检验。

## 吞吐与首任务延迟的取舍

| 形状 | 模式 | 较早完成 ms | 两任务完成 ms | tasks/s | forward 增量峰值 MiB |
|---|---|---:|---:|---:|---:|
| prefill-128 | serial | 12.709 | 25.197 | 79.2 | 8.25 |
| prefill-128 | parallel | 14.097 | 23.252 | 85.8 | 8.25 |
| prefill-128 | batch | 13.617 | 13.617 | 146.2 | 12.81 |
| prefill-1024 | serial | 23.587 | 46.872 | 42.6 | 175.09 |
| prefill-1024 | parallel | 31.163 | 40.471 | 49.3 | 175.09 |
| prefill-1024 | batch | 34.917 | 34.917 | 不采纳 | 325.28 |
| decode-128 | serial | 8.602 | 16.892 | 118.0 | 0.76 |
| decode-128 | parallel | 8.601 | 16.868 | 118.1 | 0.76 |
| decode-128 | batch | 8.647 | 8.647 | 229.6 | 0.96 |
| decode-1024 | serial | 8.956 | 17.251 | 115.4 | 3.94 |
| decode-1024 | parallel | 8.904 | 17.244 | 115.5 | 3.94 |
| decode-1024 | batch | 9.085 | 9.085 | 218.3 | 7.32 |

设备就绪时间用公共 origin event 到各 terminal event 的 elapsed_time，包含提交供给造成的空闲；wall time 还含 host 等待返回。first/last 与 A/B 身份区分：AB/BA 交替后按每次较早／较晚任务统计，A/B 单独分布另存 JSON。batch 两任务同时就绪。

prefill 双流缩短整体完成时间，但较早任务比串行更晚完成，不能把吞吐收益表述为两个任务都降低延迟。三个数值通过的形状里，batch 优于双流；batch 使用更大的算子临时内存。显存增量从准备完成后计算，decode 初始 KV 不在增量内；所有模式共享同样的已保留基线和权重。

## 真实 kernel 执行图

24 个诊断 trial；完整采集含 58,981 个设备任务、56,828 个计算任务（包括输入准备／校验等 scope 外任务）。所有 CSV 计算条目已核对精确身份。下表重叠来自各 forward 内实际 kernel 区间的并集求交，不是两个 forward 起止范围的交集。

| 形状 | 双流计算重叠 µs（两次） | 实际计算 stream |
|---|---|---|
| prefill-128 | 0.000 / 0.000 | 43 / 44 |
| prefill-1024 | 12829.794 / 12853.750 | 43 / 44 |
| decode-128 | 0.000 / 0.000 | 43 / 44 |
| decode-1024 | 0.000 / 0.000 | 43 / 44 |

1024-token prefill 两次诊断均观察到约 12.8 ms 的跨任务计算重叠；两条计算 stream 为本轮物理 43 / 44，默认 origin stream 为 46。短 prefill 和两个 decode 形状的诊断均为 0。短 prefill 的无 profiler 耗时改善尚不能直接归因于计算重叠。

任务边界关系：`输入／独立初始 KV 就绪 → origin event → A/B 各自 wait → 完整 forward → 各自 terminal event → host join → 校验／释放`。串行额外由同一 stream 的 FIFO 将先提交的完整 forward 排在后提交者之前；双流两条任务链仅共享起点与主机汇合，没有 A→B 数据依赖。batch 只有一条共同执行链，每个 kernel 处理两个 batch 行。

独立校验了 80 条输入就绪／输出完成要求和 16 项跨任务顺序检查；未将这些要求添加到同步图中自证。实际设备 event-wait 边 24 条，只有 API 证据的跨流等待 16 次。同流等待由 FIFO 覆盖。

该图是完整模型运行的 **kernel execution DAG**，包含实际任务、FIFO、event 和 host completion。它仍不是全模型每个 native kernel 的完整精确内存数据依赖 DAG；没有据此给出自动 stream 分配或理论最大加速比。跨 trial 的所有主机阶段也未建成完整因果图；HB 结论限定于已声明的每个 trial 内边界。

Profiler 与性能采集分开，诊断只增加外层 forward/event scope，仍可能改变主机提交时序。设备诊断有无重叠都不能直接当作无 profiler 的逐微秒重放；耗时和 trace 是两类互补证据。

## 数值失败项

全部 serial/parallel logits 与 24 层 KV 逐元素相同。三个通过形状的 batch 也逐元素相同。1024-token prefill 的 batch 两任务贪心 token 与 serial 相同，但 logits 最大绝对差分别为 0.3125 / 0.25，KV 最大绝对差分别为 2.453125 / 1.736328125；预设 `atol=0.0625, rtol=0.02` 不通过。

最终保留 14 个失败对照（12 次性能、2 次诊断），没有调整容差。独立资格复核证明该差异随 long-prefill batching 出现，串行重复与双流完全一致。尚未定位到具体原生算子／tiling 舍入机制，不能直接宣称是正常 BF16 舍入，也不能当作 stream 竞态。进一步量化 batching 收益前需定位首次分歧或建立可靠高精度参考。

后续进展见[数值定位报告](NUMERICS.md)：已找到首次分歧并完成高精度核验，另有独立全 FP32 控制组的有效比较；本页保留原始 BF16 实验结论。

## 重放与下一步

[证据包](results/published/evidence.tgz)保留 trace、kernel CSV、源码、参数、全部样本及中断／资格检查；[文件哈希](results/published/evidence_manifest.json)和[归档说明](results/published/archives.json)支持独立重放。初始化硬件历史 Alarm 仍存在，实验前后记录中未见其他 NPU 作业；未修改或重启设备。复现命令见 [README](README.md)。

后续[数值定位](NUMERICS.md)已完成，并补充独立全 FP32 控制组；原始 BF16 失败保持不变。下一项为 graph capture/replay 条件核验及同精度对照。P3b 视觉／语言并发仍需已有适配模型及图像输入，尚未开展。
