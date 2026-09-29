# 实测结果：少核算子和提交空隙同时存在，单流不足以解释全部现象

本次完成 4 个无 profiler 服务对照与 4 个独立 profiler 服务采集。模型为 Qwen2.5-0.5B-Instruct，BF16、batch 1、10 输入 / 64 输出。打开 [交互报告](report/index.html) 查看全部步骤、CPU / CANN / NPU 时间线和原始核数字段。

## 1. graph 更快，但没有跨流计算重叠

| 无 profiler 完整请求 | eager | PIECEWISE graph |
|---|---:|---:|
| 正式样本 | 10 | 10 |
| 完成时间中位数 | 713.277 ms | 308.295 ms |
| 最小–最大 | 711.114–724.109 ms | 306.472–360.064 ms |

本轮 graph 中位完成时间减少 **56.8%**。E/G/G/E 两对独立启动的 graph/eager 比分别为 0.428、0.454，方向一致。样本量有限，报告范围和中位数，不作稳定 P99 或普遍加速承诺。

四份诊断均没有观察到跨 stream 计算区间重叠。eager 用 1 条活动流，graph 用 26 条；graph 每个 decode replay 25 个分区，共 **63 × 25 = 1,575 次**。这些诊断记录不支持把当前收益解释为多个分区的计算重叠。

HTTP 完成时间包含 prefill、生成、logprob 和响应处理，并非纯 decode 时间。graph 也改变编译和捕获路径，不是只改变流数的因果对照。

## 2. 最明显的变化是设备任务之间的间隙减少

下表来自各自独立的 **plain profiler**，每项为预先选定的 decode 16–47 的 32 个步骤中位数。不同列的中位数不应直接相减求另外一列。

| 每个稳定 decode 步 | eager | graph |
|---|---:|---:|
| 首个关联设备任务开始至最后任务完成 | 12.376 ms | 4.969 ms |
| 计算任务区间并集 | 2.853 ms | 3.114 ms |
| 计算时间覆盖率 | 23.04% | 62.62% |
| 没有已记录设备任务覆盖的区间 | 9.473 ms | 1.718 ms |
| 与下一步骤设备窗口的间隔 | 0.624 ms | 0.607 ms |

两组完整请求的计算任务总数均为 **19,999**。graph 的计算区间并集并未更短，但设备窗口明显缩短；与无 profiler 性能改善一起看，支持“当前执行路径的提交／调度间隙值得优先优化”。

**23% 与 63% 不是芯片 core 利用率。** 存在一个计算任务就计入覆盖，其内部可能只使用少量核。该定义还包含单列标注的 AI_CPU 计算。未覆盖区间也不直接证明设备硬件完全空闲。

## 3. 空隙是否与 CPU 下发有关，有更具体的证据

分析器对每段无设备任务记录的区间，找到随后开始的任务，通过精确 flow 取其 CPU / CANN 调用；graph 内任务对应到该次 replay 的提交。

稳定步骤中，在这些空隙里，**后续任务的原生 CANN 下发调用尚未开始**的时长，每步中位数为：

- eager：4.829 ms。
- graph：0.841 ms。

对存在直接 CPU operator flow 的任务，后续 CPU operator 调用尚未开始的对应空隙时长中位数为 3.517 / 0.448 ms。graph 内 kernel 没有每次新的 CPU operator 调用，不伪造该数据。

这些值取空隙与“调用开始之前”的实际交集，没有把整段空隙都归为 CPU 开销。它们说明相当一部分间隙出现时，后续提交还未发生；并不能进一步区分 Python 运算、线程调度、依赖阻塞或未记录的工作。另有已经提交后才出现的空隙，保留为运行时／设备侧待解释区间。

## 4. 算子使用的核数差异很大

当前设备查询为 **24 Cube / 48 Vector**。稳定 decode 的原始字段示例：

| kernel | 报告类型与核数字段 | 含义边界 |
|---|---|---|
| `ReshapeAndCacheNdKernel` | Vector / Block Num=1 | 本次单 token KV 写入只报告 1 核 |
| `AddRmsNormBias`、`SwiGlu` | Vector / Block Num=1 | 小规模算子并未铺满全部 Vector 核 |
| `_compute_slot_mapping_kernel` | Vector / Block Num=2 | slot 映射与 KV 写入的分核方式不同 |
| `aclnnMatmul_MatMulCommon_MatMulV2` | AI_CORE / Block Num=19、22 或 24 | 同名 kernel 的样本报告不同核数；未恢复内部 tiling |
| `FusedInferAttentionScore` | MIX_AIC / Block Num=24、Mix Block Num=48 | 两个字段分别保留，不相加成利用率 |

部分 graph 内 kernel 报告 0，显示 unknown，不解释为“用了零核”，也不拿 eager 的字段直接填补。

PipeUtilization 进一步显示，“报告用了多个核”仍不等于计算流水线持续满载。例如 eager 的同名 matmul 样本聚合 `aic_mac_ratio` 中位数约 0.134、`aic_mte2_ratio` 约 0.537；KV 写入的 `aiv_vec_ratio` 约 0.010、`aiv_scalar_ratio` 约 0.477。这些是该采样下的内核指标，不能解释为整卡 13.4% / 1% 利用率，也不足以证明 HBM 带宽饱和。

硬件指标采集会改变时序：graph 的 Pipe 诊断步骤窗口中位数约 6.342 ms，而 plain 为 4.969 ms。因此性能结论使用无 profiler 对照，机制时间线主要读 plain，硬件计数器从 Pipe 独立读取。

## 5. 本轮选择与结论

本轮最清楚的证据是提交间隙，因此沿已设计的 eager/graph 对照进一步核验了空隙后的提交时间，没有扩展为人工改流或增大 batch。现有结果不能证明“单 stream 导致低 core 利用率”，也不能推导“再增加 stream 就会更快”。

当前可支持的表述是：**小 batch decode 同时存在少核算子与明显提交间隙；graph 在未出现跨流计算重叠的情况下，提高了计算任务的时间覆盖并缩短完整请求时间。** 独立计算并发能否进一步改善资源使用，需要另外构造保持数据依赖和计算内容一致的对照。

## 验证与复现边界

- 20 个无 profiler 正式响应、4 个诊断响应及 24 个预热响应，共 **48 个响应**，64 个贪心 token 全部逐项一致，logprob 有限、长度和结束原因通过。
- 4 条完整请求、256 个步骤；**99,946 个设备任务、79,996 个计算任务**完成身份与归属核验，未归属 runtime 任务为零。graph 共 3,150 次 replay；捕获对象身份、Model ID、stream/task 序列与执行／完成边界核对。
- graph dump 的示意时间不参与计算。精确 NOTIFY ID 配对、逐物理核活动时间、硬件 stall 原因、完整 kernel 内存依赖不在本次证据覆盖范围。
- 原始 profiler/source/request 证据按 478 文件 manifest 归档；CANN 大型缓冲和编译缓存留在远端。远端与本地主机时钟存在偏移；所有关联和耗时均使用同一远端采集时钟，未跨主机混用。
- 历史设备 Alarm 状态仍保留在设备快照中；结论限定本机本轮环境，没有将其作为性能原因。

独立图：[eager plain](report/eager-plain-decode32.svg) · [graph plain](report/graph-plain-decode32.svg) · [eager Pipe](report/eager-pipe-decode32.svg) · [graph Pipe](report/graph-pipe-decode32.svg)。

14 项区间／真实证据负向测试通过，覆盖错误 CSV、graph stream/task、缺失任务和提前完成等反例。离线浏览器检查四组、全部 256 个步骤选项、任务／计数器详情与窄屏布局通过，见 [验证记录](results/validation/validation.json) 和 [浏览器记录](results/validation/browser.json)。
