# P32 结果：地址复用、stream 登记与设备执行依赖

**[VERIFIED] 本轮完成 198 轮无 profiler 对照和 12 轮独立 profiler 诊断。** 未登记和 owner 回交等待两组均在旧 copy 未完成时重新分配了 A 的地址；`record_stream` 组没有提前复用。**本轮 B 没有任何设备访问，未证明实际冲突访问、数据竞争或漏洞。** 这一结果支持此前源码调查的 Case B：allocator 能保护已登记的跨 stream 用途，但不能从任意调用者的异步操作自动推知所有用途。完整源码结论及例外见[源码报告](../references/torch_npu_allocator_lifetime/README.md)。

## 环境与纳入范围

2026-10-02 在 Ascend910B2C 上运行，容器逻辑设备 0 对应物理设备 5。Python 3.12.13，CANN 9.0.0，torch `2.10.0+cpu`、torch-npu `2.10.0`，native allocator。实际 torch / torch-npu commit 分别为 `449b1768410104d3ed79d3bcfe4ba1d65c7f22c0` / `94f8a8e6b523d7ba553e1b80d5b5248478391526`。每个配置 × 模式 × 线程数均为新进程；命令、安装 Python 源文件、配置和哈希保存于各案例。例：[baseline run.json](results/formal-02/baseline-t1-omit/run.json)、[运行计划](results/formal-02/plan.json)。

**[VERIFIED] 测试前后没有其他 NPU 进程。设备已有 `80C98001` / Health Alarm，本轮前后相同。** 没有重置设备或更改安装、服务、全局配置。所有 warm-up 和结果校验通过，但这不能证明硬件健康；结论仅描述该环境中的观测。证据：[设备状态](results/profile-01/device_after.json)、[健康状态](results/profile-01/health_before.json)、[结束健康状态](results/profile-01/health_after.json)。

| 正式证据 | 配置与控制 | 案例 / 试验 |
|---|---|---:|
| [formal-02](results/formal-02/summary.json) | baseline、direct、queue2、perstream、expandable；三模式 × 单/双线程 × 五次 | 30 / 150 |
| [lazy-03](results/lazy-03/summary.json) | lazy；三模式 × 单/双线程 × 五次 | 6 / 30 |
| [custom-owner-01](results/custom-owner-01/summary.json) | baseline，自定义 S0；三模式 × 单/双线程 × 三次 | 6 / 18 |
| [profile-01](results/profile-01/summary.json) | baseline / direct；三模式 × 单线程 × 两次 | 6 / 12 |

正式无 profiler 结果合计 198 轮，每模式 66 轮。Profiler 的 12 轮另列，不混入配置矩阵，也不用于性能比较。六配置的差异和复现命令见[实验说明](README.md#远端复现)。

## 三组结果与状态转换

A 是 S0 上初始化的 4 MiB FP32 Tensor。所有组都在开始积压工作前完成初始化，并让 S1 等待生产事件。S1 提交 64 次独立矩阵乘法，再把 A 复制到一直存活的 `observed`。最后一个 A 引用在 S1 当前时销毁；S0 随即保留最多 32 个同尺寸候选 B，只观察地址。代码见 [probe.py:run_trial / submit_and_release](probe.py)。

| 生命周期控制 | 释放后 A 的 block 状态 | 旧任务标记未完成时同地址复用 | 完成后的回收 |
|---|---|---:|---|
| `omit`：不登记 S1 | `inactive` | **66/66**，第一候选即复用 | A 已属于新候选 |
| `record`：`A.record_stream(S1)` | `active_pending_free` | **0/66** | 普通配置 56/56 在后续分配时复用；lazy 10/10 经 cache miss 后复用 |
| `join`：S0 等待 S1 末尾事件 | `inactive` | **66/66**，第一候选即复用 | A 已属于新候选；S0 执行依赖已插入 |

**[VERIFIED]** 两个有早期复用的组，首次复用前后 `read_start`、`read_end`、`done` 三个 query 都为 False。登记组没有早期地址复用。所有 198 轮 Python weakref 已失效，并有 allocator `free_requested` 历史；全部复制值正确，地址观察窗口没有 OOM 或 allocation retry 增量。见三组正式 summary 及其链接案例的 `run.json`。**[UNKNOWN]** event query 本身不能指出某条设备读指令是否已发生，设备时序证据单独列于下一节。

| 默认 S0 配置 | omit 提前复用 | record 提前复用 | join 提前复用 |
|---|---:|---:|---:|
| baseline | 10/10 | 0/10 | 10/10 |
| direct：task queue 关闭 | 10/10 | 0/10 | 10/10 |
| queue2 | 10/10 | 0/10 | 10/10 |
| perstream | 10/10 | 0/10 | 10/10 |
| expandable | 10/10 | 0/10 | 10/10 |
| lazy | 10/10 | 0/10 | 10/10 |

自定义 S0 的 baseline 对照也是 omit 6/6、record 0/6、join 6/6。每个配置的单线程与双线程各占一半，效果相同。双线程中 worker 从 host queue 取走唯一引用、提交 S1 工作并释放 A；主线程在 S0 分配 B。实际 native thread ID 已核对。**[INFERRED]** 本轮行为由存储的 owner stream、登记集合和执行依赖决定，不能靠“另一个 host 线程释放”自动获得额外保护；这与[源码报告的锁及 thread-local stream 分析](../references/torch_npu_allocator_lifetime/README.md#6-cross-thread-behavior)一致。

## 登记与延迟回收

**[VERIFIED — 源码]** `DeviceCachingAllocator::recordStream` 将 S1 放入 `block->stream_uses`；`free` 将 block 标为未分配后，对登记用途调用 `insert_events`，没有登记用途则调用 `free_block`。前者在各登记 stream 上 record event，存入 `npu_events[stream]` 并增加 `event_count`；`process_events` query 完成事件，在计数降为零时调用 `free_block`。代码：[NPUCachingAllocator.cpp:recordStream，1557–1564](../references/torch_npu_allocator_lifetime/sources/torch-npu/torch_npu/csrc/core/npu/NPUCachingAllocator.cpp#L1557)、[free，1445–1483](../references/torch_npu_allocator_lifetime/sources/torch-npu/torch_npu/csrc/core/npu/NPUCachingAllocator.cpp#L1445)、[insert_events，2984–3009](../references/torch_npu_allocator_lifetime/sources/torch-npu/torch_npu/csrc/core/npu/NPUCachingAllocator.cpp#L2984)、[process_events，3030–3062](../references/torch_npu_allocator_lifetime/sources/torch-npu/torch_npu/csrc/core/npu/NPUCachingAllocator.cpp#L3030)。

**[VERIFIED — 实验]** record 组释放后是 `active_pending_free`；显式同步 S1/S0 后，快照仍是该状态。普通配置的后续分配处理完成事件并复用地址。lazy 组在同步后四次能命中缓存的普通分配中仍不复用 A；大于最大 S0 空闲缓存 block 的请求触发回收后，10/10 复用 A。实际触发大小为 64、82、100、118、136 MiB，均低于 192 MiB 上限，且没有 OOM/retry 增量。例：[lazy 单线程 record](results/lazy-03/lazy-t1-record/run.json)、[双线程 record](results/lazy-03/lazy-t2-record/run.json)。

**[VERIFIED — 源码]** lazy 关闭时，分配先处理事件；lazy 打开时，首次缓存查找失败后才处理事件并重试查找。[malloc，1170–1213](../references/torch_npu_allocator_lifetime/sources/torch-npu/torch_npu/csrc/core/npu/NPUCachingAllocator.cpp#L1170)。因此 `active_pending_free` 并不表示设备仍在访问，设备完成与 allocator 处理完成事件是两个时刻。本轮不能证明任意条件下都没有后台回收，仅观测了上述路径。

各轮保存释放、观察结束、同步结束和后续分配后的 gzip 快照，lazy 还保存 cache miss 后快照。查询按包含原地址的 block 进行，允许空闲 block 合并。正式无 profiler 窗口末尾 `reserved_bytes` 为 316,669,952–750,780,416 bytes；lazy 触发后最高记录 895,483,904 bytes。这是选定采样点的 allocator reservation，**不是进程全程峰值或设备总占用**；旧触发缓存与 runtime/workspace 也占内存。

## 独立 profiler 的设备顺序

[交互时间线](results/profile-01/index.html)可离线选择配置、模式、重复和观察窗口，点击查看原始 task / flow 身份。原始分析见 [profile_summary.json](results/profile-01/profile_summary.json)。

**[VERIFIED]** 每轮旧 copy 对应一个 `aclnnInplaceCopy_TensorMoveAiCore_TensorMove` / `AI_VECTOR_CORE` task。以精确 `async_npu` 和 `HostToDevice` flow endpoint 关联到对应 host copy scope 和 CANN launch，核对 connection ID；12 个 copy task 全部又与 `kernel_details.csv` 的名称、task ID、物理 stream、开始时间和 duration 一致。被实验 scope 覆盖的 860 个设备 task 已关联，未解析 flow 为零。未声称重建完整内存访问 DAG。证据：[独立复核](results/validation/evidence.json)、[分析方法](analyze_profile.py)。

| 配置 / 模式 | t00：同地址分配返回领先旧 copy 开始 | t01 |
|---|---:|---:|
| baseline / omit | 26.478 ms | 26.649 ms |
| baseline / record | 窗口内无同地址分配 | 窗口内无同地址分配 |
| baseline / join | 26.320 ms | 26.547 ms |
| direct / omit | 26.300 ms | 26.479 ms |
| direct / record | 窗口内无同地址分配 | 窗口内无同地址分配 |
| direct / join | 26.088 ms | 26.300 ms |

这里比较同一远端 trace 中 allocation host scope 的结束与旧 copy 设备 task 的开始，不比较本地/远端时钟。8 个 omit/join 试验的地址分配均早于旧 copy 开始，4 个 record 试验没有窗口内复用。Profiler 和事件查询有测量开销，以上间隔不是性能收益或普遍竞态窗口大小。

**[INFERRED]** join 可以在主机提前交回地址，因为新 B 若只在 S0 使用，设备操作会排在 S0 的 `wait_event(done)` 之后，旧 S1 用途已完成。这个保护依赖后续用途遵循该 stream 顺序；若把 B 再交给另一个未等待的 stream，不能沿用此推断。本轮没有执行 B 的设备操作，因此没有实测 B 的最终执行顺序。

## 证据修订与验证

初轮采集曾在测量中读取 `npu_stream`。该属性调用原生 `NPUStream::stream()`，会影响 host queue 进度，Python stream hash 也会触发读取。正式采集在窗口前缓存 handle，并用 `id(stream)` 索引。源码位置和额外冻结证据见[实验说明](README.md#工作负载和测量边界)。`smoke-01` 的 6 轮和 `formal-01` 的 180 轮因此保留为 pilot，不混入主结论；formal-01 本地仅保留摘要及来源，全量紧凑证据在远端归档。

formal-02 原计划六配置，进入 lazy 单线程 record 时，固定 64 MiB 请求已不能保证大于前轮留下的缓存 block，测量代码主动终止。该失败日志、快照及先完成的 lazy omit 案例保留，但正式 summary **显式只纳入五个已完成配置**。lazy-03 将完成后的触发请求改为依据缓存最大 block 自适应、仍有固定上限，单独补齐六个 lazy 案例。改变仅在旧设备工作完成后的回收控制；旧窗口内地址观测方式相同。各运行保存实际执行的 probe 源码，不以当前代码替换历史文件。[排除记录](results/excluded_pilots.json)、[失败日志](results/formal-02/lazy-t1-record.log)。

离线校验通过：1,703 个归档文件的大小及 SHA256、210 轮 summary 重算、630 个主阶段快照状态、12 个 copy 的原始 flow / CSV 时间，以及 6 个分析单元测试。浏览器离线覆盖六个案例 × 两个重复 × 三个窗口，点击证据无 JavaScript 错误。证据：[文件与原始数据验证](results/validation/evidence.json)、[单元测试](results/validation/unit_tests.txt)、[浏览器](results/validation/browser.json)。复核命令见 [README](README.md#离线分析和验证)。

## 已确认、风险及仍待实验的问题

- **[VERIFIED]** 本轮普通跨 stream copy 不使 allocator 自动保护未知 S1；显式登记防止了观察窗口内的地址复用。单/双 host 线程和默认/自定义 S0 均有对应证据。
- **[INFERRED]** 若调用者既不登记 S1，也不在后续实际设备用途前建立足够的依赖，提前交回地址可能构成跨 stream 生命周期风险。地址重新分配与真正有冲突的设备访问仍需分开判断。
- **[UNKNOWN]** 本轮没有 B 读写、逐指令访问证据或数据损坏，不能定性为漏洞。是否发生实际冲突、哪些上层调用遗漏登记，需要额外定向实验和调用点审计。
- **[UNKNOWN]** HCCL 的 Work 保留/等待/撤销登记、graph private pools 与 replay 外部存储、自定义 operator 的生命周期，以及既有硬件告警对其他工作负载的影响，均未由本轮验证。它们需要独立实验，不将 eager 结果直接推广过去。

本阶段完成的是 allocator 特征验证：记录跨 stream 用途与建立设备依赖是两种不同的保护路径，host 地址分配时间不能单独作为设备访问是否安全的判据。
