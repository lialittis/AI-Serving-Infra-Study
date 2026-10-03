# P36 结果：graph 捕获/重放的存储边界

**[VERIFIED] 本轮完成 20 轮正式试验（4 模式 × 5 轮）与 6 轮 smoke。** P32 遗留的 graph 边界问题得到
实测回答：**捕获前在普通池分配、被图内烘焙地址的"外部存储"，释放后地址可被普通分配复用，随后的
replay 会全量读到复用者写入的数据**——external-free 两组各 5/5 轮 `fully_consumed_replacement`，
有序与跨流重放同样中招。源码侧 `NPUGraph`/allocator 对此**没有任何登记或保护**（与 CUDA 文档一致，
外部存储的存活是使用者的责任）。私有池侧：带 pending replay 删除 graph 与 static tensor 后，
后续普通分配**不与私有池地址重叠**（0/5），replacement 完好、进程存活——与 `releasePool` 源码一致
（freeable 池的块走 npuFree 归还驱动，不回普通池）。

## 环境与测量方式

2026-10-03 在 Ascend910B2C（物理设备 5）运行，torch `2.10.0+cpu` / torch-npu `2.10.0` / CANN 9.0.0，
native allocator，baseline 配置（NPUGraph 要求 `TASK_QUEUE_ENABLE≠2`），每模式独立进程。
图工作负载：非默认流上捕获 `static_out.copy_(external)`；`external`（4 MiB FP32，`arange × 轮次系数`）
在捕获前于普通池分配；`static_out` 在捕获内进入私有池。每轮先做一次有序重放并逐元素校验，
再引入该模式唯一变量。哨兵值为 −777.0（负数，与 arange 数据无碰撞）。设备健康状态与 P31–P35
相同（既有 `Alarm / 80C98001`，前后一致）。代码见 [probe.py](probe.py)。

| 模式（各 5 轮） | 唯一变量 | 地址复用 | 结果 |
|---|---|---:|---|
| `keep-alive` | external 保持存活；replacement 写哨兵（必然异地址） | 0/5 | **intact 5/5**（校验器对照） |
| `external-free-ordered` | 释放 external → 候选命中其地址 → S0 写哨兵 → S0 有序重放 | 5/5 | **fully_consumed_replacement 5/5** |
| `external-free-cross` | 同上，重放在另一条 stream（无等待） | 5/5 | **fully_consumed_replacement 5/5** |
| `pool-release-pending` | 重放入队 → 立即删除 graph 与 static_out → 普通分配 replacement 写哨兵 | 与私有池重叠 0/5 | survived 5/5；replacement 完好 |

全部试验：replacement 自身同步后仍为全量哨兵；无 OOM / allocation retry；候选命中数与地址记录在
各 run.json（[summary](results/formal-01/summary.json)）。

## 读数与源码对应

- **外部存储无保护 [VERIFIED — 实验 + 源码]**：`NPUGraph::capture_begin/end` 只管理捕获流的私有池
  （[NPUGraph.cpp:212](../references/torch_npu_allocator_lifetime/sources/torch-npu/torch_npu/csrc/core/npu/NPUGraph.cpp#L212)、
  [249](../references/torch_npu_allocator_lifetime/sources/torch-npu/torch_npu/csrc/core/npu/NPUGraph.cpp#L249)）；
  捕获前普通池分配的 tensor 不进入任何登记，`replay()` 直接以烘焙地址执行
  （[283](../references/torch_npu_allocator_lifetime/sources/torch-npu/torch_npu/csrc/core/npu/NPUGraph.cpp#L283)）。
  释放→复用→重放的每一步都没有 happens-before。跨流模式证明这不是同流顺序可以挽救的问题。
- **私有池不泄漏回普通池 [VERIFIED — 实验 + 源码]**：`releasePool` 在 use_count 归零后把池标为
  freeable，未用（unsplit）块由 `free_cached_blocks` npuFree 归还驱动
  （[NPUCachingAllocator.cpp:2144](../references/torch_npu_allocator_lifetime/sources/torch-npu/torch_npu/csrc/core/npu/NPUGraph.cpp)，
  同文件 releasePool 段），不会作为普通地址复用。pending replay 与删除竞争的驱动级行为
  （npuFree 与未完成执行的内存）无法从 Python 观测，本轮仅记录进程存活与 replacement 完好，
  **不据此宣称驱动层安全**。
- 与 P33 的关系：这是同一竞态类在 graph 路径上的实例——"存储生命周期由引用计数管理、消费者是
  烘焙了裸地址的重放体"。区别在于 graph 的消费者不经过任何 stream 顺序（地址直接嵌入模型），
  因此 join 式保护不可用；唯一的安全做法是**外部输入在 graph 存续期间保持存活**（vLLM 的
  static input copy 模式正是这样用的：输入先拷进私有池的静态 buffer，图只读静态 buffer）。

## 证据修订

smoke-01 曾出现 keep-alive 组"污染"的假阳性：轮次系数为 1 时 `arange[777]` 恰等于哨兵 777.0，
造成单元素双重计数。哨兵改为 −777.0 后消除；该轮 2 个案例保留为 pilot，不混入主结论。
另一处探针缺陷：`arange × k` 产生的中间临时量使池内存在多个同尺寸块，单次分配可能落在非
external 块上（smoke-02 前 external-free 组 0/2 复用、结果 intact 属于未触发而非受保护），
改为逐候选分配直至命中 external 地址后实测 5/5 复用。

## 边界

- 单图、单外部输入、copy 工作负载；未测多图共享池（`graph_pool_handle`）、私有池内 tensor 的
  释放复用、以及 graph 模式下 vLLM 完整管线的输入拷贝路径。
- pool-release 的驱动级 npuFree 竞争不可观测，见上。
- 离线校验通过：70 个归档文件大小及 SHA256、formal summary 重算一致、6 个分析单元测试。
  原始数据保留在远端 `ascend910:/data/tianchi/practice_36_graph_pool_boundary/results/`。

## 结论

P32 遗留的 graph 边界两项均闭合：**外部存储的存活责任完全在使用者**（释放后重放必然读到复用者
的数据，5/5），私有池与普通池**地址空间隔离**（freeable 块 npuFree 不回普通池）。vLLM-Ascend 的
ACLGraph 按"输入先拷入静态 buffer"的方式使用图，避开外部存储路径；任何绕过该模式、把普通池
tensor 直接捕获进图的调用都会落入本轮实测的竞态类。P32 任务清单的 graph 遗留项至此完成。
