# Practice 16 结果：两个计算分支确实同时在执行

**三个双 stream 轮次均出现约 3.3ms 的设备计算区间重叠；三个单 stream 对照均为零。**

后续增加了[同步策略对照](SYNC_COMPARISON.md)，单独测量每轮等待与最后统一等待的性能，并验证不等待就读取结果的后果；不与本页的profiler耗时混合。

这次是独立算子实验，没有运行模型。环境为 Ascend 910B2C、CANN 9.0.0、torch 2.10.0+cpu、torch-npu 2.10.0。
两种模式使用同一批输入/输出 tensor 和相同的 Host 提交顺序，只改变向量分支所在 stream。
预热后采集六轮，每轮六次矩阵乘法和六次向量乘法；每轮等待完成后逐元素校验，两类输出全部正确。

## 设备时间线

打开[交互时间线](results/2026-09-24-run02/analysis/index.html)或[完整 SVG](results/2026-09-24-run02/analysis/timeline.svg)。蓝色为矩阵乘法，橙色为向量乘法，绿色短条为实际区间交集。

| 轮次 | 物理 stream | 计算 kernel 总跨度 | 两分支重叠时间 |
|---|---|---:|---:|
| 01 serial | 44 | 5.222369ms | 0 |
| 01 parallel | 44 / 43 | 4.952058ms | 3.293572ms |
| 02 parallel | 44 / 43 | 5.004360ms | 3.304652ms |
| 02 serial | 44 | 5.225149ms | 0 |
| 03 serial | 44 | 5.221669ms | 0 |
| 03 parallel | 44 / 43 | 4.997660ms | 3.294012ms |

任务跨度是**最后一个计算 kernel 结束 − 第一个计算 kernel 开始**，不含 Host 初始化、校验、profiler 导出，也不等于完整调用耗时。
重叠是两分支各个计算区间的交集并集；没有使用 Host 调用范围，也没有用分支首尾包围范围代替实际计算时间。

例如第二轮并行中，`mm-00` 与 `mul-01` 的实际重叠为 **562.782μs**。每个并行轮次共有八对相交的 kernel 区间。
向量分支累计执行时间中，超过 99.99% 与矩阵分支的计算区间重叠。这里表示 profiler 记录的任务执行区间重叠，并非逐周期的硬件单元利用率测量。

## 为什么只创建两个 stream 不够

真正被执行的任务为：

| 分支 | Python 调用 | 实际 kernel | 任务类型 | 并行模式的物理 stream |
|---|---|---|---|---|
| 矩阵 | `torch.mm(X, X, out=Y)` | `aclnnMm_MatMulV3Common_MatMulV3` | AI_CORE | 44 |
| 向量 | `torch.mul(V, 1.5, out=W)` | `aclnnMuls_MulAiCore_Mul` | AI_VECTOR_CORE | 43 |

数据没有跨分支依赖，且使用不同存储。Host 交替提交两类任务；两条 stream 各自按顺序运行。
我们保留到轮末才等待，使两条流都有机会同时工作。与 Practice 15 不同，这次没有在两个分支之间插入 event wait。

本次 runtime stream 句柄 `140551924371456`、`140551924383744` 分别由真实 flow 对应到物理 stream 44、43。
句柄、torch-npu 的 stream_id 和物理 stream 编号是不同标识，不能互相直接比较，也不能跨进程复用这些对应关系。

## 并行不等于成倍加速

第一轮中：

| 指标 | 单 stream | 双 stream |
|---|---:|---:|
| 六次 mm 的累计执行时间 | 2.690948ms | 4.951918ms |
| 六次 mul 的累计执行时间 | 2.531141ms | 3.293632ms |
| 两分支计算 kernel 总跨度 | 5.222369ms | 4.952058ms |

虽然大部分向量计算与矩阵计算重叠，两类算子的累计执行时间都变长了。因此总跨度只小幅缩短；三轮分别约缩短 **5.18%、4.23%、4.29%**。
共享资源竞争是可能的解释，但本练习没有采集带宽或逐流水线利用率，不能确定具体瓶颈。profiler 本身也有开销，这些数字不是无插桩吞吐基准，更不能推广到模型性能。

## 已核验与未覆盖

- 72 个计算 kernel 全部匹配实际 CPU flow、CANN flow、connection ID 及 kernel CSV。
- 九次终点 event record 与九次 Host 等待配对；每个 event 排在所属 stream 的计算之后，等待返回晚于 event 完成。
- 同一物理 stream 内已观测任务不重叠。双 stream 重叠来自跨 stream 的计算任务。
- 25 个设备任务属于结果校验或 profiler 控制，单独记录，不混入 72 个计算 kernel 的统计。
- 两分支四张存储地址区间不重叠；所有 tensor 保留到两条流完成，每轮全部输出元素验证通过。

八项离线测试覆盖真实结果、错误输出、存储别名、缺 flow、错误 event、错误 stream、CSV 不一致，以及“包围区间相交但 kernel 不相交”的反例。

没有测试 KV block 释放/复用、抢占、并发 vLLM 请求、graph replay 或 Host 多线程。
也没有验证两个占满同类计算资源的算子是否能并行。这个实验只证明：**在当前负载与环境下，两个独立计算分支能够在不同 NPU stream 上重叠执行，且输出正确。**

原始参数与地址见 [run.json](results/2026-09-24-run02/run.json)，跨层核验见 [evidence.json](results/2026-09-24-run02/analysis/evidence.json)，复现入口见 [README](README.md)。
