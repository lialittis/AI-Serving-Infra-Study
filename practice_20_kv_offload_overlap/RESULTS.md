# P1 实测结果：KV 回载确有重叠，但本轮重算更快

在当前单卡 Qwen2.5-0.5B eager 负载中，真实 CPU prefix 命中、D2H 保存、H2D 回载和 block 复用均已验证。原生传输与独立请求 B 的计算发生重叠；强制串行组没有计算／DMA 重叠。回载没有带来端到端优势。

打开[离线交互图](report/index.html)，或查看[机器可读验证结果](results/published/validation.json)。

## 无 profiler 性能

下表为中位数，单位 ms；括号内为首 token 延迟的 Q1–Q3。每模式每长度 10 个正式周期，另有等量预热。

| 前缀 token | 模式 | A′ 首 token（IQR） | A′ 完成 | B 完成 |
|---|---|---:|---:|---:|
| 1024 | native | 50.88（49.99–51.70） | 857.56 | 3179.52 |
| 1024 | serialized | 43.94（43.32–44.42） | 855.90 | 3203.90 |
| 1024 | recompute | 24.40（24.15–24.97） | 751.96 | 2859.04 |
| 3072 | native | 51.41（50.85–52.61） | 862.36 | 3190.03 |
| 3072 | serialized | 49.00（47.46–49.80） | 872.57 | 3206.69 |
| 3072 | recompute | 38.83（38.68–39.13） | 767.98 | 2875.97 |

原生组相对串行组的首 token 延迟，两种长度均变慢；A′ 总完成时间和 B 完成时间相对串行组的差异未形成一致结论。相对重算组，原生组在上述指标均更慢。这里的“一致”仅指前后两对服务方向相同且合并 IQR 不重叠，不是统计显著性检验。

服务顺序为 native → serialized → recompute → recompute → serialized → native。计时在远端 loopback 客户端进行，等待实际流式 token 到达与完整响应。两请求完成窗口由较长的 B 主导，不将其解释为持续到达场景的服务容量。

性能测量关闭 profiler 和重型 Python 观察器，保留少量命中／传输记账；延迟包含这部分开销。串行组包含主机屏障，不能据此单独量化“换一个 stream ID”的成本。

## 真实传输与依赖

| 模式 | 设备任务 | 计算任务 | KV 约束 | 未满足 | 物理 stream |
|---|---:|---:|---:|---:|---|
| native | 272550 | 213431 | 786 | 0 | 43, 44, 46 |
| serialized | 272550 | 213431 | 786 | 0 | 43, 44, 46 |
| recompute | 240802 | 213525 | 0 | 0 | 46 |

原生组两次 H2D 分别搬运 12 / 36 MiB，与 B 计算区间实际交叠 **345.931 / 725.110 µs**。串行组仍使用独立的传输 stream，但全部传输的计算重叠均为 0。stream 数字仅在各自运行内关联。

每周期 D2H 总量分别为 205.5 / 229.5 MiB，包含 A、压力请求和 B 的已完成 block；两个 offload 模式严格一致。A′ 的 NPU prefix hit 为 0，CPU hit 分别为 1024 / 3072 token。重算组没有 KV DMA。

两种 offload 图各验证 786 条约束：386 条同代数据依赖、382 条存储换代约束、16 条 CPU 缓存发布和 2 条回载释放约束。第二个周期出现 CPU pool 的真实驱逐／复用。全部要求在独立构建的同步图中可达。

同步链包括：

- KV producer → 计算流中采样结果的 event → 原生主机等待 → 后续调度与后台提交 → D2H。`confirmed_tokens` 本身不是设备屏障。
- D2H → 完成 event 查询 → worker 完成消息 → scheduler 发布 CPU prefix，并释放额外的 NPU block 引用。
- H2D → 完成查询 → request 恢复调度 → attention 消费；CPU/NPU 的额外引用在完成确认后释放。
- 同一物理 block 换代前，旧传输访问必须先结束。图分别保存逻辑分配与设备覆盖，不把主机分配时刻当作设备写入。

## 正确性与观测边界

60 个正式周期及 60 个预热周期在三模式下逐 token 一致；六个独立诊断周期的输出也一致。独立真实 DMA 检查覆盖 K/V 分离、非零 storage offset、块映射和未覆盖区域，实验前后均逐字节通过。

后台线程不继承 PyTorch CPU scope，原始异步关联会把其 event 错归到主线程。两份 offload 诊断共纠正 36 条这类归属，使用 CANN connection、enqueue/dequeue correlation 与源码约束下的 FIFO 批次顺序。记录数、句柄和每批 memcpy 数均须一致；没有按最近时间猜测，也没有用两套时钟的数值相近补边。

三份诊断保留了 9 条无 host 关联的 `PLACE_HOLDER_SQE` runtime 任务，仅连接物理 stream FIFO，不虚构数据含义。KV 访问以完整批次和 24 层 attention 调用边界保守建模，原生 workspace 仍未知，`complete_exact_model_data_dag=false`。

本轮仅覆盖正常 prefix 缓存驱逐，不覆盖活动请求抢占／取消。源码检查发现 Ascend runner 未调用新版 `handle_preemptions`；本实验遇到活动抢占会停止，不把当前结果推广到那条路径。没有升级或覆盖系统安装。

环境为 Ascend 910B2C、CANN 9.0.0、torch/torch-npu 2.10.0、vLLM 0.21.0、vLLM-Ascend 0.21.0rc1。既有硬件 Alarm 仍在；基础数值和往返拷贝前后通过，但不能据此认证硬件健康。最后已无实验 NPU 进程。

## 归档与复现

- [完整性能样本压缩包](results/published/benchmark-evidence.tgz)：包含原始响应、预热、命中／搬运记录、服务命令、源码与仪器快照。
- [性能汇总](results/published/performance.json)、[环境](results/published/environment.json)、[数值检查](results/published/numerics-after.json)。
- [诊断文件哈希清单](results/published/p1-diagnostic-evidence.manifest.json)与[归档位置](results/published/evidence_locations.json)。完整诊断包保留在远端 `/data/tianchi/practice_20_kv_offload_overlap/results/p1-diagnostic-evidence.tgz`，本地为 `/tmp/p1-diagnostic-evidence.tgz`。
- 完整图在本地／远端对应诊断的 `analysis/execution_graph.json`；本地原始诊断位于本 practice 的 `results/diagnostic-r03/`。这些大文件不加入 Git。

诊断包 SHA256：`eba3bb9372cc7441e1303b2b50aa187989c1e1389fa9b3b51469efd5f6901c1c`。

首轮短前缀资格检查、串行长前缀资格检查及早期诊断仍保留在远端 results；一次端口 TIME_WAIT 导致的诊断启动失败未采集设备数据，修复后使用新目录重试。它们均未混入正式性能样本。

当前结论支持继续研究更大模型或更长前缀下的收益交叉点；无需假设“有重叠就一定加速”。本次没有完成对回载／调度成本的逐项归因。
