# P33 结果：地址复用后的实际冲突访问

**[VERIFIED] 本轮完成 40 轮无 profiler 对照和 6 轮独立 profiler 诊断。** omit 组 10/10 轮：A 的地址在旧 copy 设备任务开始前就交给候选 B，B 在 S0 上写入哨兵值后，旧 copy 在 S1 上**读到的全部 1,048,576 个元素都是哨兵值**——P32 观察到的"提前地址复用"在本工作负载下构成实际冲突访问，旧任务消费了新任务写入的数据。record 组 0/10 复用、10/10 完好；join 组 10/10 提前复用但 10/10 完好，profiler 证实其保护来自设备顺序（旧 copy 先完成、候选写入后执行），P32 的 [INFERRED] 升级为实测。synced 校验器对照 10/10 完好。

这是受控探针下的机制验证：**它证明该风险模式在本硬件上真实可发生，不证明生产 vLLM 路径已经发生**。哪些调用点可能踩中见[调用点审计](../references/cross_stream_call_site_audit/README.md)。

## 环境与纳入范围

2026-10-03 在 Ascend910B2C 上运行，容器逻辑设备 0 对应物理设备 5。Python 3.12.13，CANN 9.0.0，torch `2.10.0+cpu`、torch-npu `2.10.0`，native allocator，baseline 配置（`TASK_QUEUE_ENABLE=1`、`PER_STREAM_QUEUE=0`、`expandable_segments:False,multi_stream_lazy_reclaim:False`），单 host 线程。运行前后设备健康状态与 P31/P32 相同（既有 `Alarm / 80C98001`），前后一致，无重置或安装变更。每模式一个新进程；命令、安装 Python 源文件、配置和哈希保存于各案例。例：[plan.json](results/formal-01/plan.json)、[omit run.json](results/formal-01/baseline-t1-omit/run.json)。

smoke-01（2 轮 omit 先导）与 formal-01 分开保留，不混入主结论。

| 正式证据 | 配置与控制 | 案例 / 试验 |
|---|---|---:|
| [formal-01](results/formal-01/summary.json) | baseline；omit / record / join / synced × 10 轮 | 4 / 40 |
| [profile-01](results/profile-01/profile_summary.json) | baseline；omit / record / join × 2 轮，独立 profiler | 3 / 6 |

## 四组结果

工作负载沿用 P32：S1 等待生产事件后提交 64 次 FP16 矩阵乘，把 A（4 MiB FP32，值为轮次号）复制到一直存活的 `observed`，最后一个 A 引用在 S1 当前时销毁。S0 随后最多分配 8 个同尺寸候选；本实验新增：找到复用 A 地址的候选后**立即在 S0 上 `fill_(777.0)`**（synced 组推迟到全部同步后）。全部设备工作完成后对 `observed` 逐元素分类。代码见 [probe.py:run_trial](probe.py)。

| 模式 | 释放前保护 | 窗口内地址复用 | observed 元素分类 | 判定 |
|---|---|---:|---|---|
| `omit`（10 轮） | 无 | 10/10，首个候选即复用，done 事件均未完成 | **10/10 全量哨兵** | 冲突访问发生：旧 copy 消费了 B 写入后的存储 |
| `record`（10 轮） | `A.record_stream(S1)` | 0/10 | 10/10 全量原值 | 登记阻止复用，无冲突窗口 |
| `join`（10 轮） | S0 `wait_event(done)` | 10/10，首个候选即复用，done 事件均未完成 | 10/10 全量原值 | 提前交回地址 + 设备顺序保护，无损坏 |
| `synced`（10 轮） | 无（写入推迟） | 10/10 | 10/10 全量原值 | 校验器对照：同代码路径无污染 |

**[VERIFIED]** 全部 40 轮：Python weakref 已失效、allocator `free_requested` 在案、写入目标自身同步后仍为全量哨兵（排除写入丢失）、窗口内无 OOM / allocation retry。10/10 omit 轮的 `observed` 不含任何 A 原值或第三种值——旧 copy 任务读到的每一个元素都来自 B 的写入之后。

## 独立 profiler 的设备顺序

[交互时间线](results/profile-01/index.html)可离线选择模式、重复和窗口（完整 / 释放与分配 / 写入与旧 copy），点击条目查看原始 task / flow 身份。

**[VERIFIED]** 每轮旧 copy 对应一个 `aclnnInplaceCopy_TensorMoveAiCore_TensorMove` / `AI_VECTOR_CORE` task，候选写入对应 fill 设备 task；以精确 `async_npu` 与 `HostToDevice` flow endpoint 关联到 host scope 和 CANN launch，connection ID 逐一核对，并全部与 `kernel_details.csv` 的名称、Task ID、物理 stream、开始时间和 duration 一致（12 个任务）。未解析 flow 为零。

| 模式 / 轮 | 设备顺序（profiler 实测） | observed | 间隔 |
|---|---|---|---:|
| omit / t00 | 写入完成 → 25,770 µs → 旧 copy 开始 | 全量哨兵 | 写入早于旧 copy 约 25.8 ms |
| omit / t01 | 写入完成 → 25,946 µs → 旧 copy 开始 | 全量哨兵 | 同上量级 |
| join / t00 | 旧 copy 完成 → 0.62 µs → 写入开始 | 全量原值 | wait_event 排序生效 |
| join / t01 | 旧 copy 完成 → 0.62 µs → 写入开始 | 全量原值 | 同上 |
| record ×2 | 写入（异地址）早于旧 copy | 全量原值 | 无复用，顺序无冲突含义 |

omit 的 25.8 ms 间隔与 P32 profile-01 观察的旧 copy 延迟（约 26 ms，64 次 backlog 矩阵乘之后）一致。profiler 与事件查询有测量开销，间隔不用于性能结论。

## 证据修订与验证

首轮上传的源码树混入 macOS AppleDouble 元数据文件（`._*.py`），被 run.py 按惯例一并指纹归档；它们从未被执行，仅作为字节保留在 sources/ 与归档清单中，正式结论不受影响。

离线校验通过：486 个归档文件的大小及 SHA256、46 轮 summary 重算、138 个主阶段快照状态、12 个 copy/write 任务的原始 flow / CSV 时间、6+10 个分析单元测试，以及浏览器离线覆盖三案例 × 两重复 × 三窗口的点击验证（无 JavaScript 错误）。原始 profiler 数据（约 4 MB）保留在远端 `ascend910:/data/tianchi/practice_33_conflict_access/results/profile-01/`，本地已恢复并全量校验。证据：[文件与原始数据验证](results/validation/evidence.json)、[单元测试](results/validation/unit_tests.txt)、[浏览器](results/validation/browser.json)。复核命令见 [README](README.md#离线分析和验证)。

## 已确认、风险及仍待实验的问题

- **[VERIFIED]** 在无登记、无执行依赖的跨 stream 用途后释放存储，本工作负载下地址复用与冲突访问**每轮必然发生**（10/10 + 2/2 profiler），损坏形态为旧任务全量消费新数据。
- **[VERIFIED]** `record_stream` 的登记路径与 `wait_event` 的顺序路径都能阻止损坏：前者阻止地址提前复用，后者允许主机提前拿回地址但由设备顺序保护。两者保护机制不同，P32 的源码分析与本轮实测一致。
- **[INFERRED]** 发生条件可从本轮工作负载外推：tensor 的存储在"其他 stream 的设备用途尚未完成"时被 host 释放，且后续同地址设备操作不遵循能建立 happens-before 的 stream 顺序。满足该模式的任何调用点都有同类风险；强度（是否必然损坏）取决于新旧任务的提交间隔。
- **[UNKNOWN]** 生产 vLLM-Ascend 路径是否满足该模式：静态审计见[调用点审计](../references/cross_stream_call_site_audit/README.md)，当前已识别站点均为 wait 型保护（join 等价），未发现 omit 等价站点，但静态审计不覆盖运行时全部行为。
- **[UNKNOWN]** HCCL Work 的存储保留/等待/撤销登记、graph 私有池与 replay 外部存储（P32 遗留）、以及自定义算子内部切流（op-plugin 已知在 4 处显式 recordStream）均未由本轮验证。

**结论**：P32 证明"地址会提前交回"，本轮证明"提前交回的地址被写入后，未完成的旧任务会读到新数据"。记录用途与建立顺序依赖是两种都被实测验证有效的保护；未受保护的跨 stream 生命周期在本硬件上不是理论风险。
