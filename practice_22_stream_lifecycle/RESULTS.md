# 实测结论：stream 数量由哪些层决定

本次已完成四组诊断、四组无观察器基线及独立采样数值检查。打开 [交互报告](report/index.html) 可按场景、请求阶段和 stream 查看任务与创建来源。

## 1. eager 的单流来自提交路径

正式请求的全部 **3,008 个设备任务**在物理 stream 46 上执行，计算任务 2,678 个。模型依次调用各层，当前路径没有把 forward 内的算子重新分配到其他流。

启动时记录到 **2 次原生 stream 创建**，并不代表请求使用两条流。默认初始化资源的原生调用者是 `libtorch_npu.so`，已保存偏移；未获得其完整 Python 创建调用栈。

没有观察到 `aclrtSetStreamResLimit` 调用，也没有发现模型因某个算子核数少而自动拆流的证据。这个结论限于本次版本、配置和观测边界，不是对所有后端策略的否定。

## 2. graph 的 26 条流可以解释为主流和捕获分区资源

| 项目 | 本次观测 |
|---|---:|
| 正式请求设备任务 | 3,458 |
| 计算任务 | 2,678 |
| 请求使用的物理流 | 26 |
| 正式使用的捕获分区 | 25 |
| 两个请求的 replay 总数 | 150 |
| 精确匹配 dump 的图内计算任务 | 1,314 |
| 归属相同实例的图内 NOTIFY_RECORD | 150 |
| connection ID 关联的执行／等待边界任务 | 300 |

**主流 46 承载普通提交和 replay 边界；另外 25 条流分别属于 25 个正式捕获分区。** 两个请求各有三个单 token decode，每次 decode 重放 25 个分区。10-token prefill 不匹配 capture size=1，走编译后的 callable。

创建过程还分两层：`model_runner_v1.py:217` 的 `graph_capture()` 显式调用 `torch.npu.Stream()`；`torch.npu.graph()` 在没有显式 stream 参数时另外使用自己的 `default_capture_stream`。最终 dump 中的内部流又属于 CANN 运行实例，不能把这三种角色合成同一个 stream。

这补上了 Practice 15 中无法把图内任务归到具体分区/replay 的一部分缺口：现在可通过捕获实例、有效 Model ID、stream/task ID、重复任务序列及边界一起核验。并未捕获底层 NOTIFY 标识，所以不能宣称精确 NOTIFY 配对也已补齐。

## 3. 一次 Python Stream() 不一定只创建一个底层资源

graph 组第一次显式 `Stream()` 调用期间，记录到 **32 次 `aclrtCreateStreamWithConfig`**；加上更早的两个资源，共 34 次原生创建。调用栈来自 `GPUModelRunner.profile_cudagraph_memory()` → `graph_capture()` → `Stream.__new__()`。

这表明框架首次申请会触发批量建池，此后 Python 对象可以取用已有资源或包装已有 stream。34 个原生句柄不是 34 条同时活跃的物理流；未被取用的池资源没有强行补齐物理 ID。

## 4. 启动时捕获了两轮 graph，编号发生复用

一共导出 **50 次捕获**：先做 25 张图的显存估算，原生 `aclmdlRIDestroy` 销毁这批实例，再捕获正式使用的 25 张图。源码的 `profile_cudagraph_memory()` 在估算后执行 `clear_all_graphs()`，与日志吻合。

旧、新实例间存在 Model ID 和原生句柄复用。仅按数字匹配会把任务连到错误的分区；分析器同时检查捕获、销毁及使用时间。正式请求没有每次 replay 重新创建这组 graph。

## 5. 采样开关改变提交时机和汇合方式

两组都创建 34 个原生 stream 资源，正式使用 stream 44 和 46；每组 4,528 个设备任务、4,112 个计算任务。

| 场景 | 首次申请辅助流的路径 | 正式采集的汇合证据 |
|---|---|---|
| 提前生成关闭 | `random_sample()` → `global_stream()` → `Stream()` | 10 条精确设备 event 等待边；另有结果回传的 CPU 完成等待 |
| 提前生成开启 | `execute_model()` → `do_async_exponential()` → `global_stream()` → `Stream()` | CPU 等待预生成 q 的 event 后继续采样，连同结果回传共 20 个正式 CPU 完成等待节点 |

不能把开关解释为“第二条 stream 存在／不存在”。native 日志中的 `wait_stream` 调用也不必然对应跨流依赖，需核对生产和等待的实际句柄，避免将同流等待标成跨流同步。

## 正确性与局限

- 16 个正式 HTTP 请求全部完成。eager / graph 与各自无观察器基线的输出 token 完全一致，两模式之间也一致：`[40666, 102, 34794, 105133]`。
- 随机请求检查所有序列的 token 数、结束原因和有限 logprob；不要求跨进程随机文本相同。独立固定 q 的六项采样数值测试全部通过，并验证真实 q 有限且为正。
- 本次四种诊断中均未观察到不同流上的计算 kernel 时间交叠。重型观察器会影响主机时序，这不推翻 Practice 17 的其他采集，也不能解释成多流必然无效。
- 已解释所有活动 stream 的资源或 graph 归属，但 **25 条 graph 内部流的底层创建调用仍不可见**。这与创建者已追到具体 Python 行的辅助流不同。
- 未直接配对 NOTIFY ID，没有恢复完整 kernel 内部读写依赖，也没有测量逐物理核利用率。零次资源限制调用只适用于已拦截的 API 路径。
- 服务退出时没有观察到全部资源的显式 destroy；进程结束记录保留，不能据此认定泄漏。

验证记录见 [基线比较](results/baseline_validation.json)、[采样数值检查](results/sampling-numerics/result.json) 和 `results/validation/`。
