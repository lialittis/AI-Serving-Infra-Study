# P31：单服务多用户请求与 NPU 并发

目标：以真实、独立 HTTP 请求关联用户、调度合批、host 提交和 NPU kernel，建立可复现且逐步深入的实验。代码和方法见 [P31 README](../practice_31_multiuser_streams/README.md)。

## P31a：固定请求 eager 基线

- [x] 独立 HTTP/SSE 客户端：1/2/4/8 用户，每用户连续 3 个请求，初始屏障启动。
- [x] 单 vLLM-Ascend 服务；TP1 BF16 eager；固定输入128/输出64 tokens；greedy。
- [x] benchmark 与 diagnostic 独立服务；正式请求窗口外预热与 profiler 控制。
- [x] 记录 frontend/external/internal 请求 ID 映射、调度前进度和 scheduled-token 请求集合。
- [x] host flow、CANN connection、kernel CSV 精确关联，保留未解析项和覆盖率。
- [x] 多 stream 计算区间并集、copy/compute 单独计量、共享批次归属。
- [x] 离线交互 HTML、结构化 JSON/CSV、源码和版本快照。
- [x] 完成 1/2 用户 smoke 关联验证。
- [x] 完成 1/2/4/8 用户正式矩阵和原始 trace 抽查。
- [x] 完成真实报告浏览器检查、结果归档和结论（四档诊断均通过，160个归档文件哈希核对）。

## P31b：采样辅助 stream 对照

固定独立 HTTP 请求形式，改用随机采样，逐项对比异步 exponential 预计算开/关。参考 P17，添加32用户档位并匹配服务容量；不设置 per-request seed，以免走入不同 sampler 分支。验证多 stream/计算重叠分析在真实正对照下的响应；不保证该负载必然重叠。

- [x] eager + temperature=0.8/top_p=0.9；只切换提前生成，不设 per-request seed。
- [x] 8／32独立用户、统一32序列容量；独立服务 ABBA benchmark 与开／关诊断。
- [x] 核对 q 元数据、event 身份、host CANN 同步与设备 EVENT_WAIT；按实际 batch/phase 分组。
- [x] 将全窗口重叠、同 step q/forward 重叠、DSA 与其他随机计算分别计量。
- [x] 完成720请求、787诊断 step、302,069计算任务关联及结果笔记。
- [x] 17项测试、四组原始证据抽查／离线浏览器检查、221个归档文件哈希验证；大文件排除。
- [ ] 若要确认性能收益：隔离 host 后台分析、增加独立服务重复，并降低诊断插桩对提交时序的影响。

结论见 [P31b 结果](../practice_31_multiuser_streams/SAMPLING_RESULTS.md)：两组都有 stream44/46；C8 提前生成后 q 在模型首 kernel 前已完成，未重叠；C32 开启后164/199 steps发生 q/forward 重叠，累计10.351 ms。关闭组的重叠对象是采样排序／top-p／softmax。总重叠更多不能直接推导吞吐更高。

## P31a 结果与独立后续问题

正式180个测量请求完成；45个诊断请求、274,410个计算 kernel 全部关联。1/2/4/8用户均出现目标 HTTP 在途并发；多用户进入共享批次，四档位诊断计算都在物理 stream46 上，跨 stream 计算重叠为0。无插桩吞吐约91→645 tokens/s。详见 [结果](../practice_31_multiuser_streams/RESULTS.md)。

- [x] 核对 greedy 输出变体：2和8用户组的无插桩重复运行自身也有差异；diagnostic 的输出都匹配某次 benchmark 变体。
- [ ] 独立研究输出确定性：固定到达/合批形态，比较首次分歧前的 logits 或数值路径；与性能采集分开，避免读回同步干扰。

## P31c：调度策略与异构长度

先构造一请求 decode 期间加入另一请求 prefill，再测试长短请求混合。分别开启 chunked prefill、async scheduling；保持其他因素固定。重新核对 scheduler→worker 对应关系，不直接沿用 P31a 的同步执行匹配契约。

## P31d：graph 对照

复用相同客户端工作负载；记录实际 capture/replay、shape、padding 和回退。关联 graph 内任务，区分独立 native launch 与 replay 内 kernel。分别比较 host 开销、stream 分布、计算重叠和无 profiler 性能。

## P31e：多轮会话与缓存

每用户维护独立历史，先关闭再开启 prefix caching。记录实际 prompt 增长、缓存命中、scheduled tokens 与延迟。之后再评估视觉与语言请求、KV offload/reload 等扩展。

## 判定边界

- 多用户在途、合批、多 stream、kernel 重叠分别给出证据，不能相互替代。
- 共享 kernel 归属于完整请求集合，不能人为平摊到单用户。
- 关联缺失时标记 incomplete，不能报告“没有重叠”作为完整结论。
- 性能只由无 profiler、无 observer 的 benchmark 支持；诊断运行允许调度发生变化。
- 保留当前设备健康状态、原始 trace 路径及校验值；只管理本实验创建的进程，不改变安装源码。
