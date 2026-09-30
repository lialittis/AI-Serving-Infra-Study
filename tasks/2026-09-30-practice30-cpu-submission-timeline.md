# Practice 30：拆解单请求 CPU 时间线

- 日期：2026-09-30。
- 状态：首轮已实施，见[复现说明](../practice_30_cpu_submission_timeline/README.md)、[结果](../practice_30_cpu_submission_timeline/RESULTS.md)和[离线报告](../practice_30_cpu_submission_timeline/report/index.html)。

## 要回答的问题

Practice 29 显示提前排队能产生跨 stream 重叠，但尚不能区分 CPU 提交慢与 CPU 被同步阻塞。本轮只研究一次正常原生请求：CPU 各阶段做什么、由哪个线程提交、哪里等待，以及 NPU 空隙出现时后续工作提交到哪里。

用户选择：推理引擎到设备，不含 HTTP；先只做 eager。坚持“先做薄，再做厚”，不扩展模型、graph、门控、并发或硬件利用率计数器。

## 实施范围

- Qwen2.5-0.5B-Instruct、BF16、单 NPU、单在途请求；固定 10 输入 / 64 输出，重点 decode 32。
- 保留真实 `LLM.generate()`、调度器、KV 增长和结果回收，不使用固定步 capsule 替代正常引擎。
- 临时包装调度、输入准备、forward、logits、采样和结果处理，记录墙钟与线程 CPU 时间；关联 PyTorch→Enqueue→Dequeue→CANN→NPU。
- 不把墙钟减线程 CPU 直接归为设备等待，不把队列起止边界差称为纯排队时间，不累加跨线程或嵌套范围。
- 三个自有进程：reference（预热 2 + 无 profiler 3）、diagnostic（预热 2 + 诊断 1）、recovery（预热 2 + 原生 1）。安装源码及配置不改。

## 首轮回填

- [x] 远端实际 revision、导入位置和源码调用关系已核实。
- [x] 1,089 个阶段标记、64 个完整步骤、19,298 个计算任务和 21,512 对队列记录关联通过。
- [x] decode 32 三个最大设备任务空隙已逐项说明；一个 RoPE Host 范围值得下一轮继续放大。
- [x] 11 次响应（含预热）的 token/logprob 精确相等，包装恢复和原生推理恢复通过。
- [x] 25 个文件前后审计未变；实际 BalanceScheduler 源码另作明确标注的运行后补采，并核对 Git HEAD。
- [x] 离线 HTML、完整请求与 decode 32 SVG、复现说明、原始证据及反例测试已补齐。

诊断请求约 936 ms，提交线程 CPU 时间约 935 ms；256 次已观测 CANN 同步范围合计约 0.84 ms。不能把当前主要开销笼统称为 CPU 等 NPU，也不能据此宣称完全没有隐藏等待或自旋。诊断比参考中位数慢约 26.5%，明确披露观测成本与运行差异。

## 三个问题的当前答案

以下结论来自首轮诊断，详细证据见[结果](../practice_30_cpu_submission_timeline/RESULTS.md)。阶段时间为 CPU 侧函数范围，不等于 NPU 纯计算时间；嵌套范围不能直接累加。

### 1. CPU 的时间花在哪里？

以 decode 32 为例，整个 step 墙钟约 **14.77 ms**：

| 阶段 | 墙钟时间 |
|---|---:|
| forward | 11.95 ms，约占整个 step 的 80.9% |
| 输入准备 | 0.56 ms |
| 采样提交 | 0.24 ms |
| token 回传与列表转换 | 0.10 ms |
| logprob 回传 | 0.09 ms |

forward 内包含逐层框架执行、参数准备、算子调用和任务提交。其线程 CPU 时间约 11.946 ms，接近墙钟时间，但本轮尚未拆清 Python、C++、Triton launcher、运行时及观测开销各自占比，不能全部归为模型计算或某个具体函数的性能问题。

### 2. CPU 什么时候提交任务，什么时候等待？

主线程在输入准备、forward、采样等阶段调用算子和拷贝接口，产生 Enqueue；torch-npu 下发线程执行 Dequeue 和 CANN launch，NPU 执行已提交任务。主线程继续工作与 NPU 执行可以交叠。

需要在 CPU 上读取输出时，观察到明确同步：

- token 回收：异步拷贝到 pinned CPU 内存，记录并等待 transfer Event，随后读取 CPU 列表。
- logprob 回收：token IDs、logprobs、selected ranks 的 `.cpu().numpy()` 路径中，每步观察到三次 Stream synchronize。

整次请求共 64 次 Event 同步、192 次 Stream 同步，CANN 调用范围合计 **0.836 ms**，不是约 936 ms 诊断请求的主要耗时。提交线程 CPU 时间约 935 ms，说明该线程大部分时间获得 CPU 执行；其中仍可能包含运行时自旋和观测开销。这些同步范围不能代表所有潜在等待。

### 3. NPU 任务之间出现间隙时，CPU 在干什么？

decode 32 检查的三个较大设备任务间隙约 **127–135 µs**，其中约 **97–104 µs** 发生在下一项任务开始入队之前：

- 输入/元数据准备范围中，观察到 `pin_memory`、`aten::to`、`_to_copy` 等操作。
- `_prepare_inputs` 中，观察到 slot mapping 提交、slice、sub 等操作。
- forward 中，定位到 `vllm::npu_rotary_embedding` Host 调用范围，后续对应 `_triton_rope` 入队和下发。

这些具体间隙表明：下一项任务尚未开始入队，CPU 提交进度值得进一步分析。前两项所列操作没有覆盖全部间隙，RoPE 也尚未拆到 launcher 内部；时间重叠不是完整因果证明。不能推广为所有间隙都由 CPU 导致，也不能把单条 stream 的任务间隙等同于整颗 NPU 空闲。

诊断比无 profiler 参考中位数慢约 **26.5%**，混合观测开销和独立运行差异。因此当前证据支持继续拆解 CPU 提交路径，尚不能证明整个推理完全受 CPU 限制。

## 新增 subtask：分析 forward 内部各部分占比与执行情况

- [ ] **状态：待开始；本次仅记录任务，不推进分析、插桩或远端实验。**
- 目标：解释 decode 32 中约 11.95 ms 的 forward CPU 侧耗时，哪些部分在准备参数、执行框架逻辑、调用 launcher、入队或等待，以及它们与 NPU 执行如何交叠。
- 范围：沿用当前单请求、BF16、eager 配置，先聚焦 decode 32；遵循“先做薄，再做厚”，不立即增加模型、模式、并发或 benchmark 矩阵。
- 第一步：先用已有证据整理 forward 内已记录范围的包含关系，区分包含子调用的时间与扣除子调用的自身时间；占比统一以前述 forward 范围为分母，明确未归属时间，不拼凑出缺少证据的完整分解。
- 最小深入范围：优先选择已定位的 **RoPE Host→Enqueue**，沿实际调用关系区分 Python 参数处理、Triton launcher、C++ 入队边界；观察结果后再决定是否扩大到其他部分。
- 证据要求：分别报告主线程与下发线程、墙钟与线程 CPU 时间、Host 范围与 NPU 任务时间；保留跨线程关联，避免嵌套重复计数。新增观测若有必要，继续采用可恢复的临时包装，并核验输出与观测成本。
- 预期产出：一张 forward 内部耗时与占比表（含未解释部分）、一条细化的 RoPE 提交时间线，以及已确认事实、仍待验证的原因和下一小步建议。
