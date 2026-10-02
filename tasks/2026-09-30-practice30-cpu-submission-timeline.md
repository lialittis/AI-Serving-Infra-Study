# Practice 30：拆解单请求 CPU 时间线

- 日期：2026-09-30。
- 状态：eager、graph 同类分析及首轮 forward 细分均已完成。graph 工作已在 `d14b25a` 提交并推送；本轮细分见 [forward 结果](../practice_30_cpu_submission_timeline/FORWARD_RESULTS.md)和[报告](../practice_30_cpu_submission_timeline/report/forward/index.html)。另见[复现说明](../practice_30_cpu_submission_timeline/README.md)、[首轮结果](../practice_30_cpu_submission_timeline/RESULTS.md)、[graph 结果](../practice_30_cpu_submission_timeline/GRAPH_RESULTS.md)和[双模式报告](../practice_30_cpu_submission_timeline/report/comparison/index.html)。

## 要回答的问题

Practice 29 显示提前排队能产生跨 stream 重叠，但尚不能区分 CPU 提交慢与 CPU 被同步阻塞。本轮只研究一次正常原生请求：CPU 各阶段做什么、由哪个线程提交、哪里等待，以及 NPU 空隙出现时后续工作提交到哪里。

用户选择：推理引擎到设备，不含 HTTP；首轮只做 eager。坚持“先做薄，再做厚”：做厚的第一步补充 graph mode 对相同三个问题的分析，再深入 forward 内部。不扩展模型、门控、并发或硬件利用率计数器。

## 实施范围

- Qwen2.5-0.5B-Instruct、BF16、单 NPU、单在途请求；固定 10 输入 / 64 输出，重点 decode 32。
- 保留真实 `LLM.generate()`、调度器、KV 增长和结果回收，不使用固定步 capsule 替代正常引擎。
- 临时包装调度、输入准备、forward、logits、采样和结果处理，记录墙钟与线程 CPU 时间；关联 PyTorch→Enqueue→Dequeue→CANN→NPU。
- 不把墙钟减线程 CPU 直接归为设备等待，不把队列起止边界差称为纯排队时间，不累加跨线程或嵌套范围。
- 三个自有进程：reference（预热 2 + 无 profiler 3）、diagnostic（预热 2 + 诊断 1）、recovery（预热 2 + 原生 1）。安装源码及配置不改。

## 首轮回填

- [x] 远端实际 revision、导入位置和源码调用关系已核实。
- [x] 1,089 个阶段标记、64 个完整步骤、19,298 个计算任务和 21,512 对队列记录关联通过。
- [x] decode 32 三个最大设备任务空隙已逐项说明；一个 RoPE Host 范围作为后续深入分析候选，先补 graph mode 对照。
- [x] 11 次响应（含预热）的 token/logprob 精确相等，包装恢复和原生推理恢复通过。
- [x] 25 个文件前后审计未变；实际 BalanceScheduler 源码另作明确标注的运行后补采，并核对 Git HEAD。
- [x] 离线 HTML、完整请求与 decode 32 SVG、复现说明、原始证据及反例测试已补齐。

诊断请求约 936 ms，提交线程 CPU 时间约 935 ms；256 次已观测 CANN 同步范围合计约 0.84 ms。不能把当前主要开销笼统称为 CPU 等 NPU，也不能据此宣称完全没有隐藏等待或自旋。诊断比参考中位数慢约 26.5%，明确披露观测成本与运行差异。

## 三个问题的当前答案

以下结论仅来自首轮 eager 诊断，不能直接推广到 graph mode；详细证据见[结果](../practice_30_cpu_submission_timeline/RESULTS.md)。阶段时间为 CPU 侧函数范围，不等于 NPU 纯计算时间；嵌套范围不能直接累加。

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

## Subtask 1：补充 graph mode 的同样分析（做厚的第一步）

- [x] **状态：已实施并完成远端验证，结果见 [GRAPH_RESULTS.md](../practice_30_cpu_submission_timeline/GRAPH_RESULTS.md)。**
- 目标：在 graph mode 下同样回答 CPU 时间花在哪里、何时提交与等待、NPU 任务间隙时 CPU 在做什么，并与 eager 对照。
- 最小范围：保持同模型、BF16、单 NPU、单请求、10 输入 / 64 输出、采样与 logprob 配置，仍以正常 `LLM.generate()` 和 decode 32 为重点；只增加一种明确记录配置的 graph 路径，不铺开模式矩阵。
- 先确认实际生效路径：记录 graph 配置、捕获范围与 replay 证据，逐项区分 prefill/decode、图内部分与仍走 eager 的部分；不能仅凭关闭 `enforce_eager` 就认定整个请求都在 replay。
- 采集口径：沿用调度、输入准备、forward、采样、结果回收的阶段计时与线程身份；分开记录初始化/编译/捕获和预热后的请求执行，不将一次性捕获成本混入稳态 replay 时间。
- 图执行重点：记录实际存在的输入/元数据更新、graph task update、replay 提交和结果同步边界。依据可用的 flow、运行时标识和设备记录建立关联；不能强套 eager 的逐算子 Enqueue 对应关系，无法确认的图内部行为保留为未知。
- 对比重点：阶段墙钟与线程 CPU 时间、Host 提交活动、同步次数与耗时、设备任务间隙及对应 CPU 活动。区分 CPU 阻塞与设备依赖；不能把图内部 stream 的局部间隙当成整颗 NPU 空闲。
- 验证与恢复：沿用隔离的 reference/diagnostic/recovery 流程、可恢复包装及源码审计，分别披露两种模式的观测成本。核验 token、结束原因和 logprob；如存在数值差异，记录差异及判据，不预设跨模式必须逐位一致。若环境或基础配置已变化，补采匹配的 eager 对照。
- 预期产出：graph 下三个问题的答案、与 eager 使用相同口径的阶段表及 CPU→运行时→NPU 时间线，并明确哪些 CPU 工作减少、哪些仍存在、哪些原因还不能确认。

### Graph subtask 回填

- [x] 加入 `--mode eager|graph`，graph 采用 PIECEWISE / capture size=1，保留正常请求流程；同环境重采 `eager-02` 作为对照。
- [x] 25 个捕获图、每 decode 25 次 replay、63 个 decode 共 1,575 次，prefill 无 replay；正式测量期间无 capture，也未观察到 task update 调用。初始化单独记录。
- [x] graph 的 CPU 时间：decode 32 forward 约 4.249 ms，其中 replay 调用合计 0.316 ms；剩余约 3.933 ms 尚未细分。输入准备、图外算子与结果回收仍存在。
- [x] graph 的提交与等待：replay CANN 调用位于主线程，无普通 Enqueue/Dequeue 关联；图外工作仍有队列下发。64 次 Event + 192 次 Stream 同步合计约 0.695 ms，不是诊断请求的大头。
- [x] graph 的间隙：最大示例 223.309 µs 中，201.793 µs 发生在下一次 replay CANN 调用开始之前；另外两项定位到输入拷贝及 slot mapping 提交。未强行归因全部空隙。
- [x] 性能对照只用无 profiler 参考：eager / graph 中位数 725.879 / 291.454 ms，约 2.49 倍；诊断观测成本单独披露。26 条物理 stream 没有计算任务跨流重叠。
- [x] 两模式各 11 次响应及跨模式 token/logprob/结束原因精确相等；27 个文件前后与跨模式审计一致，包装恢复、原生恢复和设备空闲检查通过。
- [x] 补充 graph 和匹配 eager 的 HTML/SVG、对照页、便携证据；保留未知内部行为。更细粒度的 forward 归因未在本 subtask 推进。

## Subtask 2：分析 forward 内部各部分占比与执行情况

- [x] **状态：首轮已完成。已有 trace 的非重叠占比和一次 RoPE 细读已实现；剩余时间仍明确保留为未归因。**
- 目标：细分 decode 32 的 forward CPU 侧耗时，解释哪些部分在准备参数、执行框架逻辑、调用 launcher、入队或等待，以及它们与 NPU 执行如何交叠。首轮 eager 为约 11.95 ms；现有匹配对照为 eager 13.099 ms / graph 4.249 ms，后续以明确标注轮次的证据为准，不混算。
- 范围：以现有单请求、BF16、eager 的 decode 32 为起点，结合 Subtask 1 的 graph 对照决定优先细分范围；不立即增加模型、并发或 benchmark 矩阵。
- 第一步：先用已有证据整理 forward 内已记录范围的包含关系，区分包含子调用的时间与扣除子调用的自身时间；占比统一以前述 forward 范围为分母，明确未归属时间，不拼凑出缺少证据的完整分解。
- 最小深入范围：优先选择已定位的 **RoPE Host→Enqueue**，沿实际调用关系区分 Python 参数处理、Triton launcher、C++ 入队边界；观察结果后再决定是否扩大到其他部分。
- 证据要求：分别报告主线程与下发线程、墙钟与线程 CPU 时间、Host 范围与 NPU 任务时间；保留跨线程关联，避免嵌套重复计数。新增观测若有必要，继续采用可恢复的临时包装，并核验输出与观测成本。
- 预期产出：一张 forward 内部耗时与占比表（含未解释部分）、一条细化的 RoPE 提交时间线，以及已确认事实、仍待验证的原因和下一小步建议。

### Forward subtask 首轮回填

- [x] 复用已有 `eager-02` / `graph-01`，按同线程包含关系计数，父子不重复相加。占比分母采用 forward profiler 范围，并与内部双时钟边界明确区分。
- [x] eager：attention 28.70%、RoPE 18.78%、GEMM 12.46%、add RMSNorm 9.03%，其他已记录操作 6.32%，未覆盖 24.71%。graph：图外 attention 66.60%、replay profiler 范围 15.55%、未覆盖 17.85%。
- [x] graph 的 24 次 attention 进一步分到直接子范围；扣除这些子调用后，自身余量合计 1,013.285 µs，未强行命名为 Python 或等待。
- [x] 新增 `--forward-detail`：只选中 eager decode 32 的第一次 RoPE。Python 包装、JIT.run、动态 binder、native launcher 均记录双时钟，并通过队列/CANN/CSV 身份关联到一个 3.320 µs 的 NPU kernel。
- [x] 选中调用无编译，预热缓存指纹不变；30 个已审计文件未变，11 次输出精确一致，包装和原生执行恢复通过。
- [x] 明确记录观察扰动：被包装的 RoPE 外层约 268.323 µs，其余 23 次中位数约 108.489 µs。新增计时用于读调用结构，不作为稳定性能占比；native C++ 内部未完整采集。
- [x] 输出占比表、可展开调用树、RoPE SVG、机器可读证据、复现命令及便携归档。

### Subtask 2 后续：一次 graph 图外 attention（2026-10-02）

- [x] 只细化预热后 decode 32 第一层的一次调用，保留原生 PIECEWISE 请求；新增 `--attention-detail` 与 8 个可恢复的方法范围。
- [x] 沿远端源码确认 context → backend.forward → KV 写入 → FIA 参数准备/提交 → output 整理；实际 DecodeOnly 使用主机长度列表 `[42]` 和 NPU KV view，没有在此路径 `.tolist()` 回收长度。
- [x] 精确关联两项计算与一项设备拷贝：KV 写入 1.900 µs、FIA 23.581 µs、MEMCPY 0.600 µs。同一次调用的两个 copy 队列只有一个设备任务；另一个无设备任务，与源码中的 output 别名自拷贝相容，native no-op 判断仍未追入。
- [x] 分开报告内部双时钟和 profiler 包含树，保留各方法余量；不将旧的 24 次 attention 约 1.013 ms 余量与新单次重度观测直接相减。
- [x] 披露扰动：选中 Host 范围 395.780 µs，其余 23 次中位数 109.082 µs；诊断请求约为原生参考中位数的 1.530 倍，不能作为稳定性能占比。
- [x] 11 次响应完全一致、29 个源文件不变、绑定与原生推理恢复；测量期间无重新 capture，保持每 decode 25 次 replay。选中范围没有显式 CANN 同步记录，不推断 native 绝无等待。
- [x] 输出 [结论](../practice_30_cpu_submission_timeline/ATTENTION_RESULTS.md)、[HTML/SVG](../practice_30_cpu_submission_timeline/report/attention/index.html) 与便携证据。

RoPE 的补充源码分析已独立整理到 [Triton 参考笔记](../references/Triton/README.md)：历史 `rope_native` 包含 Python launcher 包装，不能当作纯 C++ 耗时。核数与 CANN 工具另见[未来任务](2026-10-02-triton-core-execution-observability.md)。

下一小步候选：减少嵌套观测，只继续解释一个 FIA 调用或输出 copy 边界；尚未开展，不扩展 benchmark 矩阵。
