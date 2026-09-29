# 2026-09-28 Ascend 多 Stream 测试任务

- 计划日期：2026-09-28（Europe/Berlin）。
- 状态：环境复查、P0、P1、P2 与 P3 阶段实验已完成；P3b 元数据后续见 Practice 27，P4 保持待调查。
- 目标：今天按优先级推进多 stream 实测，分别回答依赖是否正确、设备是否重叠、端到端是否获益，并形成可复现的 kernel execution graph。
- 执行顺序：环境复查 → P0 采样双流 → P1 KV offload → P2 MoE 分支 → P3 多模态 / 多任务 → 结果汇总。P4 记录可行性与所需条件。
- 范围：优先单卡 eager。每项按实际完成情况验收；受阻、只完成微基准或只有源码证据，均不得标为完整实机验证。

## 1. 环境与已有证据

最近记录：单张 Ascend 910B2C、64 GiB HBM，CANN 9.0.0、torch / torch-npu 2.10.0、vLLM 0.21.0、vLLM-Ascend 0.21.0rc1。开始前重新核验，不把历史记录视为实时状态。

- 远端：SSH 别名 `ascend910`；工作目录 `/data/tianchi`。沿用现有认证，凭据不写入任务文档、脚本或结果。
- 已有模型：`/data/huggingface_home/hub/Qwen2.5-0.5B-Instruct`、`/data/huggingface_home/hub/Llama-3.2-1B-Instruct`；启动前确认仍存在。
- [Practice 16](../practice_16_multistream_parallel/RESULTS.md)：独立矩阵 / 向量分支存在真实重叠，不能据此推断模型收益；已有[同步策略对照](../practice_16_multistream_parallel/SYNC_COMPARISON.md)和[跨流依赖实验](../practice_16_multistream_parallel/CROSS_STREAM_DEPENDENCY.md)可复用。
- [Practice 17](../practice_17_vllm_multistream/README.md)：Qwen batch 32、Llama batch 64 的采样随机分支已出现设备重叠，尚缺无 profiler 性能闭环。
- [Practice 18](../practice_18_kernel_data_dag/RESULTS.md)：当前 Qwen 数据依赖投影中 prefill 的 W/CP 约 1.088，decode 为 1；存在原生 workspace 覆盖缺口，不能直接用于实机任意改流。
- 背景：[CUDA 多 Stream 应用与同步](../references/cuda_stream_use_cases/README.md)、[KV offload 同步问题](../references/vllm_syn_issuse_analysis/README.md)。

## 2. 今日安排与检查点

以下是从开始执行起的建议时间盒，目的是尽早暴露阻塞，不是已验证的工期。超时后先保存证据并判断是否切换到其他可执行项；未完成项继续保持未完成。

| 顺序 | 任务 | 首轮时间盒 | 检查点 |
|---|---|---|---|
| 0 | 环境、源码与资源复查 | 30 分钟 | 得到实际版本、空闲资源、候选功能和模型清单 |
| P0 | 真实采样双流性能闭环 | 60–90 分钟 | 无 profiler 对照可重复，明确收益或无收益 |
| P1 | KV offload 与计算重叠 | 2–3 小时 | 真实 D2H / H2D 触发，依赖和复用边有证据 |
| P2 | MoE shared expert 分支 | 90–120 分钟 | 完成真实层级分支的单流 / 双流对照，或明确实现阻塞 |
| P3a | 独立推理任务并发 | 45–60 分钟 | 串行、双流、batching 三种方案可比较 |
| P3b | 多模态跨请求流水线 | 60–90 分钟 | 模型可用时完成最小双请求实验，否则记录缺少的条件 |
| P4 | tile 级 / 多卡方案 | 15 分钟 | 明确所需 kernel 改造或额外设备，不冒充已实测 |
| 收尾 | 证据核验与结果汇总 | 30–45 分钟 | 每项有状态、结论、命令和证据路径 |

上述完整计划可能超过一个工作日。优先保障 P0、P1、P2 的有效结论；P3 尽早确认模型与实现是否可用。若重点调整为 forward 内部并行，可将 P2 提到 P1 之前。

## 3. 环境复查

- [x] 记录检查时间、NPU 型号 / 可见数量、可用 HBM、host 内存、磁盘与当前占用。
- [x] 记录实际 CANN、torch-npu、vLLM、vLLM-Ascend 版本及相关源码指纹。
- [x] 确认现有模型、端口与独立输出目录；只管理本次启动的服务进程。
- [x] 核对 `enable_async_exponential`、`NPUOffloadingSpec`、`multistream_overlap_shared_expert` 在安装版本中的实现和适用条件。
- [x] 确认是否已有可用的 shared-expert MoE 层实现、多模态模型；在前期暴露下载、容量和适配成本。
- [x] 评估是否需要重跑 Practice 16：本轮未重采历史实验，基础 NPU 算术与独立采样检查通过。

环境证据见 [inventory.json](../practice_17_vllm_multistream/results/2026-09-28-p0/environment/inventory.json)。
仅有两个 dense 文本模型；shared-expert 实现在 `ops/fused_moe/fused_moe.py` 中，
要求 `has_shared_experts`，并与 mix placement 不兼容，单卡 harness 尚未验证。
基础检查前后通过，但历史硬件告警 `80C98001` 仍存在，性能结论限定于本机本轮条件。

## 4. P0：补齐真实采样双流的性能闭环

**问题：提前生成采样随机数，是否在当前模型和负载下改善请求完成时间？**

- [x] 复用 Practice 17 的请求与配置：单卡 BF16 eager，随机采样，无请求级独立 seed；保留服务器级 seed 并记录。
- [x] 主对照：Qwen batch 32、Llama batch 64，各比较 `enable_async_exponential=true/false`。
- [x] 低成本边界对照：Qwen batch 1、Llama batch 32；均覆盖 4 / 64 token。
- [x] 新增独立无 profiler 性能入口；现有诊断 runner 增加输出长度参数，保留默认 4 token。
- [x] 两组保持输入、输出 token 数、采样参数及服务配置一致；实际活跃 batch / scheduler step 在独立诊断中记录，不冒称每个性能样本具有相同调度形态。
- [x] 按开—关—关—开进行四批独立服务测量，每批每条件预热 5 次、正式 5 次；开／关各 10 个正式样本，报告中位数、波动和前后两对方向。
- [x] 八份独立 trace 核对 q producer、sampler consumer、物理 stream 与同步；性能测量关闭 profiler 和 Python 重型观测。
- [x] 六项固定 q 数值检查通过；开启路径 q 一致，真实 q 的有限性 / 正值检查通过。没有要求两次随机生成文本一致。

**验收：** 有无 profiler 性能表、独立 trace 证据和正确性核验。`false` 组仍可能使用第二条 stream，必须称为“提前生成开 / 关”，不能称为“单流 / 双流”。明确区分“已重叠”和“有性能收益”。

**本次结果：** [性能报告](../practice_17_vllm_multistream/PERFORMANCE.md)及
[验证记录](../practice_17_vllm_multistream/results/2026-09-28-p0/validation.json)。
本轮没有稳定加速；Qwen batch 32 的 4 / 64 token 完成时间中位数分别增加约
4.58% / 5.05%，Llama batch 64 的 64 token 增加约 1.96%，均为本轮描述性结论。
160 个正式样本、160 个排除的预热样本、六项独立采样数值检查和八组诊断已归档；
149,991 个计算任务与 CSV 匹配，另保留两条未关联 runtime placeholder。
16 项测试、八份新证据的本地重放和八张图的无环检查通过。

## 5. P1：真实 KV offload 的传输、计算与复用 DAG

**问题：KV 保存 / 回载是否能与独立请求计算重叠，哪些依赖保证结果正确？**

实现入口参考：[对应版本的 KV Cache CPU Offload 指南](https://docs.vllm.ai/projects/ascend/en/v0.21.0rc/user_guide/feature_guide/kv_cache_cpu_offload.html)。配置以安装版本源码为准。

- [x] 确认 `OffloadingConnector` / `NPUOffloadingSpec`、prefix caching 及 eager 路径的兼容性，定位实际传输 stream 和同步代码。
- [x] 优先复用已有小模型，限制实验 KV 容量并设置有界 CPU block pool；避免通过耗尽整机内存触发实验。
- [x] 构造前缀 A → 其他请求造成 KV 容量压力 → 再访问前缀 A，证明真实 D2H 和后续 CPU 命中回载 H2D；仅打开 offload 开关不算触发成功。
- [x] 在 A 回载期间安排独立请求 B 的就绪计算，核对 scheduler 确实允许它推进。
- [x] 在同一 offload 实现上构造强制串行与正常异步对照，保持请求、缓存策略和传输量可比；如果实际调度改变了传输量，单独披露，不能归因于改流。
- [x] 增加关闭 offload、通过重新计算恢复前缀的系统层基线，分别统计重算量、传输量与命中情况。
- [x] 采集计算 / memcpy 节点、实际 stream、event、KV block ID、存储代次、字节范围及复用时刻。
- [x] 核验 `KV 写入 → D2H 读取 → 源 block 复用`、`H2D 写入 → attention 读取`，以及 CPU block 的写入、读取、回收顺序。
- [x] 比较恢复后的 KV 或相应模型输出；使用确定性采样 / 数值容差进行可解释的正确性对照。
- [x] 独立运行无 profiler 性能测量，报告请求完成时间、传输字节、命中 / 重算量；若测 TTFT、token 间延迟，使用流式客户端并记录 token 到达时间。

**验收：** 有真实 offload / reload 证据、带存储代次和同步边的 DAG、强制序列化 / 原生异步对照，以及正确性与性能结果。

**受阻路径：** 若安装版本缺少实现或不兼容，保留具体源码 / 错误证据，先完成 pinned host buffer 与 NPU buffer 的有界分块传输实验。该结果标记为“机制微基准”，真实 vLLM offload 项保持受阻，不用张量命名替代真实集成。

**本轮结果：** 已完成 [Practice 20](../practice_20_kv_offload_overlap/README.md)；见[实测报告](../practice_20_kv_offload_overlap/RESULTS.md)和[交互图](../practice_20_kv_offload_overlap/report/index.html)。安装版本的旧 `NPUOffloadingSpec` 接口不兼容，按确认方案使用已有 `AscendSimpleCPUOffloadConnector`，未修改系统安装。串行组保留相同的三条物理 stream，通过主机屏障消除重叠。

60 个正式周期及 60 个预热周期完成三模式逐 token 对照；原生 H2D 搬运 12 / 36 MiB，与独立请求 B 的计算重叠约 346 / 725 µs。三份诊断图覆盖 785,902 个设备任务，两份 offload 图各验证 786 条 KV 数据、复用、发布和回载约束，均有同步可达保证。本轮小模型上回载慢于重算，未得到端到端收益。边界为正常 prefix 驱逐与 block 级访问；不覆盖活动请求抢占／取消或完整原生 workspace 的精确访存 DAG。

## 6. P2：forward 内部的 shared-expert 分支

**问题：同一层的 shared expert 与 routed experts 能否正确并行，收益受哪些资源约束？**

实现入口参考：[vLLM-Ascend 配置说明](https://docs.vllm.ai/projects/ascend/en/v0.21.0rc/user_guide/configuration/additional_config.html)中的 `multistream_overlap_shared_expert`；必须核验具体模型、TP=1 和 eager 是否实际支持。

- [x] 找到包含 shared expert 的真实层实现；现有 dense Qwen 不适用该功能。
- [x] 优先构造单层 harness，复用真实算子路径、可控权重与输入，明确它不是完整预训练模型的端到端验证。
- [x] 核对 shared / routed 两分支的输入就绪、输出合并、gate 依赖、workspace 和 allocator 生命周期。
- [x] 使用相同输入和权重比较单 stream 与双 stream；测试 decode 类小 token 数和 prefill 类较大 token 数两个形状。
- [x] 根据 dtype 预先记录数值容差，报告最大误差；两条分支完成后合并，不能只验证单独分支输出。
- [x] 分别采集执行图和无 profiler 性能，报告实际重叠量、各分支独立 / 并行耗时、总完成时间。

**验收：** 得到 `输入 → 两分支 → 合并` 的真实层级 kernel 图、正确 event 连接及性能结果。若模型或设备路径要求多卡，明确记录限制；通用双 MLP 只能作为机制实验，不能标为真实 MoE 路径已完成。

**本轮结果：** [Practice 21](../practice_21_moe_shared_overlap/README.md) 完成真实 Qwen2 MoE 单层 eager 对照，见[报告](../practice_21_moe_shared_overlap/RESULTS.md)与[交互图](../practice_21_moe_shared_overlap/report/index.html)。固定构造权重，测试 1/32/256/1024 token；这是缩小尺寸的真实层，不是完整预训练模型验证。24 次详细诊断覆盖 538 个设备任务、192 条通过验证的数据要求，另有 24 次无方法观察器的轻量诊断。两类诊断均未观察到 shared/routed 计算交叠；48 组完整层性能配对显示双流中位数慢约 7%–9%。输出、CPU 参考、72 次输入复用及权重不变检查通过。native workspace 精确访存仍不在覆盖范围内。

## 7. P3：扩展到多任务与多模态

### P3a：独立推理任务并发

- [x] 选择已有小模型或其实际 forward 路径，准备独立输入、KV 与可变工作区，确认运行上下文允许并发。
- [x] 在相同总工作量下比较：两任务串行、两 stream 执行、合成 batch 执行；记录模型权重是否共享及额外内存。
- [x] 同时报告整体吞吐和每任务完成时间，验证输出；实际 stream 由 trace 确认，不将两个 HTTP 请求或两个进程直接视为双流。
- [x] 区分框架现有调度与自建执行 harness；若 vLLM 将请求合批，按实际行为记录。

**验收：** 三种策略有公平对照，能够判断双流相对 batching 是否仍有价值。

**本轮结果：** [Practice 23](../practice_23_independent_inference/README.md) 已完成完整预训练 Qwen2.5-0.5B 的 HF eager harness，见[报告](../practice_23_independent_inference/RESULTS.md)及[执行图](../practice_23_independent_inference/report/index.html)。四形状、144 个性能样本、24 个诊断 trial；共享只读权重、独立 KV，80 条就绪／完成约束及 16 项跨任务顺序检查通过。长 prefill 双流观察到约 12.8 ms 计算交叠，pair 时间配对中位数下降 13.7%；decode 基本持平。数值通过的三形状 batch=2 更快。

**验收限制：** 1024-token prefill 的 batch 未通过原始数值容差，保留 14 次失败对照；没有放宽门限，该格收益不采纳。三方有效比较覆盖其余三形状，长 prefill 只采纳 serial/parallel 比较。该任务实验已收尾，长 prefill batching 的首个数值分歧另列后续定位；不是原生 vLLM scheduler、HTTP serving 或完整 kernel 内存 DAG。

**数值后续（2026-09-29）：** [首次数值分歧定位已完成](../practice_23_independent_inference/NUMERICS.md)：第 0 层 `mlp.gate_proj`，固定相同输入仍随 M=1024/2048 变化；12 个隔离 kernel 精确关联，FP64 误差界及 28 个精确有理数点积复核通过。局部 FP32 替换仍失败；完整 FP32 配置完成另外 144 个三方样本，四形状均通过，原始 BF16 失败继续保留。内部 tiling／归约树未恢复，不声明原生 BF16 已修复。

- [x] 定位首次分歧、验证独立算子与高精度参考。
- [x] 在明确区分精度配置的前提下补充有效三方对照。
- [x] 核验静态输入／KV 与 graph capture 条件，完成同精度 serial／双流／batch replay 对照；capture、预热、replay 分开计时，采集实际 stream 与同步。

**Graph 后续（2026-09-29）：** [Practice 24](../practice_24_graph_replay/README.md) 已完成完整 FP32、固定形状／固定 decode 步的 eager 与 graph 六配置对照，见[报告](../practice_24_graph_replay/RESULTS.md)和[执行图](../practice_24_graph_replay/report/index.html)。288 个无 profiler pair、48 个诊断 trial、336 次数值对照及 24 次输入／初始 KV 复用检查通过；40 次 replay 的 48,204 个内部任务关联完整，160 条边界要求和 32 项跨任务顺序检查通过。Graph 双流四形状都快于 graph 串行，但 batch=2 仍更快。串行 graph 使用不同内部 stream，由同一调用 stream 的 completion → launch 顺序串行化。独立 graph pool、固定 KV 长度，不代表持续生成、native vLLM 调度或完整 native 内存 DAG。后续 P3b 已完成，见下一节。

### P3b：视觉编码与另一请求的语言计算

**输入准备：** 已确认 `/data/huggingface_home/hub/Qwen2.5-VL-3B-Instruct`，索引引用的两个权重分片及 processor/tokenizer 配置均存在。已下载并同步两张[公开示例照片](../datasets/p3b_images/README.md)至 `/data/tianchi/datasets/p3b_images/`：`beach.jpeg`（2048×1365）与 `beijing.jpeg`（614×410）；本地／远端完整解码和 SHA256 核对通过。后续运行资格及正式实验已完成，见下方结果。

- [x] 确认已有适配模型、权重、图像输入和剩余资源，再确定最小视觉 / 语言执行路径。
- [x] 构造请求 A 处于语言阶段、请求 B 进行视觉编码的负载；比较串行调度与双流调度。
- [x] 保留 B 的 `视觉特征就绪 → B 的语言消费` 依赖，隔离不同请求的 KV 与可变状态。
- [x] 校验中间特征 / 输出，记录实际设备重叠、每请求延迟和总完成时间。

**验收：** 至少两个真实请求阶段构成可解释 DAG。只有 vision / language 两段人工张量计算时标为机制微基准；不宣称复现了 HydraInfer 或框架原生多模态双流。

**本轮结果：** [Practice 25](../practice_25_multimodal_overlap/README.md) 完成 Qwen2.5-VL-3B 的真实视觉／语言阶段 eager harness，见[报告](../practice_25_multimodal_overlap/RESULTS.md)和[执行图](../practice_25_multimodal_overlap/report/index.html)。两张图片、两档实际视觉长度 54 / 247、A prefill / decode，共 8 格；192 个性能样本、32 个诊断 trial、224 次输出校验均逐元素一致，8 格另与完整原生 forward / generation 对照通过。344,586 个设备任务，256 条数据／边界要求与 32 项显式顺序检查通过。先提交语言 prefill 再提交视觉时双流耗时下降 6.76%–13.16%，但 A 本身完成变慢；先视觉后语言无同样收益，单步 decode 基本持平。每次视觉诊断含 91 条原生同步 API 调用，说明单 CPU 提交顺序与隐式主机阻塞会限制 overlap。不是原生 vLLM 多模态调度，native workspace 和隐式 CPU 因果图仍未完整恢复。

**元数据后续（2026-09-29）：** [Practice 27](../practice_27_vision_metadata/README.md) 完成 native / lengths / cached 三组消融。视觉同步 API 91 → 7 → 0；完整预计算后 VL 的 prefill 和 decode 都有真实重叠，576 个正式样本及含预热的 816 次输出检查通过。准备成本、单流优化与并发收益分别报告；见[任务记录](2026-09-29-vision-metadata-sync.md)和[结果](../practice_27_vision_metadata/RESULTS.md)。

## 8. P4：后续能力边界

- [ ] tile 级重叠：记录候选 producer / consumer、tile 依赖、设备侧同步与内存可见性需求。cuSync 的 CUDA 实现不能直接作为 Ascend 可用实现。
- [ ] 多卡通信重叠：复查可见设备；单卡条件下不安排真实 HCCL 多卡 overlap 测试。
- [ ] 明确这两项的状态是可行性调查，实机测试在具备 kernel 实现或额外设备后另排。

## 9. 统一证据与收尾

每项结果至少保存：精确命令 / 配置、环境和源码版本、负载参数、原始重复测量、正确性结果、trace 与分析摘要。输出使用独立目录，不覆盖历史正式结果；新 practice 编号在实施前检查是否空闲。

图中至少区分 `data_dependency`、`stream_order`、`event_wait`、`host_sync`、`storage_reuse`。记录边的证据来源和覆盖缺口；没有 happens-before 保证时，不能用“本次碰巧先执行”补边。stream 标识按本次运行关联，不沿用历史物理 stream 编号。

以下勾选仅表示本轮环境复查与 P0 的收尾情况，后续实验执行时需重新核验。

- [x] P0 重叠按实际计算区间求交，未用分支包围范围替代。
- [x] profiler 诊断与无 profiler 性能分开；性能等待完整 HTTP 响应。
- [x] 报告所有已完成对照，包括无重叠、退化及不确定结果。
- [x] 八张执行图、代表时间线及结论已保存；未宣称完整模型原生数据依赖已全部恢复。
- [x] 本轮更新下表和证据链接；P1 另有独立验收，未用 P0 结果替代；P2 已另行完成层级验收；P3a 已完成并明确数值未通过格；P3b 阶段 harness 已完成；P4 仍待能力调查。

| 任务 | 状态 | 证据目录 | 核心结论 / 阻塞 | 下一步 |
|---|---|---|---|---|
| 环境复查 | 已完成 | [环境记录](../practice_17_vllm_multistream/results/2026-09-28-p0/environment/inventory.json) | 基础数值通过；历史告警仍存在；后续功能源码入口存在，模型仅有两个 dense 文本模型 | P1 已复查环境与 connector；告警仍在 |
| P0 采样双流 | 已完成 | [P0 报告](../practice_17_vllm_multistream/PERFORMANCE.md) / [验证记录](../practice_17_vllm_multistream/results/2026-09-28-p0/validation.json) | 160 个正式样本、六项数值检查、八组诊断；未发现稳定加速，完整重放通过 | P1、P2 单层实验已完成 |
| P1 KV offload | 已完成 | [P1 报告](../practice_20_kv_offload_overlap/RESULTS.md) / [验证记录](../practice_20_kv_offload_overlap/results/published/validation.json) | 60 个正式周期；真实 D2H/H2D 与计算重叠，1,572 条 KV 约束通过；本轮重算更快 | P2 单层实验已完成；后续可测更大模型／更长前缀的收益交叉点 |
| P2 MoE 分支 | 层级实机实验已完成 | [P2 报告](../practice_21_moe_shared_overlap/RESULTS.md) / [验证记录](../practice_21_moe_shared_overlap/results/published/validation.json) | 原生双 stream、同步正确；四种形状未见计算重叠，完整层双流慢约 7%–9% | 按计划进入 P3；收益交叉点与 graph 另行实验 |
| P3a 多任务 | harness 实验完成；一格数值未通过 | [P3a 报告](../practice_23_independent_inference/RESULTS.md) / [验证](../practice_23_independent_inference/results/published/validation.json) | 长 prefill 双流确有交叠；decode 收益不明显；三个有效形状 batch 更快 | 数值定位、全 FP32、固定步 graph 及后续 P3b 已完成；P4 待能力调查 |
| P3b 多模态 | 真实模型阶段 harness 已完成 | [P3b 报告](../practice_25_multimodal_overlap/RESULTS.md) / [验证](../practice_25_multimodal_overlap/results/published/validation.json) | 先语言后视觉的 prefill 双流快 6.76%–13.16%；VL 与 decode 无稳定收益；实际特征 handoff 与 256 条约束通过 | 视觉元数据同步消融已完成（Practice 27）；graph／资源竞争计数器待后续；P4 仍为能力调查 |
| P4 能力调查 | 待执行 | — | — | 记录实现和设备前提 |
