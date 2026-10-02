# 笔记：视觉 attention 的 lengths、Sub 计算与后续研究问题

记录日期：2026-09-30。本文整理 Practice 27 中已经确认的概念和证据，并记录三个研究方向。待办 1、2 已完成可观测层的源码核验与独立小数组实验，见 [tolist / D2H 报告](TOLIST_D2H.md)和 [Sub 提交与内存报告](SUB_EXECUTION.md)；2026-10-01 完成待办 3 的隔离正确性实验，见 [校验覆盖笔记](CORRECTNESS.md)和 [HTML 报告](correctness_probe/report/index.html)。同步操作的整体分布见[同步分析笔记](README.md)。

实验范围：Qwen2.5-VL-3B-Instruct、HF eager、Ascend 910B2C、torch / torch-npu 2.10、Transformers 5.5.4。以下设备归属和调用链对应这次固定实现，不能直接推广到所有 attention 后端。

## 1. lengths 表示什么

`lengths` 是各 attention 分段的长度列表或 tensor。窗口 attention 把按窗口排列的视觉序列切成多个片段，分别计算 attention；Q/K/V 需要使用相同的分段边界。

`cu_seqlens` 是这些片段的累计边界，`lengths` 是相邻边界的差值：

```text
示例，非本次实测数值：
cu_seqlens = [0, 4, 10, 12]
lengths    = [   4,  6,  2]

第 0 段：[0, 4)   长度 4
第 1 段：[4, 10)  长度 6
第 2 段：[10,12)  长度 2
```

代码对应：

```python
lengths = cu_seqlens[1:] - cu_seqlens[:-1]
splits = [
    torch.split(tensor, lengths.tolist(), dim=2)
    for tensor in (query_states, key_states, value_states)
]
```

这里需要区分三种长度：`lengths.numel()` 是分段数量；`lengths[i]` 是某个分段的序列长度；`sum(lengths)` 是这组分段覆盖的总序列长度。它们也不等于 merger 之后送入语言模型的视觉 token 数。

例如本次 `prefill-beach-v256` 的首个窗口 attention 有 20 个分段、21 个累计边界；Q/K/V 被切分的序列维为 988，merger 后实际视觉 token 数为 247。`v256` 是场景的视觉 token 预算标识，不是 Sub 输入数组长度。

## 2. 长度原本来自 CPU，为什么又在 NPU 上计算

`get_window_index(grid_thw)` 根据 CPU 上的图像网格生成窗口索引和累计长度。其实现先统计窗口中的有效位置数，再累加，并乘以 `spatial_merge_unit`，得到视觉 attention 使用的序列边界；去除相邻重复边界可以消除无有效位置的窗口对应的重复累计值。

原 visual forward 显式执行：

```python
cu_window_seqlens = torch.tensor(
    cu_window_seqlens,
    device=hidden_states.device,
    dtype=torch.int32,  # 本次非 tracing 路径
)
cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)
```

因此，窗口累计长度被放到了 NPU。窗口 attention 再对这个设备 tensor 做差分，结果也在 NPU。这里的长度取决于网格和窗口布局，不取决于像素经过网络后的特征值。

原方法共享累计长度的准备路径：Flash Attention 分支直接接收 `cu_seqlens`，而本次 eager 分支需要 Python 长度列表来切分 Q/K/V。源码结构解释了为何 eager 会出现设备往返，但不能据此确定作者的设计意图。

本次 32 个视觉 block 中，28 个窗口 attention 使用这条 NPU 长度路径；4 个整图 attention 使用由 CPU grid 得到的 CPU 累计长度。不是所有 attention 的 `lengths` 都在 NPU。

### 2.1 为什么一行 Python 减法会变成 NPU Sub

决定执行后端的关键不是这行代码的写法，而是参与运算的对象类型和 `device`。本次窗口路径先用 `torch.tensor(..., device=hidden_states.device, dtype=torch.int32)` 创建 `cu_window_seqlens`；`hidden_states` 位于 NPU，所以 `cu_seqlens` 是 `device='npu'` 的 PyTorch Tensor，其数值存储在 NPU 内存中。Python 侧仍持有 Tensor 对象及 shape、dtype、device 等元信息。

```python
left = cu_seqlens[1:]    # NPU Tensor view
right = cu_seqlens[:-1]  # NPU Tensor view
lengths = left - right
```

两个切片仍是 NPU Tensor view，共享父 tensor 的设备存储。`-` 先进入 PyTorch 的 Tensor 运算绑定，形成标准 ATen 算子 `aten::sub.Tensor`；dispatcher 再根据输入 tensor 的 NPU dispatch key（本版本为 `PrivateUse1`）选择 torch-npu 注册的实现。该实现分配 NPU 输出并进入 `aclnnSubGetWorkspaceSize → aclnnSub`，最终由 CANN runtime 把 `aclnnSub_SubAiCore_Sub` 提交到当前 NPU stream。

```text
Transformers 模型中的 Python 表达式
  → Tensor.__sub__ / aten::sub.Tensor
  → PyTorch dispatcher 根据 device 选择 PrivateUse1
  → torch-npu / op-plugin 的 Sub 实现
  → aclnnSubGetWorkspaceSize → aclnnSub
  → CANN runtime → NPU Sub kernel
```

同一行表达式因对象不同会有不同结果：

| `cu_seqlens` 的对象 | `cu_seqlens[1:] - cu_seqlens[:-1]` 的行为 |
|---|---|
| Python `list` | 切片仍是 list，list 不支持减法，直接报错 |
| CPU PyTorch Tensor | dispatcher 选择 CPU Sub，在 CPU 上计算 |
| NPU PyTorch Tensor | dispatcher 选择 torch-npu Sub，提交 NPU kernel |

因此不能简称为“torch-npu 定义了一种特殊 tensor”。上层对象仍是 `torch.Tensor`；torch-npu 为 NPU device 接入 PyTorch dispatcher、内存和算子实现。也不是 Python 解释器把源码直接编译成 NPU kernel。本次模型业务表达式位于 Hugging Face Transformers 的 Qwen2.5-VL eager forward，我们的实验程序负责调用和观测；这条实测路径不经过 vLLM-Ascend。

`lengths` Tensor 返回 Python 后，NPU kernel 可能尚未完成。只有后续 CPU 需要数值，例如调用 `.tolist()`，才会在当前实现中等待 stream、执行 D2H 并构造 Python `list[int]`。详细的 dispatcher、主机队列和 CANN 两阶段接口见 [Sub 提交与执行报告](SUB_EXECUTION.md#2-cpu-到-npu-的实际调用链)。

## 3. Sub 的实际调用与计算证据

差分使用普通逐元素减法，不是一个专门的“窗口长度计算”kernel：

```text
cu_seqlens[1:]、cu_seqlens[:-1]   → 两个切片
                 ↓
             aten::sub           → PyTorch 算子
                 ↓
             aclnnSub            → Ascend 主机接口
                 ↓
       aclnnSub_SubAiCore_Sub    → NPU kernel
                 ↓
             lengths            → NPU int32 tensor
```

取 `native / prefill-beach-v256 / parallel / 重复 1（VL）` 的首组窗口 attention：

| 项目 | 真实记录 |
|---|---|
| 两个切片的主机算子 | `aten::slice`，trace_index 116727 / 116729 |
| 减法主机算子 | `aten::sub`，trace_index 116732，时长 9.845 µs |
| Ascend 接口 | `aclnnSub`，trace_index 116731，时长 1.926 µs |
| 减法输入 | 两个长度为 20 的 int32 tensor |
| 设备 kernel | `aclnnSub_SubAiCore_Sub`，trace_index 528908 |
| 执行类型 | `AI_VECTOR_CORE` |
| 物理 stream / task ID | 43 / 19157 |
| 设备 kernel 时长 | 1.580 µs |
| 对应图节点 | `prefill-beach-v256:k:528908` |
| kernel CSV 行索引 | 26130，解析数组从 0 开始计数，不含表头 |

这些主机与设备记录通过已生成图的 flow 关联，不是按时间接近猜测匹配。

进一步比较同一 trace 的绝对时间（单位 µs）：`aten::sub` 在 `1790697769325533.765` 开始，约在 `1790697769325543.610` 返回；设备 Sub 在 `1790697769325586.594` 才开始。**这个实例中，主机 Sub 调用已经返回，设备 Sub 尚未开始，因此它没有等待设备减法执行完成。** 这不等于主机调用没有开销，也不证明首次调用、分配或其他队列状态下永远不等待。

上面的主机区间不是 Python 整行的全部耗时，设备 1.580 µs 也不包括前面的排队。随后 `.tolist()` 的同步才在这条路径上建立主机等待。

## 4. 内存与数据流

```text
CPU grid
  → CPU 窗口累计长度
  → H2D：NPU 累计长度 tensor
  → NPU unique_consecutive
  → 两个切片作为 Sub 输入
  → NPU Sub 输出 lengths
  → .tolist()：等待 + D2H，得到临时 CPU tensor
  → CPU 构造 Python list[int]
  → CPU 用列表描述 Q/K/V 分段，Q/K/V 数据仍在 NPU
```

Tensor 的主机描述信息与设备存储不同。CPU 持有 Tensor 对象、shape、stride、device 等，不代表 CPU 已经拥有数值。`.tolist()` 生成主机副本和 Python 列表，不把原始 NPU tensor 原地迁移或改写。

这次 Sub 输出包含 20 个 int32，有效数值载荷 80 字节。原列表推导式对 Q/K/V 各做一次 `.tolist()`，因此有三次重复读取。首组实际同步 API 时长分别为 42.705、0.530、0.381 µs；其后的 memcpy API 时长分别为 15.409、8.197、8.005 µs。等待时长不能都算到 Sub 上，memcpy API 时长也不是纯数据传输时间。

P27 的 `lengths` 变体在 CPU 上预计算相同长度并保存为 tuple，绕过每层设备差分和读回。这使窗口 attention 的 84 条相关同步 API 消失；其他视觉元数据同步仍保留。变量 `lengths` 与实验变体名称 `lengths` 相关，但一个是数据，一个是实验配置。

## 5. 待研究问题 1：tolist 同步在哪，如何连接到 D2H

**状态：2026-09-30 完成当前可观测层的分析和独立小数组对照。** 详细结论、实际源码行号、内存与二进制证据见 [TOLIST_D2H.md](TOLIST_D2H.md)。驱动内部暂存/DMA 等不可见项在报告中保留边界。

**问题：** `.tolist()` 触发的同步具体在哪一层？Python、PyTorch、torch-npu 和 CANN 之间怎样连接到 D2H？

已确认：PyTorch `tensor_to_list` 对非 CPU tensor 转到 CPU，然后生成 Python 对象；实际 NPU `_to_copy` 和 OpApi copy 路径中，`non_blocking=false`，CPU 副本未 pinned，先 `aclrtSynchronizeStream`、后 `aclrtMemcpy`。原笔记引用的 WithTimeout 是另一条拷贝实现，已按实际路径修正。

- [x] 对照实际安装版本，补全从 `Tensor.tolist` Python 绑定、`tensor_to_list`、ATen dispatcher 到 NPU copy 实现的函数/文件/行号；记录源码与二进制版本。
- [x] 跟踪 `toBackend(CPU)` 的参数传递，确认 `non_blocking`、目标分配器和 D2H 方向在哪一层确定。
- [x] 核对同步封装、主机任务队列和 CANN runtime 的关系，确定等待的是哪条 stream、哪些已提交任务；区分 Python 提交线程与运行时 worker。队列排空没有单独计时。
- [x] 记录源/目标缓冲区、大小、是否使用 pinned host memory、是否有中间缓冲；框架到 CANN 的地址已核对，驱动内部缓冲/DMA 保留未知。
- [x] 分别记录同步与 memcpy API 耗时，以 CPU-only 调用测量列表构造路径，并完成三次 `.tolist()` 与一次复用对照；不将测量解释为精确拆分同一调用的所有成本。

补充说明：`non_blocking` 对显式异步 D2H 有作用，但 `.tolist()` 内部固定为 false，且 CPU 消费异步结果前仍需等待，见 [non_blocking 分析](TOLIST_D2H.md#7-non_blocking-能改变什么)。

预期产物：带源码位置的完整调用链，以及同一次调用中同步、D2H 和主机消费的时间线。不要将公开同版本源码等同于已验证的安装二进制内部执行轨迹。

## 6. 待研究问题 2：Sub 是否阻塞 CPU，怎样编译、提交及寻址

**状态：2026-09-30 已完成可观测层专项实验。** 见 [Sub 提交、执行与内存地址报告](SUB_EXECUTION.md)。CANN 描述符、两段 API、缓存和 runtime/device 时间线已核对；最终设备参数块和内部 tiling 仍不可见。

**问题：** Sub 是否完全阻塞 CPU？从 host 到 device 怎样完成编译和传递？输入输出地址是什么，kernel 如何被调用？

已有实例证明主机 Sub 返回早于设备执行，但“阻塞 CPU”需要具体到线程和阶段：调用线程可能进行参数检查、内存分配或提交；运行时 worker 可能处理队列；设备可能尚在执行之前的任务。这几种状态不能合称为“CPU 被完全阻塞”。

- [x] 对比仅 Sub、Sub 后立即 `.tolist()`、Sub 后显式同步三组，分别记录主机调用返回、设备 kernel 起止和等待区间；比较首次与预热后的调用。
- [x] 跟踪 `aten::sub → aclnnSub → runtime launch → 设备任务`，说明参数检查、输出分配、算子选择、workspace、入队与实际 launch 分别在哪发生；tiling 仅定位到 CANN 执行准备边界，具体函数和选中 key 未捕获。
- [x] 确认该路径使用已有 kernel 二进制、首次加载/缓存，还是涉及即时编译；若有编译，记录发生位置和产物。不要预设“每次 Python 减法都会在设备上编译”。
- [x] 采集父 tensor、两个 slice、Sub 输出的 `data_ptr()`、storage 基址、`storage_offset()`、shape、stride、dtype、有效字节范围，以及输出创建/释放引用和同步区间；未跟踪底层 malloc/free 的精确时刻。
- [x] 对连续 int32 一维父 tensor，验证 `[1:]` 的逻辑起点是否较父 tensor 向高地址移动 4 字节、`[:-1]` 是否共享基址。区分正常切片偏移与错误地址；地址按同一次运行解释，不把虚拟地址当作物理地址。
- [x] 对照 kernel 实际参数或可获得的运行时记录，核对 storage 基址 + offset 描述符及输出、workspace 的关系；未见额外转换 kernel，最终设备参数块未解码。
- [x] 将线程、队列、stream、task ID 和内存生命周期关联起来，区分算子 launch 的输入输出地址与 Python 对象身份。

已产出：一次 Sub 的主机—设备时序图、三组首次/后续对照、输入/输出内存表、现有 ELF 加载与执行缓存证据。以上是隔离小数组实验；P27 全模型的完整长度数组指针和最终 kernel 参数尚未据此恢复。

## 7. 待研究问题 3：偏移、溢出、翻转与校验

**状态：2026-10-01 完成单流小数组正确性与软件异常注入。** 26 个样例中 12 个合法样例全部通过、14 个注入异常均被专项校验检出；7 个异常来源样例仍被 split 接受。详见 [CORRECTNESS.md](CORRECTNESS.md) 和 [交互 HTML](correctness_probe/report/index.html)。未开展跨 stream race、分配器压力或硬件 ECC 测试。

**问题：** Sub 输出是否可能出现偏移、溢出、翻转等异常？已有检查能覆盖什么，还需补什么？

先明确“翻转”的含义：可能指减法顺序反了导致符号反转、元素顺序颠倒，也可能指内存 bit flip；这些需要不同实验。“偏移”也应区分合法 slice 的 storage offset 与错误读写位置。正常模型实验尚未发现这些异常；新实验在隔离小 tensor 上主动注入错误验证检测能力，不能据此宣称真实模型或硬件出现故障。

P27 已做源码哈希/实现契约检查、grid 与像素形状匹配检查，以及 features、logits、KV 等输出对照；含预热/资格的 816 次输出比较全部逐元素一致，8 个场景另与原生完整 forward/generate 对照通过。**这些属于已测场景的集成正确性证据，不是每次 Sub 输入输出的独立审计，也不是溢出或 bit flip 检测器。** 详见 [P27 结果](../../practice_27_vision_metadata/RESULTS.md) 与 [metadata.py](../../practice_27_vision_metadata/metadata.py)。

- [x] 保存 CPU 原始 grid/累计边界，以 Python 整数或足够宽的整数建立独立参考；转换为 int32 **之前**检查范围，避免参考值与被测值先发生同样的截断或溢出。
- [x] 校验去重后的累计边界从 0 开始、严格递增，末项等于实际 attention 序列长度；校验差分元素为正、元素数比边界数少 1、逐元素等于参考且总和等于序列长度。
- [x] 对“非负、单调且最大累计边界不超过 int32 最大值”的前提做显式检查。此时相邻差值也在非负 int32 范围内；重点追踪累计求和、乘法和 dtype 转换是否在此前就越界。
- [x] 验证切片 stride、storage offset、可访问范围、输入是否被改写、输出是否发生非预期别名；设计非零 offset、stride=2、边界窗口、重复边界等最小样例。完成前保留引用并同步，未专门制造提前复用或释放后使用。
- [x] 分别注入元素颠倒、操作数互换、单个长度错误、超范围累计边界及单 bit 修改，确认各校验能捕获什么。测试注入应使用隔离的小 tensor，不改模型权重或正式结果。
- [x] 核对当前版本 `torch.split` / ATen 源码及实际 Sub 调用路径的参数报错；CANN 闭源内部检查和直接 C API 调用仍未独立审计。已特别验证“长度总和正确但边界错位”的情况，不能仅靠总和检查判断分段正确。
- [x] 区分算术错误、地址/生命周期错误、跨 stream 缺失依赖和硬件故障；若研究硬件 bit flip，再核对 ECC/设备错误报告的覆盖范围，不把数值对照等同于硬件故障诊断。

已产出：校验覆盖矩阵、原始数值与地址记录、可切换样例的 HTML，以及独立参考和反例测试。读取/校验引入同步，本轮只评价正确性，不提供性能比较。

## 8. 证据入口与建议研究顺序

建议按 **问题 1 → 问题 2 → 问题 3** 推进：问题 1、2 已完成可观测层分析，问题 3 已完成单流正确性与软件异常注入；每项均有独立报告。后续可选择给 CPU 元数据预计算加入前置契约，或继续解码设备参数；跨 stream 生命周期与硬件故障需另设实验。

| 证据 | 位置 |
|---|---|
| 原模型与变体源码依据 | [SOURCE_AUDIT.md](../../practice_27_vision_metadata/SOURCE_AUDIT.md) |
| CPU 长度预计算与局部替换 | [metadata.py](../../practice_27_vision_metadata/metadata.py) |
| 逐条同步关联 | [sync_audit.json](../../practice_27_vision_metadata/results/published/sync_audit.json) |
| Sub 的设备节点、flow、CSV 对应 | [native-graph.json.gz](../../practice_27_vision_metadata/results/published/native-graph.json.gz) |
| 交互设备时间线 | [native.html](../../practice_27_vision_metadata/report/native.html) |
| 原始 trace/CSV 解包方法 | [P27 README](../../practice_27_vision_metadata/README.md) |
| PyTorch 主机列表构造源码 | [v2.10.0 tensor_list.cpp](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/csrc/utils/tensor_list.cpp) |
| torch-npu 主机/设备拷贝源码 | [v2.10.0 CopyKernel.cpp](https://github.com/Ascend/pytorch/blob/v2.10.0/torch_npu/csrc/aten/common/CopyKernel.cpp) |

本文使用的原始 trace/CSV 位于归档的 `formal-r01/diagnostic/native/prefill-beach-v256/` 下。trace_index 只在对应 trace 文件内有意义；时间单位和原点必须一致，比较时保留小数精度。
