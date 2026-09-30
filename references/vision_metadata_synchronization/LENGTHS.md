# 笔记：视觉 attention 的 lengths、Sub 计算与后续研究问题

记录日期：2026-09-30。本文整理 Practice 27 中已经确认的概念和证据，并保留三个待研究方向；没有新增远端实验。同步操作的整体分布见[同步分析笔记](README.md)。

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

**问题：** `.tolist()` 触发的同步具体在哪一层？Python、PyTorch、torch-npu 和 CANN 之间怎样连接到 D2H？

已确认的基础：PyTorch `tensor_to_list` 对非 CPU tensor 转到 CPU，然后生成 Python 对象；本次 trace 可见 `aten::to → aten::_to_copy → aten::copy_`，copy 区间内先同步、后 `aclrtMemcpy`。公开 torch-npu 2.10 源码的普通阻塞 D2H 路径先等待当前 stream，再调用拷贝封装。详见[同步笔记第 8 节](README.md#8-深入-tolist内存拷贝与主机等待)。

- [ ] 对照实际安装版本，补全从 `Tensor.tolist` Python 绑定、`tensor_to_list`、ATen dispatcher 到 NPU copy 实现的函数/文件/行号；记录源码与二进制版本。
- [ ] 跟踪 `toBackend(CPU)` 的参数传递，确认 `non_blocking`、目标分配器和 D2H 方向在哪一层确定。
- [ ] 核对同步封装、主机任务队列和 CANN runtime 的关系，确定等待的是哪条 stream、哪些已提交任务；区分 Python 提交线程与运行时 worker。
- [ ] 记录源/目标缓冲区、大小、是否使用 pinned host memory、是否有中间缓冲；若底层不可见，明确标为未知，不由 API 名称推断硬件 DMA 路径。
- [ ] 分别测量同步等待、memcpy API、Python list 构造；验证重复三次 `.tolist()` 与只做一次再复用的区别。

预期产物：带源码位置的完整调用链，以及同一次调用中同步、D2H 和主机消费的时间线。不要将公开同版本源码等同于已验证的安装二进制内部执行轨迹。

## 6. 待研究问题 2：Sub 是否阻塞 CPU，怎样编译、提交及寻址

**问题：** Sub 是否完全阻塞 CPU？从 host 到 device 怎样完成编译和传递？输入输出地址是什么，kernel 如何被调用？

已有实例证明主机 Sub 返回早于设备执行，但“阻塞 CPU”需要具体到线程和阶段：调用线程可能进行参数检查、内存分配或提交；运行时 worker 可能处理队列；设备可能尚在执行之前的任务。这几种状态不能合称为“CPU 被完全阻塞”。

- [ ] 对比仅 Sub、Sub 后立即 `.tolist()`、Sub 后显式同步三组，分别记录主机调用返回、设备 kernel 起止和等待区间；比较首次与预热后的调用。
- [ ] 跟踪 `aten::sub → aclnnSub → runtime launch → 设备任务`，说明参数检查、输出分配、算子选择、tiling、workspace、入队与实际 launch 分别在哪发生。
- [ ] 确认该路径使用已有 kernel 二进制、首次加载/缓存，还是涉及即时编译；若有编译，记录发生位置和产物。不要预设“每次 Python 减法都会在设备上编译”。
- [ ] 采集父 tensor、两个 slice、Sub 输出的 `data_ptr()`、storage 基址、`storage_offset()`、shape、stride、dtype、有效字节范围、分配和释放时间。
- [ ] 对连续 int32 一维父 tensor，验证 `[1:]` 的逻辑起点是否较父 tensor 前移 4 字节、`[:-1]` 是否共享基址。区分正常切片偏移与错误地址；地址按同一次运行解释，不把虚拟地址当作物理地址。
- [ ] 对照 kernel 实际参数或可获得的运行时记录，确认是否直接使用切片地址、是否插入格式转换/连续化临时 tensor，以及输出、workspace 的存储关系。
- [ ] 将线程、队列、stream、task ID 和内存生命周期关联起来，区分算子 launch 的输入输出地址与 Python 对象身份。

预期产物：一次 Sub 的主机—设备时序图和输入/输出内存表，分别标明计算、排队、等待、地址别名及编译/加载证据。当前 P27 的完整长度数组指针和 kernel 参数尚未据此恢复。

## 7. 待研究问题 3：偏移、溢出、翻转与校验

**问题：** Sub 输出是否可能出现偏移、溢出、翻转等异常？已有检查能覆盖什么，还需补什么？

先明确“翻转”的含义：可能指减法顺序反了导致符号反转、元素顺序颠倒，也可能指内存 bit flip；这些需要不同实验。“偏移”也应区分合法 slice 的 storage offset 与错误读写位置。目前没有发现这些异常，但现有结果不足以证明所有输入、所有访存都不会出错。

P27 已做源码哈希/实现契约检查、grid 与像素形状匹配检查，以及 features、logits、KV 等输出对照；含预热/资格的 816 次输出比较全部逐元素一致，8 个场景另与原生完整 forward/generate 对照通过。**这些属于已测场景的集成正确性证据，不是每次 Sub 输入输出的独立审计，也不是溢出或 bit flip 检测器。** 详见 [P27 结果](../../practice_27_vision_metadata/RESULTS.md) 与 [metadata.py](../../practice_27_vision_metadata/metadata.py)。

- [ ] 保存 CPU 原始 grid/累计边界，以 Python 整数或足够宽的整数建立独立参考；转换为 int32 **之前**检查范围，避免参考值与被测值先发生同样的截断或溢出。
- [ ] 校验去重后的累计边界从 0 开始、严格递增，末项等于实际 attention 序列长度；校验差分元素为正、元素数比边界数少 1、逐元素等于参考且总和等于序列长度。
- [ ] 对“非负、单调且最大累计边界不超过 int32 最大值”的前提做显式检查。此时相邻差值也在非负 int32 范围内；重点追踪累计求和、乘法和 dtype 转换是否在此前就越界。
- [ ] 验证切片 stride、storage offset、可访问范围、输入是否被改写、输出是否发生非预期别名或提前复用；设计非零 offset、边界窗口、重复边界等最小样例。
- [ ] 分别注入元素颠倒、操作数互换、单个长度错误、超范围累计边界及单 bit 修改，确认各校验能捕获什么。测试注入应使用隔离的小 tensor，不改模型权重或正式结果。
- [ ] 审计当前版本 `torch.split`、ATen 和 aclnnSub 的实际参数校验与报错行为；特别验证“长度总和正确但边界错位”的情况，不能仅靠总和检查判断分段正确。
- [ ] 区分算术错误、地址/生命周期错误、跨 stream 缺失依赖和硬件故障；若研究硬件 bit flip，再核对 ECC/设备错误报告的覆盖范围，不把数值对照等同于硬件故障诊断。

预期产物：校验覆盖矩阵，列出异常类型、触发输入、检查位置、是否检出和残余盲区。新增读取/校验可能引入同步，因此正确性诊断与性能采集应分开执行。

## 8. 证据入口与建议研究顺序

建议按 **问题 1 → 问题 2 → 问题 3** 推进：先明确主机等待与 D2H 的链路，再恢复 Sub 的提交和内存关系，最后针对这些关系设计校验。这里仅登记后续问题，未执行新增测试。

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
