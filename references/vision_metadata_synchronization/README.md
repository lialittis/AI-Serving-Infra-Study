# 笔记：视觉 forward 中的同步何时发生，怎样影响多 Stream 重叠

记录日期：2026-09-30。依据 [Practice 27](../../practice_27_vision_metadata/README.md) 的 `formal-r01` 实验及其源码、trace 审计。

关于分段长度的定义、实际 Sub kernel 和后续的同步/D2H、编译提交/地址、结果校验问题，见独立的 [lengths 专题笔记](LENGTHS.md)。

本实验说明：**两个任务没有数据依赖、也已经分配到不同 stream，仍可能因为 CPU 在提交途中等待 NPU，无法产生计算重叠。同步的位置与等待时间，比同步 API 的条数更能解释这种现象。**

本文范围是 Ascend 910B2C 上 Qwen2.5-VL-3B-Instruct 的 HF eager 阶段实验：BF16、单图视觉输入、32 个视觉 block、一个 CPU 提交线程。结论对应本次固定实现和采集路径，不直接代表其他 attention 后端、graph replay 或 vLLM 在线调度。

## 1. 先区分三种“同步”

| 机制 | 本实验中的位置 | 对调度的作用 |
|---|---|---|
| 同一 stream 的执行顺序 | 各 stream 内提交的任务 | 由流内顺序保证先后；不意味着 CPU 每发射一个 kernel 都必须等它完成 |
| 跨 stream 的 event 等待 | B 语言消费 B 视觉特征之前 | 在消费 stream 上建立设备依赖；CPU 可以继续提交后续工作，设备消费必须等特征就绪 |
| 主机等待设备 | 视觉 forward 内隐式触发的同步 API，以及实验末尾的 event synchronize | 调用线程等待返回；若它也是唯一的提交线程，尚未提交的独立任务也会被延迟 |

本文重点分析第三种机制中发生在 `vision_B` 范围内的隐式同步。trace 中可见 `AscendCL@aclrtSynchronizeStream` 和 `AscendCL@aclrtSynchronizeStreamWithTimeout`，上层 Python 不一定显式写了 `synchronize()`。

“视觉同步 API 为 0”只描述这个观测范围。准备阶段同步、跨 stream event 和末尾 host join 仍然存在；已经在其他 stream 上运行的任务，也不会仅因 CPU 等待就必然停止。

## 2. 同步发生在视觉 forward 的三个位置

下面按主机执行/提交源码的顺序排列，不能将“调用了 merger”理解为设备已经完成 merger。

```text
patch embedding
  → 位置、窗口与累计长度元数据准备，输入重排       [6 条]
  → 32 个视觉 block
      其中 28 个窗口 attention：Q/K/V 分段长度读取 [28 × 3 = 84 条]
  → merger
  → 用反向索引恢复输出顺序                      [1 条]
  → 返回视觉特征 tensor
```

### 2.1 开头：元数据准备和输入重排，6 条

本次输入的 `grid_thw` 在 CPU。原实现依据它生成位置索引、窗口索引、累计窗口长度，再用于 NPU tensor 的处理。

trace 中开头有 4 条关联到 `aten::copy_`，另外 2 条关联到 `unique_consecutive` 调用。结合固定源码顺序，前者对应位置索引消费、累计长度构造为设备 tensor，以及 hidden states 和位置 tensor 的窗口索引消费；后者出现在设备累计长度的去重路径。

这些小型元数据操作也可能引入主机等待。这里的证据是本实现的实际调用链，不表示任何 CPU 索引或任何拷贝在所有实现下都会同步。`unique_consecutive` 的关联说明同步发生在该路径中；当前审计没有进一步拆解其内部每个等待的实现原因。

### 2.2 各窗口 attention 内：分段长度读取，84 条

原 attention 的 eager 分支包含：

```python
lengths = cu_seqlens[1:] - cu_seqlens[:-1]
splits = [
    torch.split(tensor, lengths.tolist(), dim=2)
    for tensor in (query_states, key_states, value_states)
]
```

`torch.split` 在这里需要 Python 长度列表。窗口分支的 `cu_seqlens` 在 NPU，`.tolist()` 要把长度值取回 CPU，主机需要等到这些值可用。列表推导式对 Q、K、V 各执行一次 `.tolist()`。

32 个视觉 block 中，28 个采用窗口 attention，所以有 `28 × 3 = 84` 条相关同步 API。其余 4 个整图 attention 在本实验中使用由 CPU grid 生成的 CPU 累计长度，不产生这一组设备长度读取同步。这个计数不能脱离本次 tensor 所在设备和 attention 实现来推广。

这些长度来自图像网格与窗口划分，不依赖本次网络算出的视觉特征。因此 `lengths` 变体可以提前准备 Python tuple，保留分段语义，避免在每层读取设备长度。

### 2.3 末尾：恢复输出顺序，1 条

原视觉 forward 最后执行：

```python
merged_hidden_states = self.merger(hidden_states)
reverse_indices = torch.argsort(window_index)
merged_hidden_states = merged_hidden_states[reverse_indices, :]
```

`window_index` 和这里的反向索引在 CPU，输出特征在 NPU。末尾 trace 显示同步位于以下调用链中：

```text
aten::index → aten::to → aten::_to_copy → aten::copy_ → 同步 API
```

调用链和时间位置是 trace 的直接证据；将这个 index 定位到反向索引消费，是结合固定源码执行顺序得到的判断。不是 CPU 上的 `argsort` 本身在等待视觉计算。

因为这次等待发生在视觉 forward 返回前，先视觉后语言时，唯一的 CPU 提交线程仍可能长时间无法开始提交另一请求的语言计算。等待时间也不能简单解释成“拷贝这几个索引需要这么久”：同步调用可以包含等待已提交设备工作的时间。

## 3. 一个可复核的时间实例

选择 `lengths / prefill-beach-v256 / parallel / 重复 1`，即 VL（先提交 B 视觉，再提交 A 语言）。以下数据直接来自 [sync_audit.json](../../practice_27_vision_metadata/results/published/sync_audit.json)。

时间原点为该次 **`vision_B` 主机 scope 开始**；单位均为毫秒。开始时间不与交互报告的全局设备时间原点混用。

| 序号 | 同步开始 | API 持续时间 | 关联调用/源码位置 | 原 trace_index |
|---|---:|---:|---|---:|
| 1 | 0.506580 | 0.008160 | copy / 位置索引消费 | 667080 |
| 2 | 0.772664 | 0.005055 | copy / 累计长度转设备 tensor | 667086 |
| 3 | 0.923126 | 0.112971 | aclnnUniqueConsecutive | 667096 |
| 4 | 1.082759 | 0.001150 | aten::unique_consecutive 内 WithTimeout 同步 | 667098 |
| 5 | 1.113827 | 0.000488 | copy / hidden states 窗口索引消费 | 667099 |
| 6 | 1.155314 | 0.031231 | copy / 位置 tensor 窗口索引消费 | 667105 |
| 7 | 58.449583 | 21.302954 | index → copy / 输出反向索引消费 | 693364 |

最后一次 API 大约在 `79.752537 ms` 返回。在它返回之前，CPU 还停留在视觉调用内，不能进入接下来的 A 语言提交。因此，**只看“还剩 7 次”会漏掉末尾一次持续约 21.30 ms 的等待。** 本次对应的 A/V compute kernel 重叠为 0。

作为对照，同一场景和提交顺序下，native 的末尾同步约在 `82.378729 ms` 开始、持续 `1.871141 ms`。不能因此认为 lengths 使总性能恶化：前面反复等待被移除后，CPU 与设备的进度差也会变化，末尾等待可能变长。应结合整段提交时序与独立性能测量判断，不能只比较最后一个 API。

这些都是 profiler 诊断时间，不是无 profiler 的正式性能时间。API 记录可能嵌套，条数不等于独立等待次数；本文不累加 duration，也不把 21.30 ms 直接当作可获得的性能收益。

## 4. 为什么需要 native、lengths、cached 三组对照

| 变体 | 提前准备什么 | 视觉范围内同步 API | 8 个场景的 VL 实际计算重叠 |
|---|---|---:|---|
| native | 保留原始视觉路径 | 6 + 84 + 1 = 91 | 均为 0 |
| lengths | attention 的 Python 分段长度 | 6 + 0 + 1 = 7 | 均为 0 |
| cached | 再准备窗口/反向索引、RoPE cos/sin，索引提前驻留 NPU | 0 | 均观察到重叠 |

cached 缓存的是元数据，不是图像特征或 KV cache。每次仍执行 patch embedding、全部 32 个视觉 block、merger 和特征重排。元数据构建/上传/同步被移到 pair 计时前；本轮一次准备成本约 0.568–0.665 ms，持久化设备数据为 139104 或 636272 字节。首次使用时仍需承担这些成本。

lengths 验证“只移除 attention 长度读取是否足够”；cached 验证进一步预计算其他固定元数据后的效果。本轮结果支持前者不足、后者可以解除当前 VL 路径的重叠障碍。cached 同时改动多种元数据路径，尚不能把全部收益单独归因于反向索引。

正式无 profiler 测量中，cached 双流相对 **cached 自身单流** 的 VL pair 耗时下降：prefill 为 7.18%–7.98%，decode 为 5.17%–10.74%。native 与 cached 的耗时差同时包含元数据优化，不能全部算作并发收益。完整对照见 [RESULTS.md](../../practice_27_vision_metadata/RESULTS.md)。

## 5. CPU 提交依赖与任务数据依赖不是一回事

请求 A 的语言计算与请求 B 的视觉编码没有需要相互等待的特征/KV 数据依赖。但本实验通过同一个 CPU 线程顺序调用它们：

```text
VL，native / lengths：
CPU：提交 B 视觉，期间多次等待，末尾仍等待 → 返回 → 开始提交 A 语言
NPU：B 视觉计算                              → A 语言计算

VL，cached（示意，不按时间比例）：
CPU：提交 B 视觉 → 返回 → 提交 A 语言 → 提交 B 特征等待及 B 语言
V 流：B 视觉计算 ───────────────────────→ V_done
L 流：       A 语言计算 ─────→ 等待 V_done → B merge / B 语言
```

移走元数据同步后，视觉 forward 返回 tensor 时，视觉 kernel 可以仍在设备上执行，CPU 有机会继续向 L 流提交 A。实际是否重叠仍需检查设备时间线，不能仅根据 Python 返回或 stream 数量判断。

对于 LV（先提交 A 语言），A 已经在设备上排队或执行，随后视觉中的主机等待不必然阻止两者重叠。这解释了为何相同代码换一种提交顺序会有不同表现；也不意味着所有 LV 场景都有可观收益。

B 语言消费的是 **B 自己的视觉特征**，所以必须保留 `V_done → B_features_ready` event 依赖。它与 A/V 之间因主机阻塞而出现的串行化不同。末尾 A/V/B event 的主机等待用于完整完成与计时。对应实现见 [P25 execute](../../practice_25_multimodal_overlap/model.py)。

## 6. 怎样从记录中复核

源码依据见 [SOURCE_AUDIT.md](../../practice_27_vision_metadata/SOURCE_AUDIT.md)，变体实现见 [metadata.py](../../practice_27_vision_metadata/metadata.py)。采集源码 SHA256 为 `9d6d15040bdb985d9518117ed029f6a318a38abc06091462d96f16cf1d3d7820`；原文位于证据归档的 `formal-r01/sources/installed/transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.py`。归档解包方法见 [P27 README](../../practice_27_vision_metadata/README.md)。

[同步审计程序](../../practice_27_vision_metadata/sync_audit.py) 先按同线程 CPU 算子区间包含关系关联同步；对于 worker 线程，通过精确的 `async_task_queue` enqueue/dequeue flow 回溯到主机算子，不使用最近时间戳猜测。上表第 3 条通过 queue ID `1538866` 关联，其余条目通过同线程区间关联。

在仓库根目录执行以下命令，可以提取上表原始记录，无需解包整个归档：

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path('practice_27_vision_metadata/results/published/sync_audit.json')
audit = json.loads(path.read_text())
for variant in ('native', 'lengths', 'cached'):
    rows = audit[f'{variant}-prefill-beach-v256-sync']['calls']
    selected = [r for r in rows if 'r1-parallel/' in r['scope']]
    print(variant, 'vision sync API count:', len(selected))
    if variant == 'lengths':
        for r in selected:
            print(r['start_from_vision_us'], r['duration_us'],
                  r['host_ancestors'], r['trace_index'], r['evidence'])
PY
```

查看设备计算重叠时，打开 [cached 交互报告](../../practice_27_vision_metadata/report/cached.html)，选择 `prefill-beach-v256 → parallel → 重复 1`。该 trial 的 A/V compute kernel 重叠约 **34.306637 ms**；设备时间线上检查蓝色视觉与绿色 A 语言任务的区间交集。重复 0 对应 LV，重复 1 对应 VL，不是生成 token 的编号。切换 [lengths](../../practice_27_vision_metadata/report/lengths.html) 或 [native](../../practice_27_vision_metadata/report/native.html) 的同一配置，VL 重叠为 0。

交互报告的设备时间线用于确认 kernel 重叠；逐条主机同步位置以 `sync_audit.json` 为入口，再凭 `trace_index` 查对应变体、场景的原始 `trace_view.json`。不同 trace 文件的索引不能混用。图中的依赖边本身不表示时间重叠，现有图也没有完整恢复所有隐式 CPU 因果边或原生 workspace 访存依赖。

## 7. 后续可验证的问题

进一步定位各项贡献，可以分别增加“仅反向索引提前驻留 NPU”“再提前窗口索引”“再预计算位置编码”的对照，检查末尾同步是否消失、A 首个 kernel 是否提前、pair 与单请求延迟如何变化。这些是后续实验建议，本笔记没有新增采集。

若评估实际部署价值，还需要把元数据准备成本与复用次数纳入请求延迟，并在观察到重叠后进一步测量硬件资源竞争。当前证据能证明所记录 compute task 的时间区间相交，不能据此量化算力/带宽占用，或保证每个请求都更快。

## 8. 深入 `.tolist()`：内存、拷贝与主机等待

### 8.1 Tensor 的描述信息与实际数值分开存放

Python 中的 `lengths` 是主机上的 Tensor 对象，其底层描述信息包含 shape、dtype、device、stride 和设备存储指针。但这不代表主机已经拥有它的数值。对本次窗口 attention，长度数组的数值在 NPU 设备内存中。读取 shape 与把所有元素转成 Python 整数，是两件不同的事。

例如累计长度 `[0, 4, 10, 12]` 相减得到 `[4, 6, 2]`（仅为示意，并非本次真实长度）。`lengths = cu_seqlens[1:] - cu_seqlens[:-1]` 的结果仍是 NPU tensor；CPU 可以拿到这个对象，而对应设备计算尚未完成。

原路径包含一次元数据往返：

```text
CPU：grid → 窗口累计长度列表
                 │ H2D，构造设备累计长度 tensor
                 ▼
NPU：累计长度 → unique_consecutive → 每层差分 → lengths 数值
                                                   │ D2H
                                                   ▼
CPU：临时 tensor 的数值缓冲区 → 逐元素转换 → Python list[int]
                                                   │
                                                   ▼
                            用这些整数描述 Q/K/V 的分段
```

累计长度在 visual forward 开头准备，差分与 `.tolist()` 在窗口 attention 中重复执行。这里复制回 CPU 的是长度元数据；Q/K/V 激活仍在 NPU。

### 8.2 `.tolist()` 实际分两步：先得到 CPU tensor，再构造列表

[PyTorch v2.10.0 的 tensor_list.cpp](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/csrc/utils/tensor_list.cpp) 中，`tensor_to_list` 检查设备类型：非 CPU tensor 先通过 `toBackend(Backend::CPU)` 获取 CPU tensor，再由 `recursive_to_list` 按 shape/stride 遍历 CPU 内存、构造 Python 列表和标量对象。这与 [2.10 的 API 文档](https://docs.pytorch.org/docs/2.10/generated/torch.Tensor.tolist.html) 描述一致。

因此有三种不同表示：

| 表示 | 所在位置 | 内容与生命周期 |
|---|---|---|
| 原始 `lengths` tensor 的存储 | NPU 设备内存 | 定长整数数组；调用 `.tolist()` 不会把原 tensor 原地变成 CPU tensor |
| 临时 CPU tensor 的存储 | 主机内存 | 设备数组的主机副本，供 C++ 转换逻辑读取；转换结束后临时引用可释放 |
| 返回的 Python list | 主机 Python 对象内存 | 列表及 Python 整数对象，独立于原设备存储；不再是 int32 tensor 的内存布局 |

这不是直接让 CPU 解引用 NPU 数据指针，也不是每个 Python 整数都单独进行一次 D2H。普通连续数组路径先复制数值缓冲区，再在主机上逐元素构造列表。实际分配器可能复用内存，不能把“临时 tensor”理解成每次必然发生一次操作系统级分配。

### 8.3 本次阻塞拷贝为什么要等 stream

[torch-npu v2.10.0 CopyKernel.cpp](https://github.com/Ascend/pytorch/blob/v2.10.0/torch_npu/csrc/aten/common/CopyKernel.cpp) 的普通连续、相同 dtype 的 D2H 路径，进入 `copy_between_host_and_device`。其阻塞分支先取得当前 NPU stream，调用 `AclrtSynchronizeStreamWithTimeout`，再执行 `AclrtMemcpyWithModeSwitch`，方向为 `ACL_MEMCPY_DEVICE_TO_HOST`。

这意味着调用线程先等当前 stream 的已提交工作完成，随后进行 D2H，最后才能读主机副本并生成列表。等待范围由 stream 顺序决定，不是只针对这几个长度值进行最小依赖等待；它可能包含在前面排队的 Q/K/V 投影等计算。其他 stream 上已经提交的任务不必因此停止。

PyTorch 在转 CPU 的这段代码中释放了 GIL，但当前调用线程仍要等函数返回。本实验只有一个 CPU 提交线程，所以释放 GIL 并不会自动让它转去提交另一请求。

这里使用公开的同版本源码解释机制，并用下述真实 trace 确认“先同步、后 memcpy”的调用顺序；没有声称已对安装的 torch-npu 二进制做逐指令审计。当前 trace 也没有完整记录 host buffer 是否 pinned、运行时内部暂存区或物理传输引擎，不能进一步断言具体 DMA 通道或内部复制次数。

### 8.4 真实 trace：80 字节也会触发等待

取 `native / prefill-beach-v256 / parallel / 重复 1` 的第一组窗口 attention 长度读取，三个 `aten::copy_` 的输入形状都是 `20`、类型为 `int`，结合累计长度的 int32 源码类型，有效数值载荷为 `20 × 4 = 80` 字节。每次读取的主机调用区间内，都有同步 API，随后有 `AscendCL@aclrtMemcpy`。

| 对应读取（按 Q/K/V 源码顺序） | copy_ trace_index | 同步 API 时长 µs | memcpy trace_index | memcpy API 时长 µs |
|---|---:|---:|---:|---:|
| Q 的分段长度 | 116736 | 42.705 | 671940 | 15.409 |
| K 的分段长度 | 116763 | 0.530 | 671942 | 8.197 |
| V 的分段长度 | 116790 | 0.381 | 671944 | 8.005 |

原始文件位于归档 `formal-r01/diagnostic/native/prefill-beach-v256/` 下的 `trace_view.json`。形状、API 顺序和时长直接来自 trace；Q/K/V 对应关系结合源码列表推导式的执行顺序确定。

三次读取的是同一个长度 tensor，代码没有复用第一次得到的 Python 列表。第一段同步较长、随后两段很短，与前一次等待后设备队列已推进的机制一致。表中 memcpy 数字是主机 API 时长，包含运行时开销，不能据此计算 80 字节的纯硬件传输带宽；也不能把同步时长当作 `.tolist()` 全部耗时。

### 8.5 列表返回后，Q/K/V 并不跟着搬到 CPU

CPU 把列表作为 `torch.split` 的分段大小。按照 [PyTorch 2.10 split 语义](https://docs.pytorch.org/docs/2.10/generated/torch.split.html)，各分段是原 tensor 的 view，仍引用 NPU 上的 Q/K/V 存储。后续 attention 继续在 NPU 计算。控制分段所需的少量整数回到 CPU，不等于激活数据回到 CPU。

改成在循环外只做一次 `sizes = lengths.tolist()` 再给 Q/K/V 复用，可以减少重复读取，但每个窗口层仍保留一次主机取值等待；本轮没有单独测试这个变体。P27 的 `lengths` 变体进一步利用长度可由 CPU grid 提前计算这一点，直接复用 CPU tuple，消除这条设备差分→D2H→Python 列表的循环路径。

异步 D2H 也不能让 Python 安全地提前读取尚未完成的目标缓冲区。即使自行安排异步拷贝，只要紧接着必须用实际数值进行 split，消费之前仍要保证拷贝完成。本例更有价值的优化是让这份原本可由 CPU 确定的元数据留在 CPU，并保留网络计算所需的真正设备依赖。
