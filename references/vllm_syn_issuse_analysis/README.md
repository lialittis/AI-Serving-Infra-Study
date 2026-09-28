# vLLM KV Offload 跨 Stream 同步问题分析

整理日期：2026-09-25。

本文以 vLLM 的 [Issue #45704：SimpleCPUOffloadConnector corrupts restored KV under load](https://github.com/vllm-project/vllm/issues/45704) 为主要案例，并结合后续的 [Issue #47282](https://github.com/vllm-project/vllm/issues/47282)，分析 KV Cache 异步迁移中的跨 stream 同步、内存版本和资源生命周期问题。

这类问题与普通的“两个计算算子位于不同 stream，访问同一 Tensor”直接相关，但其依赖链更长：竞争双方可能分别是计算 kernel 和 DMA，错误会跨越 GPU、CPU cache、后台线程和多个请求，最终在未来的一次 prefix cache 命中中才表现为乱码。

## 1. #45704 的核心竞争

SimpleCPUOffloadConnector 将已经计算的 KV block 从 GPU 异步保存到 pinned CPU memory。vLLM 的 [KV offloading 文档](https://github.com/vllm-project/vllm/blob/main/docs/features/kv_offloading_usage.md) 说明，这类传输通过 DMA 与模型计算重叠执行。

#45704 中的关键执行关系可以简化为：

```text
compute stream                         store/offload stream

Attention kernel
写 GPU KV block
       │
       │ 缺少 happens-before
       ▼
                                       D2H DMA 读取 GPU KV block
                                       写入 CPU cache
```

这是标准的跨 stream RAW（Read After Write）依赖缺失：

```text
W_compute(GPU_KV)  →  R_store(GPU_KV)
```

GPU→CPU 拷贝运行在独立的低优先级 CUDA stream 上。如果 store stream 没有等待产生 KV 的 compute stream，DMA 就可能在 Attention kernel 完成写入前读取该区域。CPU cache 中保存的将是旧数据、部分新数据，或者二者的混合。

错误通常不会在 store 时立即可见。之后另一个请求命中 prefix cache，将这份损坏的 KV 恢复到 GPU，Attention 才消费错误数据，最终表现为偶发乱码或混杂字符。因此，从输出异常回溯到原始竞争，中间可能隔着一次完整的 store/load 周期和不同请求。

Issue 中还指出，Host 侧对 token 或 block 的确认状态，只能说明调度器认为相应工作已经提交或逻辑上完成，不能充当设备同步原语。Host 状态并不能证明：

```text
产生 KV 的 GPU kernel 已经执行完成；
该写入已经对另一个 CUDA stream 上的 DMA 可见。
```

同样，两个 Host 调用具有先后关系，也不自动建立两个设备 stream 之间的 happens-before。这里仍然需要明确的 event/wait、stream wait 或具有等价语义的运行时依赖。

## 2. 为什么它不只是普通的跨算子同步

| 普通跨 stream 算子依赖 | KV offload 问题 |
|---|---|
| 算子 A 写 Tensor，算子 B 读 Tensor | Attention 写 KV，DMA 读取或覆盖 KV |
| 通常发生在一次 forward 内 | 可以跨请求和 cache store/load 周期 |
| 两端通常都是设备计算 kernel | 一端可能是 kernel，另一端是 DMA |
| 错误往往在当前计算中出现 | 错误可能在未来 prefix 命中时出现 |
| 主要观察 stream、event 和 Tensor | 还要观察 block 生命周期、版本和发布时机 |

因此，该问题属于 cross-stream synchronization，但它同时也是异步缓存协议和资源版本管理问题。只给图中的计算 kernel 标注 stream，无法完整表达故障原因。

## 3. 对称的 load 侧竞争

后续 [Issue #47282](https://github.com/vllm-project/vllm/issues/47282) 报告了另一个对称问题：修复 store 方向后，load stream 仍可能在 compute stream 使用 GPU KV 时覆盖同一块内存。

```text
compute stream                         load stream

Attention kernel                      H2D DMA
读取 GPU KV block       ╳              覆盖 GPU KV block
```

这是跨 stream WAR（Write After Read）依赖缺失：

```text
R_compute(GPU_KV)  →  W_load(GPU_KV)
```

这说明 #45704 并非孤立地漏掉一次 `wait_stream`。KV offload 需要一个双向协议：

- store 前要等待最后一次 GPU 写入；
- load 覆盖 GPU slot 前要等待旧内容的最后一次读取；
- 恢复后的计算要等待 H2D 完成；
- block 在异步传输完成前不能被回收、重新分配或发布为另一份有效内容。

只修复其中一条边，不能证明 connector 整体正确。

## 4. 三层正确性契约

一个 KV block 的异步迁移至少要同时满足三层条件。

### 4.1 调度和逻辑状态

调度器需要知道：

- block 当前属于哪个请求或 prefix；
- block 是否被 pin、是否允许回收；
- store/load 操作是否已经提交；
- 哪个消费者将使用恢复后的 block。

这类状态用于表达所有权和调度意图，但本身不产生设备同步。

### 4.2 设备 happens-before

compute stream、D2H stream 和 H2D stream 之间必须存在正确的 event/wait 关系：

```text
producer write → D2H read
last compute read → H2D overwrite
H2D completion → restored-KV consumer read
```

这些边负责保证同一物理内存上的冲突访问具有顺序。

### 4.3 生命周期、版本和发布

即使所有设备访问已经排序，仍要保证访问的是正确一代逻辑资源：

- DMA 完成前不能回收源和目标 block；
- CPU 副本完成前不能发布为有效 cache entry；
- 同一 GPU slot 被复用时，要区分不同 generation；
- 后台任务携带的 block 映射必须与实际执行时的资源版本一致。

因此，`block_id` 或物理地址相同，并不表示两次访问属于同一份逻辑 KV。

## 5. 建议的完整依赖链

store 方向可以表示为：

```text
ComputeWrite(KV block, generation=N)
    → record(write_done)
    → store_stream.wait(write_done)
    → D2H(KV N)
    → store_done
    → Host 发布 CPU cache entry N 为 VALID
```

load 方向需要同时约束旧读者和新消费者：

```text
ComputeLastRead(GPU slot, old generation)
    → record(read_done)
    → load_stream.wait(read_done)
    → H2D(CPU entry N → GPU slot)
    → record(load_done)
    → compute_stream.wait(load_done)
    → AttentionRead(GPU slot, generation=N)
```

保守实现可以让传输等待整个 compute stream 到达某个 event。更细粒度的实现可以为 block、block group 或内存范围维护 readiness event，从而保留更多计算与传输重叠。后者需要更准确地追踪一个 block 的最后生产者、最后读者和 generation。

## 6. 对 kernel execution graph 的扩展

这类案例说明，kernel execution graph 不能只包含设备计算 kernel。为了表达真实的正确性约束，图中还应包含：

- D2H 和 H2D DMA；
- event record、event wait 和 stream wait；
- CPU cache entry 的有效状态发布；
- block 的 pin、release 和 reassign；
- 后台线程提交操作；
- 请求、prefix 和 block generation 的关联。

一个跨请求的因果链可能是：

```text
Compute Kernel
    ↓ stream/event edge
D2H DMA
    ↓ completion/publication edge
CPU KV Cache Entry
    ↓ later request / version edge
H2D DMA
    ↓ stream/event edge
Consumer Compute Kernel
```

图中的边至少需要区分：

| 边类型 | 表达的含义 |
|---|---|
| `stream_order` | 同一 stream 内的提交和执行顺序 |
| `event_wait` | 不同 stream 之间的设备 happens-before |
| `data_dependency` | 生产者与消费者之间的读写关系 |
| `completion_publish` | 异步任务完成后，Host 才能发布有效状态 |
| `lifetime` | 访问完成前资源不能释放或复用 |
| `version` | 消费者必须读取预期 generation 的逻辑内容 |

被访问资源的标识也应从单纯的 `data_ptr` 或 `block_id` 扩展为类似：

```text
(device, physical_block, byte_range, generation, access_mode)
```

其中 `access_mode` 至少区分 read、write 和 overwrite。对于 batched copy，还需要将一次 DMA 拆解或摘要为多个实际访问的 block/range。

## 7. 对动态检测工具的启示

PyTorch CUDA Stream Sanitizer 一类工具主要根据已观察到的 Tensor 访问和 stream happens-before 检查冲突。#45704 说明，面向 serving 系统的检测还需要覆盖以下信息：

- 底层 driver/runtime DMA，例如 batched async copy；
- 后台线程提交的异步任务；
- pinned host memory；
- 大块分配内部的 KV block 子区域；
- block 的逻辑 generation 和复用；
- Host cache 元数据的发布时机。

如果检测器只拦截 PyTorch 算子，它可能看不到低层 DMA；如果只按物理地址检查，它又可能漏掉地址相同但 generation 错误的 stale access。反过来，没有检测到数据竞争，也不能证明请求读取了正确版本的 KV。

因此可以将检测目标表述为：

> 一次异步访问既要与冲突访问具有正确的 happens-before，也要访问调用方预期的资源版本。

这与 [PyTorch CUDA Stream Sanitizer 研究笔记](../pytorch_cuda_stream_san/RESEARCH_DIRECTIONS.md) 中“从物理地址竞争扩展到逻辑资源生命周期”的方向一致。

## 8. PyTorch CUDA Stream Sanitizer 是否是直接解决方案

结论是：现有 PyTorch CUDA Stream Sanitizer（CSAN）不是 #45704 的直接修复方案，但它提供了适合这类问题的动态检测算法骨架。

这里需要区分两种“解决”：

- **运行时正确性修复**：connector 必须建立正确的 stream/event 依赖、DMA 访问顺序和 block 生命周期协议。CSAN 不会替应用自动插入这些依赖。
- **发现和验证问题**：检测器观察资源访问与同步，报告缺失的 happens-before。CSAN 的核心算法适合承担这一职责，但现有接入范围不足以完整覆盖 vLLM KV offload。

### 8.1 现有 CSAN 已经具备的算法能力

CSAN 将运行行为抽象为：

```text
Access(stream, resource, READ / WRITE)
Record(event, stream)
Wait(stream, event)
Synchronize(stream / event / device)
Allocate(resource)
Free(resource)
```

它记录每个资源最近的读写访问，并通过 stream、event 和 Host 同步维护 happens-before。如果把 #45704 的关键行为完整地转换为事件：

```text
Write(compute_stream, GPU_KV_block)
Read(store_stream, GPU_KV_block)       # D2H
```

两次访问作用于同一资源、至少一次为写，且 store stream 没有继承 compute stream 的进度，状态机就能够报告 RAW race。

load 侧问题同理：

```text
Read(compute_stream, GPU_KV_block)
Write(load_stream, GPU_KV_block)       # H2D overwrite
```

缺少 `compute → load` 顺序时，可以报告 WAR race。因此，CSAN 的 happens-before 判定算法能够表达这两类错误。

### 8.2 当前实现的观测边界

根据本仓库保存的 [CSAN 实现分析](../pytorch_cuda_stream_san/README.md)，当前实现主要依赖两路信息：

- `TorchDispatchMode` 根据 PyTorch 算子 schema、输入和输出推导 CUDA Tensor 的读写；
- `torch.cuda._gpu_trace` 接收 stream、event、同步和内存分配/释放等回调。

它并不分析 kernel 指令，也没有在当前代码中为每个底层 DMA 自动生成内存访问事件。对 #45704 而言，各项覆盖情况大致如下：

| #45704 所需信息 | 当前 CSAN 的覆盖情况 |
|---|---|
| compute stream 上的 PyTorch Tensor 读写 | 通常可以观察 |
| CUDA stream/event 同步 | 可以观察 |
| `cuMemcpyBatchAsync` 等底层 DMA 的源和目标访问 | 若绕过 PyTorch dispatcher，很可能无法观察 |
| pinned CPU memory 的读写 | 不按 CUDA Tensor 资源追踪 |
| KV 大 Tensor 内部的 block/range | 仅按 `data_ptr()` 追踪，缺少区间语义 |
| block generation、owner 和 prefix 身份 | 不支持 |
| CPU cache entry 的有效状态发布 | 不支持 |
| 自动添加缺失的 `wait_stream` | 不支持 |

如果底层 D2H 路径绕过 dispatcher，CSAN 可能看到了 stream 和 event，却没有得到下面这条资源访问：

```text
Read(store_stream, GPU_KV_block)
```

没有 D2H read 事件，就无法把它与 compute write 配对并报告冲突。对于 H2D，也可能缺少对 GPU 目标 block 的 write 事件。

另外，CSAN 当前按 Tensor 的起始 `data_ptr()` 建立访问历史，没有完整描述地址范围、stride 和 storage alias。KV cache 通常是在一块大分配中管理多个 block；只比较 Tensor 起始地址既可能遗漏部分重叠，也可能因粒度过粗而产生误报。

### 8.3 哪些情况下现有 CSAN 可能直接检出

如果生产和拷贝都通过 CSAN 可见的 ATen 算子执行，并且两端能够归并到同一个被追踪地址，例如一个简化实验：

```python
with torch.cuda.stream(compute_stream):
    kv.copy_(new_value)

with torch.cuda.stream(store_stream):
    cpu_buf.copy_(kv, non_blocking=True)
```

CSAN 有机会观察 compute stream 对 `kv` 的写和 store stream 对 `kv` 的读，并报告缺少同步。但这只说明算法和受支持的 PyTorch 路径有效，不能据此认为 vLLM 中的底层 batched copy、后台线程和 KV block 子分配已经被覆盖。

CSAN 在真实算子调用返回后执行检查，因此它主要提供诊断；发现竞争时，相关 CUDA 工作可能已经提交。它不是阻止错误访问发生的隔离机制。

### 8.4 将其扩展为 serving 检测方案

要覆盖 #45704 这一类真实路径，需要在 CSAN 的状态机之前增加 serving/runtime 接入层：

```text
cuMemcpyBatchAsync 或其他 async copy
    ↓ 解析每组源地址、目标地址和长度
D2H: Read(GPU range) + Write(CPU range)
H2D: Read(CPU range) + Write(GPU range)
    ↓
提交给 happens-before 与访问冲突状态机
```

还需要补充以下能力：

1. **地址区间追踪**：以 `[address, address + length)` 判断实际重叠。
2. **KV block 子分配映射**：把大 buffer 内的物理区间映射到 block、request 和 prefix。
3. **generation 与所有权**：区分同一地址或 block ID 的不同逻辑生命周期。
4. **Host 发布事件**：只有 DMA 完成后，CPU cache entry 才能变为 `VALID`。
5. **生命周期检查**：异步传输或消费尚未结束时，禁止回收和重新分配相关 block。
6. **底层执行覆盖**：覆盖自定义 kernel、driver/runtime copy、后台线程以及 graph replay。

扩展后的检测事件可以使用：

```text
Access(queue, physical_range, generation, READ / WRITE)
Record(event, queue)
Wait(queue, event)
Publish(logical_resource, generation)
Acquire / Release(logical_resource, generation)
```

由此可以形成如下定位：

```text
现有 PyTorch CSAN
    = 普通 PyTorch 跨 stream Tensor race 检测器

扩展后的 serving CSAN
    = kernel + DMA + KV block/version + 生命周期检测器
```

对于 #45704，真正的修复仍然是建立 producer→store 的同步关系，并采用正确的拷贝访问顺序。扩展后的 sanitizer 用于发现缺边、解释因果链、验证修复以及防止回归。它可以与 kernel execution graph 共享同一套 `Access / Record / Wait / Lifetime / Version` 事件模型。

## 9. 建议的验证实验

仅通过最终文本是否乱码来测试，会导致故障定位困难。可设计以下分层实验：

1. **Store RAW 实验**：延迟 KV producer 或提前 D2H，验证缺少 producer→store edge 时能稳定复现错误。
2. **Load WAR 实验**：延长 Attention 对旧 GPU block 的读取，同时发起 H2D overwrite。
3. **Load→consume 实验**：让计算在 H2D 完成前读取目标 block，验证恢复完成边。
4. **生命周期实验**：在 DMA in-flight 时尝试回收和重新分配同一物理 block。
5. **版本实验**：复用相同 `block_id` 和地址，但为其分配新的 generation，检查 stale 后台任务。
6. **数据完整性实验**：在 D2H 后校验 CPU 副本，在 H2D 后校验 GPU 副本，从而定位首次损坏的位置。
7. **图断言实验**：直接检查 trace 中是否存在必要的 happens-before 和 publication edge，而不只检查最终输出。

每类实验都应包含缺失同步的负例和正确同步的对照组。高并发、prefix reuse、copy 延迟扰动和 block 复用可以提高竞争窗口的可观测性。

## 10. 结论

#45704 是典型的跨 stream RAW 缺边，但故障发生在计算与 DMA 之间，并跨越了设备内存、CPU cache 和未来请求。#47282 展示了对称的 load 侧 WAR 缺边，进一步说明 KV offload 的正确性需要一个完整的双向内存迁移协议。

研究 kernel execution graph 时，应同时表达设备执行顺序、数据访问、Host 发布和资源版本。只有 stream 信息而没有读写范围、生命周期和 generation，无法解释这类 serving 系统中的异步错误。

## 参考资料

- [vLLM Issue #45704：SimpleCPUOffloadConnector corrupts restored KV under load](https://github.com/vllm-project/vllm/issues/45704)
- [vLLM Issue #47282：SimpleCPUOffloadConnector load-side cross-stream race](https://github.com/vllm-project/vllm/issues/47282)
- [vLLM KV Offloading 文档](https://github.com/vllm-project/vllm/blob/main/docs/features/kv_offloading_usage.md)
- [本仓库：PyTorch CUDA Stream Sanitizer 研究方向](../pytorch_cuda_stream_san/RESEARCH_DIRECTIONS.md)
