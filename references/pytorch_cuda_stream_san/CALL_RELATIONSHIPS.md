# CUDA Stream Sanitizer：对象关系与实际调用时序

本文配合 [实现分析](./README.md)，以本目录 [`_sanitizer.py`](./_sanitizer.py) 为依据。图中的实例名、属性名和方法名对应真实代码；PyTorch dispatcher 与 GPU trace 的内部实现位于该文件之外，以外部边界表示。

这些图来自静态源码分析，没有采集真实 GPU 调用轨迹。图中回调箭头表示运行时通知 Python 回调，不代表 GPU kernel 完成通知。

## 1. 对象持有关系：哪些状态长期保留

下图的实线表示属性引用或容器成员，虚线表示创建或注册关系，**不表示执行先后顺序**。

```mermaid
flowchart TD
    CS["cuda_sanitizer: CUDASanitizer<br/>模块级全局实例"]
    DM["dispatch: CUDASanitizerDispatchMode"]
    EH["event_handler: EventHandler<br/>seq_num"]
    SS["syncs: StreamSynchronizations<br/>current_sync_states<br/>recorded_sync_states<br/>host_sync_state"]
    TA["tensors_accessed: _TensorsAccessed"]
    TI["TensorInfo<br/>allocation_stack_trace"]
    WA["write: Access 或 None"]
    RA["reads: list[Access]"]
    AH["argument_handler: ArgumentHandler<br/>一次普通 dispatch 创建一个"]
    GT["gpu_trace 的回调注册表<br/>保存 EventHandler 的绑定方法"]

    CS -->|dispatch| DM
    DM -->|event_handler| EH
    EH -->|syncs| SS
    EH -->|tensors_accessed| TA
    TA -->|accesses 中每个 data_ptr 对应| TI
    TI -->|write| WA
    TI -->|reads| RA
    DM -.->|每次调用时创建局部对象| AH
    DM -.->|register_callback_for_*| GT
    GT -.->|绑定方法引用同一个实例| EH
```

关键区别：

- `EventHandler`、`StreamSynchronizations`、`_TensorsAccessed` 随 dispatch mode 长期存在，跨算子保留状态。
- `ArgumentHandler` 是一次普通 `__torch_dispatch__` 调用的局部对象，只负责本次输入输出；`record_stream` 分支会提前返回，不创建它。
- GPU trace 注册的是这个 `event_handler` 实例的绑定方法，因此同步事件与算子检测修改的是同一套状态。
- `TensorInfo`、`Access` 是数据记录，不会主动执行检测。

源码依据：`EventHandler.__init__` 第 349 行，`CUDASanitizerDispatchMode.__init__` 第 562 行，局部 `ArgumentHandler()` 第 606 行，`CUDASanitizer.__init__` 第 637 行。

## 2. 初始化与启用：构造对象和进入 dispatch mode 是两步

```mermaid
sequenceDiagram
    participant U as 模块导入 / 调用方
    participant CS as cuda_sanitizer: CUDASanitizer
    participant DM as dispatch: CUDASanitizerDispatchMode
    participant EH as event_handler: EventHandler
    participant TA as tensors_accessed: _TensorsAccessed
    participant SS as syncs: StreamSynchronizations
    participant RT as torch._C / gpu_trace

    U->>CS: CUDASanitizer()，模块末尾
    CS->>DM: CUDASanitizerDispatchMode()
    DM->>EH: EventHandler()
    EH->>TA: _TensorsAccessed()
    EH->>SS: StreamSynchronizations()
    SS->>SS: create_stream(DEFAULT_STREAM_ID)
    Note over EH: seq_num = 0
    DM->>RT: torch._C._activate_gpu_trace()
    loop 十类 trace 回调
        DM->>RT: register_callback_for_*(EH 的绑定方法)
    end
    Note over CS: enabled = False

    U->>U: enable_cuda_sanitizer()
    U->>CS: enable()
    CS->>DM: __enter__()，继承自 TorchDispatchMode
    Note over CS: enabled = True

    opt 调用 disable，或满足条件的析构清理
        CS->>DM: __exit__(None, None, None)
        Note over CS: enabled = False
    end
```

构造时就激活 trace 并注册回调；`enable()` 才把 mode 放入 dispatch 上下文。`disable()` 在本文件中只退出 mode，没有对应的 trace 回调注销逻辑。

析构函数仅在 `sys` 尚可用、解释器未进入 finalizing、且 `enabled=True` 时调用 `disable()`。

源码依据：第 562、637、641、645、649、661、672 行。

## 3. 一次普通算子的实际调用顺序

这是跨对象的主要执行路径。`ArgumentHandler` 不调用 `EventHandler`；它保存集合，随后由 dispatch mode 读取并传给 `EventHandler`。

```mermaid
sequenceDiagram
    participant P as PyTorch dispatcher（外部）
    participant DM as dispatch: CUDASanitizerDispatchMode
    participant AH as 本次 argument_handler: ArgumentHandler
    participant ZA as zip_arguments：模块级函数
    participant PT as pytree
    participant OP as func：被拦截的真实算子
    participant CU as torch.cuda
    participant EH as 共享 event_handler: EventHandler

    P->>DM: __torch_dispatch__(func, types, args, kwargs)
    alt func 是 aten.record_stream.default
        DM->>OP: func(*args, **kwargs)
        OP-->>DM: 返回值
        DM-->>P: 直接返回
    else 普通路径
        Note over DM: 根据 func._schema.name 计算 is_factory
        DM->>AH: ArgumentHandler()
        DM->>AH: parse_inputs(schema, args, kwargs, is_factory)
        AH->>ZA: zip_arguments(schema, args, kwargs)
        ZA-->>AH: 迭代得到 argument 与 value
        loop 每个传入参数
            AH->>PT: tree_map_(partial(_handle_argument), value)
            PT->>AH: _handle_argument(叶子值, 属性)
            Note over AH: 收集 CUDA Tensor 地址、读写分类、参数名
        end
        AH-->>DM: 完成输入解析

        DM->>OP: func(*args, **kwargs)
        Note over OP,EH: 算子执行期间也可能触发已注册的内存或同步回调
        OP-->>DM: outputs

        DM->>AH: parse_outputs(schema, outputs, is_factory)
        AH->>PT: tree_map_(partial(_handle_argument), value)
        PT->>AH: _handle_argument(输出叶子值, 属性)
        AH-->>DM: 完成输出解析
        DM->>CU: current_stream()
        CU-->>DM: stream 对象，读取 cuda_stream 属性
        Note over DM: 读取 AH 的集合；read_only = read - written
        DM->>EH: _handle_kernel_launch(stream, read_only, written, outputs, schema, aliases)
        EH-->>DM: errors 列表
        alt errors 非空
            Note over DM: print(error, file=sys.stderr)，调用错误对象的 __str__
            DM-->>P: raise CUDASanitizerErrors(errors)
        else 没有检测到冲突
            DM-->>P: return outputs
        end
    end
```

`zip_arguments` 是模块级生成器函数，由 `ArgumentHandler` 迭代以获取 schema 参数与实参的对应关系。`_handle_argument` 是经 `pytree.tree_map_` 调用的绑定方法回调，不是直接对整个参数对象只调用一次。

`_handle_kernel_launch` 由 Python dispatch mode 显式调用。此文件没有注册“每个硬件 kernel launch 都调用这个函数”的 trace 回调。检查发生在真实算子返回之后，也不要求 GPU 工作已经完成。

源码依据：`parse_inputs` 第 518 行、`parse_outputs` 第 543 行、`__torch_dispatch__` 第 596 行。

## 4. EventHandler 内部如何调用状态对象完成检测

为避免把一张图展开得过宽，下图将第 363 行的局部函数 `check_conflict` 单独画成执行参与者。它是 `_handle_kernel_launch` 内的闭包，**不是独立对象或类**。

```mermaid
sequenceDiagram
    participant DM as dispatch mode
    participant EH as event_handler: EventHandler
    participant SS as syncs: StreamSynchronizations
    participant TA as tensors_accessed: _TensorsAccessed
    participant CK as check_conflict：局部闭包

    DM->>EH: _handle_kernel_launch(...)
    Note over EH: seq_num += 1；创建空 error_list
    EH->>SS: update_seq_num(stream, seq_num)
    Note over EH: 提取一次 stack_trace

    loop 每个 read_only 地址
        EH->>TA: ensure_tensor_exists(data_ptr)
        Note over EH: 构造 current_access = Access(READ, ...)
        EH->>TA: get_write(data_ptr)
        TA-->>EH: previous_write 或 None
        EH->>CK: check_conflict(data_ptr, current_access, previous_write)
        Note over CK: 按下方公共检查逻辑处理
        CK-->>EH: 检查完成；错误写入共享 error_list
        EH->>TA: add_read(data_ptr, current_access)
    end

    loop 每个 written 地址
        EH->>TA: ensure_tensor_exists(data_ptr)
        Note over EH: 构造 current_access = Access(WRITE, ...)
        EH->>TA: were_there_reads_since_last_write(data_ptr)
        TA-->>EH: 是否存在读历史
        alt 存在读历史
            EH->>TA: get_reads(data_ptr)
            TA-->>EH: reads
            loop 每条 previous_read
                EH->>CK: check_conflict(data_ptr, current_access, previous_read)
                CK-->>EH: 检查完成
            end
        else 没有读历史
            EH->>TA: get_write(data_ptr)
            TA-->>EH: previous_write 或 None
            EH->>CK: check_conflict(data_ptr, current_access, previous_write)
            CK-->>EH: 检查完成
        end
        EH->>TA: set_write(data_ptr, current_access)
        Note over TA: 替换 write，并清空 reads
    end

    EH-->>DM: error_list
```

上图每次 `check_conflict(...)` 调用，都在调用位置执行以下逻辑后返回：

```mermaid
sequenceDiagram
    participant EH as event_handler: EventHandler
    participant CK as check_conflict：局部闭包
    participant SS as syncs: StreamSynchronizations
    participant TA as tensors_accessed: _TensorsAccessed
    participant ER as UnsynchronizedAccessError

    EH->>CK: check_conflict(data_ptr, current_access, previous_access)
    alt previous_access 为 None
        Note over CK: 直接 return
    else 存在历史访问
        CK->>SS: is_ordered_after(current.stream, previous.seq_num, previous.stream)
        SS-->>CK: 是否已建立顺序
        opt 未建立顺序
            CK->>TA: get_allocation_stack_trace(data_ptr)
            TA-->>CK: 分配栈或 None
            CK->>ER: 构造错误(data_ptr, 分配栈, current, previous)
            Note over CK: error_list.append(error)
        end
    end
    CK-->>EH: 返回；必要时已向 error_list 追加错误
```

调用方向是 `EventHandler → StreamSynchronizations` 查询先后关系，以及 `EventHandler → _TensorsAccessed` 查询或更新历史。两种状态对象不会相互调用。

`check_conflict` 发现错误时只向列表追加对象，不当场抛出。`EventHandler` 仍会按代码更新本次访问历史，最终由 dispatch mode 打印并抛出包装异常。

源码依据：第 354 至 426 行；`_TensorsAccessed.set_write` 第 223 行；`is_ordered_after` 第 333 行。

## 5. CUDA 同步与内存回调：如何进入同一个 EventHandler

这些回调绕过 `ArgumentHandler`，直接更新共享 `EventHandler` 内的状态对象。

```mermaid
flowchart LR
    RT["PyTorch 底层 trace 触发点<br/>外部实现"]
    GT["gpu_trace 回调注册表"]
    RT -->|调用已注册回调| GT

    subgraph EH["同一个 event_handler: EventHandler"]
        ER["_handle_event_record(event, stream)"]
        EW["_handle_event_wait(event, stream)"]
        ES["_handle_event_synchronization(event)"]
        ST["_handle_stream_synchronization(stream)"]
        DS["_handle_device_synchronization()"]
        MA["_handle_memory_allocation(data_ptr)"]
        MF["_handle_memory_deallocation(data_ptr)"]
    end

    subgraph SS["syncs: StreamSynchronizations"]
        RS["record_state(event, stream)"]
        WE["stream_wait_for_event(stream, event)"]
        AE["all_streams_wait_for_event(event)"]
        AS["all_streams_wait_for_stream(stream)"]
        SA["sync_all_streams()"]
        MG["_state_wait_for_other(state, other)"]
    end

    TA["tensors_accessed: _TensorsAccessed"]

    GT --> ER
    GT --> EW
    GT --> ES
    GT --> ST
    GT --> DS
    GT --> MA
    GT --> MF
    ER --> RS
    EW --> WE
    ES --> AE
    ST --> AS
    DS --> SA
    WE --> MG
    AE -->|每个已有 stream| WE
    AE -->|更新 host 状态| MG
    AS --> MG
    SA --> MG
    MA -->|ensure_tensor_does_not_exist；采集栈后 create_tensor| TA
    MF -->|ensure_tensor_exists；delete_tensor| TA
```

event record 保存状态副本；其他同步路径通过 `_state_wait_for_other` 合并进度。图中省略了防御性的 `_ensure_*` 调用。下表补全全部十类注册关系：

| `gpu_trace` 注册函数后缀 | EventHandler 绑定方法 | 直接调用的状态方法 |
|---|---|---|
| `event_creation` | `_handle_event_creation(event)` | `syncs.create_event(event)` |
| `event_deletion` | `_handle_event_deletion(event)` | `syncs.delete_event(event)` |
| `event_record` | `_handle_event_record(event, stream)` | `syncs.record_state(event, stream)` |
| `event_wait` | `_handle_event_wait(event, stream)` | `syncs.stream_wait_for_event(stream, event)` |
| `memory_allocation` | `_handle_memory_allocation(data_ptr)` | `tensors_accessed.ensure_tensor_does_not_exist`，然后 `create_tensor` |
| `memory_deallocation` | `_handle_memory_deallocation(data_ptr)` | `tensors_accessed.ensure_tensor_exists`，然后 `delete_tensor` |
| `stream_creation` | `_handle_stream_creation(stream)` | `syncs.create_stream(stream)` |
| `device_synchronization` | `_handle_device_synchronization()` | `syncs.sync_all_streams()` |
| `stream_synchronization` | `_handle_stream_synchronization(stream)` | `syncs.all_streams_wait_for_stream(stream)` |
| `event_synchronization` | `_handle_event_synchronization(event)` | `syncs.all_streams_wait_for_event(event)` |

注册函数的完整前缀为 `register_callback_for_`。特别注意 event wait 回调收到的参数是 `(event, stream)`，转调状态方法时顺序变为 `(stream, event)`。

源码依据：绑定方法第 428 至 467 行；回调注册第 565 至 594 行；状态方法第 266 至 338 行。

## 6. 两条调用链在哪里汇合

```text
算子路径：
dispatcher
  → dispatch.__torch_dispatch__
  → dispatch 先调用 ArgumentHandler 解析，再执行 func，再解析输出
  → dispatch 调用 event_handler._handle_kernel_launch
  → event_handler 查询 syncs，并查询 / 更新 tensors_accessed

同步路径：
底层 trace 触发点
  → gpu_trace 中已经注册的绑定方法
  → 同一个 event_handler._handle_event_* / _handle_*_synchronization
  → 同一个 syncs 更新同步进度
```

`CUDASanitizerDispatchMode` 负责接入和调度，`ArgumentHandler` 负责本次访问分类，`EventHandler` 负责把访问历史与同步状态结合起来判断。实际同步进度由 `StreamSynchronizations` 保存，实际资源历史由 `_TensorsAccessed` 保存。
