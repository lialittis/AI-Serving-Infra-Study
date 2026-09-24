# Practice 14：KV池初始化的逐步结果

**KV池不是按每个请求、每个block分别申请设备内存。** 初始化时先为各层建立大块K/V tensor，之后CPU上的vLLM才管理其中的block编号。

本次还确认：实际服务走的是**可扩展内存段**路径，即“已有虚拟地址区域 + 按需申请物理内存 + 映射”。普通torch-npu小探针的分配路径不能直接代替真实vLLM服务的结果。

运行：`2026-09-24-run02`，Ascend 910B2C，Qwen2.5-0.5B-Instruct，单卡BF16 eager。服务正常就绪、正常退出，没有发送业务推理请求。配置与版本见[environment.json](results/2026-09-24-run02/environment.json)及[command.json](results/2026-09-24-run02/command.json)。

## 第一步：启动时建立分配器地址区域

vLLM-Ascend在[platform.py](results/2026-09-24-run02/sources/vllm_ascend/vllm_ascend/platform.py)中默认配置`PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`。我们没有手工打开它；KV分配入口记录了生效后的环境，快照也确认实际segment为可扩展类型。

承载本次KV的地址区域来自一次实际调用：

```text
aclrtReserveMemAddress
返回起始虚拟地址：20698521141248
预留地址范围大小：73651978240 bytes
```

**这个调用发生在KV池分配之前。** 模型加载等启动工作已经使用该分配器区域，KV池继续使用它。预留约68.59GiB虚拟地址，不代表当场占用同样大小的HBM；实际物理内存按需提供。

## 第二步：vLLM计算能给KV多少内存

实际`NPUWorker.determine_available_memory()`结果：

```text
配置的设备内存预算                  19,635,634,176 bytes
减：权重、峰值激活和非torch内存       1,037,270,016 bytes
减：本次应用的graph内存估计                      0 bytes
───────────────────────────────────────────────────
可用于KV的预算                      18,598,364,160 bytes
```

随后vLLM根据模型结构计算block容量：

```text
每层、每个block的K与V：
128 tokens × 2 KV heads × 64维 × 2 bytes × 2份(K和V)
= 65,536 bytes

24层合计：65,536 × 24 = 1,572,864 bytes / block编号

floor(18,598,364,160 / 1,572,864) = 11,824 个block
```

最终tensor所需总字节数为18,597,543,936，预算剩余820,224字节，不足以再增加一个跨24层的block。分析器核对了预算计算及该配置向worker、runner的传递。

源码入口：[worker.py](results/2026-09-24-run02/sources/vllm_ascend/vllm_ascend/worker/worker.py)的`determine_available_memory`与[配置计算](results/2026-09-24-run02/sources/vllm/vllm/v1/core/kv_cache_utils.py)的`get_kv_cache_configs`。

## 第三步：Ascend runner申请48个原始存储

24层，每层K和V各一张tensor，共48张。以第一层K为例：

```python
# 实际调用的核心语句，self.device为npu:0
torch.zeros(numel, dtype=torch.int8, device=self.device)
```

这时用INT8表示原始字节存储，不代表KV最终使用INT8量化。

| 第一层K的记录 | 实际值 |
|---|---:|
| 申请数据字节数 | 387,448,832（369.5MiB） |
| 返回data_ptr / storage_ptr | 20699522482176 |
| allocator block大小 | 387,449,344 |
| 分配器额外空间 | 512 bytes |

源码：[model_runner_v1.py](results/2026-09-24-run02/sources/vllm_ascend/vllm_ascend/worker/model_runner_v1.py)的`_allocate_kv_cache_tensors`和`_allocate_int8_cache_tensor`。

## 第四步：torch-npu补充物理内存并映射

第一层K的这次申请，在已有映射区域中使用了5,292,032字节。为了覆盖剩余存储并按分配器粒度扩展，实际调用了：

```text
19次 aclrtMallocPhysical，每次20MiB
19次 aclrtMapMem，每次映射对应的20MiB
总共新增映射380MiB = 398,458,880 bytes
```

其中第一对真实调用是：

```text
aclrtMallocPhysical
    size   = 20,971,520
    返回handle = 139661845296768
             │ 同一个handle
             ▼
aclrtMapMem
    handle = 139661845296768
    size   = 20,971,520
    虚拟地址 = 20699527774208
```

这里的handle是**物理内存对象的句柄**，不是HBM物理地址。`data_ptr`及`aclrtMapMem`的目标是设备虚拟地址。物理块在HBM中是否相邻，不能由这些数字判断。

为什么新增映射380MiB，大于该tensor剩余需求？因为分配器以本次20MiB的粒度扩展；多出的空间留在池里，供后续tensor使用。下一张V tensor会继续利用已经映射的剩余空间。**不是每次申请都从零开始，也不是申请多少字节就恰好向底层取得多少字节。**

分析器对全部48张tensor逐一验证：

- Python申请大小与allocator `alloc`记录一致。
- 每个物理句柄和对应映射一一匹配，申请先于映射，大小一致。
- 多个CANN映射合起来与allocator `segment_map`的地址和长度完全一致。
- 返回的tensor存储区间被已有映射与新增映射完整覆盖，并位于之前预留的虚拟地址区域内。

同一地址会出现在多次启动调用中，因此没有仅凭地址或“最近的调用”关联：还要求相同PID/TID及对应raw tensor调用的时间范围。分配器内存历史没有时间戳，使用其中的地址、大小与记录顺序核对，不伪造时间。

## 第五步：变成BF16 KV视图，绑定到模型

原始K存储随后被整理为：

```text
原始tensor：INT8 [387448832]
                     ↓ view dtype / shape
KV tensor：BF16 [11824, 128, 2, 64]
                 block  token head dim
```

前后`data_ptr`、`storage_ptr`、存储字节数保持一致。48张K/V全部如此；`reshape`和模型绑定期间没有新的allocator `alloc`记录，也没有新的CANN内存申请。

实际绑定函数来自Ascend的[bind_kv_cache补丁](results/2026-09-24-run02/sources/vllm_ascend/vllm_ascend/patch/worker/patch_qwen3_next_mtp.py)。我们记录了调用返回后的attention上下文，确认各层持有的正是这些KV存储。

因此，这一步是**用不同dtype和shape解释已有存储，再保存引用**，不是再申请一套KV数据。

## 第六步：CPU建立block编号管理

设备上的大块KV存储建立后，CPU上的`BlockPool`记录：

```text
block总数       = 11,824
保留null block  = B0
空闲可分配block = 11,823
```

这些block对象是CPU上的管理信息。未来分配B1给请求，只是在已有KV tensor中选定相应位置；不会为B1重新申请一个20MiB物理内存块。本练习停在初始化结束，不发送请求验证后续生命周期。

## 总数核对

| 范围 | 记录结果 |
|---|---:|
| 模型层数 / K与V存储数 | 24 / 48 |
| KV tensor数据字节数 | 18,597,543,936 |
| allocator allocated增量 | 18,597,568,512 |
| 两者差额 | 24,576 = 48 × 512 |
| allocator reserved增量 / 新增映射字节数 | 18,601,738,240 |
| KV阶段`aclrtMallocPhysical` | 887次 |
| KV阶段`aclrtMapMem` | 887次 |
| KV阶段新`aclrtReserveMemAddress` | 0次，使用启动阶段已有区域 |
| KV阶段普通`aclrtMalloc` / `aclrtMallocAlign32` | 0 / 0 |
| reshape、bind期间新增存储 | 0 |

统计只针对KV阶段；模型权重、内存profiling及其他初始化调用没有混入。分配器的allocated/reserved也不是整张卡所有HBM占用。

完整记录见[summary.json](results/2026-09-24-run02/analysis/summary.json)、[逐存储CSV](results/2026-09-24-run02/analysis/allocations.csv)和[跨层关联JSON](results/2026-09-24-run02/analysis/allocation_evidence.json)。

当前观察到CANN API调用、返回句柄、映射地址和分配器状态；驱动/固件怎样选择具体HBM物理页没有直接证据。清零kernel的执行完成时间也不在本练习范围内。所有时间是带观测开销的主机时间，不用于性能比较。
