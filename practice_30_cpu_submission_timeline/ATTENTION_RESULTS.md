# 一次 graph 图外 attention：CPU 方法、任务提交与设备执行

2026-10-02，`attention-01`。这是 P30 forward 细分的下一小步：保留完整原生请求，只放大预热后 **decode 32、第一层的一次图外 attention**。

入口：[HTML 报告](report/attention/index.html) · [精确时间线 SVG](report/attention/attention.svg) · [机器可读关联](report/attention/attention.json) · [便携证据](results/attention-01-archive/archive.json)。

## 1. 这次确认了什么？

本次 PIECEWISE graph 的 attention 仍经 Python 后端处理并向 NPU 提交任务。实际选中的是 `model.layers.0.self_attn.attn`，位于 `g1` 和 `g2` 两次 replay 的 Host 提交之间；不能把这一点理解为 CPU 完成了 attention 数学计算，也不能仅凭 Host 边界推断前一张图的设备计算已全部结束。

调用路径是：

```text
vllm::unified_attention_with_output
  ├─ get_attention_context：获取已有 metadata、layer、KV cache
  └─ AscendAttentionBackendImpl.forward
       ├─ reshape_and_cache
       │    ├─ 对 K / V / slots 做 slice
       │    └─ torch_npu._npu_reshape_and_cache → KV 写入提交
       └─ forward_impl
            └─ forward_fused_infer_attention
                 ├─ _get_fia_params：KV pool 的 view + 已有长度列表
                 ├─ query slice
                 ├─ torch_npu.npu_fused_infer_attention_score → FIA 提交
                 └─ view / slice / copy_ → 写回 output
       随后 forward 自身再次对 output 执行 slice / copy_
```

主要新认识是：**图外 attention 的 Host 范围包含上下文查询、形状与参数准备、多个算子提交及输出整理，不能等同于单个 FIA 的调用时间。** 本次长度参数来自已有 Python 列表；没有在这条 DecodeOnly 分支通过 `.tolist()` 从 NPU 回收长度。

## 2. 实际分支与参数

观测记录表明：`DecodeOnly`、`capturing=False`、causal 开启、无 sinks、无 sliding window、无 hamming sparse、不是 KV producer；KV cache 引用已初始化。`forward_impl` 实际进入 FIA，没有进入 paged-attention 分支。

| 参数 | 本次值 | 含义 |
|---|---|---|
| Q | BF16 / NPU，`[1,14,64]` | 一个 token，14 个 query heads |
| 本步 K/V | BF16 / NPU，`[1,2,64]` | 写入 KV pool 的新数据 |
| query 长度 | 主机列表 `[1]` | 本次计算的 query 数 |
| KV 长度 | 主机列表 `[42]` | 本次可读的有效历史长度 |
| FIA 接收的 K/V | NPU view，`[10,128,128]` | 10 个物理块，每块 128 token，最后维度合并 KV heads/head size |
| block table | NPU INT32，`[1,2]` | 本请求逻辑块到物理块的映射 |
| slot mapping | NPU INT32，`[1]` | 本步 KV 写入的位置 |
| FIA 配置 | `TND`、block size 128、14/2 heads、scale 0.125、sparse mode 3 | 原始调用参数保持不变 |

KV pool 的容量不等于本次 attention 的有效序列长度：FIA 结合 block table 和长度 `[42]` 访问历史。`view` 改变访问形状，不意味着复制整个 KV pool。观测只保存形状、stride、dtype、device 和已有主机标量/列表，不读取 Tensor 内容。

## 3. 远端源码依据

本次直接检查远端安装源码；采集后将对应文件按 SHA256 冻结归档。vLLM revision 为 `ad7125a431e176d4161099480a66f0169609a690`，vLLM-Ascend 为 `80610e4438dba05011b05f89fc45d91e96992671`。

| 远端位置 | 对应工作 |
|---|---|
| `/vllm-workspace/vllm/vllm/model_executor/layers/attention/attention.py:620` | `get_attention_context` 查找当前层的上下文 |
| 同文件 `:706、721、723` | custom-op 的 Python 实现获取上下文，再调用 `impl.forward` |
| `/vllm-workspace/vllm-ascend/vllm_ascend/attention/attention_v1.py:1279` | backend `forward`、分支判断、KV 写入与计算调用 |
| 同文件 `:1229、1245` | `reshape_and_cache` 对 K/V/slots 切片，经 DeviceOperator 写 KV |
| `/vllm-workspace/vllm-ascend/vllm_ascend/device/device_op.py:44` | 设备适配器调用 `torch_npu._npu_reshape_and_cache` |
| `attention_v1.py:1258、1275` | `forward_impl` 选择 FIA 路径 |
| 同文件 `:985、1022` | DecodeOnly 的 `_get_fia_params` 创建 K/V view，复用 block table 与 `seq_lens_list` |
| 同文件 `:1045、1146` | 非 capture、causal、无 sliding window 的 FIA 调用 |
| 同文件 `:1162、1163、1342` | 内层整理并写回 output；外层再次赋值 |

`attention_observer.py` 包装这些原始函数/方法的现有边界，不复制或改写其函数体。包装安装在图捕获和两次完整请求预热之后，仅在选中调用中启用细分范围。`get_attention_context` 是前置独立范围，其余七个范围按原始嵌套关系记录。

## 4. 计时细化到了哪里？

下面是每个包装范围内部的双时钟，不包括自身 profiler 标记的全部边界成本；各行存在包含关系，不能串行相加。

| 记录范围 | 墙钟 µs | 线程 CPU µs | 扣除直接子 scope 后墙钟 µs |
|---|---:|---:|---:|
| 上下文查询 `attn_context` | 2.594 | 2.068 | 2.594 |
| 后端入口 `attn_backend` | 303.517 | 302.973 | 52.891 |
| KV 准备 `attn_kv_prepare` | 63.525 | 63.193 | 37.615 |
| KV 算子调用 `attn_kv_submit` | 25.910 | 25.431 | 25.910 |
| 路径选择 `attn_dispatch` | 187.101 | 186.745 | 21.507 |
| FIA 方法 `attn_fia` | 165.594 | 165.199 | 88.905 |
| FIA 参数准备 `attn_fia_params` | 9.846 | 9.434 | 9.846 |
| FIA 算子调用 `attn_fia_submit` | 66.843 | 66.361 | 66.843 |

`attn_fia` 的余量仍包含 query slice、结果整理、参数包装及观测成本；`attn_backend` 的余量包含分支、末尾输出赋值及观测成本。线程 CPU 接近墙钟不能区分有效工作、运行时自旋和 profiler 成本。

报告还使用 **profiler 边界** 将方法 scope 和 `cpu_op` 合成同线程包含树。例如 FIA 方法的 profiler 范围为 **175.203 µs**，分成直接子 scope **98.571 µs**、直接子 CPU op **23.349 µs**、剩余 **53.283 µs**。这三个数恰好相加，但不能与上表内部计时的 165.594 µs 混算。

这使原先宽泛的 Host 余量定位到具体方法区域，但没有把剩余时间全部解释为某种纯业务开销。上一轮 24 次 attention 的自身余量约 1.013 ms 与本次重度插桩样本不是同一口径，不能相减，也不能将单次结果乘以 24 外推。

## 5. 从 Host 提交关联到实际设备任务

所有以下设备任务均精确关联到选中调用，位于物理 stream 46，且不属于 graph 内部 replay task：

| Host 调用与队列 | CANN connection | NPU task | 设备任务 | 设备 µs |
|---|---:|---:|---|---:|
| KV submit / `q:24389` / `ReshapeCacheOperation` | 45278 | 28819 | `ReshapeAndCacheNdKernel` | 1.900 |
| FIA submit / `q:24390` / `aclnnFusedInferAttentionScoreV3` | 45283 | 28820 | `FusedInferAttentionScore` | 23.581 |
| FIA 方法的 copy / `q:24391` / `aclnnInplaceCopy` | 45289 | 28821 | `MEMCPY_ASYNC` | 0.600 |

`q:...` 是由队列关联 ID 生成的分析标识；Enqueue/Dequeue 的 correlation 与 flow 身份已经检查。每条设备任务还核验了 CANN connection、stream/task 以及计算任务 CSV 身份，不仅凭名称或相近时间匹配。

外层 `forward` 还有一个 `aclnnInplaceCopy` 队列 `q:24392`，没有关联到设备任务。因此本次是 **两个 `copy_` Host 调用、两次 copy 队列记录、一个设备拷贝任务**。源码中 FIA 返回传入的 `output`，外层再对该 output 赋值；这一现象与别名自拷贝跳过设备执行相容，但本轮没有追入 native no-op 判断，不将推断当作底层实现证明。

选中 custom-op 范围内没有记录到显式 CANN `Synchronize`。这支持本次可见路径以准备和提交为主，不证明 native 内部绝无阻塞，也不表示整个请求不需要同步。完整请求仍记录到 256 次显式同步调用，属于已有的结果回收等路径。

## 6. 观察成本与验证

- 被选中的 custom-op Host 范围为 **395.780 µs**；同一步其余 23 次中位数为 **109.082 µs**，范围 102.740–138.360 µs。额外 scope 和参数元数据检查明显扰动了样本，层次与顺序差异也存在；不能把差值当成精确的纯插桩开销。
- 三次无 profiler 原生参考耗时中位数 **288.011 ms**；诊断请求 **440.768 ms**，约为参考的 **1.530 倍**；独立恢复请求 **292.557 ms**。诊断差值混合 profiler、包装和进程间差异，不作为性能优化结果。
- 11 次响应（含预热）的 token、logprob、结束原因精确一致；8 个细分 scope 各出现一次且均在 decode 32。attention/基础/graph 绑定均恢复。
- 29 个被审计文件前后未变，未修改安装源码、配置或已有服务；实验子进程结束，设备空闲，原生推理恢复。
- 64 个步骤、2,672 个阶段范围、9,227 对队列、26,517 个设备任务完成关联检查，其中 19,298 个计算任务核验 CSV。25 个捕获对象、1,575 次 replay；测量期间没有重新 capture，decode 32 保留 25 次 replay。
- 18 项测试通过，覆盖跨线程/错误父范围、嵌套重复计数、队列身份及异常后继承方法恢复等反例。139 个归档文件逐项校验哈希；本地仅使用归档工具重新分析后，HTML、SVG、JSON 与远端产物逐字节一致。报告通过离线浏览器、展开详情、宽/窄屏及独立 SVG 检查。

首轮止步于一次调用的结构和证据。后续已聚焦 [FIA 的提交、参数与 binary 选择](fia_probe/README.md)：复用本轮模型时间线，并单独做三个配置的合成输入探针，确认文件 → 加载字节 → 入口 → launch；不把两次实验混成一条原始记录。输出 copy 的 native 判断、Triton 核数 / msprof / msdebug 仍未在此推进。
