# Hugging Face eager 与 vLLM-Ascend：调用入口、模型实现和多用户执行

记录日期：2026-10-02。本文用于解释 Practice 27 的 Hugging Face eager 实验与真实 vLLM-Ascend 服务之间的关系。软件行为随版本变化；涉及 vLLM 的通用机制以记录日期时的官方文档为准，涉及 `lengths` 的结论则以仓库内冻结的 Transformers 5.5.4 源码和 Ascend 实测为准。

核心结论：**两条路径可以加载同一个 Qwen2.5-VL 架构和同一份权重，但调用入口、运行模型的 Python 实现、attention/KV cache 接口和请求调度可能不同。** 它们最终都可能到达 PyTorch、torch-npu、CANN 和 NPU，但不能因为底层相同，就认定中间会执行相同的 Python forward。

## 1. 先区分“模型”的三个含义

日常说“使用同一个模型”可能指三件不同的事：

| 层次 | 例子 | 两条路径是否可以相同 |
|---|---|---|
| 架构 | Qwen2.5-VL-3B-Instruct | 可以相同 |
| checkpoint | hub 中的配置、safetensors、tokenizer/processor | 可以相同 |
| 执行实现 | Transformers 模型类，或 vLLM 原生/适配模型类 | 可能不同 |

权重文件只保存参数，不规定服务引擎必须逐行运行哪一份 Python forward。不同实现只要保持模型语义和权重映射，就可以用不同的数据布局、attention backend、KV cache 和融合算子完成推理。

本次 HF 实验使用的模型实现来自安装的 Transformers 包，而不是模型 hub 中的自定义 remote code。实际文件已经冻结为 [installed_model.py](../vision_metadata_synchronization/correctness_probe/results/run-r01/installed_model.py)，SHA-256 为 `9d6d15040bdb985d9518117ed029f6a318a38abc06091462d96f16cf1d3d7820`。权重、配置和 processor 位于模型 hub 目录；模型类来自 `site-packages/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py`。

## 2. Hugging Face eager：用户直接调用模型

典型入口是：

```python
processor = AutoProcessor.from_pretrained(model_path)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path).to("npu")
inputs = processor(image=image, text=prompt, return_tensors="pt")
outputs = model.generate(**inputs.to("npu"))
```

简化调用关系：

```text
用户脚本
  → Transformers generate / forward
  → Qwen2.5-VL 的视觉和语言模块
  → PyTorch ATen 算子
  → torch-npu
  → CANN runtime / 算子库
  → NPU kernel
```

用户程序决定什么时候调用、一次传入多少样本、如何排队。如果写成逐请求循环，A 完成后才开始 B；Transformers 不会凭空提供 vLLM 的在线请求 scheduler、连续批处理或块式 KV cache 管理。

Practice 27 固定视觉 attention 为 Transformers 的 eager attention 实现。这条真实源码会执行：

```python
lengths = cu_seqlens[1:] - cu_seqlens[:-1]
splits = [
    torch.split(tensor, lengths.tolist(), dim=2)
    for tensor in (query_states, key_states, value_states)
]
```

对应冻结源码见 [installed_model.py:264](../vision_metadata_synchronization/correctness_probe/results/run-r01/installed_model.py#L264)。`cu_seqlens` 已按 `hidden_states.device` 创建成 NPU Tensor，所以减法经 PyTorch dispatcher 进入 torch-npu Sub；`.tolist()` 再引入同步和 D2H。完整实测链见 [lengths 笔记](../vision_metadata_synchronization/LENGTHS.md#21-为什么一行-python-减法会变成-npu-sub)。

## 3. vLLM-Ascend：用户先把请求交给推理引擎

离线入口可能是：

```python
from vllm import LLM

engine = LLM(model=model_path)
outputs = engine.generate(prompts)
```

在线入口通常是：

```bash
vllm serve /path/to/model
```

客户端再发送 OpenAI-compatible HTTP 请求。简化调用关系为：

```text
多个客户端请求
  → vLLM engine / 请求队列
  → scheduler 选择本轮请求和 token
  → KV cache manager 分配或查找块
  → model runner 构造本轮输入 batch
  → 选定的模型实现和 attention backend
  → vLLM-Ascend 平台适配与算子
  → PyTorch / torch-npu / CANN
  → NPU kernel
```

vLLM-Ascend 是 vLLM 与 Ascend 软件栈之间的硬件插件，而不是另一套模型权重格式。官方安装架构把 vLLM 列为 inference engine、vLLM-Ascend 列为 hardware plugin，并继续依赖 PyTorch、torch-npu 与 CANN：<https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/getting_started/installation.md>。

## 4. vLLM 到底执行哪一份模型代码

当前 vLLM 的 `model_impl` 有三种与本文相关的选择：

| 选择 | 含义 |
|---|---|
| `auto` | 有 vLLM 原生实现时优先选择它，否则尝试 Transformers modeling backend |
| `vllm` | 强制使用 vLLM 模型实现 |
| `transformers` | 强制使用 vLLM 的 Transformers modeling backend |

官方定义见 [ModelConfig](https://docs.vllm.ai/en/latest/api/vllm/config/model/) 和 [Supported Models](https://docs.vllm.ai/en/latest/models/supported_models/)。实际部署必须记录版本、`model_impl` 和加载后的 Python 类，不能只凭模型名称判断。

### 4.1 vLLM 原生模型实现

如果目标架构有 vLLM 原生实现，vLLM 可以把相同 checkpoint 加载到自己的模块中。模型可能使用 vLLM Attention 接口、块式 KV cache、批处理 metadata 和融合算子。例如概念上可能是：

```python
output = self.attention(
    query,
    key,
    value,
    kv_cache=kv_cache,
    attention_metadata=metadata,
)
```

这与 Transformers eager 中先执行 `lengths.tolist()`、再用 `torch.split` 切三次 Q/K/V 的组织方式不同。模型语义可以一致，kernel execution graph、同步点和中间 tensor 却可以不同。

### 4.2 Transformers modeling backend

`--model-impl transformers` 会复用兼容的 Transformers 模型实现，但模型仍运行在 vLLM engine/model runner 内。vLLM 官方要求兼容 attention 通过 `ALL_ATTENTION_FUNCTIONS` 接口，以便为 attention 模块连接 vLLM Attention layer 和 KV cache。因此它比原生 vLLM 实现更接近 Transformers 源码，但仍不等同于用户直接调用 `transformers_model.generate()`。

要判断某一行 Transformers 源码是否实际执行，仍需核对加载后的类、attention 实现和 trace，不能由 `model_impl="transformers"` 单独推断所有内部路径完全相同。

## 5. 三种常被混淆的执行方式

| 名称 | 谁负责请求调度 | 谁提供模型执行框架 | eager 的含义 |
|---|---|---|---|
| HF eager | 用户脚本 | Transformers | 使用 Transformers 的普通 Python/PyTorch attention/forward 路径 |
| vLLM eager | vLLM scheduler/model runner | vLLM 原生或 Transformers backend | 仍是 vLLM，只是不使用支持的 graph capture/replay 路径 |
| vLLM graph | vLLM scheduler/model runner | vLLM 原生或 Transformers backend | 支持的计算段由 ACL Graph 等捕获和重放 |

所以：

```bash
vllm serve model_path --enforce-eager
```

不等价于：

```python
transformers_model.generate(...)
```

`--enforce-eager` 改变的是 vLLM 的设备执行方式，不会取消 vLLM 的 scheduler、model runner 和 KV cache 管理，也不会自动选择 Transformers 原始模型类。vLLM-Ascend 的 graph/eager 用法见官方 [Graph Mode Guide](https://docs.vllm.ai/projects/ascend/en/v0.11.0/user_guide/feature_guide/graph_mode.html)。

## 6. 多用户服务如何执行

多用户在线场景通常使用 vLLM/vLLM-Ascend，或者使用另一个具有同类调度能力的服务引擎。直接 HF eager 也能包成 Web 服务，但排队、批处理、KV cache 和并发策略都要由应用自己实现。

假设 A 正在 decode，B 和 C 新到达：

```text
轮次 1：A prefill
轮次 2：A decode + B prefill
轮次 3：A decode + B decode + C 的可调度部分
轮次 4：仍活跃的请求重新组成下一批
```

每个请求保留自己的 token、采样参数、结束条件和 KV block 映射，但模型权重和 KV cache 内存池可以共享。model runner 每轮处理 scheduler 选出的 token batch，不是为每个用户顺序调用一次完整 `generate()`。vLLM 的 scheduler 配置包括 active sequences、token budget、chunked prefill、异步调度和多模态输入约束，见官方 [SchedulerConfig](https://docs.vllm.ai/en/latest/api/vllm/config/scheduler/)。

多用户也不自动等于多 stream：

```text
多个用户
  → scheduler 合并为一个设备 batch
  → 一次模型执行
  → 常见情况下由主计算 stream 执行这一批 kernel
```

额外 stream 可能用于 KV 搬运、通信、图像编码或其他可重叠任务，但不是“一位用户一条 stream”。多用户描述请求调度；多 stream 描述设备任务并发；二者不是同一层。

## 7. 对当前 lengths 结论的适用范围

已经证明的是：

```text
固定 Transformers 5.5.4 Qwen2.5-VL 源码
  + HF eager 视觉 attention
  + Ascend 910B2C / torch-npu 2.10
  → 真实执行 Sub → .tolist() → torch.split
```

尚未证明的是：

```text
任意 vLLM-Ascend Qwen2.5-VL 服务
  → 必然执行同一段 lengths.tolist()
  → 必然产生相同数量的同步和 D2H
```

如果 vLLM 原生模型或 Ascend attention backend 直接消费累计边界，或者提前在 CPU/model runner 生成长度，这段往返可能不存在。如果实际选中的 Transformers backend 仍进入冻结源码中的这个分支，则可能重新出现同类同步；必须用实际部署证据确认。

## 8. 怎样确认一个真实 vLLM-Ascend 服务走哪条路径

建议按以下顺序记录：

1. 固定 vLLM、vLLM-Ascend、Transformers、torch、torch-npu 和 CANN 版本。
2. 保存启动参数，特别是 `model_impl`、`--enforce-eager` 和 compilation/graph 配置。
3. 在模型 worker 内打印实际加载类；官方建议可用 `LLM.apply_model(lambda model: print(type(model)))` 判断是否为 Transformers modeling backend。
4. 记录实际 attention backend 和模型源码位置，不以 checkpoint 名称代替实现身份。
5. 对一个受控请求采集 CPU/NPU trace，查找 `aten::sub.Tensor`、`aclnnSub`、`aclrtSynchronizeStream`、D2H 和 `torch.split` 的主机范围。
6. 用 flow 关联主机算子、runtime launch 和设备 kernel；不要仅按名称或时间接近猜测。
7. 再做多请求对照，区分 scheduler 合批带来的变化与模型内部同步。

只有第 3–6 步都指向同一链路，才能把当前 HF eager 的 `lengths` 发现推广到那次 vLLM-Ascend 部署。

## 9. 与现有研究的关系

- [视觉同步分析](../vision_metadata_synchronization/README.md)：HF eager 中 91→7→0 的元数据同步消融。
- [lengths 专题](../vision_metadata_synchronization/LENGTHS.md)：`cu_seqlens`、Sub、`.tolist()` 和分段含义。
- [Sub 提交与内存](../vision_metadata_synchronization/SUB_EXECUTION.md)：PyTorch dispatcher、torch-npu、CANN 两阶段 API 和 NPU kernel。
- [vLLM/vLLM-Ascend stream 源码分析](../streams_in_vllm_source_code/README.md)：服务引擎内 stream 的创建、绑定和 eager/graph 差异。

这篇笔记连接两条研究线，但不把 HF trace 当作 vLLM trace，也不把 vLLM 的多请求调度当作多 stream 证据。
