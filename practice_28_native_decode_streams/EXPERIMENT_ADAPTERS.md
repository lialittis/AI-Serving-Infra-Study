# 临时包装与实验适配：为什么这样做，怎样恢复

本文集中说明 Practice 28 的实现思路，供后续练习复用。记录日期：2026-09-30；代码入口对应当前练习脚本。实际运行版本以结果归档中的 `collector/` 为准。

**基本方法是：先通过原生请求取得真实状态与参考结果，再在实验进程中控制资源和 stream，仍调用原生模型与算子实现。** 没有编辑已安装的 vLLM / vLLM-Ascend 源码文件，但确实改变了实验进程中的函数绑定、资源组织与提交方式。

这不是透明地旁观默认 serving：为了研究一个固定 decode 步，我们从原生请求中截取状态，随后由实验脚本直接调用 runner。结论适用于这个受控执行单元，不能直接当作默认 scheduler 或 HTTP 服务的性能结论。

## 三种操作应当分开理解

| 操作 | 为什么需要 | 代码入口 | 生效范围与时机 | 对计时的影响 |
|---|---|---|---|---|
| 临时包装 `runner._model_forward` 和 `_sample` | 在真实请求中保存选定步的输入、KV、metadata 和原生结果 | [collect_states](native.py#L162) | 当前 runner 实例；仅原生状态采集期间 | 保存状态包含同步、克隆和 D2H；不在正式 pair 计时中 |
| 创建第二 runner，建立两个 capsule | 两个独立任务要有各自的可写资源，又要执行同一种模型计算 | [make_second](native.py#L288)、[Capsule](native.py#L216) | 实验子进程生命周期 | 模型加载、KV 初始化、图捕获在计时外 |
| 临时切换模块级引用和执行上下文 | 原生模块的一些状态保存在全局变量中，不能让两个 runner 无意共用可写状态 | [GlobalState.active](native.py#L116)、[Capsule.submit](native.py#L250) | 每次模型提交期间 | **包含在正式 pair 计时内**，属于实验适配成本 |
| 选择当前 stream、提交任务和等待终点 | 构造单流／双流对照 | [pair](native.py#L325) | 每个 pair；调用现有 stream API，并非替换算子 | 提交、事件控制及最终 CPU 等待计入 wall time |
| 临时包装 `torch.npu.NPUGraph.replay`，增加 profiler scope | 将 replay、graph 对象与 A/B 任务精确关联 | [Observer.installed](experiment.py#L34) | 类方法替换，对本实验进程内该类生效；仅诊断期间 | plain / Pipe 诊断使用；无 profiler 正式测量不安装此观察器 |

它们分别承担“取得原生参考”“构造受控执行条件”“观察运行过程”的作用，不能笼统地说所有包装都只做日志、都没有运行开销。

## 先读一个最小包装：原函数仍然被调用

`collect_states()` 先保存原来的绑定方法：

```python
original_forward, original_sample = runner._model_forward, runner._sample
```

新的 `forward` 在指定步保存执行前状态，最后调用原来的计算：

```python
return original_forward(*args, **kwargs)
```

新的 `sample` 先取得原生输出，再保存数值参考，返回的仍是原生输出：

```python
output = original_sample(logits, spec)
# 指定步同步并保存 logits、sample、KV 的参考值
return output
```

包装只在下面这个范围内安装：

```python
with Bindings([(runner, '_model_forward', forward),
               (runner, '_sample', sample)]):
    response = generate(llm, prompt)
```

离开 `with` 后恢复方法。正式固定步执行时，使用的已经是原来的 `_model_forward` 和 `_sample`，不会每次再次截取整份参考状态。

采集也有主动准备动作：每条原生参考请求前清零 KV 池，并在选定步同步。这样未使用的池内容也可比较。因此这段运行用于获取受控的数值参考，不用于衡量未插桩服务延迟。

## 然后读资源切换：恢复引用，保留资源所有者

`GlobalState` 保存 workspace manager、RoPE 相关引用、GraphParams、全局 graph pool 以及 Ascend 工具函数缓存的当前 stream。为 B 创建新的状态时，分配独立 workspace manager 和 graph pool，并在对应状态下初始化 runner。

每次 `Capsule.submit()` 的顺序是：

```text
绑定当前 capsule 的状态和 forward context
    → 原生 _model_forward
    → 原生 compute_logits
    → 原生 _sample
    → 保存输出引用
退出上下文，恢复之前的 Python 绑定
```

`GlobalState.active()` 退出前还会保存本次原生调用产生的新引用，交给当前 capsule 持有，再恢复进入前的全局绑定。当前 stream 缓存按这次实际选中的 stream 绑定，避免一直使用首次缓存的流。

这里有两个不同的时间点：

1. **本实现在 CPU 提交函数返回时恢复 Python 全局引用。** NPU 此时可能仍在执行已经提交的工作；这依赖所提交任务的资源继续存活，不能推广成任意异步框架中随时换全局状态都安全。
2. **等待终点确认设备完成后，才可以改写或释放这些工作使用的存储。** capsule、runner、图对象等在此期间继续持有资源；下一轮恢复输入和 KV 必须在上一轮完成之后。

因此 `Bindings` 不负责设备同步。`pair()` 在 `finally` 中等待已记录的终点 event；若提交中途失败、终点尚未 record，就等待已经提交工作的 stream。它不会等待从未提交的分支的 event。

这些模块级绑定并不是线程局部隔离。本实验只使用一个 CPU 模型提交线程；不能把这套实现直接放进两个并发 Python 模型提交线程中，就假定全局引用仍然安全。

## 哪些资源共享，哪些由实验重新组织

- B 使用原生 `NPUModelRunner` 加载模型，再让参数的 `.data` 引用 A 的只读参数存储；这改变内存所有权关系，不改参数数值。
- 原生 `_ROPE_DICT` 会复用固定的 `cos_sin_cache`。在当前 Qwen 路径中，它是只读查找表，另外记录地址和前后哈希；不把任意模型 buffer 都视为只读。
- KV、固定输入、metadata、采样状态、workspace、可变 RoPE 缓冲和图池按实例持有并检查可见存储范围。
- Graph 沿用原生 dummy capture 和 runner 的固定输入地址。恢复状态时向固定输入存储复制数值，不能以新 Tensor 地址代替捕获时地址。
- 计时中的原生模型／算子计算继续执行；请求调度、选择哪份固定状态、安排到哪个 stream，则由实验脚本控制。

## 恢复不是一句“退出进程即可”

[Bindings](common.py#L21) 记录属性的原值以及它是否原本属于该对象。退出时恢复原属性，或删除临时添加的属性，让原来的类／继承查找重新生效；部分安装失败也执行回滚。

更大范围的恢复由不同层负责：

| 层次 | 责任 | 证据 |
|---|---|---|
| Python 绑定 | 上下文退出时恢复，含异常与部分安装失败 | [本地测试](test_analysis.py)、子进程 `recovery.json` |
| 已提交的设备任务 | 保持资源存活，等待已提交的流／终点，再复用存储 | `pair()` 的 `finally`；提交首个任务后注入异常的资格检查 |
| native 错误或析构挂起 | 控制器只清理自己启动的进程组；必要时升级终止，不 reset 设备 | [控制器](run_suite.py)、归档中的 launcher / exit / recovery 记录 |
| 实验后的原生环境 | 核对受审计文件哈希、NPU 遗留进程，独立新进程重跑原生请求 | [最终审计](results/final_audit.json) |

关键文件哈希一致不是“扫描所有库文件”的证明；可见 tensor 地址不相交也不是穷尽所有 native 隐式工作区的证明。源码、数值、时间线和恢复检查共同限定本次结论。

## 读代码与复用时的顺序

建议先读 `collect_states()` 和 `Bindings`，理解“保存原函数 → 安装包装 → 调用原函数 → 恢复”；再读 `Capsule.submit()` 与 `pair()`，理解资源绑定和设备完成是两件事；最后读仅在诊断中使用的 `Observer`。

历史实现错误及修正见 [RUN_HISTORY.md](RUN_HISTORY.md)，包括用真实 Attention metadata 重捕获、推理上下文和只读 RoPE 表分类的问题。正式采集版本与后来增强的审计版本有明确区分，未把新代码冒充采集时版本。

后续 Practice 29 按“先做薄，再做厚”推进：先复用已理解的必要隔离与恢复逻辑，只新增门控这一项实验变量；观察、计时和干预分别记录，不一开始就叠加大量包装与 benchmark。
