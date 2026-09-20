# Practice 06：制造第一个“假的 race”

只回答一个问题：**A 保存了旧 block reference，而这个 block 已被 B 复用，会怎样？**

我们人工安排错误顺序，构造一个最小 KV lifetime violation detector。
纯 Python、单线程；不需要 vLLM、GPU/NPU 或 Transformer。

## 运行与预期输出

在仓库根目录执行，要求 Python 3.10+，无额外依赖：

```bash
python3 practice_06_stale_block_access/stale_block_access.py
python3 -m unittest discover -s practice_06_stale_block_access -p 'test_*.py' -v
```

演示输出：

```text
ALLOC A B0:g1

A remembers B0:g1

FREE  A B0:g1

ALLOC B B0:g2
WRITE B B0:g2

A READ B0:g1
STALE BLOCK ACCESS
request=A
block=0
expected_generation=1
current_generation=2
```

`A READ` 记录尝试访问；detector 在读取 payload 之前抛出 `StaleBlockAccess`。
演示捕获这个预期异常并打印报告，因此正常完成时退出码为 0。
检测器若没有拒绝旧引用，演示会报断言失败。

## 人工安排的错误顺序

核心代码就是：

```python
allocator = Allocator(num_blocks=1)
a_block = allocator.allocate("A")
allocator.free(a_block)
b_block = allocator.allocate("B")
allocator.write(b_block, "B's KV data")
allocator.read(a_block)  # 抛出 StaleBlockAccess
```

池只有一个块，所以 B 必定复用 A 释放的 B0。没有概率、线程调度或 sleep。

`a_block` 是不可变的 `BlockReference(request_id="A", block_id=0, generation=1)`。
释放 B0 不会删除这个变量；B 复用 B0，也不会把旧引用自动更新为 g2。
每个 reference 保存分配当时的 generation，池保存物理块当前的 generation。

## detector 做什么？

`check_generation` 只做一次比较：

```python
if reference.generation != current_generation:
    raise StaleBlockAccess(reference, current_generation)
```

在本次错误中：

| 信息 | 来源 | 值 |
|---|---|---|
| request | A 保存的旧引用 | A |
| block | A 保存的旧引用 | 0 |
| expected_generation | 分配给 A 时保存的版本 | 1 |
| current_generation | 池中 B0 的当前版本 | 2 |

`1 != 2` 表示：这块物理存储已经进入另一段分配生命周期。
报告中的 request 是错误访问的发起方 A，不能用当前 owner B 替代。
异常同时提供这四个属性，测试可直接校验，不必解析日志。

如果只用 `block_id=0` 读取当前存储，A 就会拿到 B 的数据。
本实验在返回数据之前阻止这种访问；同样的检查也保护 write 和 free。
新旧 owner 即使都叫 A，generation 比较仍然能发现错误。

## “假的 race”是什么意思？

这里没有并行执行，因此不是实际的线程 data race。
我们用确定的执行顺序重现了“释放 → 复用 → 旧引用访问”的生命周期错误。
错误关键在于引用存活得比分配实例更久，并非必须有两个线程同时运行。

[Practice 05](../practice_05_block_lifecycle/README.md) 已引入 generation 和旧引用校验。
Practice 06 将一个错误读取过程单独提取出来，给出明确的检测点和结构化诊断。

## 边界

- 每个物理块只保存一个字符串 KV 占位数据，聚焦生命周期，不重复 token 到块的映射。
- free 清空占位数据但不重置 generation；成功 allocate 才递增对应块的 generation。
- 释放后尚未复用，generation 仍相等。仅比较 generation 无法检测这种访问，
  因此 allocator 另有 FREE 状态检查；owner 和块编号也单独检查。
- reference 只用于签发它的同一个池；这里不实现跨池身份或防伪。
- generation 比较不提供线程同步；本实验也没有检测所有 KV 正确性问题。

测试覆盖完整错误日志、诊断字段、旧引用读写及释放、同名 request 复用、
合法访问不误报，以及释放后尚未复用的状态错误。
