# Practice 05：物理块的生命周期

只回答一个问题：**block 从 allocation 到 free 再到 reuse，到底发生了什么？**

这是独立的纯 Python、单线程实验，不需要模型、NPU 或 vLLM。
`A0`、`B0` 等字符串仅代表 token 的 KV 数据。

## 运行

在仓库根目录执行（Python 3.10+，无额外依赖）：

```bash
python3 practice_05_block_lifecycle/block_lifecycle.py
python3 -m unittest discover -s practice_05_block_lifecycle -p 'test_*.py' -v
```

只有一个物理块，每块有四个 token 槽位。输出为：

```text
ALLOC request=A block=0 generation=1
WRITE request=A block=0 generation=1 offset=0 value=A0
FREE  request=A block=0 generation=1
ALLOC request=B block=0 generation=2
WRITE request=B block=0 generation=2 offset=0 value=B0

Physical identity: B0 == B0
Allocation identity: B0:g1 != B0:g2

Try writing through A's old handle:
REJECT stale handle B0:g1; current generation is g2
B still reads: B0
FREE  request=B block=0 generation=2
```

`B still reads: B0` 中的 `B0` 是 B 写入的数据标签，表示 B 的 token 0；
日志中的 `block=0` 才是物理块编号。

## 一块物理存储，两次分配

| 操作 | owner | 当前 generation | 物理块内容 |
|---|---|---|---|
| 初始 | FREE | 0 | `[-- -- -- --]` |
| allocate(A) | A | 1 | `[-- -- -- --]` |
| A 写入 | A | 1 | `[A0 -- -- --]` |
| free(A) | FREE | 1 | `[-- -- -- --]` |
| allocate(B) | B | 2 | `[-- -- -- --]` |
| B 写入 | B | 2 | `[B0 -- -- --]` |

物理块编号一直是 0；生命周期发生了更替。

```text
FREE(g0) → ALLOCATED(A, g1) → FREE(g1) → ALLOCATED(B, g2)
                 │                              │
             handle A                       handle B
              B0:g1                          B0:g2
```

- **physical identity**：同一个池中的 `block_id`，表示哪个物理存储槽。
- **logical / allocation identity**：`(block_id, generation)`，表示该槽的哪一次分配。

这里的“logical identity”指分配实例的身份，与 Practice 04 的逻辑块下标不是同一个概念。
generation 是每个物理块独立维护的计数器，从 0 开始，每次成功分配加 1。
写入和释放不会增加或重置它。两个不同块可以同时处于 g1，因此 generation 单独不能标识对象。

## 为什么只检查 block_id 不够？

A 释放后，其变量 `a` 仍保存旧引用 `B0:g1`。释放存储不等于自动删除所有引用。
B 随后获得 `B0:g2`；如果旧引用只保存编号 0，延迟执行的一次 A 写入就可能覆盖 B 的数据，
一次旧的 free 也可能释放 B 正在使用的块。完全单线程也可以按顺序触发这种错误。

本实验让 `allocate` 返回不可变的 `BlockHandle(block_id, generation)`。
每次 `read`、`write`、`free` 都先检查：

1. 物理块编号是否合法。
2. handle 的 generation 是否与该块当前 generation 相同。
3. 该块是否仍处于已分配状态。
4. 操作的 request 是否为当前 owner。

检查全部通过后才读取或修改数据。释放后、尚未复用时，旧 handle 因 FREE 状态被拒绝；
复用后，它因 generation 不匹配被拒绝。即使新旧 request 都叫 A，generation 仍能区分两次分配。
generation 必须随引用一起保存并在操作前校验，仅在日志里打印它不能阻止错误。

## 和 Practice 04 的关系

[Practice 04](../practice_04_paged_kv_cache/README.md) 解决空间映射：

```text
token index → logical block → physical block + offset
```

Practice 05 解决时间上的有效性：

```text
这个 physical block 的引用，是否仍属于当前这一次分配？
```

如果以后合并两者，block table 的表项就可以从 `block_id` 升级为 `BlockHandle`；
token 的除法和取余不变。本次独立展示生命周期，不扩展 Practice 04。

## 实验边界

本例在 free 时原地清空数据，便于观察复用；真实内存池不一定立即清零。
清空数据和检查 generation 解决不同的问题：即使清空过，旧引用仍可能破坏新写入的数据。

handle 只用于签发它的同一个池；没有跨池身份或防伪能力。Python 整数在本例中不会固定宽度溢出。
没有线程、锁、共享块或引用计数；generation 检查也不等于并发同步机制。

测试验证事件顺序、释放后的访问、复用后的旧引用读写及释放、同名 request 的复用、
数据清空、各块独立计数、池耗尽、owner 校验及边界检查。
