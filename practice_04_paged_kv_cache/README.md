# Practice 04：一个 token 怎样映射到 physical KV block？

**request 看到的是 logical sequence；allocator 管理的是 physical blocks。**

这个实验只回答上面的问题。使用 Python 标准库，不需要模型、PyTorch、NPU 或 vLLM。
多个 request 的操作按顺序执行，没有并发或调度器。

## 1. 三步找到一个 token 的位置

设 request 内的 token 下标为 `i`，每块容纳 `block_size` 个 token：

```python
logical_block = i // block_size
physical_block = request.block_table.physical_blocks[logical_block]
offset = i % block_size
```

结果是 **`(physical_block, offset)`**。必须使用该 request 自己的 block table：
`i` 相同，不代表不同 request 使用相同的物理块。
代码中这个查询由 `Request.locate(i)` → `BlockTable.locate(i)` 完成。

```text
request 的 token 下标 i
        │
        ├── i // block_size ──> logical block ──> 查该 request 的 block table
        │                                                   │
        └── i % block_size ──> offset                       ▼
                                          physical block ID + offset
```

`A0`、`A1` 等字符串代表相应 token 的 **K/V 记录占位符**。真实缓存存的是各层 attention 的
K/V 数值；这里不存真实张量，也不计算 attention，专注于地址映射。

## 2. 三个类各自负责什么

| 类 | 负责的数据和操作 |
| --- | --- |
| `BlockPool` | 固定数量的物理块；分配、读取、写入、释放；每次选择编号最小的空闲块 |
| `BlockTable` | 单个 request 的 `logical block index → physical block ID` 表；按需扩容和查表 |
| `Request` | request 名称、逻辑 token 数和自己的 block table；追加与按逻辑下标读取 |

`BlockPool` 不需要知道 `A0` 属于哪个逻辑位置，只接收物理块号和 offset。
`Request` 不保留一个连续的 KV 副本；`read_all()` 会逐 token 查表，临时重建逻辑顺序以便观察。

## 3. 运行

在仓库根目录使用 Python 3.10+：

```bash
python3 practice_04_paged_kv_cache/paged_kv_cache.py
python3 -m unittest discover -s practice_04_paged_kv_cache -p 'test_*.py' -v
```

程序固定使用 `num_blocks=4`、`block_size=4`，以便输出可以逐步手算。

## 4. 从连续映射走到不连续映射

初始状态：

```text
B0: FREE
B1: FREE
B2: FREE
B3: FREE
```

**A 追加 6 个 token**，需要 `ceil(6 / 4) = 2` 个块：

```text
A logical block 0 -> B0
A logical block 1 -> B1

B0: [A0 A1 A2 A3]
B1: [A4 A5 -- --]
B2: FREE
B3: FREE
```

`--` 是已分配块中的空槽，不等于 FREE 块。此时 A 的容量为 8 个槽，但逻辑长度为 6，
所以 `A.read(6)` 必须失败。B1 的剩余两个槽仍属于 A，不会分配给另一个 request。

**B 追加 4 个 token**，占用 B2。然后 **A 再追加 A6、A7、A8**：
A6/A7 填满原来的尾块，A8 才触发新块分配。因为 B2 已被占用，A 获得 B3：

```text
A block table = [0, 1, 3]
B block table = [2]

B0: [A0 A1 A2 A3]
B1: [A4 A5 A6 A7]
B2: [B0 B1 B2 B3]   # 方括号内是 request B 的 K/V 标签
B3: [A8 -- -- --]
```

| A 的 token 下标 | logical block | 查表后的 physical block | offset |
| ---: | ---: | --- | ---: |
| 0 | 0 | B0 | 0 |
| 3 | 0 | B0 | 3 |
| 4 | 1 | B1 | 0 |
| 7 | 1 | B1 | 3 |
| 8 | 2 | **B3** | 0 |

对 A8，`8 // 4 = 2`，但物理块号不是 2；必须查 A 的表中第 2 项，得到物理块号 3。
直接把 logical block index 当 physical block ID 会读到 request B 的数据。
A 原有 token 不需要搬动；按表读取仍得到 `[A0, A1, ..., A8]`。

## 5. 回收与边界

演示最后释放 A 的 B0、B1、B3，B 的 B2 保持不变。新的 request C 追加 5 个 token，
重新使用 B0 和 B1；未写入的槽会是 `--`，不会保留 A 的旧标签。

接口约定：

- 空 request 不分配物理块；填充尾块空槽也不会分配新块。
- 一次 `append()` 若空间不足，会在写入或分配之前抛出 `MemoryError`，原状态保持不变。
- `Request.locate/read` 只接受已有 token 下标；`BlockTable.locate` 只检查已分配容量。
- `Request.release()` 可以重复调用；释放后该对象不可继续追加，请创建新 request。
- 直接操作 pool 的分配/释放接口时，不能绕过仍持有该块的 block table 提前释放它。
  正常使用通过 `Request` 完成生命周期管理。

测试覆盖 6-token 示例、跨块追加、不连续物理块、request 数据隔离、回收重用、
空间不足时不部分写入、逻辑边界、空 request 及非法物理访问。

## 6. 这个实验的边界

这里实现的是 block 分配与查表，不是 PagedAttention 算子。没有真实 K/V 张量、
层/头维度、前缀共享、引用计数、copy-on-write、换出或抢占。
`BlockPool` 为了易读使用线性扫描；这不是生产分配器的性能实现。

与 Practice 01 的联系是：逻辑上的 token 顺序保持不变，存储位置从连续 tensor 中的下标，
变成了 **先查 block table，再访问块内 offset**。
