# Practice 36：graph 捕获/重放的存储边界

P32 遗留项之一：私有池（private pools）、capture mark 删除与 replay 对外部存储的使用。
源码侧（[NPUGraph.cpp](../references/torch_npu_allocator_lifetime/sources/torch-npu/torch_npu/csrc/core/npu/NPUGraph.cpp)、
[NPUCachingAllocator.cpp releasePool](../references/torch_npu_allocator_lifetime/sources/torch-npu/torch_npu/csrc/core/npu/NPUCachingAllocator.cpp#L2144)）：
捕获时 `beginAllocateToPool` 把捕获流的分配导向私有池；`reset()` 只把池标记 freeable，块以
npuFree 归还驱动而不回普通池；**对捕获前在普通池分配、被图内烘焙地址的"外部存储"没有任何保护**。
本实验用受控探针实测这两条边界。

## 四种模式（每模式独立进程）

图工作负载：捕获 `static_out.copy_(external)`，`external` 在捕获前于普通池（S0）分配，
`static_out` 在捕获内进入私有池。每轮先做一次有序重放并校验输出，确保捕获机制本身正确，
然后引入该模式的唯一变量：

| 模式 | 变量 | 预期问题 |
|---|---|---|
| `keep-alive` | external 保持存活；分配同尺寸 replacement（不同地址）并写哨兵 | 校验器对照：重放输出必须完好 |
| `external-free-ordered` | 释放 external → replacement 复用其地址（S0）写哨兵 → 在 S0 上重放 | 有序重放是否读到 replacement 的数据（消费已释放外部存储） |
| `external-free-cross` | 同上，但重放在另一条 stream（无等待） | 跨流无序版本 |
| `pool-release-pending` | 重放入队后立即删除 graph 与 static_out（池 freeable）→ 再分配 replacement 写哨兵 | 待完成重放与池释放/后续分配是否冲突；地址是否重叠（源码预测不重叠：块 npuFree 不回普通池） |

输出分类沿用 P33/P34：`intact` / `fully_consumed_replacement` / `mixed` / `unexpected_values`；
replacement 自身同步后必须仍为全量哨兵。pool-release 模式的观测是地址重叠、replacement 完整性
与进程存活；驱动级 npuFree 与待完成重放的关系无法从 Python 观测，见 RESULTS 边界。

- [结果](RESULTS.md)。

## 远端复现

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
PY=/usr/local/python3.12.13/bin/python
npu-smi info   # 必须空闲

$PY practice_36_graph_pool_boundary/run.py \
  --output /data/tianchi/practice_36_graph_pool_boundary/results/my-formal \
  --modes keep-alive external-free-ordered external-free-cross pool-release-pending \
  --repeats 5
$PY practice_36_graph_pool_boundary/analyze.py \
  /data/tianchi/practice_36_graph_pool_boundary/results/my-formal
```

baseline 配置（`TASK_QUEUE_ENABLE=1`，NPUGraph 不支持 =2）。捕获必须在非默认流上进行
（源码强制），本实验用独立 capture stream。

## 离线分析和验证

```bash
python3 -m unittest discover -s practice_36_graph_pool_boundary -p 'test_*.py' -v
python3 practice_36_graph_pool_boundary/analyze.py practice_36_graph_pool_boundary/results/formal-01
```
