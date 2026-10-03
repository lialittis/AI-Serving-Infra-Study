# Practice 33：地址复用后的实际冲突访问

[P32](../practice_32_allocator_lifetime/README.md) 证明了 omit/join 两组在旧 copy 设备任务未完成时
就把 A 的地址交给候选 B，但 B 当时没有任何设备读写，因此不能定性为竞态或漏洞。本实验补上缺失的
一半：**B 真正在 S0 上写入该地址，然后在全部设备工作完成后逐元素检查旧 copy 的输出 `observed`**。

四种模式只在生命周期保护和写入时机上不同，其余工作负载与 P32 baseline 一致：

| 模式 | 释放前保护 | B 的写入 | 预期 |
|---|---|---|---|
| `omit` | 无 | 窗口内立即写 | 旧 copy 可能读到 B 的哨兵值 |
| `record` | `A.record_stream(S1)` | 窗口内立即写（地址不同） | 无提前复用，observed 保持 A 值 |
| `join` | S0 `wait_event(done)` | 窗口内立即写（复用地址） | 设备顺序保护，observed 保持 A 值 |
| `synced` | 无 | 全部完成后才写 | 校验器对照：observed 必须保持 A 值 |

`observed` 的元素分类只有三种出口：全部为 A 原值（intact）、全部为哨兵
（fully_overwritten，旧 copy 消费了 B 写入后的存储）、两者混合（mixed）。写入目标自身在同步后
必须仍是完整哨兵。地址复用、事件进度和 allocator 状态的采集方式沿用 P32。

- [结果和证据边界](RESULTS.md)。
- [无 profiler 正式矩阵](results/formal-01/summary.json)：四模式 × 10 轮。
- [设备顺序证据](results/profile-01/index.html)：omit/record/join × 2 轮独立 profiler，
  点击条目查看原始 task / flow 身份。

## 工作负载和测量边界

与 P32 相同：S1 使用独立 FP16 `[4096,4096]` 矩阵提交 64 次预热过的 `torch.mm(..., out=scratch)`，
随后把 A（4 MiB FP32，值为 `index+1`）复制到一直存活的 `observed`；最后一个 A 引用在 S1 当前时
销毁。A 的初始化在 backlog 之前完成，S1 显式等待生产事件；`join` 在释放前让 S0 等待 S1 末尾事件。

本实验新增：S0 最多分配 8 个同尺寸候选；找到复用 A 地址的候选（omit/join 通常为第一个）后，
在观测窗口内于 S0 上 `fill_(777.0)`。`synced` 模式候选分配与查找路径相同，但 fill 推迟到
`consumer.synchronize()` + `owner.synchronize()` 之后。窗口内不读回设备值、不写文件、不调用
`empty_cache`；元素分类在全部完成后的 `validate` scope 中进行。

测量窗口内不读取或 hash stream handle（沿用 P32 的 `npu_stream` 排空教训，handle 事先缓存）。
每模式一个新进程，每进程 10 轮；profiler 单独进程运行（`record_shapes=False`、
`profile_memory=False`），不作为性能基准，也不与无 profiler 结果混排。

内存上界：4096 FP16 矩阵 96 MiB、候选每个 4 MiB（≤8 个）、A/observed 各 4 MiB；另有历史缓存与
runtime/workspace。未做 B 的主机侧读写，未做逐指令访问观测；设备任务区间是 profiler 上报值。

## 远端复现

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
PY=/usr/local/python3.12.13/bin/python
npu-smi info   # 必须空闲

$PY practice_33_conflict_access/run.py \
  --output /data/tianchi/practice_33_conflict_access/results/my-formal \
  --modes omit record join synced --repeats 10
$PY practice_33_conflict_access/analyze.py \
  /data/tianchi/practice_33_conflict_access/results/my-formal

$PY practice_33_conflict_access/run.py \
  --output /data/tianchi/practice_33_conflict_access/results/my-profile \
  --modes omit record join --repeats 2 --profile
$PY practice_33_conflict_access/analyze.py \
  /data/tianchi/practice_33_conflict_access/results/my-profile
$PY practice_33_conflict_access/analyze_profile.py \
  /data/tianchi/practice_33_conflict_access/results/my-profile
$PY practice_33_conflict_access/render.py \
  /data/tianchi/practice_33_conflict_access/results/my-profile
```

配置固定为 P32 的 baseline（`TASK_QUEUE_ENABLE=1`、`PER_STREAM_QUEUE=0`、
`expandable_segments:False,multi_stream_lazy_reclaim:False`），单 host 线程；P32 已证明该维度
不改变结论方向。实际环境、安装版本、source hashes、命令、PID、设备状态和每轮地址/事件/快照/
元素分类均保存。

## 离线分析和验证

```bash
python3 -m unittest discover -s practice_33_conflict_access -p 'test_*.py' -v
python3 practice_33_conflict_access/analyze.py practice_33_conflict_access/results/formal-01
python3 practice_33_conflict_access/analyze.py practice_33_conflict_access/results/profile-01
python3 practice_33_conflict_access/analyze_profile.py practice_33_conflict_access/results/profile-01
python3 practice_33_conflict_access/render.py practice_33_conflict_access/results/profile-01
```

原始 profiler 数据保留在远端，Git 仅包含代码、摘要、报告与 SHA256 清单；恢复与复核命令见
[RESULTS](RESULTS.md)。
