# Practice 34：fill_exponential 复用模式的间隔扫描

[审计](../references/cross_stream_call_site_audit/README.md)判定 vllm-ascend 默认采样路径
`fill_exponential`（`enable_async_exponential=False` 时的非 greedy 采样）在结构上与 P33 的
omit 模式等价：q 在 global_stream 分配写入、主流消费后即释放、下一轮无先导等待直接再分配。
本实验把"时间性缓解"变成定量结论：**扫描两轮之间的间隔，测损坏概率如何随间隔变化**，
并同时评估两种修复（`record_stream` 登记与先导等待）的有效性。

## 结构与替代说明

每轮迭代复刻 fill_exponential 的调用结构：

1. producer stream（global_stream 模拟）：`empty` + `fill_(轮次号)`（替代 `exponential_`，
   保持同 stream 同提交语义的设备写任务，便于元素分类）；
2. 主流 `wait_stream(producer)`（对应 [sampler.py:41](../references/cross_stream_call_site_audit/sources/vllm-ascend/vllm_ascend/sample/sampler.py#L41)）；
3. 主流先执行 `consumer_backlog` 个矩阵乘（模拟采样链中 `div_` 之前的算子与主流拥堵），
   再 `copy_(q)` 到预分配、永不释放的 `observed[i]`（`probs.div_(q)` 的读端模拟）；
4. 删除 q（"函数返回"释放）；
5. host `sleep(gap)` 模拟调度/forward 间隔，期间无任何同步。

每个间隔结束后统一同步并逐元素分类：`observed[i]` 应为轮次号 `i+1`；出现其他值即旧读端
消费了后续轮次写入的数据。两个自变量：**消费端延迟**（backlog 0/8/32 个矩阵乘）与**两轮间隔**
（0–50 ms）。三种模式：

| 模式 | 差异 | 目的 |
|---|---|---|
| `asis` | 与默认路径结构一致 | 测损坏概率 vs（延迟, 间隔） |
| `record` | 释放前 `q.record_stream(main)`，跑最大 backlog | 评估登记修复 |
| `leading-wait` | fill 前 `producer.wait_stream(main)`，跑最大 backlog | 评估先导等待修复（do_async_exponential 的做法） |

先导 smoke（smoke-01/02）已确认：backlog=0 时间隔 0 也 0/4 损坏（消费端在照片终点稳赢），
backlog=32 时 gap 0 与 1 ms 均 3/3 可损坏轮次全损坏——损坏需要"消费端被延迟"这一现实条件。

- [结果](RESULTS.md)。
- [间隔矩阵](results/formal-01/summary.json)。

## 远端复现

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
PY=/usr/local/python3.12.13/bin/python
npu-smi info   # 必须空闲

$PY practice_34_sampling_gap/run.py \
  --output /data/tianchi/practice_34_sampling_gap/results/my-formal \
  --modes asis record leading-wait --gaps-us 0 200 1000 5000 25000 50000 \
  --iterations 20 --backlogs 0 8 32
$PY practice_34_sampling_gap/analyze.py \
  /data/tianchi/practice_34_sampling_gap/results/my-formal
```

配置固定 P32/P33 的 baseline，每模式一个新进程。每轮记录地址、相邻复用、逐元素分类；
allocator 统计与首个地址的分配/释放历史一并保存。间隔为 host 睡眠，不在主流上排设备工作；
这一简化与真实采样步的差异在 RESULTS 的边界一节讨论。

## 离线分析和验证

```bash
python3 -m unittest discover -s practice_34_sampling_gap -p 'test_*.py' -v
python3 practice_34_sampling_gap/analyze.py practice_34_sampling_gap/results/formal-01
```
