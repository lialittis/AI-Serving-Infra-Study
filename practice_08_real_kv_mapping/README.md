# Practice 08：真实 token 怎样映射到 NPU 上的 KV block？

沿用 Practice 07 的真实 vLLM / vLLM-Ascend 服务，只新增对 KV 映射链路的观测。
本实验不实现自己的 allocator，不修改已安装源码。

已于 2026-09-22 在远端运行完成：输入 126 个 token、输出 8 个 token，
观测到 block table 从 `[1]` 增长到 `[1, 2]`，第一层 133 个位置的实际 K/V 写入校验全部通过。
先看 [RESULTS.md](RESULTS.md)，再看 [原始事件汇总](results/2026-09-22-run01/summary.md)。

## 要核对的链路

```text
KVCacheManager.allocate_slots
    ↓ 本 request 的 block IDs（按 KV group）
Scheduler → NPUModelRunner
    ↓ CPU block table
AscendAttentionBackendImpl.forward
    ↓ NPU block table + slot_mapping + KV tensor
DeviceOperator.reshape_and_cache
    ↓ 写入后的第一层 KV 数据
```

所有层面的观测必须相互吻合：

```python
logical_block = token_position // block_size
physical_block = block_table[logical_block]
offset = token_position % block_size
slot = physical_block * block_size + offset
```

`token_position` 是 request 内从 0 开始的位置，`token_id` 是词表编号，二者不同。
`slot` 是展平后的 token 槽编号，不是设备物理内存地址。

## 为什么用 126 个输入 token？

当前后端 block size 是 128。我们用本地 tokenizer 将 `hello` 编成一个 token ID，
然后在 HTTP 请求中直接传入这个 ID 重复 126 次的列表，准确控制输入长度。
输出固定为 8 个 token，关闭 EOS 提前结束。

第一步处理位置 0～125；后续步骤处理 126、127、128……132。
因此第 4 个调度步会跨过边界，必须使用第二个逻辑块。
最后一个输出 token 位于 133，但请求已结束，不再为它执行 forward。
这个输入用于验证映射，不用于评价生成文本质量。

## 在远端复现

目录依赖：保留同级 `practice_03_ascend_start/collect_environment.py` 和
`practice_07_real_request_trace/trace_hooks.py`。运行使用远端 Python 3.12。

```bash
cd /data/tianchi
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export PATH=/usr/local/python3.12.13/bin:$PATH
npu-smi info

python practice_08_real_kv_mapping/run_mapping.py \
  --output /data/tianchi/practice_08_real_kv_mapping/results/my-new-run
```

确认没有其他模型服务占用这张卡；输出目录必须不存在。
脚本自动启动 `127.0.0.1:8008` 服务、发送唯一一个推理请求、保存结果并关闭自己的服务。
基线固定单卡 BF16、eager、block_size=128、max_num_seqs=1、max_model_len=2048；
关闭 prefix caching、chunked prefill 和 async scheduling。

## 新代码怎样阅读

1. `run_mapping.py`：与 Practice 07 启动脚本相同的结构，增加精确长度的 token-ID 输入。
2. `sitecustomize.py`：通过 `P08_TRACE_DIR` 开关加载本次追踪器。
3. `kv_trace.py`：复用 Practice 07 的基础事件，增加 allocation、block table、slot 和 KV 写入观测。
4. `summarize_mapping.py`：用各层真实记录相互校验，并生成逐 token 的 `token_mapping.csv`。

先阅读 `kv_trace.py` 中的三个观测点：`allocate`、`first_attention`、`cache_write`。
它们分别回答“拿到哪些块”“算子收到哪些地址信息”“写入后数据在哪里”。
检查器不接受仅打印公式计算结果作为证据；它同时要求设备索引和实际数据校验通过。

## 观察的侵入性与范围

本次与 Practice 07 有明确区别：为了读取真实 NPU 索引以及确认 KV 写入，
追踪器会调用 `.cpu()`，引入设备到主机的数据复制和同步。

- 每步只观察第一层 attention，复制有效 block table / slot 索引。
- 写入后仅复制该步触及的第一层 KV blocks，并与写入算子的 K/V 输入逐元素比较。
- 不复制整个 KV pool，不修改缓存内容，不记录完整 K/V 数值。
- 每步比较都应得到 `key_equal=true`、`value_equal=true`。
- 该运行会改变执行时间和异步重叠，不用于吞吐、kernel 耗时或 race 检测结论。
- 当前检查针对 Qwen2.5-0.5B 的单 full-attention KV group、128-token blocks、每 token 两个 KV heads、head_dim=64。
  遇到不同布局或版本时明确失败，需要重新检查源码，而不是套用同一公式的存储布局假设。

block ID 标识池内的 block 槽；`data_ptr` 是 tensor 暴露的设备指针，不能解释为芯片物理地址。
设备 kernel 的精确名称、时间线和其他层的行为留给后续实验。

## 本地查看和验证（不需要 NPU）

```bash
python3 practice_08_real_kv_mapping/summarize_mapping.py \
  practice_08_real_kv_mapping/results/2026-09-22-run01
python3 -m unittest discover -s practice_08_real_kv_mapping -p 'test_*.py' -v
```

汇总器生成 `summary.md` 和 `token_mapping.csv`，并检查各观察点的一致性。
6 项测试基于真实归档及人为损坏的副本，验证错误的 block table、slot、KV 数据校验结果
和缺失事件都会被拒绝。

归档还保存命令、环境、脚本快照、源码指纹、HTTP 请求/响应及设备结束状态。
本地与远端 `/data/tianchi/practice_08_real_kv_mapping` 有相同工作目录副本。
在实验目录中执行 `sha256sum -c SHA256SUMS` 可检查受管理文件完整性。
