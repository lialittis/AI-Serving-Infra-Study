# Practice 35：真实 vLLM 采样路径上的复用前置条件测量

P34 在受控探针中给出了损坏不等式（消费端延迟 > 间隔即损坏）。本轮在**真实 vLLM 生成**上
测量该不等式的前半部分是否会出现：离线 `vllm.LLM`（Qwen2.5-0.5B，BF16，eager，temperature=1.0
+ top-k/top-p，固定 seed）在三种引擎变体下运行，[observer](observer.py) 以与安装版本逐行同语义的
包装替换 `random_sample`（不修改安装源码，经 [sitecustomize](sitecustomize.py) 注入引擎进程），
记录每次采样调用的 q 地址、是否复用上一轮地址、以及**此时上一轮主流 `div_` 是否仍未完成**
（P34 竞态前置条件）。

| 变体 | 引擎配置 | 预期 |
|---|---|---|
| `default` | 同步调度，默认采样路径 | 前置条件应≈0：引擎读取采样结果的主机同步在结构上把 div_ 排在下一次 fill 之前 |
| `async` | `async_scheduling=True` | 去除逐步主机同步；前置条件是否出现即本轮问题 |
| `protected` | `enable_async_exponential=True` | 不再进入 `random_sample`（调用数 0），走受保护异步路径 |

另记录同进程两次生成的 token 是否一致（确定性副证据，归因谨慎）。16 个相同 prompt、
max_tokens 48、批内连续合批，构成轻负载。

- [结果](RESULTS.md)。
- [测量汇总](results/formal-01/summary.json)。

## 远端复现

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
PY=/usr/local/python3.12.13/bin/python
npu-smi info   # 必须空闲

$PY practice_35_live_sampling_reuse/run.py \
  --output /data/tianchi/practice_35_live_sampling_reuse/results/my-formal \
  --variants default async protected --prompts 16 --max-tokens 48 --repeats 2
$PY practice_35_live_sampling_reuse/analyze.py \
  /data/tianchi/practice_35_live_sampling_reuse/results/my-formal
```

每变体一个新进程（含 worker spawn）；observer 退出时落盘 `records-<pid>.json`，包含被包装
函数的原始源码与 SHA256。事件查询为 host 侧观测，不重建设备指令顺序。

## 离线分析和验证

```bash
python3 -m unittest discover -s practice_35_live_sampling_reuse -p 'test_*.py' -v
python3 practice_35_live_sampling_reuse/analyze.py practice_35_live_sampling_reuse/results/formal-01
```
