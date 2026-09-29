# Practice 26：真实 decode 的计算覆盖、核资源与提交空隙

回答：单 stream 是否导致 core 利用率低？先区分算子规模、任务之间的空隙和必要依赖，再比较真实 eager / PIECEWISE graph 行为。

编号 23–25 已被其他实践占用；本次采用 26。设计见 [PLAN.md](PLAN.md)。完成后的入口为 [实验结论](RESULTS.md)、[离线报告](report/index.html)。

## 实验设置

Qwen2.5-0.5B-Instruct、单卡 Ascend910B2C、BF16、batch 1、固定 10-token 输入、64-token 贪心输出。每次请求重新开始，实际生成时 KV 随 decode 增长；不是固定 KV 的单步重复 replay。prefix caching、chunked prefill、async scheduling、提前随机分支关闭。所有模式都请求 `logprobs=1` 和 token ID 格式，以便逐 token 核验；logprob 计算属于此配置。

- 无 profiler：E/G/G/E 四个独立服务；每组预热 3 次、正式请求 5 次。每模式 10 个完成时间样本。
- plain profiler：每模式独立服务，预热 3 次后采一个完整请求。
- PipeUtilization：再分别启动独立服务，以相同输入采一个完整请求。
- 每条诊断含 1 个 prefill 和 63 个 decode；预先选择 decode 16–47 为稳定分析区域，其他步骤仍可查看。不同位置是描述性观测，不充当独立服务重复。

Graph 是 PIECEWISE、capture sizes=[1]、custom_ops=[all]。与 eager 的比较同时改变编译／融合及执行提交方式，不是只改变 stream 数量的因果对照。

## 代码阅读顺序

1. [run_suite.py](run_suite.py)：四个无 profiler 对照，再运行四个独立诊断。
2. [run.py](run.py)：保存模型和源码指纹，启动自己管理的服务，预热、发请求、开始／停止 profiler，然后关闭服务。
3. [observer.py](observer.py)：仅诊断进程加载。用 import hook 包装 execute/sample/replay；没有全局 Python 调用追踪、LD_AUDIT、额外 device wait 或中间 tensor 回读。graph 捕获结束导出任务结构，JSON 记录缓冲到停止 profiler 时写出。
4. [analyze.py](analyze.py)：关联真实 flow、runtime connection、CSV 和 graph 任务；计算每步区间、空隙、核数字段及提交时间关系。
5. [summarize.py](summarize.py)：核验配置与输出，独立汇总无 profiler 完成时间。
6. [render_report.py](render_report.py)：输出离线交互 HTML 和四张代表 decode SVG。

Pipe 组仅在本次进程创建 profiler 时替换 `aic_metrics`；其余参数保留已安装 vLLM-Ascend 的 profiler 构造选项。系统库不修改。无 profiler 组不注入观察器。只管理自己的服务进程；NPU 已有负载时拒绝开始。

## 测量边界

- HTTP 完成时间包含 prefill、生成、logprobs 和响应处理，不是 isolated decode latency，也不是流式 token 到达间隔。
- 设备步骤窗口是该步首个关联设备任务开始到最后一个完成；步骤之间的间隔另计。CPU execute/sample 区间也显示，不把 Python 返回当成设备完成。
- 计算覆盖取实际计算任务区间的并集，包含 profiler 报告的 AI_CPU 计算，核类型分别保留。它不表示全芯片利用率。
- 等待／拷贝在“无计算”区间上的覆盖可能互相交叠，不能相加成总开销。
- 未被记录任务覆盖的空隙不直接称为硬件 idle。记录空隙后任务的直接提交或 graph replay 时间，区分原生调用尚未开始与已经返回，但不单凭此认定原因。
- graph 内算子关联到所属 replay，没有本次逐 kernel 的新 CPU 下发；捕获 dump 的示意 ts/dur 不用于计时。
- Block Num=0 或缺失显示未知；Mix Block Num 单列。流水线 ratio 不是整卡占用率、物理核时间线或带宽饱和证明。

## 远端复现

在 `/data/tianchi`，使用已安装库和本地权重。输出目录必须不存在；默认端口 8026。

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export PATH=/usr/local/python3.12.13/bin:$PATH
python practice_26_decode_utilization/run_suite.py \
  --output practice_26_decode_utilization/results/new-run
```

单独采集示例：

```bash
python practice_26_decode_utilization/run.py \
  --mode graph --profile plain --requests 1 \
  --output practice_26_decode_utilization/results/new-graph-plain
python practice_26_decode_utilization/analyze.py \
  practice_26_decode_utilization/results/new-graph-plain
```

## 本地离线重放

只需 Python 标准库及同仓库 Practice 17 的区间／flow 辅助函数；不需要 NPU。

```bash
mkdir -p /tmp/p26-replay
tar -xzf practice_26_decode_utilization/results/published/evidence.tgz -C /tmp/p26-replay
for mode in eager graph; do
  for profile in plain pipe; do
    python practice_26_decode_utilization/analyze.py \
      "/tmp/p26-replay/2026-09-29-run01/${mode}-${profile}"
  done
done
python practice_26_decode_utilization/summarize.py /tmp/p26-replay/2026-09-29-run01
python practice_26_decode_utilization/render_report.py /tmp/p26-replay/2026-09-29-run01 --output /tmp/p26-report
P26_REPLAY_ROOT=/tmp/p26-replay/2026-09-29-run01 \
  python -m unittest discover -s practice_26_decode_utilization -p 'test_*.py' -v
```

HTML 内嵌压缩数据，现代浏览器可直接离线打开；包含所有 256 个采集步骤、任务详情和原始核数字段。

归档保留请求响应、源码／采集器快照、版本与模型哈希、graph dumps、原始 trace/CSV 和 profiler 元数据。大型 CANN 原始缓冲、数据库与编译缓存留在远端；编译缓存另存哈希清单。`results/published/manifest.json` 覆盖可重放归档；练习根目录 `SHA256SUMS` 覆盖交付文件。提取后的大目录被 Git 忽略，避免与压缩归档重复提交。
