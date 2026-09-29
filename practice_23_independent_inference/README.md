# Practice 23：独立推理任务的串行、双 stream 与 batching

P3a 使用主机已有 **Qwen2.5-0.5B-Instruct 完整预训练权重**，通过 HuggingFace Transformers eager forward 比较相同工作量的三种执行策略。它是可控执行 harness，不是 vLLM 原生请求调度测试。

- `serial`：A、B 两次 forward 进入同一 stream，无中途主机等待。
- `parallel`：A、B 分别进入两条 stream；一个 CPU 线程按 AB / BA 顺序提交。
- `batch`：把相同两条序列合为 batch=2，一次 forward。

每个任务只执行一次完整模型 forward，返回最后一个位置的完整 vocabulary logits 和更新后的全部层 KV。decode 的初始前缀由相同基线预先生成。不是 HTTP 请求、完整生成、TTFT 或服务 SLO 实验。

查看[实测结果](RESULTS.md)、[离线交互执行图](report/index.html)及[验证记录](results/published/validation.json)。

后续已完成[长 prefill 数值定位](NUMERICS.md)：首次分歧来自第 0 层 MLP 投影的 batch 形状变化，附 FP64／精确有理数核验和独立全 FP32 三方对照。原始 BF16 失败状态保持不变。

## 正确性与状态隔离

共享一份只读 BF16 权重（988,065,536 bytes，约 942.29 MiB），A/B 使用不同输入、独立 `DynamicCache`；decode KV 在计时前各自 clone，batch 按 A/B 顺序拼接。mask、position、前缀均提前就绪。默认 RoPE、无 sliding window，模型 `eval()`、`inference_mode()`；模型权重及全部 buffer 前后逐 tensor SHA256 一致。安装版本源码与配置随采集归档。

保留输入、cache 和输出引用，直到两个末尾 event 均完成再校验和释放；检查各任务 live KV 的物理字节范围无交叠。临时算子 workspace 由 torch-npu 管理，本实验没有恢复内部 workspace 的全部分配／访存，也没有采用两个 CPU 线程同时进入模型。

逐次比较两个任务的完整 logits、24 层 K/V（每任务 49 个 tensor）。串行／双流要求完全一致；batch 预设 `atol=0.0625, rtol=0.02`，另要求贪心 token 一致。**1024-token prefill 的 batch 对照未通过这个门限，保留失败，不放宽容差，不采纳该格的收益结论。** 同一个 token 不代表 logits 或 KV 相同。

## 测量边界

四个形状：prefill 128 / 1024，decode context 128 / 1024。每种形状每模式预热 3 次，正式测 12 轮；轮换全部六种模式次序，并交替 AB/BA。每个样本处理两个独立任务，计时包含 forward 提交、event 控制和等待两任务完成；输入／KV 准备、数值核对、权重加载及 profiler 不计入。

公共起始 NPU event 位于默认 stream；执行 stream 先 wait，再 forward、record terminal event。报告 wall pair makespan、每任务 NPU event 就绪时间、主机观察时间与 allocated 峰值。NPU event 间隔包括主机提交不足造成的设备空闲。主机观察时间因提交和等待顺序而可能晚于设备完成；两者不能混用。batch 两个任务同时完成。这里的 tasks/s 只适用于这个固定 forward 单元。

独立 profiler 阶段每格采集两次。CPU scope 只包围 forward 和 event 操作，不安装模型方法包装或逐算子 Python dispatch observer。设备任务使用精确 PyTorch flow、CANN connection、CSV stream/task/start/duration 核对。等待有原生 API 却没有设备 SQE 时保留逻辑屏障，不虚构执行时长。调度 HB 与边界要求分别构造、独立验证；两个 forward 之间没有人为添加 data dependency。

## 运行与重放

在已安装 torch-npu / Transformers 并有本机权重的 Ascend 环境中：

```bash
cd /data/tianchi
python practice_23_independent_inference/run_experiment.py \
  --output practice_23_independent_inference/results/formal-new
```

输出目录必须不存在。机器或软件版本变化后，应先重新做资格检查和源码审计；脚本不修改系统安装。

本地只需 Python 标准库即可重放证据和生成报告（同仓库 Practice 17 / 20 提供 flow / HB 公用逻辑）：

```bash
mkdir -p /tmp/p23-replay
tar -xzf practice_23_independent_inference/results/published/evidence.tgz -C /tmp/p23-replay
python practice_23_independent_inference/analyze.py /tmp/p23-replay/formal-r02
python practice_23_independent_inference/render_report.py \
  /tmp/p23-replay/formal-r02/analysis/execution_graph.json /tmp/p23-report.html
python -m unittest discover -s practice_23_independent_inference -p 'test_*.py'
```

归档独立重放核对 58 个原始文件，执行图与汇总逐字节一致。6 项反事实／数值有效性检查通过；离线浏览器覆盖全部 24 个 trial、节点详情、失败格标记和窄屏布局。

原始 trace 约 270 MB，JSON 分析需要相应内存。证据包含分析必需的原始 trace、kernel CSV、源码、样本、配置，以及资格检查与中断轮次；CANN 二进制采集文件和 profiler 数据库仍在远端，未重复放入仓库。完整执行图用 gzip 保存，交互 HTML 内嵌所选实验节点，不需要联网。

## 文件

| 文件 | 用途 |
|---|---|
| `model.py` | 完整 checkpoint、显式输入／KV 和缓存布局 |
| `run_experiment.py` | 正式测量、逐次校验、独立 profiler |
| `qualify.py` / `qualify_long.py` | 短输入资格检查／长输入差异复核 |
| `trace_graph.py` | 精确设备归因、FIFO、event 与 host join |
| `analyze.py` | 数值有效性、配对耗时、实际计算重叠、HB 检查 |
| `render_report.py` | 离线交互时间线与任务边界图 |
| `test_analysis.py` | 删除同步边后的反事实检查 |

后续可以单独核验 graph replay 或并发 CPU 提交能否减少 eager 供给瓶颈；不能将本次结果外推到这些执行模式。P3b 多模态仍是独立待办。
