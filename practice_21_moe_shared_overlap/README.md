# Practice 21：真实 MoE shared-expert 分支与双 stream

[P2 实测结论](RESULTS.md) · [离线交互执行图](report/index.html) · [验证结果](results/published/validation.json)

本实验复用已安装 vLLM 的 `Qwen2MoeSparseMoeBlock` 与 vLLM-Ascend 的
`AscendFusedMoE`，比较单卡 eager 下关闭／开启 shared-expert overlap 的行为。
这是可控权重的真实模型单层，不是通用双 MLP，也不是完整预训练模型的端到端测试。

## 实验范围

- BF16，TP=DP=EP=1；hidden 1024，8 个 routed expert，top-2。
- routed intermediate 512；shared intermediate 1024，包含原生 sigmoid gate。
- token 数 1、32、256、1024；路由、分发、grouped matmul、激活、归并使用原生 backend。
- 关闭 gate 多流、量化、EPLB 和 graph；仅切换 shared-expert 的原生多流开关。
- 权重与输入由固定 CPU 随机种子生成；既有 dense 模型路径只用于配置中的 tokenizer
  占位，`skip_tokenizer_init=True`，不读取它的权重。

`model.py` 完成原生配置、分布式组、custom-op、weight-prefetch 与权重后处理初始化。
`run_experiment.py` 先验证数值与输入复用，再进行无 profiler 性能测量，最后安装
`observer.py` 采集方法边界和原生 event。每次调用重新建立 forward context，遵守当前
vLLM 的 MoE layer index 约定，不修改系统安装。

## 三类独立证据

1. 性能：四种模式（serial、parallel、shared_only、routed_only），每形状每模式 12 组，
   每组 20 次调用；交替反转顺序。计入 context、提交、原生同步和完成等待。
2. 详细诊断：24 个 forward，观察实际 stream、tensor 边界、event handle 与 host scope。
   `analyze.py` 对照原生 flow 和 kernel CSV，重建同步图，再独立验证数据要求。
3. 轻量诊断：`capture_light.py` 只添加整次 forward 的 profiler marker，不安装方法包装器
   或 TorchDispatchMode。`analyze_light.py` 通过原生 MoE CPU scope、物理 stream 和精确
   task 身份计算重叠，检查详细观察器是否掩盖了并发。

图中的 `runtime_wait` 是没有物理 EVENT_WAIT SQE 的原生等待 API 约束，不是设备任务，
没有虚构设备时间。边界数据 DAG 也不等于 native workspace 的精确访存 DAG。
源码版本由 `contracts.json` 固定，变化后须重新核对语义。

## 实机复现

在已有 Ascend 环境的 `/data/tianchi` 根目录执行，保留 CANN 运行时 PYTHONPATH。
先核对设备没有其他任务；使用新的输出目录，避免覆盖旧证据。

```bash
python practice_21_moe_shared_overlap/run_experiment.py \
  --output practice_21_moe_shared_overlap/results/new-formal
python practice_21_moe_shared_overlap/capture_light.py \
  --output practice_21_moe_shared_overlap/results/new-light
```

正式采集默认包含数值、72 次输入 buffer 连续复用、192 组性能样本与 24 次详细诊断。
轻量采集独立启动，另生成 24 次诊断。失败目录保留原始错误；不将资格试采混入正式统计。

## 本地离线重放

不需要 PyTorch/NPU，仅使用 Python 标准库及仓库中 Practice 17/20 的图工具。

```bash
mkdir -p /tmp/p21-replay
tar -xzf practice_21_moe_shared_overlap/results/published/formal-evidence.tgz -C /tmp/p21-replay
tar -xzf practice_21_moe_shared_overlap/results/published/light-evidence.tgz -C /tmp/p21-replay
python practice_21_moe_shared_overlap/analyze.py /tmp/p21-replay/formal-r01
python practice_21_moe_shared_overlap/analyze_light.py \
  /tmp/p21-replay/light-r01 /tmp/p21-replay/formal-r01
python practice_21_moe_shared_overlap/render_report.py \
  /tmp/p21-replay/formal-r01/analysis/execution_graph.json /tmp/p21-report.html
python -m unittest discover -s practice_21_moe_shared_overlap -p 'test_*.py' -v
```

五项测试包含从真实图移除输入等待、输出 join 或 API-only barrier 后，要求同步证明失败；
没有通过删掉设备同步再运行错误 kernel 的方式测试。

离线浏览器检查可设置 `P21_BROWSER_PACKAGE` 指向本机 Playwright 包，执行
`node practice_21_moe_shared_overlap/check_report.cjs`。覆盖 24 个试验选择、阶段与 kernel
证据、连线过滤、时间缩放及移动端布局。测试记录在
[offline_validation.json](results/published/offline_validation.json)。

## 结果与局限

本轮双流计算确实进入不同的物理 stream，数据依赖均有同步保证；详细与轻量诊断的
shared/routed kernel 区间交集均为零，无 profiler 完整层耗时的配对中位数增加约 7%–9%。

该结论仅适用于这里的层尺寸、dtype、单卡 eager 和安装版本，不代表更大模型、graph、
多卡通信或量化路径的收益。profiler 本身仍可能影响提交时间，未对无 profiler 的逐项开销
进行归因。设备的历史 Alarm 仍在，数值通过不能认证硬件健康。
