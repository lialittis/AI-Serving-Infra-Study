# Practice 15：eager / PIECEWISE 执行图对照

同一模型、BF16、单卡、10 输入 / 4 输出、同脚本与源码，重新采集 eager 与 PIECEWISE capture [1]。prefill 执行编译 callable；三个 decode 各重放 25 个普通分区，attention 保持直接调用。

| 项目 | eager | graph |
|---|---:|---:|
| 设备任务（不含 profiler 控制） | 1444 | 1669 |
| 计算 kernel CSV 行数 | 1279 | 1279 |
| 物理 stream 数 | 1 | 26 |
| 具有两条精确 host flow 的任务 | 1444 | 787 |
| 关联到异步队列的任务 | 1444 | 787 |
| NPUGraph.replay 次数 | 0 | 75 |
| MODEL_EXECUTE 任务 | 0 | 75 |
| 有 runtime connection 的 replay 边界任务 | 0 | 150 |
| 有 Model Id、缺少逐算子 flow 的任务 | 0 | 732 |
| 原生结果完成边界 | 4 | 4 |
| Attention 调用（24 层 × 4 步） | 96 | 96 |
| 本次观察到的 Triton 编译 | 8 | 8 |
| 正式请求期间编译 | 0 | 0 |
| RoPE 的确定 RAW 边 | 192 | 48 |
| KV 存储候选边 | 144 | 144 |

输出 token IDs 一致：`True`。输出：` 天空之所以`。

26 条物理 stream 不等于 26 路计算并行。缺少图内 flow 时不建立逐 kernel 的 Python/FX 归属；NOTIFY 任务没有配对证据，不补画跨流同步边。无数据边表示未覆盖，不表示独立。本实验带插桩、只有一个正式请求，不报告性能加速比。

[打开 eager 图](../2026-09-28-eager-run01/analysis/index.html) · [打开 graph 图](../2026-09-28-graph-run01/analysis/index.html)
