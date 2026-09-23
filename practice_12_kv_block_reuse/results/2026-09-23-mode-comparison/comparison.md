# Practice 12：eager / graph 实测对比

主对照：`2026-09-23-run03-eager` 与 `2026-09-23-run04-graph`。请求、模型配置、安装源码、观测脚本和非模式参数一致。

| 项目 | eager | graph |
|---|---|---|
| 执行配置 | eager | PIECEWISE · capture [1] |
| 126-token prefill | eager | compiled callable（设备图 runtime NONE） |
| 1-token decode | eager | 25 个普通分区重放；attention 保持直接调用 |
| A/B 复用 / 层数 | B1 / 24 | B1 / 24 |
| 原生等待后释放、再复用 | 验证通过 | 验证通过 |
| 核对 KV/FIA 链数 | 192 | 192 |
| 主机观测范围数 | 310 | 410 |
| 全窗口设备任务数 | 1542 | 1692 |
| 计算 kernel CSV 行数 | 1374 | 1374 |
| MODEL_EXECUTE 数量 | 0 | 50 |
| 带有效 Model Id 的任务 | 0 | 488 |
| 上述任务缺少 torch flow | 0 | 488 |
| 上述任务缺少 CANN flow 起点 | 0 | 488 |
| 请求 A 输出 |  syntax, |  syntax, |
| 请求 B 输出 |  = class |  = class |

两种模式均验证了24层 KV 的同块复用和原生完成边界。图模式的 attention 分区未被设备图捕获，KV/FIA 保持直接调用，因此192条 KV/FIA 链仍可逐一关联。普通分区重放内部缺少关联，不等于本次 KV 生命周期证据缺失。

图模式新增50个 MODEL_EXECUTE、50个 NOTIFY_RECORD 和50个 NOTIFY_WAIT，总任务数多150；计算kernel CSV行数相同。部分RoPE任务名称变为 `_triton_rope_1`。这些计数不构成逐FX节点映射，也不能据此判断哪种模式更快。主机观测范围多100，是每步新增25个ACL wrapper范围造成的观测差异。

图模式仅捕获尺寸1。prefill 的 runtime NONE 不代表没有编译；这里不验证 FULL graph，也不验证 attention 被完整捕获后的生命周期。

A/B 两请求的采样 token IDs 在模式间相同：`True`。不要求跨进程设备地址相同，只核对每次运行内部 A/B 共享同一存储。

以下耗时包含逐层插桩、profiler、Python记录与串行HTTP开销，只有每模式一次A/B请求，**不是性能基准，不计算加速比**。

| 诊断观察 | eager | graph |
|---|---:|---:|
| A 最后 KV 访问结束 → free入口（µs） | 2693.375 | 2526.025 |
| A free返回 → B allocate入口（µs） | 5535.994 | 5781.515 |
| 请求A HTTP往返（ms，含观测） | 93.222408 | 86.187231 |
| 请求B HTTP往返（ms，含观测） | 90.503423 | 80.730315 |

原始关联见两个run的 `analysis/`，具体重放缺口见各自 `analysis/reuse_evidence.json` 的 `replay_coverage`。
