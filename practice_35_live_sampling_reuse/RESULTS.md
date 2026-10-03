# P35 结果：真实 vLLM 采样路径上的复用前置条件

**[VERIFIED] 在真实 vLLM 生成（Qwen2.5-0.5B，BF16，eager，temperature=1.0 + top-k/top-p，固定
seed，16 并发请求 × 2 次重复）上，默认采样路径每一部都在复用上一轮 q 的地址（64/68），但
P34 竞态的前置条件——下一轮 fill 提交时上一轮主流 `div_` 尚未完成——在同步与
`async_scheduling` 两种引擎模式下均为 0 次。** 真实步进间隔中位数约 12.0–12.4 ms，远大于
采样消费端的实际延迟；引擎结构（`div_` 与后续 forward 同流排序、采样结果逐步回读）使
P34 的时序余量在实测负载下恒为正。`enable_async_exponential=True` 的 protected 变体
0 次进入 `random_sample`，开关有效。三变体的两次重复生成 token 完全一致。

## 环境与测量方式

2026-10-03 在 Ascend910B2C（物理设备 5）运行，vllm `0.21.0+empty`（`ad7125a`）、
vllm-ascend `0.21.0rc1`（`80610e44`）、torch `2.10.0+cpu` / torch-npu `2.10.0`、CANN 9.0.0。
离线 `vllm.LLM`（enforce_eager、`max_model_len=1024`、`max_num_seqs=16`、`gpu_memory_utilization=0.3`），
`SamplingParams(temperature=1.0, top_k=50, top_p=0.9, max_tokens=48, seed=12345)`。
[observer](observer.py) 经 [sitecustomize](sitecustomize.py) 注入 EngineCore/worker 进程，以与安装版本
逐行同语义的包装替换 `random_sample`；被包装函数的原始源码及 SHA256 随记录落盘
（`original_source` 字段）。设备健康状态与 P31–P34 相同（既有 `Alarm / 80C98001`，前后一致）。

| 证据 | 变体 | 采样调用 | 地址复用 | 前置条件 | 步进间隔（ms） | 重复一致 |
|---|---|---:|---:|---:|---|---|
| [formal-01](results/formal-01/summary.json) default | 同步调度 | 68 | 64 | **0** | 12.0 / 12.4 / 21.6（min/中位/max） | 是 |
| formal-01 async | `async_scheduling=True` | 68 | 64 | **0** | 11.5 / 11.8 / 20.9 | 是 |
| formal-01 protected | `enable_async_exponential=True` | 0 | — | — | — | 是 |
| [smoke-01](results/smoke-01/summary.json) default | 4 请求先导 | 17 | 15 | 0 | — | — |

68 次调用 = 2 次重复 × 34 个采样步；每轮首次调用无前驱，故复用分母为 64。

## 读数与归因

- **地址链式复用是常态**：除每轮第一次外全部复用上一轮地址（64/64），与 P32/P34 的
  allocator 行为预测一致——同 stream 池内即刻复用在生产路径上同样发生。
- **前置条件 0 次的原因 [INFERRED — 结构]**：`div_` 与该步全部采样算子同在主流，下一步
  forward 也在主流排队其后；host 在采样结果回读（D2H）前不推进下一步调度。因此下一轮
  fill 提交时（间隔约 12 ms），上一轮 `div_` 早已执行完毕。async 模式下引擎最多提前约一步，
  主流上的顺序关系仍然成立，本负载未见反转。
- **与 P34 不等式的对接**：损坏需要消费端延迟 > 间隔。实测间隔约 12 ms，消费端在该结构下
  实际延迟远小于一步（µs–ms 量级且先于下一步 forward），余量恒为正。P34 中 b32（约 13 ms
  延迟）恰与本负载间隔同量级——若主流出现超过一步的设备积压（更深合批、重负载、或未来
  引擎更激进的提前提交），余量仍可能翻负，本轮未构造出该场景。
- **确定性副证据**：三变体重复生成完全一致；负载下未见 P31 那类 greedy 变体。这不构成
  竞态不存在的证明（前置条件本身未发生），仅与主结论一致。

## 边界

- 单卡、单模型、16 并发、48 token 的轻负载；未测重负载/长生成/大 batch 下的主流积压。
- 前置条件由 host 侧事件查询判定（`Event.query()`），不重建设备指令顺序；若查询时刻与
  fill 提交之间 `div_` 恰好完成，前置条件可能被低估一个极小窗口。
- observer 包装与安装实现语义一致，但运行的是包装后的代码；其开销为 host 侧记录与
  事件创建，不改变 stream 结构。
- protected 变体未插桩异步路径内部（该路径已有 P17 实测与源码分析）。

离线校验通过：50 个归档文件大小及 SHA256、summary 重算一致、3 个分析单元测试。
原始数据保留在远端 `ascend910:/data/tianchi/practice_35_live_sampling_reuse/results/`。

## 结论

P32→P35 证据链闭合：地址提前交回（P32）在无保护时必然损坏（P33）、损坏条件是
消费端延迟超过间隔（P34），而**真实引擎的当前结构使该条件在实测负载下不成立**
（P35：复用每步发生、前置条件 0、间隔约 12 ms）。默认采样路径的安全性依赖引擎结构与
负载余量而非显式保护；任何让主流积压超过一步间隔的改动都可能翻转余量。
结构性修法（先导等待或 `enable_async_exponential`）已被 P34 证明在全部间隔下有效。
