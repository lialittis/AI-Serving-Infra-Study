# Ascend 910B2C：实验结果与结论

实验日期：2026-09-18。模型：Qwen2.5-0.5B-Instruct，单卡 BF16、eager。
复现入口：[README](README.md)。本报告使用远端原始 JSON，而不是从终端显示的舍入数字反推。

## 1. 证据与完成范围

| 实验 | 状态 | 证据 |
| --- | --- | --- |
| NPU 矩阵运算 | 通过 | 会话实测，CPU/NPU 结果一致；可用 check_npu.py 复查 |
| Transformers 生成 | 通过 | [日志](results/2026-09-18/logs/transformers-smoke-2026-09-18.log)；npu:0，18 个新 token |
| vLLM 离线批处理 | 通过 | [spawn 成功日志](results/2026-09-18/logs/vllm-smoke-spawn-2026-09-18.log)；两个请求完成，进程正常关闭 |
| HTTP chat API | 用户确认通过 | 会话中确认；原始聊天响应未归档 |
| HTTP completions 压测 | 两组均通过 | 下列原始 JSON，各 20 成功 / 0 失败 |

- [concurrency-1.json](results/2026-09-18/concurrency-1.json)：13:30:31 UTC 输出结果。
- [concurrency-4.json](results/2026-09-18/concurrency-4.json)：13:30:47 UTC 输出结果。
- [environment.json](results/2026-09-18/environment.json)：13:34:49 UTC 事后环境快照与模型指纹。
- [历史 fork 失败日志](results/2026-09-18/logs/vllm-smoke-2026-09-18.log)：记录实际排错过程。

两份原始 JSON 按字节保留，不往其中补写原本没有记录的启动参数。
服务配置根据本实验 `serve.sh` 和用户执行步骤整理，原始服务端启动日志未单独归档；
首次压测客户端完整命令来自会话。新复现脚本会自动记录实际命令和环境。
重建的参数单独记录在 [experiment_parameters.json](results/2026-09-18/experiment_parameters.json)，
其中明确标注了来源，避免与原始工具输出混淆。

## 2. 并发 1 与 4 的对比

两组均为输入 128 token、输出 64 token、20 个正式请求、2 个预热请求。
客户端 request-rate=inf，temperature=0，忽略 EOS，range ratio=0；seed 分别为 1 和 4。

| 指标 | 并发 1 | 并发 4 |
| --- | ---: | ---: |
| 成功 / 失败 | 20 / 0 | 20 / 0 |
| 总输入 / 总输出 token | 2560 / 1280 | 2560 / 1280 |
| 正式测试耗时 (s) | 13.61 | 3.79 |
| 请求吞吐 (req/s) | 1.47 | 5.28 |
| 输出吞吐 (token/s) | 94.06 | 337.66 |
| 平均 TTFT (ms) | 27.32 | 39.21 |
| 平均 TPOT (ms) | 10.36 | 11.41 |
| P99 TTFT (ms) | 28.07 | 47.95 |
| P99 TPOT (ms) | 10.66 | 11.70 |

输出吞吐提高到 **3.590 倍**（约增加 259%）；平均 TTFT 增加约 11.89 ms，
平均 TPOT 增加约 10.1%。单请求生成速度略降，多请求合计产出明显提高。
这是本轮最主要的观察，与批处理改善设备利用率的机制一致，但并未测出具体硬件瓶颈。

按固定输出 64 token 估算：

```text
平均请求耗时 ≈ 平均 TTFT + (64 - 1) × 平均 TPOT
并发 1：约 680.21 ms
并发 4：约 757.81 ms
```

这里使用 JSON 的未舍入均值。TTFT/TPOT 从真正发起 HTTP 请求后计时，
不包含客户端等待并发信号量的时间。上述约 78 ms 的单请求增加与整体 20 请求更快完成并不矛盾。
输出吞吐是全体请求合计，不是单个请求的 token/s；total token throughput 还计入输入 token，
不能当作纯生成速度。

## 3. Peak concurrent requests 的解释

原始结果显示 3 和 8，不能据此断言客户端并发限制失效。
vLLM 0.21 的统计将每个请求计入它活跃过的所有 1 秒时间桶；
一个桶可以包含先后发生、从未同时执行的请求。因此这是秒桶中的活跃请求数，
并非精确的瞬时并发。参考 [vLLM 统计源码](https://github.com/vllm-project/vllm/blob/v0.21.0/vllm/benchmarks/serve.py#L479-L508)。

## 4. KV cache 与模型质量

离线初始化日志报告约 0.93 GiB 权重和 17.25 GiB 可用 KV cache 预算。
后者来自 `gpu_memory_utilization=0.3` 的预算规划，不代表两个请求实际消耗了 17 GiB K/V。
本轮没有采集压测期间的 KV cache 使用率或峰值 HBM 时间序列，不能据此比较并发的内存增量。

模型成功生成不等于回答正确。该 0.5B 模型对 KV cache、prefill 的解释存在概念错误，
本轮仅验证推理功能与性能行为，没有完成模型质量评测。

## 5. 可复现范围与限制

- 样本每组只有 20 个，总时长较短；P99 不作为可靠的尾延迟结论。
- 两组随机输入的长度一致，但 seed 不同，token 内容不完全相同；这是一轮机制学习实验。
- 服务默认前缀缓存仍启用；预热会复用首个样本，正式样本可能有缓存收益。
  不把本结果表述为严格的无缓存 prefill 性能。重复整组且复用 seed 也可能增加缓存命中。
- 小模型、短输入、eager 模式的结果不能外推到长上下文、大模型或图模式。
- 当前卡存在历史 AIC 告警；用户明确要求暂不处理。本轮通过不构成整卡健康或稳定性认证。
- 未执行输入长度扫描、graph/eager 对照、NPU profiling，也未验证 SSH 隧道性能。

后续可先重复每组并扩大样本量，再固定并发 1、输出 64，比较输入 128 / 512 / 1536 的 TTFT 与 TPOT。
若要做严格对照，应控制输入集合、缓存状态、运行顺序和重复次数，并保存逐请求明细及服务端日志。
