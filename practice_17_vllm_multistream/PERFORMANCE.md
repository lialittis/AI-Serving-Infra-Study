# P0：采样双流的无 profiler 性能闭环

执行日期：2026-09-28。沿用 Practice 17 的两个模型，比较
`enable_async_exponential=true/false`，补齐真实请求完成时间、独立数值检查和
4 / 64 token 的诊断图。关闭组也使用辅助 stream，因此不是“单流对双流”。

直接查看[完整结果表与时间线](results/2026-09-28-p0/README.md)、
[原始数据与图检查摘要](results/2026-09-28-p0/summary.json)，或
[环境复查记录](results/2026-09-28-p0/environment/inventory.json)。

[验证记录](results/2026-09-28-p0/validation.json)：16 项测试通过，八份新证据本地
重放一致，八张图均无环，实验后基础数值检查通过且无遗留实验进程。

## 测量协议

| 项目 | 本次配置 |
|---|---|
| 模型 | Qwen2.5-0.5B-Instruct、Llama-3.2-1B-Instruct |
| 提交 batch | Qwen 1 / 32；Llama 32 / 64 |
| 每请求输出 | 4 / 64 token，`ignore_eos=true` |
| 执行 | TP=1、BF16、eager；关闭 prefix caching、chunked prefill、async scheduling |
| 采样 | temperature=0.8、top_p=0.9；服务器 seed=123，无请求级 seed |
| 服务批次 | 每模型开—关—关—开；每次独立启动，单卡只运行一个实验服务 |
| 重复 | 每批每种负载预热 5 次、正式测量 5 次；每种条件开／关各 10 个正式样本 |
| 计时 | 单调时钟，HTTP 请求发出至完整响应接收；不含编码、解析、文件写入、服务启动及预热 |

客户端和服务端都运行在同一远端容器，通过 `127.0.0.1` 通信；不包含 SSH 或外部
网络往返。这里的 token 吞吐是测量窗口内的完成 token 数 / 请求耗时，不是持续到达
流量下的服务容量。

性能进程不启用 profiler，也不安装 Practice 17 的 `sys.setprofile` 观测。
`PYTHONPATH` 只保留 CANN 运行时所需路径，移除实验观测路径和空路径；同时清除
trace / profiler 环境变量。观测实际 scheduler step 的 trace 在独立服务运行中采集。
因此性能表中的 batch 表示提交请求数，不宣称每个无插桩样本具有相同的 scheduler 分组。

分析保留所有原始样本，不剔除慢样本。报告中位数、四分位区间和 AB / BA 两对的
变化方向；只有两对同向且合并样本的四分位区间不重叠时，才标记“本轮改善 / 退化
方向一致”。这是描述性规则，不是显著性检验，也不是生产 SLO 或 p99 保证。

## 数值与执行证据

独立数值脚本调用安装版本的 sampler 方法，仅将指数随机数生成临时替换成同一个
固定正值 q。实际 NPU stream 切换、event、softmax、除法和 argmax 仍执行，并与
CPU 参考比较。随后恢复真实指数随机生成，检查完成 event 后的 q 有限且大于零。
它不验证 top-k / top-p 过滤精度或整个随机分布，也不要求两次随机生成文本一致。

诊断对 Qwen batch 32、Llama batch 64 的开 / 关和 4 / 64 token 各采一次，共八组。
逐项核对 CPU/CANN flow、kernel CSV、stream、event 和可见 q 数据契约。
trace 中的时间交集不能用于解释无 profiler 性能差值的大小；硬件资源竞争、Host
等待等具体归因需要额外证据。

新 trace 出现了不同 stream 的 kernel 起始时间完全相同、`async_npu` flow ID
也相同的情况。分析器先以 HostToDevice flow 和 connection ID 确定 CANN launch，
再通过实际 dequeue 范围、enqueue/dequeue correlation 和 CPU 调用范围消歧。
不能唯一关联时仍报错，不按最近时间或 kernel 名字猜测。解析器还对 kernel CSV
建立身份索引，避免 64 token 长 trace 的逐行全表扫描；匹配条件保持不变。

长 trace 还出现 `PLACE_HOLDER_SQE`，没有主机 flow，connection ID 为 uint64 的
最大值。此类条目保留为未关联运行时节点，只接已观测的物理 stream 顺序，不归入
计算 kernel、不推断所属 step，也不生成数据依赖。摘要单列其数量与覆盖缺口，
不会为了让“全部关联”检查通过而删除条目。

## 性能结果

本轮共 160 个正式样本，另有 160 个预热样本排除在性能统计外。以下是主负载的
HTTP 完成时间中位数；更小为好。

| 模型 / batch | 输出 token | 提前生成开 / ms | 提前生成关 / ms | 开启后的耗时变化 |
|---|---:|---:|---:|---:|
| Qwen / 32 | 4 | 72.986 | 69.792 | +4.58% |
| Qwen / 32 | 64 | 884.136 | 841.644 | +5.05% |
| Llama / 64 | 4 | 74.934 | 74.326 | +0.82%，不确定 |
| Llama / 64 | 64 | 830.683 | 814.720 | +1.96% |

Qwen 两组主负载和 Llama 64 token 组的 AB / BA 两对方向一致，且四分位区间不重叠，
按预定描述规则属于本轮退化。Qwen batch 1、Llama batch 32 及 Llama batch 64 的
4 token 组均标为不确定；其中 Llama batch 32 的 64 token 中位数虽缩短约 0.83%，
两对方向却相反，不能称为稳定收益。本轮没有证据支持提前生成带来稳定端到端加速。

独立数值检查覆盖 `(batch, vocab)=(4,257)、(32,151936)、(64,128256)`，各测开／关。
六项 argmax 与 CPU 参考精确一致，三项开启路径的固定 q 逐元素一致，真实指数随机
生成的有限性和正值检查全部通过。上述范围不等于完整模型精度或随机分布认证。

机器仍报告历史告警 `80C98001`，基础数值检查通过；性能结论仅适用于本机本轮条件。
当前结果不足以把退化归因于某个特定硬件资源或某一次 host event 等待。

八组诊断共核对 149,991 个计算任务，均与 kernel CSV 唯一匹配。两份 Llama 长请求
各保留一条未关联的占位节点。Qwen 的 4 token 对照在 mixed prefill 中相差一次
`aclnnInplaceCopy_TensorMoveAiCore_TensorMove`，因此不声称两组逐 kernel 工作完全
一致；其他配对的计算任务总数相同，也不据此声称每个内部访存都相同。

| 模型 / 输出 | 开：重叠 step / 总 step | 关：重叠 step / 总 step | 开：compute 交集 / us | 关：compute 交集 / us |
|---|---:|---:|---:|---:|
| Qwen / 4 | 4 / 5 | 0 / 5 | 89.703 | 0 |
| Qwen / 64 | 62 / 65 | 0 / 65 | 1783.168 | 0 |
| Llama / 4 | 4 / 5 | 1 / 5 | 960.061 | 1019.739 |
| Llama / 64 | 64 / 65 | 0 / 65 | 12231.114 | 0 |

Llama 短请求关闭组的交集来自一个较大的 mixed prefill step；交集总量更大不代表
持续 decode 覆盖更好。这些数值来自独立插桩运行，不能与性能表直接相减估算收益。

首轮 Qwen 预启动失败是移除 `PYTHONPATH` 时误删 CANN `acl` 依赖路径，未产生正式
性能样本；修复后在新目录 r02 完成全套测量。随后修正了端口预检查对 TCP 地址复用
的处理，Llama 测量配置和计时逻辑未改变。失败与有效运行保存在不同目录。

## 复现

远端 `/data/tianchi`，使用现有 CANN 环境和 Python 3.12.13。下列输出目录应不存在。

```bash
python practice_17_vllm_multistream/run_sampling_benchmark.py \
  --model /data/huggingface_home/hub/Qwen2.5-0.5B-Instruct \
  --batch-sizes 1 32 --output-tokens 4 64 --port 8017 \
  --output practice_17_vllm_multistream/results/my-p0-qwen

python practice_17_vllm_multistream/run_sampling_benchmark.py \
  --model /data/huggingface_home/hub/Llama-3.2-1B-Instruct \
  --batch-sizes 32 64 --output-tokens 4 64 --port 8019 \
  --output practice_17_vllm_multistream/results/my-p0-llama

python practice_17_vllm_multistream/verify_sampling_numerics.py \
  --output practice_17_vllm_multistream/results/my-p0-numerics

python practice_17_vllm_multistream/run_p0_diagnostics.py \
  --output practice_17_vllm_multistream/results/my-p0-diagnostics
```

以上顺序运行，不在性能测量期间运行诊断。环境不兼容、响应 token 数异常、服务
非正常退出都会中止对应任务，并保留不完整证据；不自动重试混入正式样本。

本地分析不需要 NPU：

```bash
python3 practice_17_vllm_multistream/analyze_sampling_benchmark.py \
  practice_17_vllm_multistream/results/my-p0-qwen
python3 -m unittest discover -s practice_17_vllm_multistream -p 'test_sampling_benchmark.py' -v
```

完整图仍由 `analyze_run.py` 生成，证据压缩与重放复用 `export_results.py` /
`verify_results.py`。`build_p0_report.py` 从两份 benchmark、八份诊断图和数值结果
生成汇总表、逐 step 摘要及代表性真实 kernel 时间线。

本次完整诊断归档在远端
`/data/tianchi/practice_17_vllm_multistream/p0-diagnostics-evidence.tgz`，本地工作副本为
`/tmp/p17-p0-diagnostics-evidence.tgz`，已解压至 `/tmp/p17-p0-diagnostics-evidence`。
其 SHA-256 为 `6e80b0462141fa15fe04854904e3eabc834204a1373025742ddc173a38683b29`；
其他证据位置见 [evidence_locations.json](results/2026-09-28-p0/evidence_locations.json)。

仓库中的 `diagnostics/` 是摘要、清单和 SVG 摘录，完整图 / trace / CSV 留在证据包中；
对完整解压目录执行 `verify_results.py` 才能重放诊断。`benchmarks/` 保留全部性能
样本、源代码快照和启动记录，可直接使用 `analyze_sampling_benchmark.py` 重算。
