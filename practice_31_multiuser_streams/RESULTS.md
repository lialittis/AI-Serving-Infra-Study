# P31a 结果：多用户原生合批，单条 stream 串行执行

2026-10-02 在现有 Ascend 主机完成了 1/2/4/8 用户的单服务 eager 实验。**独立 HTTP 请求确实并发，vLLM 将请求合入共享批次；四个诊断案例的计算任务均在物理 stream 46 上，跨 stream 计算重叠为 0。** 这是本轮模型、greedy 和关闭辅助并发功能的配置结果，不代表所有 vLLM 模式都只使用一条 stream。

完整交互报告保留在本地和远端归档中，不上传 Git；新 checkout 请先按 [报告恢复步骤](README.md#完整报告与大文件) 取回。在本地打开 `results/formal-01/index.html`，建议从八用户诊断阅读。选择一个请求，再选择一个八请求的 decode step；点击 NPU kernel 可查看对应 CPU 算子、CANN 提交、flow ID 和原始 trace/CSV 位置。无需恢复大文件也可阅读下述结论和提交中的小型证据。

## 采集范围与完整性

配置为 Qwen2.5-0.5B-Instruct、单卡 TP1、BF16、eager。各用户初始屏障启动，之后各自完成一个请求再发下一个；每用户3个独立请求，每个输入128 tokens、输出64 tokens。prefix caching、chunked prefill、async scheduling、异步采样预计算关闭。不同档位服务容量保持 max_num_seqs=8。

共8个案例，**135个无插桩 benchmark 请求 + 45个 diagnostic 请求 = 180个测量请求**，全部返回完整64 tokens。预热不计入这些数量。benchmark 每档重复3次，diagnostic 每档一次；每个案例启动独立服务。

| 诊断用户数 | 请求数 | 调度 steps | 批次内请求数分布 | 计算 kernel 数 | 计算 streams | 计算重叠 |
|---:|---:|---:|---|---:|---:|---:|
| 1 | 3 | 192 | 1请求：192 steps | 57,309 | 1 | 0 µs |
| 2 | 6 | 195 | 1请求：6；2请求：189 | 72,186 | 1 | 0 µs |
| 4 | 12 | 195 | 1请求：3；3请求：3；4请求：189 | 72,552 | 1 | 0 µs |
| 8 | 24 | 195 | 1请求：3；7请求：3；8请求：189 | 72,363 | 1 | 0 µs |

四个诊断中，HTTP 峰值在途请求数分别达到1/2/4/8；设备任务均位于物理 stream 46。对框架已有 stream 对象的查询将其关联到 Python stream ID 0。copy/compute 重叠也均为0；这些数字不表示整个服务生命周期中没有其他已分配但未执行任务的 stream。

全部 **274,410 个计算 kernel** 完成 trace/CSV 身份核对，并关联到所属执行 step。45个诊断请求全部完成客户端、前端、内部 ID 映射；每请求的 scheduled tokens 均为 `128 + 64 - 1 = 191`，没有依赖“全局必须固定192 steps”的假设。各案例无 observer error、无未解析 flow、无同 stream 计算区间相交。状态及覆盖率见各报告的 `analysis/summary.json`。

各档位额外抽查了一个 prefill/混合批次和一个纯 decode 批次的 MatMul：原始 trace、CSV、CPU 算子及 native connection 一致。例如八用户：

- prefill/混合批次 MatMul：stream46、Task ID54287、trace index1078349。
- 纯 decode MatMul：stream46、Task ID54720、trace index1079189。

具体原始行见 [八用户抽查证据](results/formal-01/c8-diagnostic/analysis/raw_spot_checks.json)。这里的请求集合是**所属批次**，不证明一个 kernel 一定读写批次内所有请求的数据。

## 无插桩性能

下表仅来自 benchmark。吞吐是每次重复中总输出 tokens / 所有请求的完整客户端时间窗口，再对3次重复取平均；延迟和 TTFT 为该档位所有测量请求的中位数。

| 用户数 | 请求数 | 输出 tokens/s | TTFT 中位数 | 完整请求延迟中位数 |
|---:|---:|---:|---:|---:|
| 1 | 9 | 91.3 | 14.09 ms | 701.39 ms |
| 2 | 18 | 174.6 | 22.60 ms | 720.57 ms |
| 4 | 36 | 338.2 | 23.99 ms | 754.55 ms |
| 8 | 72 | 645.3 | 24.37 ms | 783.44 ms |

从1到8用户，吞吐约增加7.1倍。与诊断中的共享批次结果一起看，本轮说明：**多用户服务可以通过合批提高吞吐，并不需要给每个用户建立独立计算 stream。** 一个批次的 kernel 串行排在 stream 上，和一个 kernel 内部处理多个请求、利用多个设备计算单元是不同层次。

benchmark 没有采集 kernel 时间线，不能用 diagnostic 的时长推算其设备利用率。Python hook/profiler 会改变 CPU 提交速度和调度到达关系；报告将两类运行分开，不能用二者的延迟差当作模型优化收益。

## 输出差异：已记录，原因尚未确定

1和4用户组中，同输入的各次输出一致。2和8用户组存在 greedy 输出变体，**无插桩 benchmark 自身的重复运行也出现差异**。

以每个用户/轮次的第一次 benchmark 输出为参照，共9条后续 benchmark 响应和4条 diagnostic 响应不同，涉及8组输入。所有 diagnostic 输出都能在对应输入的某次 benchmark 输出中找到完全一致的 token IDs，没有出现 diagnostic 独有的新变体。首次分歧位于输出 token 索引20、45或46（从0开始）。

见 [原始比较](results/formal-01/output_comparison.json) 和 [变体核对](results/formal-01/output_variants.json)。目前只证明了输出变体也存在于无插桩运行，**尚未证明具体数值原因**。到达顺序、合批形态及其数值路径值得进一步核对；不能直接归因于 profiler，也不能宣称逐 token 确定性已经通过。

因此，本实验的“可复现”指固定输入、配置、客户端程序、采集和分析流程，并保留实际执行结果；不承诺并发到达顺序、批次切分和生成 token 每次完全相同。请求身份与 kernel 时间线的关联验证已通过，输出确定性作为独立问题保留。

## 环境、归档与验证

- vLLM `0.21.0+empty`，vllm-ascend `0.21.0rc1`，torch `2.10.0+cpu`，torch-npu `2.10.0`，Transformers `5.5.4`；源码 revision、原文件 SHA256 与命令均已保存。
- 本次运行前后 NPU 健康状态均为 `Alarm / 80C98001`。请求与采集成功不等于硬件健康校验通过；本表描述该主机本次运行，不作为健康设备的通用性能基准。没有执行 reset，也没有修改安装源码。
- 原始约2.51 GB profiler 数据保留在 `ascend910:/data/tianchi/practice_31_multiuser_streams/results/formal-01/`；约88 MB的完整分析归档保留在本地和远端，不上传 Git。Git 仅包含代码、文档、结果摘要、抽查证据、压缩请求记录及原始文件 SHA256。精确路径见 [归档清单](results/formal-01/archive_manifest.json)。重新运行 raw flow join 需要恢复清单中的原始 profiler 文件；完整报告本身可直接离线查看。
- [早期 smoke 的排除记录](excluded_pilots.json) 解释端口复用修正和前端 wrapper 观测修正。正式结果仅使用修正后的 `formal-01`。
- 10项本地测试通过；真实案例的 HTML 离线交互和窄屏显示另有浏览器验证记录。实验结束后服务已退出，NPU 无本次遗留运行进程。

## 下一步

按任务清单进入 **P31b：独立 HTTP 请求下随机采样辅助 stream 的开/关对照**，必要时增加到32用户并调整容量。这样可以检验多用户请求遇到框架主动引入的辅助 stream 后，是否实际出现计算重叠。

输出变体作为并行的后续问题保留：固定请求到达/合批形态，再比较首次分歧 token 前的 logits 或数值路径。它与 stream 并发测量分开设计，避免读回 logits 的同步污染性能对照。
