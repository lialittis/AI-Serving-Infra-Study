# 执行计划：decode 利用情况诊断

编号 23–25 已被其他实验占用，因此本练习使用 26。

问题：一次真实 vLLM decode 的空隙与核资源使用情况是什么；是否有证据将其归因于单 stream？

1. 单卡 Qwen2.5-0.5B BF16、batch 1、10-token 输入、64-token 贪心输出。关闭 prefix cache、chunked prefill、async scheduling 和提前采样。所有请求带 logprobs 以核验精确 token；其开销属于该配置。
2. eager / PIECEWISE（capture size 1）对照。先 E/G/G/E 四个独立无 profiler 服务，每个预热 3 次、正式 5 次：每模式 10 个独立请求样本。报告每批中位数、全体中位数/范围；不以十个样本推导稳定 P99。
3. plain 和 PipeUtilization 分别独立启动两模式服务，每组预热 3 次、采集 1 个完整 64-token 请求。解释所有步骤，稳定区域预定为 decode 16–47。重复步骤不是独立运行重复，硬件指标是描述性观测。
4. 使用局部 import hook 包装 runner/sample 和 graph replay；没有 sys.setprofile、LD_AUDIT、额外设备同步或 tensor 回读。profiler 外控制组不加载观察器。graph debug dump 只发生在初始化捕获结束。记录缓存/source/model 指纹。
5. 按精确 flow / connection / 捕获 graph 任务序列关联。报告核类型、原始 Block Num/Mix Block Num、流水线指标、计算区间并集、等待/拷贝/未覆盖区间、CPU 提交时刻；记录未知，不用最近时间猜关联。
6. 依据结果只选择一个后续对照。提交空隙明显时已有 eager/graph 对照就是首项干预；若仍需 batch 或独立分支试验，先记录新假设。不能从 profiler 相减推算纯 CPU 开销，不能用计算覆盖率代替芯片利用率。

验收：20 个无 profiler 正式响应及四个诊断响应的 token 一致性；完整 64 步覆盖；原始证据、交互 HTML、精确代表步骤 SVG；负向测试覆盖区间并集、遗漏/重复任务、graph 身份、未知核数处理；远端保留原始数据，本地归档可重放证据。

## 本轮决策

2026-09-29：八个服务运行均完成。plain 诊断发现 eager 提交间隙明显，选择在已计划的 eager/graph 对照中精确核对空隙后的 CPU/CANN 下发时间；不额外增加 batch 或人为双流变量。结果见 RESULTS.md。
