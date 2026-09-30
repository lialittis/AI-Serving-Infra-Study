# Practice 30：拆解单请求 CPU 时间线

- 日期：2026-09-30。
- 状态：首轮已实施，见[复现说明](../practice_30_cpu_submission_timeline/README.md)、[结果](../practice_30_cpu_submission_timeline/RESULTS.md)和[离线报告](../practice_30_cpu_submission_timeline/report/index.html)。

## 要回答的问题

Practice 29 显示提前排队能产生跨 stream 重叠，但尚不能区分 CPU 提交慢与 CPU 被同步阻塞。本轮只研究一次正常原生请求：CPU 各阶段做什么、由哪个线程提交、哪里等待，以及 NPU 空隙出现时后续工作提交到哪里。

用户选择：推理引擎到设备，不含 HTTP；先只做 eager。坚持“先做薄，再做厚”，不扩展模型、graph、门控、并发或硬件利用率计数器。

## 实施范围

- Qwen2.5-0.5B-Instruct、BF16、单 NPU、单在途请求；固定 10 输入 / 64 输出，重点 decode 32。
- 保留真实 `LLM.generate()`、调度器、KV 增长和结果回收，不使用固定步 capsule 替代正常引擎。
- 临时包装调度、输入准备、forward、logits、采样和结果处理，记录墙钟与线程 CPU 时间；关联 PyTorch→Enqueue→Dequeue→CANN→NPU。
- 不把墙钟减线程 CPU 直接归为设备等待，不把队列起止边界差称为纯排队时间，不累加跨线程或嵌套范围。
- 三个自有进程：reference（预热 2 + 无 profiler 3）、diagnostic（预热 2 + 诊断 1）、recovery（预热 2 + 原生 1）。安装源码及配置不改。

## 首轮回填

- [x] 远端实际 revision、导入位置和源码调用关系已核实。
- [x] 1,089 个阶段标记、64 个完整步骤、19,298 个计算任务和 21,512 对队列记录关联通过。
- [x] decode 32 三个最大设备任务空隙已逐项说明；一个 RoPE Host 范围值得下一轮继续放大。
- [x] 11 次响应（含预热）的 token/logprob 精确相等，包装恢复和原生推理恢复通过。
- [x] 25 个文件前后审计未变；实际 BalanceScheduler 源码另作明确标注的运行后补采，并核对 Git HEAD。
- [x] 离线 HTML、完整请求与 decode 32 SVG、复现说明、原始证据及反例测试已补齐。

诊断请求约 936 ms，提交线程 CPU 时间约 935 ms；256 次已观测 CANN 同步范围合计约 0.84 ms。不能把当前主要开销笼统称为 CPU 等 NPU，也不能据此宣称完全没有隐藏等待或自旋。诊断比参考中位数慢约 26.5%，明确披露观测成本与运行差异。

下一轮只选一个范围（优先 RoPE Host→Enqueue）继续拆解 Python、Triton launcher、C++ 入队边界，不先做完整 benchmark 矩阵。
