# 参考资料

| 资料 | 主要内容 |
|---|---|
| [视觉 forward 同步分析笔记](vision_metadata_synchronization/README.md) | P27 同步发生位置、逐条时间证据、91→7→0 消融，以及主机等待对多 stream 提交和计算重叠的影响 |
| [lengths 概念、Sub 计算与待研究问题](vision_metadata_synchronization/LENGTHS.md) | 分段长度与累计边界、NPU Tensor 如何经 PyTorch dispatcher 进入 torch-npu/CANN、真实 Sub kernel，以及同步/D2H、地址与数值校验 |
| [tolist 同步调用链与 D2H 实测](vision_metadata_synchronization/TOLIST_D2H.md) | 实际安装版本的 NPU `_to_copy` / OpApi 分发、同步位置、CPU 缓冲区和 D2H 地址证据，以及三次读取与一次复用的对照 |
| [Sub 提交、执行与内存地址](vision_metadata_synchronization/SUB_EXECUTION.md) | 首次/预热与 Sub/tolist/sync 对照，实际主机队列、CANN 执行缓存、已有 ELF 加载及 slice 地址证据 |
| [lengths 正确性与异常校验](vision_metadata_synchronization/CORRECTNESS.md) / [HTML](vision_metadata_synchronization/correctness_probe/report/index.html) | 26 个隔离样例、独立 Python 整数参考、slice 地址、split 接受盲区与 int32 窄化/溢出反例 |
| [CUDA 多 Stream 应用与同步](cuda_stream_use_cases/README.md) | 五类应用场景、同步条件、论文与框架来源，以及与 kernel execution graph 实验的对应关系 |
| [PyTorch CUDA Stream Sanitizer](pytorch_cuda_stream_san/README.md) | 源码结构、数据访问与 happens-before 检测、研究方向 |
| [vLLM KV offload 同步问题](vllm_syn_issuse_analysis/README.md) | KV 传输、计算、存储覆盖与生命周期之间的同步约束 |
| [vLLM / vLLM-Ascend Stream 远端源码分析](streams_in_vllm_source_code/README.md) | 当前源码版本与导入路径、创建到结果消费的完整调用链、eager / graph 差异、时序图与运行证据边界 |
