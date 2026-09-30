# 参考资料

| 资料 | 主要内容 |
|---|---|
| [视觉 forward 同步分析笔记](vision_metadata_synchronization/README.md) | P27 同步发生位置、逐条时间证据、91→7→0 消融，以及主机等待对多 stream 提交和计算重叠的影响 |
| [lengths 概念、Sub 计算与待研究问题](vision_metadata_synchronization/LENGTHS.md) | 分段长度与累计边界、CPU/NPU 数据流、真实 Sub kernel，以及同步/D2H、编译提交/地址、数值与访存校验三个研究方向 |
| [CUDA 多 Stream 应用与同步](cuda_stream_use_cases/README.md) | 五类应用场景、同步条件、论文与框架来源，以及与 kernel execution graph 实验的对应关系 |
| [PyTorch CUDA Stream Sanitizer](pytorch_cuda_stream_san/README.md) | 源码结构、数据访问与 happens-before 检测、研究方向 |
| [vLLM KV offload 同步问题](vllm_syn_issuse_analysis/README.md) | KV 传输、计算、存储覆盖与生命周期之间的同步约束 |
| [vLLM / vLLM-Ascend Stream 远端源码分析](streams_in_vllm_source_code/README.md) | 当前源码版本与导入路径、创建到结果消费的完整调用链、eager / graph 差异、时序图与运行证据边界 |
