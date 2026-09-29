# 视觉元数据预计算与同步消减

承接 P3b，用户已授权执行。实现与详细设计：[Practice 27](../practice_27_vision_metadata/README.md) / [PLAN](../practice_27_vision_metadata/PLAN.md)。

- [x] 审计已安装视觉 eager 源码，定位分段长度 `.tolist()` 和窗口/位置元数据路径。
- [x] 实现 native / lengths / cached 三组对照，保持真实视觉计算及跨请求 event handoff。
- [x] 首个场景资格检查：特征、logits、完整 KV 与原实现逐元素一致。
- [x] 完成 8 场景、576 个无 profiler 样本和 96 个诊断 trial；含预热的 816 次输出检查逐元素一致。
- [x] 核验 1,031,690 个设备任务与 768 条边界要求；同步 91 → 7 → 0，完整预计算后 VL 在 prefill/decode 均产生实际重叠。
- [x] 发布结果、219 文件证据归档、交互图；重放 28 个分析输出逐字节一致，23 项检查及 96 个浏览器诊断视图通过。

边界：元数据准备有成本，需要单列；本轮不是 graph、vLLM 服务集成或硬件资源竞争计数器分析。若同步减少却没有重叠/加速，仍据实记录。首次 qualification-r01 的参数错误发生在 case 构造前，不纳入性能；修正后 qualification-r02 通过。

结果入口：[报告](../practice_27_vision_metadata/RESULTS.md) · [交互图](../practice_27_vision_metadata/report/index.html) · [验证](../practice_27_vision_metadata/results/published/replay_validation.json)。
