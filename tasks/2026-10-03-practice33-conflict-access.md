# P33：地址复用后的实际冲突访问 + 跨 stream 调用点审计

目标：把 P32 的"提前地址复用"前提推进到实际冲突访问证据，并从源码侧定位可能踩中该模式的生产调用点。
入口见 [P33 README](../practice_33_conflict_access/README.md) 与[审计报告](../references/cross_stream_call_site_audit/README.md)。

- [x] fork P32 探针：候选 B 找到复用地址后立即在 S0 写入哨兵值；新增 synced 校验器对照。
- [x] omit/record/join/synced 四组 × 10 轮无 profiler；每模式新进程；沿用事件/快照/free_requested 采集。
- [x] omit/join/record × 2 轮独立 profiler，关联旧 copy 与候选写入的全部设备任务。
- [x] 离线分析：元素分类（intact / fully_overwritten / mixed / unexpected）+ 设备顺序互证。
- [x] 归档、486 文件哈希校验、46 轮重算、138 快照、12 CSV 任务核对、浏览器验证、单元测试。
- [x] 调用点审计：vllm-ascend 全库 0 record_stream；vllm core 仅 CUDA 分支 1 处；op-plugin 4 处正向对照；
      triton launcher 仅用当前流；`fill_exponential` 判定为 omit 等价结构（mitigated-by-timing）。
- [ ] 候选 1 定向实验：复刻 fill_exponential 模式，量化真实调度间隔下的损坏概率。
- [ ] 候选 2：graph update_stream 与 replay 的存储边界（P32 遗留）。
- [ ] 候选 3：HCCL/CP 路径（需多卡）。

本轮 omit 10/10 全量污染证明：无保护跨 stream 生命周期在本硬件上必然冲突，不是理论风险；
join 的顺序保护由 profiler 设备区间实测升级为 [VERIFIED]。
