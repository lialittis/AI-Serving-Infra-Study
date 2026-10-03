# P34：fill_exponential 复用模式的间隔与延迟扫描

目标：把审计候选 1（默认采样路径 `fill_exponential` 的 omit 等价结构）从"时间性缓解"变成
定量损坏边界，并实测两种修复。入口见 [P34 README](../practice_34_sampling_gap/README.md)。

- [x] 步骤 1 源码确认：`enable_async_exponential` 默认 False → 非 greedy 采样默认走无保护路径；
      受保护的 `do_async_exponential` 需显式开启（sampler.py:184 分支）。
- [x] 探针：producer 分配+fill → 主流 wait → 主流积压 N 矩阵乘后 copy 消费 → 释放 → sleep(gap) → 下一轮。
- [x] smoke：b0 全间隔 0 损坏（照片终点消费端稳赢）；b32 在 gap 0/1ms 全损坏——确认"消费端延迟"为必要条件。
- [x] 正式矩阵：asis-b0/b8/b32 + record-b32 + leading-wait-b32 × 6 间隔 × 20 轮 = 600 轮。
- [x] 结论：损坏条件 = 消费端延迟 > 间隔；满足时 19/19 全损坏；两种修复全程 0 损坏。
- [x] 归档与校验：61 文件哈希、600 轮重算、6 单元测试通过。
- [ ] 步骤 3：真实 vLLM top-k/top-p 服务的输出确定性验证（构造主流积压 > 步进间隔的场景）。
- [ ] 候选 2（graph update_stream 存储边界）、候选 3（HCCL/CP，需多卡）仍排队。

[RESULTS](../practice_34_sampling_gap/RESULTS.md)：正常 vLLM 节奏下靠时序余量安全（间隔约 10ms > 采样链 µs–ms）；
余量为负即 100% 损坏。leading-wait 与 do_async_exponential 同构，是一行先导等待的零成本修法。
