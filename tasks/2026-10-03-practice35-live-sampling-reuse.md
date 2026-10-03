# P35：真实 vLLM 采样路径上的复用前置条件测量

目标：在真实引擎上测 P34 损坏不等式的前半部分是否出现。入口见
[P35 README](../practice_35_live_sampling_reuse/README.md)。

- [x] observer 经 sitecustomize 注入 EngineCore/worker（不修改安装源码），包装语义与安装版本逐行一致。
- [x] 排障：PYTHONPATH 覆盖导致 acl 模块丢失 → 改为追加并补 CANN python 路径。
- [x] 三变体正式运行：default / async_scheduling / enable_async_exponential，16 请求 × 2 重复。
- [x] 结果：地址复用 64/68 常态发生；前置条件 0（default 与 async 均为 0）；protected 0 调用；
      三变体重复生成一致；真实步进间隔中位约 12 ms。
- [x] 归档校验：50 文件哈希、summary 重算、3 单元测试通过。
- [ ] 重负载 / 深积压场景（主流积压超过一步间隔）未构造，留作后续按需扩展。
- [ ] 候选 2（graph update_stream）、候选 3（HCCL/CP）仍排队。

P32→P35 证据链闭合：提前交回（P32）→ 无保护必损坏（P33）→ 条件=延迟>间隔（P34）→
真实引擎实测条件不成立、复用每步发生（P35）。
