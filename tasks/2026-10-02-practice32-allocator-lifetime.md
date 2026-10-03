# P32：原生 NPU allocator 跨 stream 生命周期

目标：验证[源码调查](../references/torch_npu_allocator_lifetime/README.md)发现的缓存与事件机制，区分地址重新分配、执行依赖和实际冲突访问。入口见 [P32 README](../practice_32_allocator_lifetime/README.md)。

- [x] 在空闲设备上固定实际版本、配置和既有健康告警。
- [x] 建立 omit / record / owner-join 三组，保持生产者依赖相同。
- [x] weakref 与 allocator `free_requested` 双重核对最后存储释放。
- [x] 保留有额外 host queue drain 的首轮为独立观测对照；正式采集预缓存 stream handle。
- [x] 默认分配 stream：单/双 host 线程，六配置，每组五次；lazy 独立补齐。
- [x] 核对 lazy cache miss 后的延迟回收、统计及地址。
- [x] 自定义分配 stream 三组与跨 host 线程对照。
- [x] 独立 profiler 核对地址交回和旧 copy 的实际设备顺序。
- [x] 离线 timeline、原始证据清单、浏览器校验和[结果报告](../practice_32_allocator_lifetime/RESULTS.md)。
- [ ] HCCL：Work 等待、存储保留、记录撤销以及 owner stream 的对应关系，单独设计。
- [ ] Graph：private pools、capture mark 删除与 replay 外部存储，单独设计。

本轮候选 B 不读写。观察到同地址与 pending 事件不能独自证明竞态；设备时间线同样只能证明本轮任务顺序，不包含逐指令访问或真实 B 冲突。
