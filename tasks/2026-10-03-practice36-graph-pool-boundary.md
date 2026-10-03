# P36：graph 捕获/重放的存储边界（P32 遗留项）

目标：实测 private pools 与 replay 外部存储的生命周期边界。入口见
[P36 README](../practice_36_graph_pool_boundary/README.md)。

- [x] 源码阅读：NPUGraph 捕获切私有池、replay 任意流、reset→releasePool→freeable→npuFree；
      外部存储无任何登记。
- [x] 探针：keep-alive / external-free-ordered / external-free-cross / pool-release-pending。
- [x] 排障两处：哨兵 777.0 与 arange[777] 撞车（改 −777.0）；arange 中间临时量导致候选不命中
      （改逐候选分配直至命中 external 地址）。smoke-01 保留为 pilot。
- [x] 正式矩阵 20 轮：external-free 两组 5/5 全量消费复用者数据（有序与跨流重放同）；
      keep-alive 5/5 完好；pool-release 5/5 幸存且 0 地址重叠。
- [x] 归档校验：70 文件哈希、summary 重算、6 单元测试通过。
- [x] P32 任务清单的 graph 遗留项关闭；HCCL 项仍待多卡环境。

结论：外部存储存活责任完全在使用者（vLLM 的静态输入拷贝模式正是规避方式）；
私有池与普通池地址空间隔离。详见 [RESULTS](../practice_36_graph_pool_boundary/RESULTS.md)。
