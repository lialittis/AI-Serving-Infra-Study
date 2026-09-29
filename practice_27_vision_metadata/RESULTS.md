# 视觉元数据预计算：实测结果

本轮完成 8 个场景、576 个无 profiler 样本、96 个独立诊断 trial；含预热/资格的 816 次输出比较全部逐元素一致，8 个场景另与原生完整 forward/generate 对照通过。

视觉阶段同步 API 从 native 的每次 91 条降为 lengths 的 7 条，再降为 cached 的 0 条。仅预计算 attention 分段长度不足以让 VL（先视觉后语言）产生重叠；完整元数据预计算后，8 个场景的 VL 都观察到真实 compute kernel 重叠。

cached 相对自身单流的 VL pair 耗时下降：prefill 为 7.18%–7.98%，decode 为 5.17%–10.74%。其 LV prefill 下降 6.86%–12.04%，LV decode 基本持平。因此 P25 的 decode 无明显收益只适用于当时的元数据与提交路径，不能推广成 decode 天生无法并发。

完整元数据准备每场景观测约 0.57–0.66 ms，额外有效设备数据约 136 / 621 KiB，均在正式 pair 计时之外。

## 方法与工作量

完整预训练 Qwen2.5-VL-3B-Instruct，BF16 eager，Ascend 910B2C（容器逻辑 0 / 物理 5），单 CPU 提交线程。A 为 512-token 语言 prefill 或一个准备好前缀的 decode 步；B 为两张图片、两档实际视觉 token 数 54 / 247，随后运行自己的语言阶段。

native 保留原实现；lengths 仅改 attention 分段长度来源，保留其他视觉元数据操作；cached 再预计算窗口索引、逆序索引和 RoPE cos/sin，并让索引提前驻留设备。全部网络层和像素计算保留。三组使用相同 stream/event/handoff，A/B KV 和位置状态独立。方法替换仅在实验进程内有效，原安装文件和权重未修改。

六个变体/流配置轮换位置，每种 LV/VL 顺序下各覆盖全部六个测量位置。每配置预热 3 次，计时 12 次（每种顺序 6 次）。性能计时不带 profiler，诊断与正式计时分开。

## 性能：分开看单流优化和双流收益

下面是 ms/pair 中位数。括号内为同轮双流相对同一变体单流的配对百分比中位数；负值表示双流更快。native 与 cached 的差额包含元数据路径优化，不应全算作并发收益。完整 IQR/极值、变体相对 native 的配对变化、A/V/B 就绪和主机提交记录见 [summary.json](results/published/summary.json)。

| 场景 | 变体 | LV 单流 / 双流 | LV 双流变化 | VL 单流 / 双流 | VL 双流变化 |
|---|---|---:|---:|---:|---:|
| prefill-beach-v64 | native | 95.543 / 83.236 | -12.80% | 95.788 / 95.529 | -0.24% |
| prefill-beach-v64 | lengths | 92.453 / 81.012 | -12.44% | 92.590 / 92.455 | -0.05% |
| prefill-beach-v64 | cached | 91.361 / 80.419 | -12.04% | 91.551 / 84.279 | -7.93% |
| prefill-beach-v256 | native | 150.655 / 139.469 | -7.40% | 150.711 / 150.828 | +0.10% |
| prefill-beach-v256 | lengths | 147.489 / 136.948 | -7.15% | 147.798 / 147.712 | -0.08% |
| prefill-beach-v256 | cached | 146.331 / 136.296 | -6.90% | 146.326 / 135.826 | -7.18% |
| prefill-beijing-v64 | native | 95.602 / 82.545 | -13.69% | 95.793 / 95.556 | -0.15% |
| prefill-beijing-v64 | lengths | 92.290 / 81.110 | -12.14% | 92.456 / 92.499 | +0.01% |
| prefill-beijing-v64 | cached | 91.420 / 80.522 | -11.97% | 91.449 / 84.171 | -7.98% |
| prefill-beijing-v256 | native | 150.525 / 139.511 | -7.37% | 150.654 / 150.803 | +0.01% |
| prefill-beijing-v256 | lengths | 147.334 / 136.617 | -7.30% | 147.640 / 147.510 | -0.12% |
| prefill-beijing-v256 | cached | 146.317 / 136.342 | -6.86% | 146.515 / 135.638 | -7.34% |
| decode-beach-v64 | native | 70.864 / 70.929 | +0.01% | 71.048 / 71.160 | +0.16% |
| decode-beach-v64 | lengths | 67.656 / 67.706 | +0.09% | 68.083 / 67.893 | -0.28% |
| decode-beach-v64 | cached | 66.912 / 66.815 | -0.15% | 64.283 / 57.423 | -10.64% |
| decode-beach-v256 | native | 126.085 / 126.042 | -0.25% | 126.328 / 126.577 | -0.08% |
| decode-beach-v256 | lengths | 122.961 / 122.781 | -0.17% | 124.039 / 123.013 | -0.76% |
| decode-beach-v256 | cached | 121.931 / 121.965 | +0.10% | 119.406 / 113.077 | -5.28% |
| decode-beijing-v64 | native | 70.704 / 70.878 | +0.24% | 70.781 / 70.827 | +0.62% |
| decode-beijing-v64 | lengths | 67.638 / 67.461 | -0.18% | 67.587 / 67.586 | -0.02% |
| decode-beijing-v64 | cached | 66.654 / 66.631 | +0.04% | 64.356 / 57.452 | -10.74% |
| decode-beijing-v256 | native | 125.998 / 126.934 | +0.72% | 126.883 / 126.289 | +0.15% |
| decode-beijing-v256 | lengths | 123.930 / 123.157 | -0.21% | 123.174 / 122.991 | -0.17% |
| decode-beijing-v256 | cached | 121.978 / 121.965 | +0.04% | 119.368 / 113.218 | -5.17% |

## 实际重叠与同步

重叠来自 A 语言与 B 视觉的实际 compute kernel 区间并集求交，单位 ms；不是两个 Python forward 包围区间相交。API 调用数可能含嵌套，不等于独立等待次数，未将其 duration 相加。这里的零同步只指 vision_B scope，跨流 event 和末尾 host join 仍然存在。

| 场景 | 变体 | LV 重叠 ms | VL 重叠 ms | 每 trial 视觉同步 API（LV / VL） |
|---|---|---:|---:|---:|
| prefill-beach-v64 | native | 19.339 | 0.000 | 91 / 91 |
| prefill-beach-v64 | lengths | 19.352 | 0.000 | 7 / 7 |
| prefill-beach-v64 | cached | 12.899 | 3.401 | 0 / 0 |
| prefill-beach-v256 | native | 19.393 | 0.000 | 91 / 91 |
| prefill-beach-v256 | lengths | 21.377 | 0.000 | 7 / 7 |
| prefill-beach-v256 | cached | 14.302 | 34.307 | 0 / 0 |
| prefill-beijing-v64 | native | 19.547 | 0.000 | 91 / 91 |
| prefill-beijing-v64 | lengths | 18.808 | 0.000 | 7 / 7 |
| prefill-beijing-v64 | cached | 22.215 | 9.661 | 0 / 0 |
| prefill-beijing-v256 | native | 16.793 | 0.000 | 91 / 91 |
| prefill-beijing-v256 | lengths | 17.210 | 0.000 | 7 / 7 |
| prefill-beijing-v256 | cached | 22.653 | 33.451 | 0 / 0 |
| decode-beach-v64 | native | 0.000 | 0.000 | 91 / 91 |
| decode-beach-v64 | lengths | 0.008 | 0.000 | 7 / 7 |
| decode-beach-v64 | cached | 0.049 | 6.901 | 0 / 0 |
| decode-beach-v256 | native | 0.006 | 0.000 | 91 / 91 |
| decode-beach-v256 | lengths | 0.015 | 0.000 | 7 / 7 |
| decode-beach-v256 | cached | 0.016 | 17.868 | 0 / 0 |
| decode-beijing-v64 | native | 0.021 | 0.000 | 91 / 91 |
| decode-beijing-v64 | lengths | 0.000 | 0.000 | 7 / 7 |
| decode-beijing-v64 | cached | 0.032 | 6.608 | 0 / 0 |
| decode-beijing-v256 | native | 0.017 | 0.000 | 91 / 91 |
| decode-beijing-v256 | lengths | 0.010 | 0.000 | 7 / 7 |
| decode-beijing-v256 | cached | 0.014 | 18.647 | 0 / 0 |

原实现的分段长度 `.tolist()` 发生于每个 window attention 的 Q/K/V 切分。lengths 去掉这类读取后，仍保留 CPU 索引进入设备时的拷贝和 `unique_consecutive` 相关同步。逐条关联见 [sync_audit.json](results/published/sync_audit.json)：依据同线程区间包含或精确 async_task_queue flow，而非最近时间戳。

首个场景中，lengths 在视觉末尾仍有 `aten::index → aten::to → aten::_to_copy → aten::copy_` 内的同步；结合固定源码的执行顺序，该末尾索引可对应到 merger 后 CPU reverse_indices 的设备索引消费；这是源码与 trace 的联合定位。cached 将索引提前驻留设备，连同其他静态元数据一起移出执行阶段。该组合对照说明只移除 attention 分段读取仍不足以解除 VL 的阻塞；本轮没有进一步独立区分窗口、逆序和位置编码各自的性能贡献。

## 准备成本与请求延迟

下列是每个场景一次实际准备的观测值，不是准备耗时分布。总准备包含元数据设备运算/上传及同步，发生于计时前。持久化字节是新增长期存活元数据 tensor 的有效字节，不等同于 allocator reserved 增量；三组对照期间都保留同一份预计算对象，显存峰值比较不是“原实现完全不持有元数据”的生产部署比较。

| 场景 | CPU 长度准备 ms | 完整准备 ms | 持久化设备字节 |
|---|---:|---:|---:|
| prefill-beach-v64 | 0.240 | 0.589 | 139104 |
| prefill-beach-v256 | 0.231 | 0.665 | 636272 |
| prefill-beijing-v64 | 0.216 | 0.568 | 139104 |
| prefill-beijing-v256 | 0.214 | 0.617 | 636272 |
| decode-beach-v64 | 0.248 | 0.630 | 139104 |
| decode-beach-v256 | 0.226 | 0.639 | 636272 |
| decode-beijing-v64 | 0.249 | 0.629 | 139104 |
| decode-beijing-v256 | 0.235 | 0.660 | 636272 |

pair 时间包含 A 语言、B 视觉、B merge/语言及结束等待；图片预处理、输入传输、A 视觉/前缀、KV clone 和上面的预计算不在计时内。不能把已复用元数据的 pair 时间直接解释为首次请求或 HTTP 端到端时间。

| cached 场景 | LV A 完成 单流 / 双流 ms | VL A 完成 单流 / 双流 ms |
|---|---:|---:|
| prefill-beach-v64 | 38.171 / 48.775 | 67.493 / 60.270 |
| prefill-beach-v256 | 38.229 / 47.552 | 113.948 / 103.403 |
| prefill-beijing-v64 | 38.211 / 48.569 | 67.392 / 60.138 |
| prefill-beijing-v256 | 38.262 / 47.792 | 113.980 / 103.197 |
| decode-beach-v64 | 13.675 / 13.627 | 40.371 / 33.007 |
| decode-beach-v256 | 13.762 / 13.766 | 86.997 / 73.186 |
| decode-beijing-v64 | 13.431 / 13.478 | 40.396 / 32.772 |
| decode-beijing-v256 | 13.807 / 13.732 | 86.968 / 73.067 |

A 完成时间以公共起始 event 为原点，包含前序排队与主机供给间隙；不是独立 kernel 服务时间。总耗时下降不保证每个请求的延迟改善，资源竞争仍需独立硬件计数器实验。

## 执行依赖与验证

| 变体 | 设备任务 | 计算任务 | 已证明边界要求 | 视觉同步 API 总数 |
|---|---:|---:|---:|---:|
| native | 344589 | 340336 | 256 | 2912 |
| lengths | 343692 | 339440 | 256 | 224 |
| cached | 343409 | 339184 | 256 | 0 |

所有 kernel CSV 记录均精确关联。真实视觉 feature 指针与消费端一致，A/B KV 等活跃存储范围隔离，feature event 等待保护 B 语言消费；要求没有作为证明边添加回图。权重/buffer 前后哈希一致，共享 rope_deltas 保持 None。

图仅覆盖显式 FIFO/event/host join 和真实 kernel 时间线；原生 workspace 的完整访存 DAG、全部隐式 CPU 因果边、算力/带宽竞争程度仍未恢复。诊断会扰动提交时序，单次诊断的重叠时长不能直接相减得到正式性能收益。

## 归档与边界

[交互对照与逐 kernel 图](report/index.html) · [归档清单](results/published/archives.json) · [验证记录](results/published/replay_validation.json)。

qualification-r01 在 case 构造前因缺失 phase 参数失败，未产生有效样本；修正后的 qualification-r02 通过。正式结果仅来自 formal-r01，失败与资格记录一起保留。大型 CANN 二进制/数据库留在远端，正式 trace_view/kernel CSV 全部可本地重放。远端与本地主机时钟有偏移，关联与耗时只使用同一采集时钟。

这仍是 HF eager 阶段 harness，没有 graph replay 或 vLLM 在线服务调度。下一步可在固定元数据路径上做 graph 对照，或先独立采集计算/带宽计数器，评估并发的资源代价；本轮未实施这些后续工作。
