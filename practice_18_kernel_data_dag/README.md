# Practice 18：全模型 kernel 数据依赖与调度分析

本 practice 沿用单卡 Qwen2.5-0.5B-Instruct eager 请求，补采 tensor 输入、输出、
原地写入、存储代次和 KV 索引来源，再做关键路径、并行度与离线 stream 分配分析。

**当前实现了完整设备任务清单上的数据依赖投影和分析链路，尚未完成精确的原生逐
kernel 数据依赖 DAG。** 原生 workspace、内部中间 tensor 的实际读写，以及改变 stream
后的 allocator 生命周期约束仍未完全暴露。程序明确输出
`complete_exact_data_dag=false`、`live_stream_reassignment_eligible=false`。

直接打开[交互报告](report/index.html)，或看[实验结果](RESULTS.md)。报告完全离线，
可切换整次请求 / 四次 forward、三种图和 1/2/4/8 条模拟 stream，查看节点、关键路径、
调用范围及跨流 event 方案。浏览器直接打开 HTML；网页源码托管平台通常不会直接渲染它。

## 这次比 Practice 15 多了什么

| 问题 | 本次证据 / 实现 | 仍有的边界 |
|---|---|---|
| 是否覆盖整模型 | 四轮各 24 层；1,444 个设备任务，含 1,260 个 forward 任务 | 任务含计算、拷贝和 event，不把它们都称为计算 kernel |
| tensor 从哪里来、谁写 | TorchDispatch 输入、输出、schema mutation；直接 Triton 与 ATB/FIA 参数 | schema 描述公开算子边界，不是原生内部访存 |
| 同一地址是否同一 tensor | StorageImpl 弱引用判活与代次；真实地址区间保留复用约束 | 原生 workspace 的分配和释放还不在此账本中 |
| 别名、原地操作 | 按字节区间生成 RAW、WAR、WAW；部分覆盖保留未覆盖区域的旧 writer | kernel 可能只访问 view 的子集；大规模稀疏 stride 使用标明的包围范围 |
| KV 写哪里、读哪里 | CPU staging 中的 block table，加同步路径的最终 position 源码契约 | 没有读取 NPU 内存指令；仅支持已核验的单卡同步配置 |
| 如何分析 | 加权最长路径、slack、W/CP、ASAP 活跃任务峰值 | 固定 kernel 时长模型，未包含资源竞争、Host 提交空隙 |
| 如何分配 stream | 按剩余路径长度排序的 list scheduling，保留 native 调用的同流连续约束，生成 event/wait | 仅离线建议；没有修改 vLLM 的真实执行 stream |

## 三种图

`observed` 保留实际单条物理 stream 的 FIFO 边。它是链，也是一种 DAG。
其 kernel 加权关键路径等于 kernel 时长之和；这不等于包含 Host 空隙的请求延迟。

`projected` 使用可见 tensor 的 RAW/WAR/WAW、allocation reuse、调用内部顺序、
未知任务屏障和阶段边界。每个节点仍是一个实际设备任务，但多 kernel 调用只在首节点
读取公开输入、末节点写出公开结果，并保留内部顺序。**这不是已经还原了内部读写。**
未知内部 workspace 可能引入漏边，保守 tensor 范围也可能引入多余边，因此它既不能
无条件充当真实依赖图的超集，也不能无条件充当子集。

`certified` 是本次“不删除已经证实的执行约束”的参照图：完整性未通过时保留原 FIFO。
这个名称不表示已经认证了多 stream 重排。三种图都只是离线数据；本 practice 没有实机
自动改流执行器。

整次请求另保留准备输入、forward、采样以及自回归步之间的顺序。不能让 decode 使用
尚未得到的 sampled token。单独分析 forward 时，将准备好的输入视为外部输入。

## 采集与构图

`run_capture.py` 复用 Practice 13 的服务生命周期、冷编译归档、预热、profiler 控制、
请求及自有进程清理。对 Practice 13 的改动是可选 observer 参数，默认仍执行原实验。

`dependency_trace.py` 在被测 runner / sampler 的范围内进入 `TorchDispatchMode`，
记录真实 dispatcher 输入和返回，并与原先的 P13 scope 一起写入 profiler。
直接 Triton 调用不经过 dispatcher，继续从实际 launcher 捕获参数。ATB/FIA 的内层
参数比 `vllm.unified_attention_with_output` 的外层 schema 更具体，关联时优先使用它们。

PyTorch 在 redispatch 自定义算子时会暂时弹出当前 mode。第四轮因此在已经进入的
attention、linear、RoPE Python 实现中重新启用 mode，记录内部公开算子的调用。
这补上了 FIA 临时结果到最终 output 的真实 `copy_` 输入；验证器要求四轮共 96 条
相应 RAW 依赖，防止把这类拷贝误判为独立分支。C++ 原生实现内部的 workspace 仍未展开。

只读取已有的 CPU 小整数 staging buffer 和同步路径已有的 NumPy positions。
没有新增 NPU 数据读回、同步或复制。CPU 读取本身会产生额外的 dispatcher 元数据调用，
因此不能把调用数增加当成设备 kernel 增加。

KV 位置需要特别小心：`query_pos` 是当前请求这一步内的相对位置；同步路径中设备
positions 是 `num_computed_tokens[req_indices] + query_pos`。本次用 runner 已计算的
`positions_np` 做版本绑定的等价契约，不能把 decode 上传的 `[0]` 当作最终 position。
根据单卡 slot-mapping 源码得到 `block_id * block_size + position % block_size`，
再把 K/V 池访问收窄到这些 token 行。prefill FIA 使用当前 Q/K/V；decode FIA 使用池和
block table。没有足够证据时回退到整池范围。

`contracts.json` 固定已人工核对的四份 Triton 源码 SHA256。runner 的 position 等价
契约也绑定源码 SHA256。源码变更后停止分析，需要重新审计，不能沿用旧契约猜测。
前两轮的附加契约源码在运行后同一会话归档；第三轮开始在运行前自动归档。

完整性检查还要求：scope 配对、原始 P13 flow/queue 校验、单物理 stream、无同流区间
重叠、无环、无悬空边、所有设备节点来自 trace、全 24 层覆盖。未知数据契约保留屏障。

## 关键路径和调度模型

所有时长转为整数纳秒，避免对绝对 profiler 时间戳使用浮点相减。`dag.py` 提供：

1. 区间切分的 last-writer / outstanding-readers 账本，生成 RAW/WAR/WAW。
2. 地址换代时保留旧分配最后访问到新分配访问的 `allocation_reuse` 边，包括读后读。
3. 拓扑排序、最早开始、最长路径、slack、`W/CP` 和 ASAP 活跃任务峰值。
4. 将同一原生调用的 kernel 收缩为调度单元，安排后再展开成逐 kernel 时间表。
5. 传递约简减少冗余 wait；每个 producer 使用独立 event generation。
6. 检查每条原始依赖的时间先后以及由 FIFO + event 形成的 happens-before 可达性，
   而不只是检查模拟时间碰巧没有重叠。同时验证 native scope 同流连续。

W/CP 是**给定图、给定时长**的工作量与跨度之比；ASAP 活跃任务峰值不是最大 antichain，
也不是 NPU 能同时承载的 kernel 数。1/2/4/8 stream 的时间表是启发式方案，不保证最优。
`--event-cost-ns` 表示假设的跨流依赖延迟，不是每次 Host event API 调用的耗时。
默认是零；`schedule_sensitivity.py` 可扫描假设值，并允许回退到单 stream。
这些模型不模拟计算单元占用、带宽争用、Host 提交或真实 allocator 改流成本。

## 复现

真实采集需要此前实验使用的 Ascend 环境和本地 Qwen 权重；分析和生成报告只需 Python
标准库。实际 capture 限定单卡、单请求、eager、关闭 async scheduling / chunked prefill /
prefix caching；不支持拿多 stream、TP、异步 spec decode 的 trace 直接套用。

```bash
# 在已准备好 Ascend / vLLM 环境的服务器上，从仓库根目录执行。
python practice_18_kernel_data_dag/run_capture.py \
  --model /data/huggingface_home/hub/Qwen2.5-0.5B-Instruct \
  --output practice_18_kernel_data_dag/results/new-run --port 8018

# 以下可在本地离线运行。
python practice_18_kernel_data_dag/build_dag.py \
  practice_18_kernel_data_dag/results/new-run
python practice_18_kernel_data_dag/verify_run.py \
  practice_18_kernel_data_dag/results/new-run
python practice_18_kernel_data_dag/render_report.py \
  practice_18_kernel_data_dag/results/new-run/analysis/data_dag.json \
  --output practice_18_kernel_data_dag/results/new-run/report
python practice_18_kernel_data_dag/schedule_sensitivity.py \
  practice_18_kernel_data_dag/results/new-run/analysis/data_dag.json \
  --output practice_18_kernel_data_dag/results/new-run/report/sensitivity.json
python -m unittest discover -s practice_18_kernel_data_dag -p 'test_*.py' -v
```

若搬运原始结果时省略了 `compiler_cache/**/precompiled.h.gch`，必须按原
`compiler_artifacts.json` 生成 `unarchived_build_intermediates.json` 记录被省略项；
P13 校验只允许这类明确标记的编译中间文件缺席，其余证据不能静默丢弃。

原始日志、地址、服务器元数据和安装源码保存在被 git 忽略的 `results/`。
可分享的 `report/` 只导出逐任务名称、相对时长、图边、调度计划及汇总，不导出原始地址、
服务器路径、PID、日志或源码。完整原始图在各轮的 `analysis/data_dag.json`。

## 完成“精确逐 kernel DAG”还需要什么

必须进一步取得 native 内部 kernel 的真实 tensor / workspace 读写与分配生命周期，
并把它们和 CANN task identity 对齐。不能把同一 aclnn 调用的公开参数复制到其每个
kernel 就称为完成。接下来需要具体解决 Cast→MatMul、Slice→Cache、Slice→FIA 等
内部缓冲的生产、消费、释放和复用，还要检查 native 全局状态及改流后的 allocator
stream ownership。取得这些证据后才能移除相应保守约束并通过完整性门槛。

当前的关键路径、条件并行度和离线 stream 分配已可运行；**这一原生内部观测缺口仍未
解决，因此用户要求的“完整精确数据依赖 DAG”尚未验收完成。**
