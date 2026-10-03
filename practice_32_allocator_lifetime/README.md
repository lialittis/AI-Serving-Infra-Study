# Practice 32：NPU allocator 的跨 stream 存储生命周期

本实验从已确认的[源码机制](../references/torch_npu_allocator_lifetime/README.md)出发，测试：A 在 S0 分配，被 S1 异步读取，最后一个存储引用消失后，S0 何时能把同一地址分配给 B。

三种模式只改变生命周期保护方式：`omit` 不登记 S1；`record` 在释放前调用 `A.record_stream(S1)`；`join` 在释放前让 S0 等待 S1 的末尾事件。所有模式都先确保 A 的初始化完成，再显式让 S1 等待生产事件。**B 仅用于地址观察，没有设备读写。**

- [结果和证据边界](RESULTS.md)。
- [无 profiler 配置矩阵](results/formal-02/summary.json)：只分析已完成的五个配置。
- [lazy reclaim 补齐矩阵](results/lazy-03/summary.json)。
- [自定义分配 stream 对照](results/custom-owner-01/summary.json)。
- [设备时间线](results/profile-01/index.html)：离线交互，可选配置、模式、重复和显示窗口；点击条目查看 flow 身份。
- [任务清单](../tasks/2026-10-02-practice32-allocator-lifetime.md)。

## 工作负载和测量边界

S1 使用独立、始终存活的 FP16 `[4096,4096]` 矩阵和输出，提交 64 次预热过的 `torch.mm(..., out=scratch)`，随后把 A 复制到始终存活的 `observed`。A 是 4 MiB 的 FP32 向量，`requires_grad=False`，没有保存 view 或 storage 别名。S0 最多分配并保留 32 个同尺寸候选。

四个事件在测量前创建和预热：生产完成、copy 前、copy 后、末尾。主机释放后检查 Python weakref，并由 allocator 历史中的 `free_requested` 核对实际存储释放。快照记录 `active_pending_free` / `inactive` / `active_allocated`；含原地址的合并 block 也会被识别。事件 query 是进度观测，不能独立证明实际 copy 访问尚未发生。

测量窗口内不写 B，不读回设备值，不调用 `empty_cache` 或设备同步，不写文件。内存快照对象留在 host；序列化、gzip、日志、全量数值校验都在本轮设备完成后进行。Profiler 单独运行，`record_shapes=False`、`profile_memory=False`，不作为性能基准。

`threads=1` 在主线程提交和释放；`threads=2` 通过 host queue 将唯一强引用转交给新线程，新线程在 S1 上提交、登记或回交依赖，并在 S1 当前时释放。替代分配在主线程的 S0 上进行。没有跨线程裸指针传递。

`record` 组同步后再申请四个候选。若 lazy 模式仍未复用，依据已完成窗口的快照选择一个大于最大 S0 缓存 block 的请求，迫使事件查询。请求大小最多 192 MiB；超界会标记 skipped，不会继续增大。这个控制只发生在旧设备工作完成之后，统计中区分“普通后续分配复用”和“cache miss 触发后的复用”。

Python `npu_stream` 属性会调用 `NPUStream::stream()`，可能排空 host 队列；`Stream.__hash__` 也读取这个属性。因此正式代码在测量前缓存 handle，并以 Python `id(stream)` 查找，窗口内不读取或 hash stream handle。源码证据见 [Stream.cpp:89](source_notes/Stream.cpp#L89)、[来源和哈希](source_notes/manifest.json)及 [NPUStream.cpp:364](../references/torch_npu_allocator_lifetime/sources/torch-npu/torch_npu/csrc/core/npu/NPUStream.cpp#L364)。

## 远端复现

使用已有 CANN/Python 环境，确认设备空闲；不更改安装源码、服务配置或全局环境。所有开关只属于新建的测试进程。每个配置、模式和线程数使用一个新的 Python 进程，每个进程执行五次试验。输出目录必须不存在。

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
PY=/usr/local/python3.12.13/bin/python
npu-smi info

$PY practice_32_allocator_lifetime/run.py \
  --output /data/tianchi/practice_32_allocator_lifetime/results/my-matrix \
  --configs baseline direct queue2 perstream expandable lazy \
  --threads 1 2 --repeats 5 --backlog 64
$PY practice_32_allocator_lifetime/analyze.py \
  /data/tianchi/practice_32_allocator_lifetime/results/my-matrix

$PY practice_32_allocator_lifetime/run.py \
  --output /data/tianchi/practice_32_allocator_lifetime/results/my-profile \
  --configs baseline direct --threads 1 --repeats 2 --profile
$PY practice_32_allocator_lifetime/analyze.py \
  /data/tianchi/practice_32_allocator_lifetime/results/my-profile
$PY practice_32_allocator_lifetime/analyze_profile.py \
  /data/tianchi/practice_32_allocator_lifetime/results/my-profile
```

配置以 baseline 为参照，分别改变 task queue `1→0/2`、per-stream queue `0→1`、expandable `False→True`、lazy reclaim `False→True`。环境变量名为 `TASK_QUEUE_ENABLE`、`PER_STREAM_QUEUE` 和 `PYTORCH_NPU_ALLOC_CONF`。实际环境、安装版本、source hashes、命令、PID、设备状态和每轮地址/事件/快照均保存。

## 离线分析和验证

```bash
python3 -m unittest discover -s practice_32_allocator_lifetime -v
python3 practice_32_allocator_lifetime/analyze.py \
  practice_32_allocator_lifetime/results/formal-02 \
  --configs baseline direct queue2 perstream expandable
python3 practice_32_allocator_lifetime/analyze.py practice_32_allocator_lifetime/results/lazy-03
python3 practice_32_allocator_lifetime/analyze.py practice_32_allocator_lifetime/results/custom-owner-01
python3 practice_32_allocator_lifetime/analyze_profile.py \
  practice_32_allocator_lifetime/results/profile-01 --check-existing
python3 practice_32_allocator_lifetime/verify_evidence.py \
  practice_32_allocator_lifetime/results \
  --output practice_32_allocator_lifetime/results/validation/evidence.json
python3 practice_32_allocator_lifetime/render.py practice_32_allocator_lifetime/results/profile-01
node practice_32_allocator_lifetime/check_report.cjs practice_32_allocator_lifetime/results/profile-01/index.html
```

分析拒绝缺失重复、未释放存储、错误地址标记、复制校验失败及窗口中的 OOM/retry；完整配置集合必须完成。`--configs` 只能显式选取原计划中的已完成配置，排除案例及其返回码会留在 summary，不能把停下来的整个 runner 伪装成成功。

Profiler 分析按 `async_npu` 与 `HostToDevice` 的精确 endpoint/origin 关联 copy、CANN call 和设备 task，并核对 connection ID。重复或缺失 flow 不按最近时间猜测。host scope 的结束时刻与旧 copy 的设备开始时刻来自同一 trace；未解析的任务另列，不声称完整的逐 kernel 内存读写 DAG。

原始 profiler 由 `.gitignore` 排除，保留在 Ascend 的对应目录。紧凑 task/flow 证据、完整 metadata、快照和哈希清单保留在 Git。需要重做 raw flow 分析时，先恢复原始文件；来源和校验值见每组的 `archive_manifest.json`。

`--check-existing` 只重算并比较，不改写归档；`verify_evidence.py` 同时核对原始 profiler CSV，需要 raw 文件已恢复。可从 Ascend 的 `/data/tianchi/practice_32_allocator_lifetime/results/profile-01` 取回各案例 `profiler/`，按清单相对路径放入本地 `results/profile-01`。全量正式紧凑证据另存于 `/data/tianchi/p32-evidence.tgz`。formal-01 是未纳入主结论的 pilot，本地只存摘要和来源；它的清单单独列出远端保留但本地省略的文件。各历史 source 副本保留原始字节，不进行格式清理。

## 当前范围

本阶段验证原生 eager Tensor allocator、默认/自定义分配 stream、单/双 host 线程和六个配置。没有 vLLM KV 逻辑块、HCCL 多 rank、graph capture/replay、自定义 kernel 或 B 的冲突读写。后续 HCCL 和 graph 必须建立独立实验，不能直接推广本轮结论。
