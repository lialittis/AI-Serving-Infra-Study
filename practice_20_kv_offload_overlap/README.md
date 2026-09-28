# Practice 20：真实 KV offload/reload 与计算重叠

本实验在 Ascend 910B2C 上运行真实 Qwen 请求，观察 NPU → CPU 保存、CPU → NPU
回载、独立请求计算和缓存块复用，构建带证据的 KV 数据与生命周期 DAG。

查看[结果](RESULTS.md)和[离线交互报告](report/index.html)。

## 实际路径

使用当前安装的 `AscendSimpleCPUOffloadConnector`：调度器与 block pool 来自 vLLM，
K/V 布局、后台拷贝线程和 DMA backend 来自 vLLM-Ascend。
`P20Connector` 通过 `kv_connector_module_path` 在实验进程中加载，未修改系统安装。
旧指南中的 `NPUOffloadingSpec` 引用的 `abstract/spec/mediums` 模块已被当前 vLLM
重构，不能直接运行；本次没有修复或替换那条旧路径。

三种模式是：

- `native`：保留原生后台队列、D2H/H2D stream、event 查询和缓存管理。
- `serialized`：保留相同 backend 和拷贝量，在提交 DMA 前等待计算流完成，
  在继续主线程前等待后台线程发布真正的完成 event，并等待该 event 完成。
  它仍有独立传输 stream，是强制串行对照；包括额外主机等待成本。
- `recompute`：关闭 CPU offload，保留相同容量的 NPU prefix cache，驱逐后重新计算。

只支持 Qwen2.5-0.5B-Instruct、BF16、TP=1、eager、同步 scheduler、单 KV group。
活动请求抢占会停止实验；多卡、取消和自动改流不在本次范围内。

## 请求与观测

NPU pool 为 97 个 128-token block，CPU pool 384 MiB。每周期先计算 A，
再用五个不同的 3104-token 请求制造缓存压力；随后启动 B，在 B 产生首 token 后
提交相同输入的 A′。A 的前缀分别为 1024/3072 token，另有 32 token 后缀。
A/A′ 生成 64 token，B 生成 256 token，压力请求各生成 4 token。

每个周期的输入在首个 block 内带不同标记，三模式复用相同输入；不通过 reset cache
替代容量驱逐。检查真实 NPU miss、CPU hit 和传输字节，并比较生成 token ID。

诊断模式记录 CPU/NPU pool 的分配与引用变化、传输批次、event 身份、完成通知、
request/block table 以及全窗口设备任务。没有新增设备等待或数值读回。
性能模式关闭这些重型观察器和 profiler，仅记录传输、命中和完成等少量协议计数。

## 如何解释 DAG

`execution_graph.json` 保存所有实际设备任务、必要主机节点、同步边、KV 访问投影
和独立生成的依赖要求。`host_program_order`、`host_queue`、`host_submission`、
`stream_order`、`event_query_complete`、`host_sync` 等构成实际 happens-before 图。
每条数据／复用要求必须在这个图中可达，不能用要求本身补出同步证明。

block generation 来自真实新分配，prefix hit 的 `touch` 不换代。缓存地址按
`base + block_id × page_bytes` 展开至所有 K/V 子张量，保留非零 storage offset。
跨 DMA 依赖在完整批次和 24 层 attention 调用边界上保守构建。
这不是原生 kernel 内部访存追踪，也没有补齐全模型 workspace 数据依赖。

后台线程不会自动继承 PyTorch CPU profiler scope。分析器通过真实 CANN connection、
enqueue/dequeue correlation 和已核对源码的 FIFO 批次顺序恢复关联；每批须有对应的
memcpy 数量、event record 和已发布的同一 event handle。不会根据最近的时间戳猜测。
`contracts.json` 固定已审计源码 SHA256，版本变化后须重新审计。

## 复现

在远端仓库根目录执行，保留 CANN 的运行时 `PYTHONPATH`：

```bash
python practice_20_kv_offload_overlap/verify_numerics.py --output /tmp/p1-numerics.json
python practice_20_kv_offload_overlap/run_experiment.py \
  --mode native --phase diagnostic --warmup 0 --repeats 1 \
  --output practice_20_kv_offload_overlap/results/new-diagnostic
python practice_20_kv_offload_overlap/run_suite.py --phase benchmark \
  --output practice_20_kv_offload_overlap/results/new-benchmark
```

本地离线检查不依赖 NPU：

```bash
python practice_20_kv_offload_overlap/analyze_benchmark.py RUN_BENCHMARK
python practice_20_kv_offload_overlap/analyze_dag.py RUN_DIAGNOSTIC
python practice_20_kv_offload_overlap/render_report.py \
  --graphs RUN_DIAGNOSTIC/analysis/execution_graph.json \
  --performance RUN_BENCHMARK/summary.json --output /tmp/p1-report.html
python -m unittest discover -s practice_20_kv_offload_overlap -p 'test_*.py' -v
```

报告的离线浏览器验证：设置 `P20_BROWSER_PACKAGE` 指向本机 Playwright 包，执行
`node practice_20_kv_offload_overlap/check_report.cjs`。已通过 3 个模式、36 个传输批次、
12 个 block 选择以及移动端布局检查；8 项因果关系与后台 event 发布测试通过。
记录见[离线验证](results/published/offline_validation.json)。

输出目录必须不存在，失败采集保留原样。服务只绑定 loopback，清理仅针对本次创建的
进程组。结果归档、诊断证据的位置和校验值见结果文档。
