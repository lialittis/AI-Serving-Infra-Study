# Practice 24：完整模型 graph replay 与双流对照

将 Practice 23 的完整预训练 Qwen2.5-0.5B forward 扩展为固定形状 `torch.npu.NPUGraph`。在**同一完整 FP32 配置**下比较 eager / graph × serial / parallel / batch=2，分开记录 capture、首次 replay、预热和正式计时。

查看[结果](RESULTS.md)、[离线交互执行图](report/index.html)和[验证记录](results/published/validation.json)。原 Practice 23 BF16 batching 失败保持原状；本实验沿用[数值后续](../practice_23_independent_inference/NUMERICS.md)已通过的独立 FP32 控制组。

## 捕获与状态约束

- 每个任务一次完整 forward，返回最后位置 logits 与 24 层 KV。prefill 长度 128 / 1024，decode 初始上下文 128 / 1024。
- **固定 decode 步**：位置、mask、KV 长度固定；重新填充相同地址的输入 token 和初始 KV 后重复该步，不是不断生成 token 的循环。
- A、B、batch 各自独立 capture 与 pool；所有 12 张图保持存活。serial 与 parallel 使用同一对图，只改变 replay 调用 stream。
- 共享只读权重、mask 和 position。input IDs 克隆；原始 input KV 单独持有引用，因为 `DynamicCache.update` 在 capture 中将 Python 字段替换为 output KV，replay 仍会读取旧地址。
- 先完成上一轮末尾 event 等待，再回填输入；全局输入就绪同步在计时外。读取输出也在末尾 event 完成之后。
- `capture_error_mode='global'`；不共享 pool、不启用 superkernel、不使用自动 capture dispatch。不是 native vLLM 调度或 HTTP serving。

serial / parallel 与同形状 eager 基线要求逐元素相同；batch 保持 `atol=0.0625, rtol=0.02` 并要求贪心 token 一致。每个 pair 校验两个任务各 49 个 tensor；输入和初始 KV 交换复用检查独立执行。

## 执行图的证据

调用 stream 和 graph 内部 stream 分开显示。普通任务核对精确 flow 与 kernel CSV；graph 核对 source-pinned 捕获对象、唯一存活 Model ID、完整 stream/task 序列及 CANN replay boundary。debug dump 的时间戳不用于性能分析。部分 replay boundary 只有 CANN connection，没有 HostToDevice flow；只采用唯一 connection 与完整 native-thread scope 包含关系，拒绝最近时间戳配对。

HB 图包含 stream FIFO、event、host join、graph launch 与 graph completion。输入就绪／输出完成要求另行构造，独立检查可达性；反事实测试删除边后必须失去相应证明。每张图内部只有一条实测 stream；更多内部流的实现需另审计，当前分析器会拒绝套用该契约。

没有恢复 native kernel 内部精确访存／workspace DAG、RI native handle、精确 NOTIFY ID 或原生流创建调用。完整任务执行图不等于完整数据依赖图。分析结论限定每个已声明 trial 的边界，跨 trial 主机准备流程没有全部建图。

## 实机复现

在已安装对应版本 torch-npu / Transformers 且已有本地权重的 Ascend 环境中，保持同仓库 Practice 23 目录可导入：

```bash
cd /data/tianchi
python practice_24_graph_replay/qualify.py \
  --output practice_24_graph_replay/results/qualification-new
python practice_24_graph_replay/run.py \
  --output practice_24_graph_replay/results/formal-new
```

输入目录必须不存在；本容器已映射物理 5 为逻辑 0，沿用容器映射；不要将 npu-smi 的物理编号直接写入 ASCEND_RT_VISIBLE_DEVICES。权重路径沿用 Practice 23 `model.py`。脚本不安装包或修改设备。分析器锁定本轮采集源码哈希，版本或 harness 改变后需要重新审计并更新契约，不能跳过检查套用旧证据。

## 离线重放

只需 Python 标准库及同仓库 Practice 17 / 20 的 flow / HB 公用代码：

```bash
mkdir -p /tmp/p24-replay
tar -xzf practice_24_graph_replay/results/published/evidence.tgz -C /tmp/p24-replay
python practice_24_graph_replay/analyze.py /tmp/p24-replay/formal-r01
python practice_24_graph_replay/render_report.py \
  /tmp/p24-replay/formal-r01/analysis/execution_graph.json /tmp/p24-report.html
python -m unittest discover -s practice_24_graph_replay -p 'test_*.py'
```

本轮从归档独立核对 62 个文件，重放汇总和完整执行图逐字节一致。7 项反事实测试通过；离线浏览器覆盖全部 48 个 trial、节点身份／依赖详情、连线切换和 390px 窄屏，见[重放验证](results/published/replay_validation.json)。

原始证据约 330 MiB，trace 包含约 143 万个事件，Python 分析需相应内存。证据包含 qualification 与正式运行；CANN 二进制原始缓冲、FRAMEWORK 与数据库留在远端，不重复纳入。HTML 内嵌压缩图数据，可以离线打开；`SHA256SUMS` 和 manifest 支持完整性校验。

| 文件 | 用途 |
|---|---|
| `harness.py` | 静态输入／KV 生命周期、capture、eager/replay 调用与计时 |
| `qualify.py` / `run.py` | 资格检查／正式平衡对照与独立 profiler |
| `contracts.json` | 已审计采集源码哈希与图对象绑定契约 |
| `trace_graph.py` / `analyze.py` | 精确任务归因、HB、性能及正确性汇总 |
| `render_report.py` / `publish_results.py` | 离线图、报告与证据发布 |
| `test_analysis.py` | 基于真实图删除同步边的反事实检查 |

后续 [P3b 视觉／语言并发实验](../practice_25_multimodal_overlap/README.md)已完成；持续 decode 的 graph 输入更新另属扩展实验。
