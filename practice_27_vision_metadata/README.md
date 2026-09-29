# Practice 27：视觉元数据预计算与同步消减

本实验延续 [P3b](../practice_25_multimodal_overlap/README.md)：检验视觉 eager 路径中的元数据读取是否阻碍另一请求的语言计算提交。沿用完整 Qwen2.5-VL-3B-Instruct、Ascend 单卡、一个 CPU 提交线程、两张图片和独立请求 KV/MRoPE。

## 三组对照

| 变体 | 改动 | 仍然执行的视觉计算 |
|---|---|---|
| native | 原实现 | 全部 |
| lengths | attention 使用从 CPU grid 提前得到的分段长度，去掉每层设备长度差与 Q/K/V 的 `.tolist()` | 原 visual forward，包括其他元数据工作、所有网络层 |
| cached | 在 lengths 上，再预计算窗口/逆序索引和 RoPE cos/sin | patch embed、32 个视觉 block、merger，以及输入/输出索引 |

预计算对象由请求的实际 grid、像素 tensor 形状和当前模型构造；执行前检查形状。所有元数据在计时前同步就绪，保持引用到执行末尾。每次仍从真实图像像素计算视觉特征，未缓存图片的网络输出。该实验只接受已核验源码哈希、eager、eval、单图；不适用于训练、动态权重或未经核验的其他实现。

[源码审计](SOURCE_AUDIT.md) 解释分段长度、索引拷贝与证据的对应关系。

`metadata.py` 局部替换本进程模型实例的 attention 方法，在退出时恢复；不修改已安装 Transformers。实验中切换方法的操作在 pair 计时外；部署固定变体时只需配置一次。`lengths` 的 attention 代码从固定哈希的原方法生成，仅改变分段长度的来源，避免重写数值计算。

## 方法与边界

详见 [PLAN.md](PLAN.md)。576 次无 profiler 性能测量与 96 次 profiler 诊断分开；正式阶段另有 144 次预热/资格输出检查。每个 case 与原生完整 forward/generate 对照；每个变体的 features、logits、全部 KV 都以原实现为参考。

测量包含 A 语言、B 视觉、B merge/语言和终止等待；沿用 P25 的计时边界。预处理/输入传输/A 已准备的视觉与前缀/KV clone 在计时外。元数据预计算单列 CPU 长度准备、总准备时间和持久化设备内存：这些是实际成本，不能把复用后的时间当作首次请求端到端延迟。

LV/VL 轮换；六个（变体、流模式）组合每两轮旋转一个位置，使各组合在每种顺序下覆盖全部六个测量位置。按同轮进行串/双流与变体/native 配对，分别报告两种顺序。这里是单次进程实验，不能据此宣称跨日稳定收益。

分析复用 P25 的 host-flow/CANN/kernel CSV 精确对应和独立依赖检查，不按相邻时间戳猜测因果。图覆盖显式 stream FIFO/event/host join，尚非完整 native workspace 访存 DAG。图中的默认 stream 可能只承载公共起始 event，双流指 L/V 两条计算 stream，不能按所有时间线行数推断计算并发度。同步 API 计数可能包含嵌套；重叠来自实际 compute kernel 区间的并集交集。没有采集逐硬件核或带宽竞争计数器。

## 运行与重放

远端容器已把物理 NPU 5 映射为逻辑 0，不另设设备覆盖：

```bash
cd /data/tianchi
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export PATH=/usr/local/python3.12.13/bin:$PATH
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python practice_27_vision_metadata/run.py --smoke --output practice_27_vision_metadata/results/qualification-new
python practice_27_vision_metadata/run.py --output practice_27_vision_metadata/results/formal-new
```

输出路径必须不存在。本地分析只需 Python 标准库及同仓库 P17/P20/P25 模块：

```bash
python practice_27_vision_metadata/analyze.py /path/to/formal-run
```

源码变化需重新核验契约，不能跳过 `contracts.json` 检查。采集源文件清单固定于本次远端目录；交付后新增的离线工具不在原采集快照中。重新采集时需审查新的清单和哈希再更新契约，旧证据的原始快照保持不变。


正式结果见 [RESULTS.md](RESULTS.md)，三组交互比较与逐 kernel 证据见 [report/index.html](report/index.html)。归档分片可独立重放：

```bash
mkdir -p /tmp/p27-replay-new
cat practice_27_vision_metadata/results/published/evidence.tgz.part-* | tar -xz -C /tmp/p27-replay-new
python practice_27_vision_metadata/analyze.py /tmp/p27-replay-new/formal-r01
python -m unittest discover -s practice_27_vision_metadata -p 'test_*.py'
```

正式原始证据约 4 GiB，归档约 258 MiB，分成小于 50 MiB 的块；分析和完整图需要额外磁盘及数 GiB 内存。图按 native/lengths/cached 分别保存，避免一个页面同时载入全部百万级任务。原始证据中保留失败和资格轮次，但统计只取 formal-r01。
