# 同步对照：少等几次，还是提前使用了结果？

已在同一 Ascend 910B2C 上完成实测。**把每轮等待合并到最后，结果仍正确，完整耗时只减少约1.75%；不等完成就读取异步回传的结果，本次读到了旧的初始化值。**

本页两条计算分支互相独立。随后补充的[A→B依赖与独立C实验](CROSS_STREAM_DEPENDENCY.md)，专门验证设备侧跨stream依赖，区分“CPU提前读取”和“消费者算子提前执行”。

打开[交互对照报告](results/2026-09-24-sync-run01/analysis/index.html)，可逐次比较提交阶段与末尾等待。原始参数、计时、每次提前读取和最终校验见 [sync_run.json](results/2026-09-24-sync-run01/sync_run.json)。

## 到底保留、去掉哪一处同步

沿用 Practice 16 的两个独立 stream，固定 FP16 `[4096,4096]` 矩阵乘法与 FP32 `[67108864]` 向量乘法。每批四轮，每轮每分支六次，共48个计算 kernel。

每轮使用独立输出存储，并从各分支输出取前4个元素，在**该分支自己的 stream**上异步拷贝到 pinned CPU 缓冲区，再记录终点 event。
CPU 缓冲区每批开始前填成 `-7`。矩阵结果应为 `1`，向量结果应为 `0.375`。
采样拷贝用于明确观察“CPU 此时读到了什么”；两种性能策略执行完全相同的拷贝和 event record。

| 策略 | 每轮结束 | 所有轮次提交后 | 读取/验证 |
|---|---|---|---|
| `per_round` | Host 等待 A/B 两个 event | 最后一轮已经完成 | 计时结束后验证全部输出 |
| `final_only` | 不等待，直接提交下一轮 | 等待两条流各自最后一个 event | 等待完成、计时结束后验证全部输出 |
| `early_read` | 不等待，立即读 pinned CPU 缓冲区 | 为清理和最终核验，补上两条流的最终等待 | 比较提前值与完成后的值；独立诊断，不计入性能对照 |

始终保留：初始化和每批开始前的设备完成屏障、同一 stream 的执行顺序、退出或复用存储前的最终等待。
本次移除的是**中间轮次的 Host 等待**，不是所有依赖关系，也没有移除生产者到采样拷贝的顺序。

```text
每轮等待：
提交第0轮 → 等A/B完成 → 提交第1轮 → 等A/B完成 → ……

最后统一等待：
Host：提交第0轮 → 提交第1轮 → 提交第2轮 → 提交第3轮 → 等最后的A/B
NPU A：mm第0轮 → copy → event → mm第1轮 → …… → 最后event
NPU B：mul第0轮 → copy → event → mul第1轮 → …… → 最后event
```

等待每条流的最后一个 event，已覆盖该流之前的计算和拷贝。因此不必重复等待中间 event。
这成立的前提是本练习中两分支独立、每轮输出独立、输入不改变、全部存储一直有效；不能照搬到跨流消费、共享缓冲区覆盖或 KV block 抢占。

## 性能：必须比较“都完成了”的耗时

预热之后，两种安全策略各重复七次并交替顺序。**性能测量没有开启 profiler，也没有在计时内调用输出校验、`.cpu()` 或 `.item()`。**
每批总计时从第一轮提交开始，到两条流均完成为止。输出逐元素校验在计时结束后进行。

| 策略 | 每批Host等待次数 | 提交阶段中位数 | 完整耗时中位数 | 完整耗时范围 |
|---|---:|---:|---:|---:|
| 每轮等待 | 8 | 20.521903ms | 20.522100ms | 20.443–20.538ms |
| 最后统一等待 | 2 | 1.201595ms | 20.163228ms | 20.140–20.394ms |

“提交阶段”在每轮等待模式里包含轮内等待；在最后统一等待模式里不包含末尾等待。
因此 **1.20ms 是 Host 提交完这些工作的时间，不是 NPU 完成计算的时间**，不能拿它与20.52ms相除来宣称十几倍加速。

公平比较是20.522100ms与20.163228ms：完整耗时中位数减少0.358872ms，约 **1.75%**。本次负载主要时间仍由设备计算决定；Host 可以更早去做别的工作，但本练习没有加入其他 CPU 业务来测这部分收益。
这是固定负载、七次重复的小规模测量，没有进行统计显著性分析，也不代表模型推理会获得同样收益。

## 正确性：提前读到了什么

独立诊断执行七批，每批四轮、两个分支，共 **56次分支采样，每次读取4个元素**：

| 观察 | 实测 |
|---|---|
| 提前读取错误的分支采样 | 56 / 56 |
| 提前读到的值 | 都是 `[-7, -7, -7, -7]` |
| 读取后查询对应event | 56 / 56仍未完成 |
| 补上最终等待后的采样 | 全部为正确值 |
| 补上等待后的完整输出tensor | 每轮逐元素验证全部正确 |

这表示**CPU 在结果尚未就绪时使用了缓冲区**。`-7` 是显式设置的哨兵值，不是矩阵乘法或向量乘法计算出来的错误答案。
`event.query()`只查询完成状态，不等待；本次是在读取之后查询，避免用阻塞等待掩盖问题。

没有使用普通阻塞式设备转CPU拷贝来制造“无同步读取”，因为它可能在内部等待完成。这里使用预先分配的 pinned CPU 缓冲区和 `copy_(..., non_blocking=True)`，CPU直接读取其现有内容。
提前读取存在时序竞争；换负载或机器可能偶尔读对，读对一次不能证明安全。本次56次全错，也不能推广为所有无同步程序必然100%读错。

## 另采profiler验证实际等待位置

性能测量与诊断结束后，另采一轮带 profiler 的三种策略，复用同一工作函数。三个模式均有48个计算kernel、8次采样拷贝和8个event record；没有通过少算工作获得结果。

| profiler组 | 计算kernel | D2H采样拷贝 | event record | Host event synchronize |
|---|---:|---:|---:|---:|
| 每轮等待 | 48 | 8 | 8 | 8 |
| 最后统一等待 | 48 | 8 | 8 | 2 |
| 提前读取（最终清理仍等待） | 48 | 8 | 8 | 2 |

144个计算kernel均匹配实际CPU/CANN flow及kernel CSV。24次拷贝和24个event记录也关联到设备任务，逐条核验同一流上的“计算 → 拷贝 → event”；12次Host等待的返回晚于对应event完成。
49个设备任务处于测量范围之外，属于校验/控制范围，单独保留不混入上述统计。

三种策略共24批（14批性能、7批诊断、3批profile），完成后核验 **192张完整输出tensor**，全部正确。
本次只验证独立计算和CPU消费边界，没有测精度损失、模型任务准确率、抢占、共享KV或设备侧跨流数据竞争。

## 复现与阅读代码

远端已有脚本；使用尚不存在的输出目录：

```bash
cd /data/tianchi
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export PATH=/usr/local/python3.12.13/bin:$PATH

python practice_16_multistream_parallel/run_sync_comparison.py \
  --output practice_16_multistream_parallel/results/my-sync-run
python practice_16_multistream_parallel/analyze_sync.py \
  practice_16_multistream_parallel/results/my-sync-run
```

`--rounds 4 --pairs 6 --repeats 7`为默认配置。输出存储按轮独立，主输入与输出约1.406GiB（不含工作区和校验临时张量）；不在计时中分配主输入/输出，也不释放任何尚在使用的存储。

先读 [run_sync_comparison.py](run_sync_comparison.py) 的 `batch()`：提交计算 → 同流拷贝 → record → 按模式选择等待/提前读 → 最终汇合 → 完整校验。
[analyze_sync.py](analyze_sync.py) 重新计算计时和错误数量，核验实际flow和等待；[render_sync.py](render_sync.py)生成报告。

本地无需NPU：

```bash
python3 practice_16_multistream_parallel/analyze_sync.py \
  practice_16_multistream_parallel/results/2026-09-24-sync-run01
python3 -m unittest discover -s practice_16_multistream_parallel -p 'test_*.py' -v
cd practice_16_multistream_parallel
sha256sum -c SHA256SUMS
```

共16项测试通过：保留原8项，新增提前读/最终正确性、提交时间冒充完成时间、profiler混入性能、输出存储别名、非pinned目标、缺flow和错误轮次等反例。
接口语义见 [torch-npu官方Stream/Event](https://github.com/Ascend/pytorch/blob/master/torch_npu/npu/streams.py)；实际已安装版本保存在本轮`sources/streams.py`。
