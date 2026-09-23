# Practice 11：把一个 FX attention 节点连接到真实执行

只回答一个问题：**第一层 `unified_attention_with_output` 实际使用哪些 tensor、
访问哪些 KV 存储，并对应哪些 NPU 执行？**

本练习沿用 Practice 10 的 Qwen2.5-0.5B-Instruct、单卡 Ascend 910B2C、BF16、
TP=1、block size 128、`mode=3`、`PIECEWISE`、设备图捕获尺寸 `[1]`。
本次重新采集 graph 配置下的 profiler，不能把 Practice 09 的 eager trace 当作本次执行证据。

结果入口：[RESULTS.md](RESULTS.md)。

已完成的阅读材料：[Prefill / Decode 交互式对照](results/2026-09-23-run02/analysis/attention_viewer.html)、
[Prefill SVG](results/2026-09-23-run02/analysis/prefill.svg)、
[Decode SVG](results/2026-09-23-run02/analysis/decode.svg)。HTML 请在本地浏览器打开。

## 实验方法

1. 启动自己的本地 HTTP 服务，保存实际版本、源码和完整命令。
2. 在原生编译阶段捕获整张 FX 图，以及第一层前、attention、本层后的三个分区。
3. 发送一次 126-input / 2-output 的预热请求。
4. 启动 profiler，再发送一个相同长度的测量请求。
5. 关联 request ID、FX layer_name、真实函数范围、tensor 存储与 profiler flow。
6. 停止采集、关闭本实验创建的服务，离线检查并生成阅读材料。

观测使用 `sys.setprofile`，不替换模型、编译器或 attention 实现。
`record_function` 只给主机范围加标签，不表示该范围返回时 NPU 已完成。
不复制 NPU tensor 数值到 CPU，不插入显式设备同步，不修改 KV 数据。
CPU block table 来自已有主机数组；NPU block table 和 slot mapping 只读取元数据。

## 在远端复现

需要已同步的这些目录：本练习、Practice 03、07、09、10。运行脚本复用 03 的环境收集器、
07 的请求追踪辅助代码、10 的原生 FX 导出与源码收集器；离线分析复用 09 的 trace 工具。

```bash
ssh ascend910
cd /data/tianchi
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export PATH=/usr/local/python3.12.13/bin:$PATH

python practice_11_attention_execution/run_attention.py \
  --output practice_11_attention_execution/results/my-run

python practice_11_attention_execution/analyze_attention.py \
  practice_11_attention_execution/results/my-run

python practice_11_attention_execution/render_attention.py \
  practice_11_attention_execution/results/my-run
```

输出目录必须不存在，端口 8011 必须空闲。运行时采集脚本保存在输出目录的
`instrumentation/`；本次实际调用的源码和文件指纹也会归档。

## 从哪里开始读代码

| 文件 | 阅读重点 |
|---|---|
| `run_attention.py` | 相对 Practice 10 增加 profiler 和一次预热请求 |
| `attention_trace.py` | `custom_attention`、`context`、`cache_write`、`partition` 四个观测点 |
| `analyze_attention.py` | 怎么验证存储对应关系，以及用真实 flow 关联设备任务 |
| `render_attention.py` | 将已验证的数据生成 SVG 和离线对照页面；生成 SVG 需要 Graphviz |

`layout()` 保存 shape、dtype、stride、storage pointer、data pointer、storage offset。
**相同存储地址在同一存活区间可以说明共享存储；不能据此断言数值相等，也不能跨释放/复用判断对象身份。**

FX 的 `view` 通常创建共享存储的视图；随后 O projection 才读取其中的数值进行计算。
因此我们既观察 attention 的输出，也观察后续编译分区的输入。

## 如何理解证据

- FX 说明算子的参数、数据依赖及可变参数声明。
- 运行时 tensor 元数据说明实际传递了哪些存储和视图。
- Python 范围说明主机执行了哪些函数。
- profiler 的关联 ID 说明哪些设备任务与哪些主机事件关联。
- graph replay 可能不重新经过原来每个 Python/算子入口；关联只能精确到有证据的范围。

这个实验没有逐元素验证 KV 或 attention 输出，也不是 race 检测器和性能基准。
Practice 08 的逐元素 KV 校验属于另一个会改变同步行为的诊断实验。
