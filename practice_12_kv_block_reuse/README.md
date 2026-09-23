# Practice 12：真实 KV block 释放与复用，eager / graph 对照

问题：**同一块真实 KV 存储先后交给 A、B 时，设备访问、原生结果等待、释放和再次使用是什么关系？切换执行模式会改变什么？**

[打开离线对照页面](results/2026-09-23-resource-comparison/index.html) ·
[对照结果](results/2026-09-23-resource-comparison/comparison.md) · [详细结论](RESULTS.md)。
HTML 用本地浏览器打开，可切换 eager / graph，逐步播放各自的真实生命周期。
新增[逐事件 / 逐次 replay 的资源记录](RESOURCE_RECORDS.md)：输入准备、存储身份、捕获快照、原生event与完成边界。

## 固定工作负载与模式开关

- Qwen2.5-0.5B-Instruct，910B2C，BF16，TP=1。
- 原生 `--num-gpu-blocks-override 2`：B0为null块，仅B1可分配。block-size=128。
- max-model-len=128、max-num-batched-tokens=128、max-num-seqs=1。
- 关闭 prefix caching、chunked prefill、async scheduling。
- 一次预热后，在同一profiler窗口中顺序发送 A、B。A输入hello ID×126，B输入world ID×126，
  各生成2 token，即prefill(126)+decode(1)。A的HTTP响应后才发送B。
- `--mode eager`（默认）：`--enforce-eager`，不编译、不设备图重放。
- `--mode graph`：编译mode=3、PIECEWISE、capture sizes=[1]、custom_ops=[all]。
  **126-token prefill执行编译callable；1-token decode重放25个普通计算分区，attention/KV保持直接调用。**
- 两种模式均显式关闭AOT和磁盘编译缓存，以相同观测脚本重采集。没有替换框架allocator，
  没有修改已安装源码或手动操纵空闲队列。

大池释放后将块放到队尾，A/B未必立即分到同一块。小池让原生allocator自然复用B1；
它是受控的串行复用基线，不模拟高并发负载，也不等同于FULL graph。

## 在远端复现两种模式

在 `/data/tianchi` 使用现有 Ascend Python 环境，依次运行（不同时启动两个服务）：

```bash
cd /data/tianchi
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export PATH=/usr/local/python3.12.13/bin:$PATH

python practice_12_kv_block_reuse/run_reuse.py --mode eager \
  --output practice_12_kv_block_reuse/results/my-eager
python practice_12_kv_block_reuse/run_reuse.py --mode graph \
  --output practice_12_kv_block_reuse/results/my-graph

for mode in eager graph; do
  python practice_12_kv_block_reuse/analyze_reuse.py \
    "practice_12_kv_block_reuse/results/my-$mode"
  python practice_12_kv_block_reuse/render_reuse.py \
    "practice_12_kv_block_reuse/results/my-$mode"
  python practice_12_kv_block_reuse/audit_resources.py \
    "practice_12_kv_block_reuse/results/my-$mode"
  python practice_12_kv_block_reuse/render_resources.py \
    "practice_12_kv_block_reuse/results/my-$mode"
done

python practice_12_kv_block_reuse/compare_modes.py \
  --eager practice_12_kv_block_reuse/results/my-eager \
  --graph practice_12_kv_block_reuse/results/my-graph \
  --output practice_12_kv_block_reuse/results/my-comparison
```

输出目录必须不存在；端口默认8012，脚本先检查占用，再启动自己的服务，预热、采集并停止
自己创建的进程组。HTTP请求仅发往127.0.0.1。源码、配置、模型文件指纹、脚本和结果均归档。

## 本地离线复核

最新主对照为 `2026-09-23-run05-eager-resources` / `2026-09-23-run06-graph-resources`，
使用相同版本的观测脚本。run02–04保持历史归档；旧记录不含完整资源字段，不追补伪造数据。
失败pilot run01有明确排除记录，不参与对比。

```bash
python3 practice_12_kv_block_reuse/compare_modes.py \
  --eager practice_12_kv_block_reuse/results/2026-09-23-run05-eager-resources \
  --graph practice_12_kv_block_reuse/results/2026-09-23-run06-graph-resources \
  --output practice_12_kv_block_reuse/results/2026-09-23-resource-comparison
python3 -m unittest discover -s practice_12_kv_block_reuse -p 'test_*.py' -v
sha256sum -c practice_12_kv_block_reuse/SHA256SUMS
```

离线分析只需Python≥3.7标准库，不需要torch/NPU。复用P07请求hook及P09–P11的分析辅助函数，
需保留仓库结构。`compare_modes.py`会拒绝不同输入、模型/源码、观测脚本、非模式参数或profiler配置的比较。
单轮生命周期使用 `analyze_reuse.py RUN` / `render_reuse.py RUN`；资源台账使用
`audit_resources.py RUN` / `render_resources.py RUN`。旧run仍可分析生命周期，资源审计会明确要求重新采集。

## 怎样观测，怎样判断

`lifetime_trace.py`只读取CPU allocator状态、CPU已有数据与NPU tensor元数据：

1. 真实allocate/free，ref_cnt、pool/block CPU对象身份、空闲队列前后状态。
2. 所有24层cache存储与FIA参数，逐层KV写入和attention调用。
3. 原生采样ID回传与`Event::synchronize`；不额外调用`.cpu()`、`.item()`、event query或设备等待。
4. graph模式下ACL wrapper的runtime mode、捕获快照、每次NPUGraph.replay和输入/输出资源核对。
5. ModelRunner输入/170个权重tensor存储、32次原生缓冲区拷贝、4条slot准备链、每次原生event的身份。

每模式均保留完整profiler CPU事件/设备任务清单；每个ATen输入输出的全部地址与隐藏workspace
生命周期仍未观测。详细字段、缺口和33项测试见[资源记录说明](RESOURCE_RECORDS.md)。

free参数可能是迭代器，观测器不提前遍历；返回时读取函数自己建立的列表。
分析使用同一profiler时间轴、真实flow ID、CANN connection ID与CSV，不按最近时间猜测kernel身份。

两种模式都保留192条直接KV/FIA链，检查它们先于原生结果等待结束，随后才释放、再分配。
graph模式另外列出重放内部缺少flow的任务，不伪造逐FX节点关联。
slot数值从CPU映射推导，未做P08式NPU数值读回；相同存储地址不等于相同逻辑分配生命周期。

带插桩的时间只作诊断，不能由每模式一次A/B请求推断加速比。本练习也不是通用race detector；
prefix sharing、并发、FULL graph或async scheduling需要新的受控实验。
