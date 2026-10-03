# 跨 stream 调用点审计：谁在无保护地跨越 stream 传递 tensor

静态扫描日期：2026-10-03。目的：[P32](../../practice_32_allocator_lifetime/RESULTS.md) 证明了地址提前交回的前提，
[P33](../../practice_33_conflict_access/RESULTS.md) 证明了无保护时冲突访问必然发生（omit 10/10 全量污染）。
本审计回答下一层问题：**安装的 vLLM / vLLM-Ascend / torch-npu / triton-ascend 源码里，哪些调用点
把 tensor 带到别的 stream 使用，各自用什么保护，是否存在 omit 等价结构。**

## 扫描范围与版本

远端 `ascend910` 安装树：vllm `0.21.0+empty`（`/vllm-workspace/vllm`，HEAD `ad7125a`）、
vllm-ascend `0.21.0rc1`（`/vllm-workspace/vllm-ascend`，HEAD `80610e44`）、
torch_npu `2.10.0`（site-packages，C++ 源见[本地副本](../torch_npu_allocator_lifetime/sources/)）、
triton_ascend `3.2.1`（`triton/backends/ascend/`）。扫描命令：

```bash
grep -rn --include='*.py' -E 'npu\.Stream\(|npu\.stream\(|stream_switch' vllm_ascend/
grep -rn --include='*.py' -E 'record_stream|wait_stream|wait_event' vllm_ascend/
grep -rn 'RecordStream' torch_npu/csrc op-plugin/   # C++ 侧
```

**全局结论：vllm_ascend 全库 0 处 `record_stream`；vllm core 唯一一处在 CUDA 分支里
（`vllm/distributed/parallel_state.py:911`，`tensor.record_stream(torch.cuda.current_stream(...))`，
无 NPU 对应）；torch-npu 的 op-plugin 在 4 个内部切流算子里显式 `recordStream`。**
即：上层框架完全依赖 wait 型顺序保护，登记型保护只出现在 C++ 算子库自己身上。

## 逐点判定

判定基准来自 P33：**join 等价**（新用途经 stream 顺序排在旧用途之后）与 **omit 等价**
（旧用途未完成时同地址新用途无任何顺序关系）。mitigated-by-timing 表示结构上 omit 等价、
但当前调用间隔在实践中掩盖了它。

### vllm_ascend/sample/sampler.py（Qwen 基线可达，P17 已实测其 stream 形态）

| 站点 | 结构 | 判定 |
|---|---|---|
| `do_async_exponential`（[sampler.py:89-102](sources/vllm-ascend/vllm_ascend/sample/sampler.py#L89)） | q 在 global_stream(44) 分配+填充，先导 `global.wait_stream(current)`；消费端 `self.async_event.synchronize()`（host 级，[L188](sources/vllm-ascend/vllm_ascend/sample/sampler.py#L188)）后才 `div_`；旧 q 由 `self.q` 持有到下一轮替换 | **join 等价，受保护**：下一轮 fill 先导等待排在旧消费之后，事件链完整。代价是 host 同步 |
| `fill_exponential`（[sampler.py:32-41](sources/vllm-ascend/vllm_ascend/sample/sampler.py#L32)） | q 在 global_stream 上 `empty_like`+`exponential_`，**无先导等待**；块随函数返回进入 global_stream 池；消费在 current(46) 上 `div_(q)`；下一轮 fill 直接从同池取回同块（P32 已证同 stream 池内即刻复用） | **omit 等价（结构）**：下一轮 fill（global 上的写）与上一轮 `div_`（current 上的读）之间无顺序关系。实践中被调度间隔缓解（div_ 早于下一轮 forward 结束执行），属 **mitigated-by-timing**，非受保护 |

`fill_exponential` 是本轮审计最重要的候选：它不是当前 greedy 基线（greedy 走 argmax 不用 q），
但任何 top-k/top-p 采样配置都会进入；结构上与 P33 omit 模式同构（S44↔S0 分配者、S46↔S1 消费者、
下一轮 fill↔B 写入）。

### vllm_ascend/attention/（按可达性分层）

| 站点 | 结构 | 判定 |
|---|---|---|
| `attention_v1.py` `update_stream`（[L419/L488/L586](sources/vllm-ascend/vllm_ascend/attention/attention_v1.py#L419)） | graph 模式下 KV 元数据与 workspace 更新提交到独立 update_stream，events 记录后由 replay 消费 | graph 路径，与 P32 遗留的 private pools/replay 外部存储同范围，未单独判定 |
| `dsa_v1.py` DSV4 overlap stream（[L52-56](sources/vllm-ascend/vllm_ascend/attention/dsa_v1.py#L52)）及 L1501-1842 的 wait_event/wait_stream 链 | aux stream 计算、main stream 多点 join | wait 链完整，属 join 等价；DeepSeek 专用，Qwen 基线不可达 |
| `attention_cp.py` / `dsa_cp.py` / `mla_cp.py` comm streams（如 [attention_cp.py:992-1044](sources/vllm-ascend/vllm_ascend/attention/context_parallel/attention_cp.py#L992)） | CP 通信流与计算流成对 wait_stream | 多卡路径；HCCL Work 的存储保留/撤销登记是 P32 遗留项，本审计不覆盖 |

### torch-npu 2.10.0（C++，本地源码副本）

| 站点 | 结构 | 判定 |
|---|---|---|
| op-plugin `DropoutKernelNpu.cpp:128`、`DropoutWithByteMaskKernelNpu.cpp:100`、`DropoutWithAddSoftmaxKernelNpu.cpp:63`、`FusedAttentionScoreKernelNpu.cpp:180` | 算子内部切流执行后，把产物 mask 的存储 `recordStream` 回原 stream | **正向对照**：库作者明确知道不登记会在复用时损坏，说明该风险在真实算子里出现过 |
| allocator `recordStream`/`insert_events`/`process_events`（NPUCachingAllocator.cpp:1557/2984/3030） | P32 已逐行核对的登记/事件回收机制 | 机制存在且 P33 实测有效；问题是上层必须显式调用 |

### triton-ascend 3.2.1 launcher

`triton/backends/ascend/driver.py` 的 `get_current_stream`（[L188](sources/triton/backends/ascend/driver.py#L188)）
取**调用者当前 stream**，`rtKernelLaunch` 直接在该 stream 下发；launcher 不自建 stream、不切流、无登记。
结论：**launcher 本身不引入跨 stream 生命周期**；风险只来自调用方自己的 stream 上下文（P13/P30 已实测其行为）。

## 候选与后续实验

1. **[候选 1] `fill_exponential` 定向实验**：用 P33 的探针结构直接复刻该模式（S44 分配→S46 消费→立即释放→S44 再分配写入），验证在 vLLM 真实调度间隔下是否复现损坏，以及缩小间隔（如连续采样循环）时是否必然复现。这能把"结构 omit 等价 + 实践缓解"变成定量结论。
2. **[候选 2] graph update_stream 与 replay 的存储边界**：P32 遗留项，attention_v1 的 update_stream 是入口。
3. **[候选 3] HCCL/CP 路径**：需要多卡环境。
4. vllm core 若未来出现 NPU 通用 record_stream 需求，目前无任何先例可循（唯一实现是 CUDA 分支）。

## 边界

静态审计不证明运行时一定发生或一定不发生：它定位结构与保护方式，发生与否由 P33 式定向实验回答。
本审计未覆盖：自定义用户算子、310P 专用路径（`_310p/`，含 `.cpu()` 同步采样）、多卡 TP/CP 的全部
通信实现、以及 graph 捕获内部的私有池行为。上述文件摘抄见 [sources/](sources/)。
