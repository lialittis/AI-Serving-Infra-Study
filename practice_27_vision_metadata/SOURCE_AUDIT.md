# 元数据消融的源码依据

采集时安装的 `transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.py` SHA256：
`9d6d15040bdb985d9518117ed029f6a318a38abc06091462d96f16cf1d3d7820`。
原文保留于归档 `formal-r01/sources/installed/transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.py`，实际生成的 attention 方法保留于 `formal-r01/patched_attention.py`。

## lengths：只改变分段长度来源

原 attention 的非 flash 分支（约 264 行）先算 `cu_seqlens[1:] - cu_seqlens[:-1]`，再在 Q/K/V 三个 tensor 的切分中各调用一次 `lengths.tolist()`。窗口分支的累计长度在设备上；整图 attention 的累计长度来自 CPU grid。当前 32 个视觉 block 中 28 个使用窗口 attention，因而原 trace 中有 84 次此类设备长度读取带来的同步 API 调用。

`metadata.py` 从已核验的方法源码生成局部方法，只把这两处改为读取预先保存的 Python tuple。矩阵运算、RoPE 作用于 Q/K、attention、softmax、投影等原语保持原实现。长度取自同一 `get_window_index()` 和 CPU grid，保留 `unique_consecutive()` 去除重复累计长度的语义；选择窗口或完整 attention 的层索引来自当前模型。

## cached：移动其他固定形状元数据

原视觉 forward（约 455–521 行）包含：patch embedding；根据 grid 构造 RoPE；窗口索引和累计长度；按窗口重排 hidden states/位置；32 个 block；merger；按窗口逆序恢复输出。

cached 预先计算位置的 cos/sin、窗口索引、逆序索引和 Python 分段长度，索引提前上传 NPU。forward 中仍执行真实像素的 patch embedding、hidden states 重排、全部 block、merger 与输出重排。缓存中不保存 hidden states 或 image features。实验限定单图/eager/eval，并检查 grid 与 pixel shape，元数据来自当前模型而非跨模型共享的全局缓存。

## 证据边界

同步审计通过 CPU 算子区间包含及精确 enqueue/dequeue flow 关联到 `aten::copy_`、`aten::index` 和 `unique_consecutive` 等调用。CPU/NPU 时间戳不能单独建立数据依赖；图的 feature/event 依赖仍由 P25 的独立证明器核验。

结合原方法的执行顺序，视觉末尾的 index→to→copy 同步可定位到逆序索引消费。cached 一次移除了多种元数据工作；本轮没有增加“仅逆序索引驻留设备”一组，所以不能把 cached 的全部收益单独归因于逆序索引。同步 API 条数也不等于独立等待次数。
