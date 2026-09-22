# Practice 08：真实 KV 映射校验

- Request：`cmpl-practice08-cross-block-0-aa81b71d`
- 输入 / 输出：126 / 8 token
- 已计算并核对第一层 K/V 写入：133 个 token 位置
- 第一层：`model.layers.0.self_attn.attn`
- 最终 request block IDs（单 KV group）：`[1, 2]`

| step | token positions | request blocks | 新分配 blocks | slots |
|---:|---|---|---|---|
| 1 | 0..125 | [1] | [1] | 128..253 |
| 2 | 126..126 | [1] | [] | 254..254 |
| 3 | 127..127 | [1] | [] | 255..255 |
| 4 | 128..128 | [1, 2] | [2] | 256..256 |
| 5 | 129..129 | [1, 2] | [] | 257..257 |
| 6 | 130..130 | [1, 2] | [] | 258..258 |
| 7 | 131..131 | [1, 2] | [] | 259..259 |
| 8 | 132..132 | [1, 2] | [] | 260..260 |

## 第一层 KV 布局

- `key_cache`：shape=`[11781, 128, 2, 64]`，stride=`[16384, 128, 64, 1]`，
  dtype=`torch.bfloat16`，device=`npu:0`，element_size=2 bytes。
- `value_cache`：shape=`[11781, 128, 2, 64]`，stride=`[16384, 128, 64, 1]`，
  dtype=`torch.bfloat16`，device=`npu:0`，element_size=2 bytes。

所有步均通过 manager block IDs → CPU block table → NPU block table → slot_mapping 校验。
每步第一层写入后，按观测到的 slot 读取实际 K/V cache，与算子输入逐元素完全相等。

这是有同步和额外读操作的诊断运行，不用于性能或 race 结论；不覆盖其他层或其他请求。
slot 是缓存中的 token 槽编号，不是 token ID，也不是 NPU 物理内存地址。
