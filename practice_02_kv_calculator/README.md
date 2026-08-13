# Practice 02: KV-cache memory calculator

For an analysis of how to reverse the memory formula into maximum token capacity,
see [Analysis.md](Analysis.md).

## Goal

Estimate the raw tensor memory required by a model's KV cache and understand which
model and workload parameters control it. The calculator uses only Python's
standard library, so it does not require PyTorch or a GPU.

## Formula

```text
KV bytes = 2 × layers × batch size × sequence length
             × KV heads × head dimension × bytes per scalar
```

The leading `2` represents the key and value caches. The estimate assumes that
every sequence in the batch has the same cached length.

## Basic usage

```bash
python practice_02_kv_calculator/kv_calculator.py \
  --layers 32 \
  --batch-size 1 \
  --sequence-length 8192 \
  --kv-heads 8 \
  --head-dim 128 \
  --dtype bf16
```

This example produces a raw KV-cache estimate of `1.00 GiB`.

You can derive the head dimension from common model configuration fields instead:

```bash
python practice_02_kv_calculator/kv_calculator.py \
  --layers 32 \
  --sequence-length 8192 \
  --kv-heads 8 \
  --hidden-size 4096 \
  --attention-heads 32 \
  --dtype bf16
```

Here, `head_dim = hidden_size / attention_heads = 4096 / 32 = 128`.

For machine-readable output, add `--json`.

## Finding inputs in a model configuration

Common configuration names are:

| Calculator input | Common model-config field |
|---|---|
| `--layers` | `num_hidden_layers` |
| `--kv-heads` | `num_key_value_heads` |
| `--hidden-size` | `hidden_size` |
| `--attention-heads` | `num_attention_heads` |
| `--head-dim` | `head_dim`, or derive it |

`--sequence-length` is a workload choice: the number of prompt and generated
tokens currently retained in the cache. `--batch-size` is the number of concurrent
sequences represented by the estimate.

If a model configuration has no separate `num_key_value_heads`, it commonly uses
standard multi-head attention (MHA), where KV heads equal attention heads. Confirm
this in the model implementation before relying on the estimate.

## Why KV heads matter

- **MHA:** KV heads equal query/attention heads.
- **GQA:** several query heads share each KV head, reducing cache memory.
- **MQA:** all query heads share one KV head, so `--kv-heads 1`.

For otherwise identical models, changing from 32 KV heads to 8 KV heads makes the
raw KV cache four times smaller.

## Supported cache dtypes

The calculator recognizes `fp32`, `fp16`, `bf16`, `fp8`, `int8`, and `int4`, plus
the longer aliases `float32`, `float16`, and `bfloat16`.

The selected type must describe the cache storage, which is not necessarily the
same as the model-weight dtype. Quantized cache formats can also require scales or
other metadata that this rough calculation does not include.

## What the estimate excludes

The result represents dense K and V tensor storage. A serving runtime may require
additional memory for:

- block-size rounding and unused slots
- allocator fragmentation
- page tables and cache metadata
- temporary attention workspaces
- model weights, activations, logits, and runtime state
- quantization scales or zero points

The calculator is therefore suitable for understanding and rough capacity
planning, not as a guarantee that a workload will fit into a device's memory.

## Tests

```bash
python -m unittest discover -s practice_02_kv_calculator -p 'test_*.py'
```

One test reuses Practice 01's configuration and confirms its four-token cache is
exactly 1,024 bytes. Another checks the 32-layer GQA example above is 1 GiB.
