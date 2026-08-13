# Practice 01: observing `past_key_values`

For the implementation architecture, tensor contracts, design tradeoffs, and
extension path, see [Code_Design.md](Code_Design.md).

## Goal

See what a KV cache contains and why autoregressive decoding reuses it.

The lab uses a deliberately tiny, randomly initialized causal language model:

- 2 Transformer layers
- 2 attention heads
- hidden size 16, so each head has dimension 8
- batch size 1
- CPU only

It is not trained, so its predicted tokens are meaningless. Its attention and
cache mechanics are the same ones we want to study.

## Run

```bash
source ~/venvs/py311/bin/activate
python practice_01_kv_cache/observe_kv_cache.py
```

Useful options:

```bash
python practice_01_kv_cache/observe_kv_cache.py --prompt 2,5,9 --decode-token 4
python practice_01_kv_cache/observe_kv_cache.py --show-values 8
```

## What to observe

Each layer returns one `(key, value)` pair:

```text
past_key_values = (
    (layer_0_key, layer_0_value),
    (layer_1_key, layer_1_value),
)
```

Every key and value tensor has shape:

```text
[batch, attention_heads, cached_tokens, head_dimension]
```

With the default four-token prompt, the prefill cache is `[1, 2, 4, 8]`.
After decoding one new token it becomes `[1, 2, 5, 8]`. The old four positions
are unchanged; only one new position is appended.

The model still computes a query, key, and value for the new token. Only the new
key and value are appended to the cache. The new query attends to all cached keys,
but queries from older tokens do not need to be computed again.

The program also checks two important invariants:

1. Cached token-by-token execution produces the same logits as recomputing the
   entire sequence, within floating-point tolerance.
2. Cache memory equals `2 * layers * batch * heads * tokens * head_dim * bytes`.
   The leading 2 represents both K and V.

## Prefill versus decode

During **prefill**, the model processes every prompt token and creates their K/V
entries. During **decode**, it receives only the latest token plus the existing
cache. The attention score matrix for one decode step therefore has one query row
instead of one row per token.

This simple implementation concatenates tensors whenever the cache grows. Real
serving engines usually preallocate storage or manage fixed-size blocks/pages to
avoid repeated allocation and copying. That is a natural next exercise after the
semantics here are clear.

## Suggested experiments

1. Change `--prompt` length and confirm that cache bytes grow linearly.
2. Change `num_layers`, `num_heads`, or `hidden_size` in `main()` and verify the
   memory formula.
3. Remove `past_key_values=cache` from the decode call and compare the shapes.
4. Print `attention_weights` from the model to see which cached positions each
   new query attends to.
