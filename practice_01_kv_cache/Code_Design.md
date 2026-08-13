# Code Design: CPU KV-Cache Practice

## 1. Purpose

This practice demonstrates the mechanics of a decoder-only Transformer's key-value
(KV) cache without requiring a GPU, model download, tokenizer, or serving framework.
It is intentionally small enough that every relevant tensor can be printed and
reasoned about directly.

The implementation answers four questions:

1. What is stored in `past_key_values`?
2. How is the cache created during prompt prefill?
3. How is it extended during token-by-token decoding?
4. Does cached decoding produce the same result as full-sequence recomputation?

The model is randomly initialized. Its token predictions have no linguistic meaning;
only its attention and cache behavior are under study.

## 2. Scope

### Included

- A minimal decoder-only Transformer implemented in PyTorch
- Multi-head causal self-attention
- One `(key, value)` cache pair per Transformer layer
- Prompt prefill and single-token decode paths
- Cache shape, selected value, and memory inspection
- Numerical comparison of cached and uncached execution
- Unit tests for the core cache invariants

### Deliberately excluded

- Training and useful language generation
- A tokenizer or text vocabulary
- Hugging Face Transformers
- Rotary position embeddings (RoPE)
- Grouped-query or multi-query attention
- Continuous batching
- Paged attention and block management
- Cache eviction, quantization, and offloading
- GPU-specific kernels and performance benchmarking

These exclusions keep the first practice focused on cache semantics rather than
framework or production-serving complexity.

## 3. Repository layout

```text
practice_01_kv_cache/
├── Code_Design.md          # This design document
├── README.md               # User guide and suggested experiments
├── observe_kv_cache.py     # Model, cache inspection, and runnable demonstration
└── test_kv_cache.py        # Correctness tests
```

## 4. Model configuration

The default model is deliberately tiny:

| Parameter | Value | Meaning |
|---|---:|---|
| Vocabulary size | 32 | Valid token IDs are 0 through 31 |
| Hidden size | 16 | Width of each token representation |
| Attention heads | 2 | Independent attention heads per layer |
| Head dimension | 8 | `hidden_size / num_heads` |
| Decoder layers | 2 | Number of cache pairs |
| Maximum positions | 128 | Size of the learned position table |
| Tensor dtype | `float32` | Four bytes per cached scalar |

The default prompt is `[1, 5, 9, 2]`, followed by decode token `7`.

## 5. High-level architecture

```text
input token IDs
      │
      ├── token embedding ───┐
      └── position embedding ┤ addition
                             ▼
                      hidden states
                             │
                  ┌──────────▼──────────┐
                  │ DecoderBlock 0      │──► layer 0 (K, V)
                  │ norm → attention    │
                  │ norm → feed-forward│
                  └──────────┬──────────┘
                             │
                  ┌──────────▼──────────┐
                  │ DecoderBlock 1      │──► layer 1 (K, V)
                  └──────────┬──────────┘
                             │
                        final norm
                             │
                         LM head
                             │
                           logits
```

The cache exists only around self-attention. Feed-forward layers operate on each
token independently and therefore do not need historical state during decoding.

## 6. Main types and interfaces

### 6.1 Cache types

```python
LayerCache = Tuple[Tensor, Tensor]
PastKeyValues = Tuple[LayerCache, ...]
```

A `LayerCache` contains the key and value tensors for one decoder layer.
`PastKeyValues` contains one `LayerCache` per decoder layer:

```text
past_key_values
├── layer 0: (key_0, value_0)
└── layer 1: (key_1, value_1)
```

This resembles the tuple-based `past_key_values` representation traditionally
exposed by Hugging Face causal language models.

### 6.2 Model input

```python
TinyCausalLM.forward(
    input_ids: Tensor,
    past_key_values: Optional[PastKeyValues] = None,
) -> ModelOutput
```

`input_ids` must have shape:

```text
[batch, input_tokens]
```

The two intended invocation modes are:

| Phase | `input_ids` | `past_key_values` |
|---|---|---|
| Prefill | Complete prompt | `None` |
| Decode | Only newly appended token(s) | Cache from previous call |

### 6.3 Model output

```python
@dataclass
class ModelOutput:
    logits: Tensor
    past_key_values: PastKeyValues
    attention_weights: Tuple[Tensor, ...]
```

| Field | Shape or structure | Purpose |
|---|---|---|
| `logits` | `[batch, input_tokens, vocab_size]` | Token prediction scores |
| `past_key_values` | One `(K, V)` pair per layer | State for the next call |
| `attention_weights` | One tensor per layer | Makes attention observable |

## 7. Tensor layout

Every cached K and V tensor uses this layout:

```text
[batch, heads, cached_tokens, head_dim]
```

For a four-token prompt with the default configuration:

```text
[1, 2, 4, 8]
 │  │  │  └── eight values per head
 │  │  └───── four cached token positions
 │  └──────── two attention heads
 └─────────── one sequence in the batch
```

The projection layers initially produce `[batch, tokens, hidden_size]`. The
`_split_heads` method transforms this as follows:

```text
[B, T, H]
    reshape
[B, T, N, D]
    transpose
[B, N, T, D]
```

where `H = N × D`, `N` is the number of heads, and `D` is the head dimension.

## 8. Attention design

For the current input, the attention layer computes:

```python
Q_new = q_proj(hidden_states)
K_new = k_proj(hidden_states)
V_new = v_proj(hidden_states)
```

If a previous cache exists, new keys and values are appended along the token
dimension:

```python
K_all = concat(K_past, K_new, dim=2)
V_all = concat(V_past, V_new, dim=2)
```

Queries are not cached. A past query was needed when producing its own output,
but it is not used to calculate the output for a future token. Future computation
needs the earlier keys for matching and earlier values for aggregation.

Attention is calculated as:

```text
softmax((Q_new × K_allᵀ) / sqrt(head_dim) + causal_mask) × V_all
```

The returned cache is `(K_all, V_all)`.

## 9. Causal mask design

The mask must work both when processing a whole prompt and when processing new
tokens after a cached prefix.

The implementation assigns absolute positions to the queries and keys:

```python
query_positions = past_tokens + arange(query_tokens)
key_positions = arange(key_tokens)
allowed = key_positions <= query_positions
```

For a four-token prefill, this produces a lower-triangular mask:

```text
        K0  K1  K2  K3
Q0      ✓   ·   ·   ·
Q1      ✓   ✓   ·   ·
Q2      ✓   ✓   ✓   ·
Q3      ✓   ✓   ✓   ✓
```

For one-token decode after that prompt:

```text
        K0  K1  K2  K3  K4
Q4      ✓   ✓   ✓   ✓   ✓
```

Disallowed attention scores are set to negative infinity before softmax, giving
those positions zero probability.

## 10. Position handling

The model uses learned absolute position embeddings. The position of a new token
must account for the length of the supplied cache:

```python
past_tokens = past_key_values[0][0].size(2)
positions = arange(past_tokens, past_tokens + input_tokens)
```

Consequently:

```text
prefill token positions: 0, 1, 2, 3
first decode position:   4
```

If decode restarted from position zero, cached decoding would no longer be
equivalent to full-sequence execution.

## 11. Execution flows

### 11.1 Prefill

Input:

```text
input_ids = [[1, 5, 9, 2]]
past_key_values = None
```

For every layer:

1. Compute Q, K, and V for all four prompt tokens.
2. Apply causal attention over a `4 × 4` query-key matrix per head.
3. Return K and V with shape `[1, 2, 4, 8]`.

The attention weight shape is `[1, 2, 4, 4]`.

### 11.2 Decode

Input:

```text
input_ids = [[7]]
past_key_values = cache for [1, 5, 9, 2]
```

For every layer:

1. Compute Q, K, and V only for token `7`.
2. Append its K and V to the four cached positions.
3. Use its single query to attend over all five key positions.
4. Return K and V with shape `[1, 2, 5, 8]`.

The attention weight shape is `[1, 2, 1, 5]`. This reduction from four query
rows to one query row is the key computational reuse illustrated by the practice.

## 12. Cache memory model

For standard multi-head attention, cache memory is:

```text
2 × layers × batch × heads × cached_tokens × head_dim × bytes_per_element
```

The leading `2` accounts for both K and V. Because
`heads × head_dim = hidden_size`, the same formula can be written as:

```text
2 × layers × batch × cached_tokens × hidden_size × bytes_per_element
```

Default prefill cache:

```text
2 × 2 × 1 × 2 × 4 × 8 × 4 bytes = 1,024 bytes
```

After one decoded token:

```text
2 × 2 × 1 × 2 × 5 × 8 × 4 bytes = 1,280 bytes
```

The cache therefore grows linearly with batch size, layer count, sequence length,
hidden width, and scalar precision.

## 13. Correctness invariants

The demonstration and tests enforce these invariants.

### 13.1 Shape invariant

For every layer:

```text
K.shape == V.shape
K.shape == [batch, heads, cached_tokens, head_dim]
```

### 13.2 Growth invariant

If a call supplies `T_past` cached tokens and `T_new` input tokens, the returned
cache contains:

```text
T_total = T_past + T_new
```

### 13.3 Prefix-preservation invariant

Appending a token must not modify existing entries:

```python
old_key == new_key[:, :, :-1, :]
old_value == new_value[:, :, :-1, :]
```

### 13.4 Numerical-equivalence invariant

For the same complete token sequence, these paths must agree within floating-point
tolerance:

```text
Path A: process all tokens in one call
Path B: process one token at a time, passing the cache forward
```

Small differences are expected because matrix operations may be grouped in a
different order. The test accepts a maximum absolute difference below `1e-5`.

### 13.5 Memory invariant

Reported cache bytes must equal the sum of:

```python
tensor.numel() * tensor.element_size()
```

for every key and value tensor in every layer.

## 14. Component responsibilities

| Component | Responsibility |
|---|---|
| `CausalSelfAttention` | Project Q/K/V, append cache, mask attention, and return layer cache |
| `DecoderBlock` | Combine normalized attention and feed-forward transformations with residuals |
| `TinyCausalLM` | Embed tokens and positions, route per-layer caches, and produce logits |
| `ModelOutput` | Give names to model outputs instead of returning an ambiguous tuple |
| `cache_bytes` | Calculate actual tensor storage occupied by the cache |
| `describe_cache` | Print cache shapes, sample values, and memory |
| `verify_cached_equals_full` | Compare full-sequence and incremental execution |
| `main` | Run the prefill/decode walkthrough and report invariants |

## 15. Design decisions and tradeoffs

### Plain PyTorch instead of Transformers

The environment already contains CPU PyTorch but not Transformers. More
importantly, implementing attention here makes the cache construction visible
instead of hiding it behind a model library.

### Tuple-based cache

A tuple of `(K, V)` pairs is easy to inspect and resembles a familiar model API.
It is not intended to be a complete cache-management abstraction.

### `torch.cat` for cache growth

Concatenation is simple and makes the append operation explicit. It is inefficient
for production decoding because every step allocates new tensors and copies the
old cache. A production-oriented follow-up should replace it with preallocated or
paged storage.

### Learned absolute positions

Absolute embeddings make the cache-position offset easy to demonstrate. Many
modern models use RoPE, where position information is applied to Q and K instead.

### Attention weights returned by default

Retaining attention weights is useful for observation, but production inference
usually avoids materializing or returning them because they consume memory.

### Random weights and integer tokens

This removes network and model-storage dependencies. It also keeps the exercise
about serving mechanics rather than generated text quality.

### Single CPU thread

The demo calls `torch.set_num_threads(1)` to keep this tiny workload predictable
and avoid excessive thread startup overhead. It is not a general CPU inference
tuning recommendation.

## 16. Known limitations

- All layers are assumed to have the same cached sequence length.
- There is no attention mask for padded batches.
- The public API does not validate every cache tensor dimension or dtype.
- `max_positions` limits total prompt plus decode length to 128.
- Cache growth copies existing tensors at every decode step.
- The implementation uses float32 only unless the model is manually converted.
- The demo performs one decode step rather than a full generation loop.
- It demonstrates standard multi-head attention, not the reduced KV-head layouts
  used by MQA or GQA models.

## 17. Extension path

A useful progression from this practice is:

1. Add a multi-step greedy generation loop.
2. Time full recomputation versus cached decoding on CPU.
3. Preallocate a static cache and update it in place.
4. Implement a block table and paged KV storage.
5. Add batched sequences with different lengths.
6. Implement GQA and compare cache memory with standard multi-head attention.
7. Add cache dtypes such as float16 or int8 and compare memory and error.
8. Reproduce the observations with a small pretrained Hugging Face model.

Each step preserves the core cache contract established here while introducing
one serving-system concern at a time.
