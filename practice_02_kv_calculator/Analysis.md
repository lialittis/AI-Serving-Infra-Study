# Analysis: How Many Tokens Fit in the KV Cache?

The KV-cache memory formula can be reversed to estimate how many tokens can be
cached at the same time.

## 1. Memory per cached token

First calculate the KV-cache memory occupied by one token across all decoder
layers:

```text
bytes per token = 2 × layers × KV heads × head dimension × bytes per scalar
```

The factor of `2` represents the key and value caches.

## 2. Maximum total cached tokens

Given a memory budget reserved for KV cache:

```text
maximum total cached tokens = floor(KV memory budget / bytes per token)
```

“Total cached tokens” means tokens across all concurrent requests:

```text
total cached tokens = tokens_request_1 + tokens_request_2 + ... + tokens_request_n
```

For a dense batch in which every sequence has the same length:

```text
tokens per sequence = floor(maximum total cached tokens / batch size)
```

## 3. Worked example

Use this model configuration:

```text
layers:        32
KV heads:       8
head dimension: 128
cache dtype:   bf16 = 2 bytes per scalar
```

Memory per cached token is:

```text
2 × 32 × 8 × 128 × 2 bytes
= 131,072 bytes
= 128 KiB per token
```

With 16 GiB available exclusively for KV cache:

```text
16 GiB / 128 KiB per token
= 131,072 total cached tokens
```

That logical capacity can be divided among equal-length concurrent sequences in
different ways:

| Concurrent sequences | Approximate tokens per sequence |
|---:|---:|
| 1 | 131,072 |
| 4 | 32,768 |
| 8 | 16,384 |
| 16 | 8,192 |
| 32 | 4,096 |

A variable-length workload could instead contain:

```text
request A:  10,000 tokens
request B:  20,000 tokens
request C:  30,000 tokens
request D:  50,000 tokens
--------------------------------
total:     110,000 cached tokens
```

This fits the 131,072-token logical budget even though the sequences have
different lengths.

## 4. Finding the KV memory budget

The entire device or host memory capacity is not normally available for KV cache.
The usable budget is approximately:

```text
KV memory budget
= total memory
- model weights
- runtime and framework memory
- activations and temporary workspaces
- safety margin
```

For example:

```text
accelerator memory:  24 GiB
model weights:      -15 GiB
runtime/workspaces:  -2 GiB
safety margin:       -1 GiB
--------------------------------
KV-cache budget:      6 GiB
```

At 128 KiB per cached token:

```text
6 GiB / 128 KiB per token = 49,152 total cached tokens
```

With eight equal-length concurrent requests, this corresponds to approximately:

```text
49,152 / 8 = 6,144 tokens per sequence
```

## 5. Prompt and generated tokens share the cache

A request's cached length includes both its original prompt and the tokens already
generated:

```text
cached tokens = prompt tokens + generated tokens so far
```

For example, a request with a 4,000-token prompt and 1,000 generated tokens
currently occupies 5,000 token positions in the KV cache.

As generation proceeds, the request consumes one additional cached token per
decode step unless an older position is evicted or a sliding-window policy is used.

## 6. Logical capacity versus usable capacity

The reversed formula gives a logical upper estimate. Actual serving capacity may
be lower because of:

- the model's configured maximum context length
- cache blocks that reserve more positions than a sequence currently uses
- allocator fragmentation and alignment
- page tables and cache metadata
- quantization scales or zero points
- temporary attention buffers
- runtime memory growth under load
- reserved capacity or a configured memory-utilization limit

For example, if a runtime allocates cache in blocks of 16 tokens, a sequence with
17 live tokens may occupy two blocks, reserving 32 token slots. Paged allocation
reduces waste compared with padding every request to a common maximum length, but
block rounding still creates some difference between live tokens and allocated
slots.

## 7. Context-length constraint

Memory capacity does not override the model's maximum supported context. The
per-sequence limit is bounded by both memory and model architecture:

```text
usable tokens per sequence
= min(memory-derived tokens per sequence, model maximum context length)
```

For example, memory might be sufficient for 32,768 tokens per sequence, but a
model configured for an 8,192-token context is still limited to 8,192 unless its
position-handling mechanism is deliberately extended.

## 8. Capacity-planning interpretation

Two related numbers are useful:

1. **Per-request context capacity:** the maximum tokens one request can retain.
2. **Total token capacity:** the sum of live cached tokens across all requests.

Serving throughput and concurrency are usually governed by the second number. A
system may accept many short requests or fewer long requests within the same total
KV-token budget.

For rough planning:

```text
maximum concurrency
= floor(maximum total cached tokens / expected tokens per request)
```

If the 6 GiB example can hold 49,152 tokens and requests are expected to retain
about 4,096 tokens each:

```text
floor(49,152 / 4,096) = 12 concurrent requests
```

This remains an estimate because real request lengths vary and the runtime may
reserve cache in blocks.

## 9. Calculator extension

A useful extension to `kv_calculator.py` is a memory-budget mode with inputs such
as:

```text
--memory-budget 6GiB
--batch-size 8
```

It could invert the existing formula and report:

- maximum total cached tokens
- approximate tokens per sequence at the requested concurrency
- unused bytes after allocating whole token slots
- an optional block-rounded estimate for paged KV caches

The existing calculator already computes `bytes_per_token`, so capacity mode can
reuse that value rather than introduce a separate formula.
