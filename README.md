# AI Serving Infrastructure Study

Exercises for understanding language-model serving: CPU-friendly KV-cache
exercises and a reproducible Ascend NPU inference and concurrency experiment.

The next stage follows a real request through the installed vLLM / Ascend stack.
See [REAL_SYSTEM_ROADMAP.md](REAL_SYSTEM_ROADMAP.md) for the recorded Practice 07–12 plan.

## Practice 01: observe a KV cache

This practice implements a tiny decoder-only Transformer in plain PyTorch. It does
not download a model or require a GPU. The model exposes a Hugging Face-style
`past_key_values` tuple so you can inspect what is cached during prefill and how
the cache grows during token-by-token decoding.

Run it with the existing Python 3.11 environment:

```bash
source ~/venvs/py311/bin/activate
python practice_01_kv_cache/observe_kv_cache.py
```

Run the correctness tests:

```bash
python -m unittest discover -s practice_01_kv_cache -p 'test_*.py'
```

See [practice_01_kv_cache/README.md](practice_01_kv_cache/README.md) for the
concepts, expected output, and suggested experiments.

## Practice 02: estimate KV-cache memory

This dependency-free command-line calculator estimates KV-cache memory from a
model's layer count, KV-head layout, context length, batch size, and cache dtype.

```bash
python practice_02_kv_calculator/kv_calculator.py \
  --layers 32 --kv-heads 8 --head-dim 128 \
  --sequence-length 8192 --batch-size 1 --dtype bf16
```

See [practice_02_kv_calculator/README.md](practice_02_kv_calculator/README.md)
for examples and guidance on finding the inputs in a model configuration.

## Practice 03: run small models on Ascend

[practice_03_ascend_start/README.md](practice_03_ascend_start/README.md) provides
Chinese-language reproduction instructions for Transformers, vLLM offline
inference, HTTP serving, and a concurrency 1 vs 4 benchmark on `ascend910`.
The [results and conclusions](practice_03_ascend_start/RESULTS.md) include original
benchmark JSON, execution logs, environment metadata, and model fingerprints.

View the archived benchmark comparison locally without an NPU:

```bash
python3 practice_03_ascend_start/summarize_results.py
```

## Practice 04: map logical tokens to physical KV blocks

This pure-Python exercise implements only `BlockPool`, `Request`, and
`BlockTable`. It shows how a request's token index maps to a physical block and
offset, including noncontiguous allocation and block reuse. No model, NPU,
PyTorch, or vLLM is required.

```bash
python3 practice_04_paged_kv_cache/paged_kv_cache.py
python3 -m unittest discover -s practice_04_paged_kv_cache -p 'test_*.py' -v
```

See [practice_04_paged_kv_cache/README.md](practice_04_paged_kv_cache/README.md)
for the mapping formula and the worked four-block example.

## Practice 05: observe block allocation lifetimes

This single-threaded, pure-Python exercise traces allocation, writing, freeing,
and reuse. Generation-tagged handles distinguish the same physical block's
successive allocations and reject access through stale references.

```bash
python3 practice_05_block_lifecycle/block_lifecycle.py
python3 -m unittest discover -s practice_05_block_lifecycle -p 'test_*.py' -v
```

See [practice_05_block_lifecycle/README.md](practice_05_block_lifecycle/README.md)
for the lifecycle trace and the distinction between physical and allocation identity.

## Practice 06: detect a stale block access

This pure-Python, single-threaded exercise deliberately reads A's old block
reference after B has reused the physical block. A minimal generation check
reports the request, block, expected generation, and current generation before
the stale read can return B's data.

```bash
python3 practice_06_stale_block_access/stale_block_access.py
python3 -m unittest discover -s practice_06_stale_block_access -p 'test_*.py' -v
```

See [practice_06_stale_block_access/README.md](practice_06_stale_block_access/README.md)
for the deterministic lifetime violation and its diagnostic output.

## Practice 07: trace a real vLLM-Ascend request

Run one real HTTP request on Ascend and correlate the API, scheduler, worker,
model runner, attention backend, and output with the installed source files.
See [practice_07_real_request_trace/README.md](practice_07_real_request_trace/README.md)
for remote execution, archived evidence, and the limits of a host-side trace.

## Practice 08: trace real KV block mappings

Follow a real request across a 128-token block boundary. Compare the KV manager's
block IDs, CPU and NPU block tables, slot mapping, and the first layer's actual
K/V data after the cache write.
See [practice_08_real_kv_mapping/README.md](practice_08_real_kv_mapping/README.md)
for the diagnostic procedure and its intentional device synchronization.

## Practice 09: trace real operator execution

Follow Python attention and KV-write calls through PyTorch operators and CANN
launch events to actual NPU kernels. The archived profiler includes one prefill
and one decode step, with verified flow IDs and a first-layer timeline excerpt.
See [practice_09_operator_trace/README.md](practice_09_operator_trace/README.md).
