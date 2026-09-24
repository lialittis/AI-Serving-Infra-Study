# AI Serving Infrastructure Study

Exercises for understanding language-model serving: CPU-friendly KV-cache
exercises and a reproducible Ascend NPU inference and concurrency experiment.

The next stage follows a real request through the installed vLLM / Ascend stack.
See [REAL_SYSTEM_ROADMAP.md](REAL_SYSTEM_ROADMAP.md) for the recorded Practice 07–18 plan.

## Stage report: the complete inference journey

Open the [offline HTML report](reports/inference_journey/index.html) in a local
browser to walk through startup, prefill, decode and request cleanup. Interactive
sequence diagrams connect CPU/NPU data movement, KV storage, operator dispatch,
asynchronous execution and graph replay to the evidence from Practices 07–11.
See the [report guide](reports/inference_journey/README.md) for viewing, rebuilding,
standalone diagrams and the distinction between measured facts and source explanations.

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
The [complete operator flow](practice_09_operator_trace/OPERATOR_FLOW.md) extends
the audit to all 33 observed device task names, four Triton kernels, ATen,
torch-npu, ATB, custom C++ operations, memory transfers, and host task queues.

## Practice 10: extract the real model computation graph

Export the actual Qwen2Model FX graph captured by vLLM on Ascend. Inspect all
24 decoder layers, tensor dependencies and shapes, mutable attention outputs,
and the native compilation partitions. An offline node browser and a first-layer
SVG connect the model graph to the operator execution studied in Practice 09.
See [practice_10_model_graph/README.md](practice_10_model_graph/README.md).

## Practice 11: connect an attention graph node to real execution

Follow the first FX attention node through actual Q/K/V storage, hidden KV-cache
context, cache writes, attention execution and its mutable output buffer. Compare
prefill with decode under the same graph configuration as Practice 10, including
the limits of attributing replayed device kernels to individual FX nodes.
See [practice_11_attention_execution/README.md](practice_11_attention_execution/README.md)
for reproduction, verified profiler links, and an offline phase comparison.

## Practice 12: trace real KV block release and reuse

Use a native two-block pool (one reserved null block, one usable block) to make
two sequential requests reuse the same physical KV storage. Trace all 24 layers,
reference counts and free queues, then connect device accesses to the native
sampled-token transfer/wait and the allocator's release/reallocation boundaries.
See [practice_12_kv_block_reuse/README.md](practice_12_kv_block_reuse/README.md)
for switchable eager / PIECEWISE graph runs, matched real-device evidence, and an
offline comparison with both lifecycle viewers. Both modes retain all 192 direct
KV/FIA links; replay-internal attribution limits are recorded separately.

Practice 12 also records each observed eager event and each graph replay, including
capture/current storage checks, native event identities and metadata preparation
chains. See [the resource ledger guide](practice_12_kv_block_reuse/RESOURCE_RECORDS.md).

## Practice 13: observe CPU submission and NPU execution

Record cold Triton compilation and binary registration, then follow a warm eager
request through actual launcher/API parameters, CPU enqueue/dequeue, CANN launch,
NPU execution and native result-transfer waits. The offline report includes
operator timelines, tensor addresses, CPU/NPU overlap and all 1,444 correlated
inference tasks. It distinguishes observed runtime registration from unobserved
kernel-code DMA timing and hidden native argument bytes.
See [the reproduction guide](practice_13_operator_submission/README.md) and
[interactive report](practice_13_operator_submission/results/2026-09-24-run03/analysis/index.html).

## Practice 14: observe actual KV pool allocation

Follow service initialization from the KV memory budget to raw K/V tensors,
torch-npu allocator blocks, actual CANN physical-memory allocation and mapping,
BF16 views and CPU block bookkeeping. The recorded expandable-segment path
reuses an existing virtual address arena; all 48 K/V storages are reconciled
against native calls and allocator snapshots. No inference request is sent.
See [the focused reproduction guide](practice_14_kv_pool_allocation/README.md)
and [six-step findings](practice_14_kv_pool_allocation/RESULTS.md).

## Practice 15: reconstruct a kernel execution graph

Connect a fresh eager model trace into typed host-dispatch, physical-stream,
event-completion and scoped tensor-dependency edges. Browse all 1,444 device
tasks, keeping partial data coverage explicit. A separate real-NPU two-stream
probe verifies cross-stream event waits and repeated event generations.
See [the guide](practice_15_kernel_execution_graph/README.md),
[model graph](practice_15_kernel_execution_graph/results/2026-09-24-model-run01/analysis/index.html)
and [two-stream graph](practice_15_kernel_execution_graph/results/2026-09-24-stream-run02/analysis/execution_graph.svg).

## Practice 16: observe actual multi-stream compute overlap

Run independent matrix and vector operators with identical inputs and submission
order on one versus two NPU streams. Three paired trials verify real device
interval overlap, exact host/CANN flows, terminal event waits and all output
elements. The offline timeline distinguishes observable parallelism from speedup.
See [the guide](practice_16_multistream_parallel/README.md),
[findings](practice_16_multistream_parallel/RESULTS.md) and
[interactive timeline](practice_16_multistream_parallel/results/2026-09-24-run02/analysis/index.html).

Practice 16 also compares [per-round waits, a final join, and premature reads](practice_16_multistream_parallel/SYNC_COMPARISON.md).
Unprofiled measurements separate submission from completed work; a separate
trace verifies unchanged operators, ordered sample copies and actual waits.
