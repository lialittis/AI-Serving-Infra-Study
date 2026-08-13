# Code Design: KV-Cache Memory Calculator

## Purpose

The calculator turns a model architecture and serving workload into a transparent,
reproducible estimate of raw KV-cache memory. It is implemented as both a reusable
Python calculation API and a command-line interface.

## Calculation model

The scalar count is:

```text
2 × layers × batch × sequence length × KV heads × head dimension
```

Multiplying by the selected dtype's bits per scalar and dividing by eight produces
bytes. `fractions.Fraction` preserves exact arithmetic, including packed 4-bit
values, until presentation.

## Components

| Component | Responsibility |
|---|---|
| `KVCacheConfig` | Hold and validate model/workload inputs |
| `KVCacheEstimate` | Hold the total and useful intermediate results |
| `estimate_kv_cache` | Apply the memory formula without CLI dependencies |
| `derive_head_dim` | Resolve an explicit dimension or derive it from model fields |
| `format_bytes` | Present byte counts in binary units |
| `estimate_as_dict` | Produce a JSON-compatible result |
| `build_parser` | Define the CLI contract and help text |
| `main` | Connect argument parsing, calculation, and output |

The pure calculation function makes the formula easy to unit-test and allows a
future notebook, web UI, or capacity-planning script to reuse it without parsing
terminal output.

## Input decisions

`kv_heads` is explicit because using total attention heads would overestimate GQA
and MQA caches. `head_dim` may be supplied directly or derived using:

```text
head dimension = hidden size / attention heads
```

The calculator rejects ambiguous combinations and non-divisible model dimensions.

Sequence length represents cached tokens per sequence. Batch size assumes a dense,
uniform batch. Variable-length and paged-cache utilization are intentionally left
for a later serving-focused practice.

## Output decisions

The human report exposes:

- bytes per token per layer
- bytes per token across all layers
- bytes per sequence
- total bytes across the batch

This breakdown makes linear scaling visible. Binary units (`KiB`, `MiB`, `GiB`)
match the way accelerator and host memory capacity is usually reasoned about.
JSON output supports composition with other tools.

## Correctness properties

- All dimensional inputs must be positive.
- Head dimension derivation must divide evenly.
- Total memory scales linearly with batch size and sequence length.
- Halving dtype width halves estimated storage.
- The result for Practice 01 must reproduce its observed cache allocation.

## Limitations and extension points

This version estimates logical dense storage, not physical runtime allocation.
Possible extensions include heterogeneous sequence lengths, block rounding, memory
budgets, maximum-token inversion, model presets, GQA comparisons, and estimates
loaded directly from model configuration files.
