# AI Serving Infrastructure Study

Exercises for understanding language-model serving: CPU-friendly KV-cache
exercises and a reproducible Ascend NPU inference and concurrency experiment.

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
