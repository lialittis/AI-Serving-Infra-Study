# AI Serving Infrastructure Study

Small, CPU-friendly exercises for understanding how language-model serving works.

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
