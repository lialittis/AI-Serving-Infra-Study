#!/usr/bin/env bash
# Run in the remote instance after loading its CANN environment (see README).
set -euo pipefail
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

model_path="${1:-/data/huggingface_home/hub/Qwen2.5-0.5B-Instruct}"
if [[ ! -f "$model_path/config.json" ]]; then
    echo "Missing $model_path/config.json; download the model first." >&2
    exit 1
fi

exec vllm serve "$model_path" \
    --served-model-name qwen-small \
    --host 127.0.0.1 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --max-model-len 2048 \
    --max-num-seqs 4 \
    --max-num-batched-tokens 2048 \
    --gpu-memory-utilization 0.3 \
    --enforce-eager
