#!/usr/bin/env bash
# Usage: bash benchmark_concurrency.sh [MODEL_PATH] [NEW_RESULT_DIRECTORY]
# Requires the API server from serve.sh to be running in another terminal.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
model_path="${1:-/data/huggingface_home/hub/Qwen2.5-0.5B-Instruct}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if [[ ! -f "$model_path/config.json" ]]; then
    echo "Missing model: $model_path/config.json" >&2
    exit 1
fi

# Check the live service before generating any requests or allocating a run directory.
python - <<'PY'
import json
import urllib.request

with urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=5) as response:
    models = json.load(response)
if not any(model["id"] == "qwen-small" for model in models["data"]):
    raise SystemExit("Expected served model qwen-small; start serve.sh first.")
print("API preflight passed: qwen-small")
PY

if [[ $# -ge 2 ]]; then
    result_dir="$2"
    mkdir -- "$result_dir"  # Deliberately refuse to overwrite an earlier run.
else
    mkdir -p /data/tianchi/benchmark_results
    result_dir="$(mktemp -d /data/tianchi/benchmark_results/concurrency-XXXXXXXX)"
fi
echo "Results: $result_dir"
python "$script_dir/collect_environment.py" --model "$model_path" \
    --output "$result_dir/environment.json"

for concurrency in 1 4; do
    command=(vllm bench serve
        --backend openai
        --base-url http://127.0.0.1:8000
        --endpoint /v1/completions
        --model "$model_path"
        --served-model-name qwen-small
        --dataset-name random
        --input-len 128
        --output-len 64
        --random-range-ratio 0
        --num-prompts 20
        --num-warmups 2
        --max-concurrency "$concurrency"
        --request-rate inf
        --seed "$concurrency"
        --temperature 0
        --ignore-eos
        --save-result
        --save-detailed
        --result-dir "$result_dir"
        --result-filename "concurrency-${concurrency}.json")
    printf '%q ' "${command[@]}" >> "$result_dir/commands.sh"
    printf '\n' >> "$result_dir/commands.sh"
    "${command[@]}" 2>&1 | tee "$result_dir/concurrency-${concurrency}.log"
done

python "$script_dir/summarize_results.py" "$result_dir" | tee "$result_dir/summary.md"
