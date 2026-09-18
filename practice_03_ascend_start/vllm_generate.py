"""Offline batched inference with the installed vLLM Ascend plugin."""

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/data/huggingface_home/hub/Qwen2.5-0.5B-Instruct")
    args = parser.parse_args()
    if not Path(args.model, "config.json").is_file():
        parser.error("Model config.json not found; download the model first (see README).")

    # NPU initialization cannot be inherited through a forked process.
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        dtype="bfloat16",
        max_model_len=2048,
        max_num_seqs=4,
        max_num_batched_tokens=2048,
        gpu_memory_utilization=0.3,
        enforce_eager=True,
    )
    conversations = [
        [{"role": "user", "content": "请用三句话解释 KV cache。"}],
        [{"role": "user", "content": "用一句话说明 prefill 和 decode 的区别。"}],
    ]
    outputs = llm.chat(
        conversations,
        sampling_params=SamplingParams(temperature=0, max_tokens=128),
    )
    for conversation, output in zip(conversations, outputs):
        print("Question:", conversation[0]["content"])
        print("Answer:", output.outputs[0].text)


if __name__ == "__main__":
    # Required for worker processes started with multiprocessing spawn.
    main()
