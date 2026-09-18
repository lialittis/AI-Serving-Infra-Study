"""Run a locally downloaded Qwen2.5 model on one logical NPU."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/data/huggingface_home/hub/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--prompt", default="请用三句话解释大语言模型中的 KV cache。")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    if not Path(args.model, "config.json").is_file():
        parser.error("Model config.json not found; download the model first (see README).")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")

    import torch
    import torch_npu  # noqa: F401
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        local_files_only=True,
    ).to("npu:0").eval()
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to("npu:0")
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0, inputs.input_ids.shape[1]:]
    print("Model device:", model.device)
    print("New tokens:", len(new_tokens))
    print(tokenizer.decode(new_tokens, skip_special_tokens=True))


if __name__ == "__main__":
    main()
