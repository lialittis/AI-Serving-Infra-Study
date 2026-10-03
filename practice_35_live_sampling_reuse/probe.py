"""Real vLLM generation on the default (sync) sampling path with P35 observation."""
import argparse
import datetime
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="/data/huggingface_home/hub/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--variant", choices=("default", "async", "protected"), required=True)
    parser.add_argument("--prompts", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    # Workers spawn with this environment; the in-process call below covers
    # engines that do not use a separate process.
    os.environ["P35_OBSERVER_DIR"] = str(out)
    sys.path.insert(0, str(HERE))
    from observer import install
    install()

    # NPU initialization cannot be inherited through a forked process.
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    import torch  # noqa: F401
    from vllm import LLM, SamplingParams

    engine_args = dict(model=args.model, tensor_parallel_size=1, dtype="bfloat16",
                       max_model_len=1024, max_num_seqs=max(4, args.prompts),
                       max_num_batched_tokens=2048, gpu_memory_utilization=0.3,
                       enforce_eager=True, seed=args.seed)
    if args.variant == "async":
        engine_args["async_scheduling"] = True
    elif args.variant == "protected":
        engine_args["additional_config"] = {"enable_async_exponential": True}
    try:
        llm = LLM(**engine_args)
    except (TypeError, ValueError) as exc:
        (out / "engine_error.json").write_text(json.dumps(
            dict(variant=args.variant, error=repr(exc)), indent=2) + "\n")
        print(f"VARIANT UNSUPPORTED: {exc}", flush=True)
        return 2

    sampling = SamplingParams(temperature=1.0, top_k=50, top_p=0.9,
                              max_tokens=args.max_tokens, seed=args.seed)
    conversations = [[{"role": "user", "content": "用两句话介绍 paged attention 的思想。"}]
                     for _ in range(args.prompts)]
    generations = []
    for repeat in range(args.repeats):
        outputs = llm.chat(conversations, sampling_params=sampling, use_tqdm=False)
        tokens = [[int(t) for t in output.outputs[0].token_ids] for output in outputs]
        generations.append(dict(repeat=repeat, tokens=tokens))
    identical = all(generations[0]["tokens"][i] == generations[r]["tokens"][i]
                    for r in range(1, args.repeats) for i in range(args.prompts))
    payload = dict(schema=1, captured_at_utc=datetime.datetime.now(
                       datetime.timezone.utc).isoformat(),
                   arguments={k: str(v) if isinstance(v, Path) else v
                              for k, v in vars(args).items()},
                   engine_args={k: str(v) for k, v in engine_args.items()},
                   identical_across_repeats=identical,
                   generations=generations)
    (out / "generation.json").write_text(json.dumps(payload, indent=1) + "\n")
    print(f"variant={args.variant} repeats_identical={identical}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
