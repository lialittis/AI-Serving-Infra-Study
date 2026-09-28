"""Capture eight separate P0 diagnostic traces after unprofiled benchmarks."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, default=Path("/data/huggingface_home/hub"))
    parser.add_argument("--port", type=int, default=8017)
    parser.add_argument("--resume", action="store_true",
                        help="reanalyze completed captures and continue; never overwrite partial captures")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=args.resume)
    root = Path(__file__).resolve().parent
    summaries = []
    for short, model, batch in [("qwen", "Qwen2.5-0.5B-Instruct", 32),
                                ("llama", "Llama-3.2-1B-Instruct", 64)]:
        for tokens in (4, 64):
            for mode in ("enabled", "disabled"):
                run = args.output.resolve() / ("%s-%s-t%d" % (short, mode, tokens))
                command = [sys.executable, str(root / "run_vllm.py"),
                           "--model", str(args.model_root / model), "--batch-size", str(batch),
                           "--mode", mode, "--max-tokens", str(tokens),
                           "--port", str(args.port), "--output", str(run)]
                if run.exists():
                    if not args.resume or not (run / "shutdown.json").exists():
                        raise ValueError("partial or unapproved existing capture: " + str(run))
                    shutdown = json.loads((run / "shutdown.json").read_text())
                    if shutdown["server_exit_code"] != 0:
                        raise ValueError("unclean existing capture: " + str(run))
                    print("REANALYZE " + run.name, flush=True)
                else:
                    print("CAPTURE " + run.name, flush=True)
                    subprocess.run(command, check=True)
                subprocess.run([sys.executable, str(root / "analyze_run.py"), str(run)], check=True)
                result = json.loads((run / "analysis" / "summary.json").read_text())
                summaries.append({"run": run.name, "output_tokens": tokens, **result["summary"]})
                (args.output / "comparison.json").write_text(json.dumps(summaries, indent=2) + "\n")
    (args.output / "complete.json").write_text(json.dumps({"runs": len(summaries)}) + "\n")


if __name__ == "__main__":
    main()
