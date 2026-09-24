"""Build the cross-run comparison from verified per-run summaries."""
import argparse
import json
from pathlib import Path


ROLES = {
    "qwen_seeded_calibration": "2026-09-24-qwen-enabled-b32-r02",
    "qwen_async": "2026-09-24-qwen-enabled-b32-r03",
    "qwen_inline": "2026-09-24-qwen-disabled-b32-r02",
    "llama_b32_threshold": "2026-09-24-llama-enabled-b32",
    "llama_async": "2026-09-24-llama-enabled-b64",
    "llama_inline": "2026-09-24-llama-disabled-b64",
}


def observations(run, kind):
    records = [json.loads(line) for path in (run / "events").glob("*.jsonl")
               for line in path.read_text().splitlines()]
    return sorted((record for record in records
                   if record.get("event") == "enter" and record.get("kind") == kind),
                  key=lambda record: record["step"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.results / "comparison.json"
    runs = {}
    for role, name in ROLES.items():
        run = args.results / name
        stored = json.loads((run / "analysis" / "summary.json").read_text())
        branch_kind = "async_exponential" if stored["summary"]["mode"] == "enabled" else "inline_exponential"
        branch = observations(run, branch_kind)
        runs[role] = {
            "directory": name,
            **stored,
            "branch_observations": [
                {key: row[key] for key in ("step", "batch_size", "vocab_size", "generator_count")
                 if key in row}
                for row in branch
            ],
        }
    result = {
        "schema": 1,
        "runs": runs,
        "observed_invariants": {
            "physical_streams": ["44", "46"],
            "default_model_stream": "46",
            "random_stream": "44",
            "random_kernel_sequence": [
                "DSARandomUniform", "Neg", "Add", "GreaterEqual", "MaskedFill", "Log", "Mul"
            ],
            "qwen_async_overlap_steps": runs["qwen_async"]["summary"]["overlap_steps"],
            "llama_async_overlap_steps": runs["llama_async"]["summary"]["overlap_steps"],
        },
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
