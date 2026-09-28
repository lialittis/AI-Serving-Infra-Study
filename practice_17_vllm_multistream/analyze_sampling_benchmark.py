"""Validate and summarize unprofiled ABBA sampling benchmark evidence."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

from run_sampling_benchmark import validate_response


def require(condition, message):
    if not condition:
        raise ValueError(message)


def distribution(values):
    quartiles = statistics.quantiles(values, n=4, method="inclusive")
    return {"n": len(values), "median_ms": statistics.median(values),
            "q1_ms": quartiles[0], "q3_ms": quartiles[2],
            "min_ms": min(values), "max_ms": max(values)}


def summarize(plan, samples):
    require(plan["blocks"] == ["enabled", "disabled", "disabled", "enabled"], "ABBA blocks")
    require(plan["profiling"] is False and plan["python_observer"] is False, "observation enabled")
    cases = [tuple(case) for case in plan["cases"]]
    require(len(cases) == len(set(cases)), "duplicate cases")
    expected = Counter((block, batch, tokens, phase, repeat)
                       for block in range(4) for batch, tokens in cases
                       for phase, count in [("warmup", plan["warmup"]),
                                            ("measure", plan["repeats_per_block"])]
                       for repeat in range(count))
    actual = Counter((r["block"], r["batch_size"], r["output_tokens"], r["phase"], r["repeat"])
                     for r in samples)
    require(actual == expected, "incomplete or duplicate benchmark samples")
    for row in samples:
        require(row["mode"] == plan["blocks"][row["block"]], "mode/block mismatch")
        require(row["validated"] is True, "response not validated")
        require(math.isfinite(row["elapsed_ns"]) and row["elapsed_ns"] > 0, "invalid duration")
        validate_response(row["response"], row["batch_size"], row["output_tokens"])
        require(row["usage"] == row["response"]["usage"], "usage mismatch")
    measured = [r for r in samples if r["phase"] == "measure"]
    result = []
    for batch, tokens in cases:
        rows = [r for r in measured if (r["batch_size"], r["output_tokens"]) == (batch, tokens)]
        modes = {}
        for mode in ("enabled", "disabled"):
            selected = [r for r in rows if r["mode"] == mode]
            info = distribution([r["elapsed_ns"] / 1e6 for r in selected])
            info["completed_tokens_per_second"] = sum(r["usage"]["completion_tokens"] for r in selected) / (
                sum(r["elapsed_ns"] for r in selected) / 1e9)
            modes[mode] = info
        block_medians = {block: statistics.median(r["elapsed_ns"] / 1e6 for r in rows
                                                 if r["block"] == block) for block in range(4)}
        pair_changes = [100 * (block_medians[off] - block_medians[on]) / block_medians[off]
                        for on, off in [(0, 1), (3, 2)]]
        on, off = modes["enabled"], modes["disabled"]
        change = 100 * (off["median_ms"] - on["median_ms"]) / off["median_ms"]
        # Descriptive consistency only: overlapping IQR or conflicting block
        # directions does not establish a repeatable speedup or regression.
        classification = "uncertain"
        if all(v > 0 for v in pair_changes) and on["q3_ms"] < off["q1_ms"]:
            classification = "consistent_improvement_in_this_run"
        elif all(v < 0 for v in pair_changes) and off["q3_ms"] < on["q1_ms"]:
            classification = "consistent_regression_in_this_run"
        result.append({"batch_size": batch, "output_tokens": tokens, **modes,
                       "median_latency_reduction_percent": change,
                       "pair_latency_reduction_percent": pair_changes,
                       "block_medians_ms": block_medians, "classification": classification})
    return {"model": plan["model"], "measurement_samples": len(measured),
            "warmup_samples": len(samples) - len(measured), "cases": result,
            "classification_rule": "Both AB/BA pair directions agree and pooled IQRs do not overlap; descriptive, not a significance test.",
            "scope": "HTTP batch completion, no profiler; diagnostic scheduler traces are separate runs."}


def analyze(run):
    require((run / "complete.json").is_file(), "benchmark not complete")
    plan = json.loads((run / "plan.json").read_text())
    expected_hash = json.loads((run / "script_hash.json").read_text())["sha256"]
    require(hashlib.sha256((run / "run_sampling_benchmark.py").read_bytes()).hexdigest() == expected_hash,
            "benchmark source hash mismatch")
    manifest = json.loads((run / "source_manifest.json").read_text())
    for name, metadata in manifest.items():
        if isinstance(metadata, dict):
            require(hashlib.sha256((run / name).read_bytes()).hexdigest() == metadata["sha256"],
                    "installed source snapshot mismatch: " + name)
    for block, mode in enumerate(plan["blocks"]):
        directory = run / ("block-%02d-%s" % (block, mode))
        command = json.loads((directory / "command.json").read_text())
        argv = command["argv"]
        require(not any("profiler" in x for x in argv), "profiler configured")
        runtime_paths = (command["PYTHONPATH"] or "").split(":")
        require(all(not p or Path(p).is_relative_to("/usr/local/Ascend") for p in runtime_paths)
                and not command["trace_environment_keys"], "observer environment")
        require(json.loads(argv[argv.index("--additional-config") + 1])["enable_async_exponential"]
                == (mode == "enabled"), "config/mode mismatch")
        shutdown = json.loads((directory / "shutdown.json").read_text())
        require(shutdown["completed"] and shutdown["server_exit_code"] == 0, "incomplete service block")
    samples = [json.loads(line) for line in (run / "samples.jsonl").read_text().splitlines()]
    return summarize(plan, samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    result = analyze(args.run)
    (args.run / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
