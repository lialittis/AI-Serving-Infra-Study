"""Offline analysis of graph storage-boundary trials."""
import argparse
import hashlib
import json
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def summarize_trial(trial):
    mode = trial["mode"]
    require(trial["statistics_after"]["num_ooms"] == trial["statistics_before"]["num_ooms"],
            "OOM confounds trial")
    require(trial["statistics_after"]["num_alloc_retries"]
            == trial["statistics_before"]["num_alloc_retries"], "retry confounds trial")
    if mode == "pool-release-pending":
        require("replacement_reused_static" in trial, "pool-release trial missing address comparison")
        return dict(id=trial["id"], mode=mode, outcome=trial["outcome"],
                    replacement_reused_static=trial["replacement_reused_static"],
                    replacement_intact=trial["replacement_intact"])
    if trial["outcome"] == "external_address_not_returned":
        # The freed external address was not handed to any same-size candidate;
        # nothing was written and the replay could not have been challenged.
        require(mode in ("external-free-ordered", "external-free-cross"),
                "unexpected no-return outcome for mode " + mode)
        return dict(id=trial["id"], mode=mode, outcome=trial["outcome"],
                    replacement_reused_external=False, replacement_intact=True,
                    candidate_attempts=trial.get("candidate_attempts"))
    counts = (trial["base_count"] + trial["sentinel_count"] + trial["other_count"])
    require(counts == trial["elements"], "element classification does not cover output")
    require(trial["replacement_intact"], "replacement lost its own sentinel values")
    if mode == "keep-alive":
        require(trial["outcome"] == "intact", "keep-alive control corrupted replay output")
        require(trial["replacement_reused_external"] is False or trial["outcome"] == "intact",
                "keep-alive trial reused a live address")
    return dict(id=trial["id"], mode=mode, outcome=trial["outcome"],
                replacement_reused_external=trial["replacement_reused_external"],
                base_count=trial["base_count"], sentinel_count=trial["sentinel_count"],
                other_count=trial["other_count"])


def analyze(run):
    cases = json.loads((run / "cases.json").read_text())
    plan = json.loads((run / "plan.json").read_text())
    require(cases, "no completed cases")
    results = []
    for case in cases:
        require(case["returncode"] == 0, "failed case: " + case["case"])
        directory = run / case["case"]
        metadata = json.loads((directory / "run.json").read_text())
        for relative, expected_hash in metadata["source_sha256"].items():
            require(hashlib.sha256((directory / relative).read_bytes()).hexdigest() == expected_hash,
                    "source fingerprint mismatch: " + relative)
        require(metadata["allocator_backend"] == "native", "non-native backend")
        require(len(metadata["trials"]) == plan["arguments"]["repeats"], "incomplete repetitions")
        trials = [summarize_trial(trial) for trial in metadata["trials"]]
        by_outcome = {}
        for trial in trials:
            by_outcome[trial["outcome"]] = by_outcome.get(trial["outcome"], 0) + 1
        results.append(dict(case=case["case"], mode=metadata["arguments"]["mode"],
                            repeats=len(trials), outcomes=by_outcome,
                            address_reuse_trials=sum(
                                trial.get("replacement_reused_external")
                                or trial.get("replacement_reused_static", False) for trial in trials),
                            trials=trials))
    return dict(schema=1, analysis_status="passed", cases=results,
                observation="Replay output classification shows whether a replayed graph consumed "
                            "replacement data written into freed external storage, and whether "
                            "destroying a graph with a pending replay disturbs later allocations. "
                            "No instruction-level access reconstruction is claimed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    result = analyze(args.run)
    (args.run / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print("case | outcomes | address reuse")
    for case in result["cases"]:
        print(case["case"], case["outcomes"], case["address_reuse_trials"], sep=" | ")


if __name__ == "__main__":
    main()
