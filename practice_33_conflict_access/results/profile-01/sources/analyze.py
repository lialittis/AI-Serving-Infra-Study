"""Offline conflict classification; element counts separate corruption from ordering luck."""
import argparse
import hashlib
import json
from pathlib import Path

CORRUPT_OUTCOMES = ("fully_overwritten", "mixed", "unexpected_values")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def summarize_trial(trial):
    require(trial["tensor_python_owner_released"], "Python owner still live")
    require(trial["write_target_all_sentinel"], "write target lost its own sentinel values")
    require(trial["observed_original_count"] + trial["observed_sentinel_count"]
            + trial["observed_other_count"] == trial["elements"], "element classification does not cover observed")
    observations = trial["observations"]
    require([row["index"] for row in observations] == list(range(len(observations))), "allocation coverage")
    require(all(row["reused"] == (row["pointer"] == trial["address_a"]) for row in observations), "address equality mismatch")
    before, after = trial["statistics_before"], trial["statistics_after_window"]
    require(after["num_alloc_retries"] == before["num_alloc_retries"], "allocator retry confounds window")
    require(after["num_ooms"] == before["num_ooms"], "OOM confounds window")
    actions = [row["action"] for row in trial["trace_entries_for_a"]]
    require("alloc" in actions and "free_requested" in actions, "storage free not verified by allocator history")
    matches = [row for row in observations if row["reused"]]

    def pending_both(row, marker):
        return not row["before"][marker] and not row["after"][marker]

    outcome = trial["observed_outcome"]
    mode = trial["mode"]
    reused = bool(matches)
    if mode == "record":
        require(not reused, "record_stream failed to prevent in-window address reuse")
        require(trial["write_address_reused"] is False, "record trial wrote the protected address")
    if mode == "join":
        require(reused, "join trial did not reuse the address (differs from P32 baseline)")
        require(trial["write_address_reused"], "join trial did not write the reused address")
    if mode == "synced":
        require(outcome == "intact", "synced checker control corrupted observed")
    if mode in ("join", "record"):
        require(outcome == "intact", f"{mode} protection path still corrupted observed: {outcome}")
    if mode == "omit" and reused:
        require(trial["write_address_reused"], "omit trial failed to target the reused address")
    interpretation = {
        "intact": "Observed kept A's values; no corruption manifested this trial.",
        "fully_overwritten": "Observed equals the candidate sentinel everywhere: the old consumer copy "
                             "read storage after the candidate write had replaced its contents.",
        "mixed": "Observed contains both A's values and sentinel values: partial interleaving.",
        "unexpected_values": "Observed contains values that are neither A's nor the sentinel.",
    }[outcome]
    if not reused and mode == "omit":
        interpretation = "No in-window reuse; the write hit a different address and proves nothing about conflicts."
    return dict(id=trial["id"], mode=mode, address_a=trial["address_a"],
                candidate_count=len(observations), write_index=trial["write_index"],
                write_address_reused=trial["write_address_reused"],
                first_reuse_index=matches[0]["index"] if matches else None,
                reuse_while_done_pending=bool(matches) and pending_both(matches[0], "done"),
                reuse_while_copy_end_pending=bool(matches) and pending_both(matches[0], "read_end"),
                reuse_while_copy_start_pending=bool(matches) and pending_both(matches[0], "read_start"),
                observed_outcome=outcome,
                observed_original_count=trial["observed_original_count"],
                observed_sentinel_count=trial["observed_sentinel_count"],
                observed_other_count=trial["observed_other_count"],
                released_states=[b["state"] for b in trial["block_after_release"]],
                synchronized_states=[b["state"] for b in trial["block_after_synchronize"]],
                write_target_all_sentinel=True,
                free_requested_verified=True,
                allocator_actions=actions,
                interpretation=interpretation)


def analyze(run, configs=None):
    cases = json.loads((run / "cases.json").read_text())
    plan = json.loads((run / "plan.json").read_text())
    selected_configs = configs if configs is not None else plan["arguments"]["configs"]
    require(set(selected_configs) <= set(plan["arguments"]["configs"]), "unknown configuration subset")
    excluded = [case for case in cases if case["case"].split("-t")[0] not in selected_configs]
    cases = [case for case in cases if case["case"].split("-t")[0] in selected_configs]
    expected = len(selected_configs) * len(plan["arguments"]["modes"])
    require(len(cases) == expected, "incomplete case matrix")
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
            by_outcome[trial["observed_outcome"]] = by_outcome.get(trial["observed_outcome"], 0) + 1
        results.append(dict(case=case["case"], mode=metadata["arguments"]["mode"],
                            environment=metadata["environment"], repeats=len(trials),
                            reuse_trials=sum(t["first_reuse_index"] is not None for t in trials),
                            write_targeted_reused_address=sum(t["write_address_reused"] for t in trials),
                            outcomes=by_outcome, corrupt_trials=sum(by_outcome.get(name, 0)
                                                                   for name in CORRUPT_OUTCOMES),
                            pending_at_first_reuse_done=sum(t["reuse_while_done_pending"] for t in trials),
                            trials=trials))
    return dict(schema=1, analysis_status="passed", cases=results, selected_configurations=selected_configs,
                excluded_case_statuses=excluded,
                trial_count=sum(case["repeats"] for case in results),
                observation="Element classification after full completion shows whether the old consumer copy "
                            "consumed candidate-written data. Device task order is only claimed when the "
                            "separate profiler evidence agrees; element counts alone do not reconstruct it.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--configs", nargs="+", help="Analyze an explicit completed subset of a stopped matrix")
    args = parser.parse_args()
    result = analyze(args.run, args.configs)
    (args.run / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print("case | reuse | wrote reused | intact | overwritten | mixed | unexpected | pending done")
    for case in result["cases"]:
        outcomes = case["outcomes"]
        print(case["case"], case["reuse_trials"], case["write_targeted_reused_address"],
              outcomes.get("intact", 0), outcomes.get("fully_overwritten", 0),
              outcomes.get("mixed", 0), outcomes.get("unexpected_values", 0),
              case["pending_at_first_reuse_done"], sep=" | ")


if __name__ == "__main__":
    main()
