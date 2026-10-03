"""Offline characterization; pending terminal events never imply a data race."""
import argparse
import hashlib
import json
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def summarize_trial(trial):
    require(trial["tensor_python_owner_released"], "Python owner still live")
    require(trial["copy_all_elements_correct"], "copy validation failed")
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
    early = [row for row in matches if pending_both(row, "done")]
    end_pending = [row for row in matches if pending_both(row, "read_end")]
    start_pending = [row for row in matches if pending_both(row, "read_start")]
    return dict(id=trial["id"], mode=trial["mode"], threads=trial["threads"],
                address_a=trial["address_a"], candidate_count=len(observations),
                terminal_pending_after_release=not trial["progress_after_release"]["done"],
                first_reuse_index=matches[0]["index"] if matches else None,
                reuse_while_terminal_marker_pending=bool(early),
                reuse_while_copy_end_marker_pending=bool(end_pending),
                reuse_while_copy_start_marker_pending=bool(start_pending),
                released_states=[b["state"] for b in trial["block_after_release"]],
                synchronized_states=[b["state"] for b in trial["block_after_synchronize"]],
                polled_states=[b["state"] for b in trial["block_after_poll"]],
                reuse_after_completion=any(row["reused"] for row in trial["allocations_after_completion"]),
                cache_miss_reuse_after_completion=bool(trial.get("completion_cache_miss_trigger")
                                                      and trial["completion_cache_miss_trigger"]["reused"]),
                release_thread=trial["release"]["thread_id"], main_thread=trial["submitting_main_thread"],
                free_requested_verified=True, correct_output=True,
                allocator_actions=actions,
                interpretation="Address selection and marker progress only; no candidate device access and no demonstrated race.")


def analyze(run):
    cases = json.loads((run / "cases.json").read_text())
    plan = json.loads((run / "plan.json").read_text())
    expected = len(plan["arguments"]["configs"]) * len(plan["arguments"]["threads"]) * len(plan["arguments"]["modes"])
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
        require(all((trial["release_thread"] == trial["main_thread"]) == (trial["threads"] == 1)
                    for trial in trials), "host-thread control not effective")
        results.append(dict(case=case["case"], mode=metadata["arguments"]["mode"],
                            threads=metadata["arguments"]["threads"], environment=metadata["environment"],
                            repeats=len(trials), reuse_trials=sum(t["first_reuse_index"] is not None for t in trials),
                            pending_release_trials=sum(t["terminal_pending_after_release"] for t in trials),
                            reuse_pending_done_trials=sum(t["reuse_while_terminal_marker_pending"] for t in trials),
                            reuse_pending_copy_end_trials=sum(t["reuse_while_copy_end_marker_pending"] for t in trials),
                            reuse_pending_copy_start_trials=sum(t["reuse_while_copy_start_marker_pending"] for t in trials),
                            post_completion_reuse_trials=sum(t["reuse_after_completion"] for t in trials),
                            cache_miss_reuse_trials=sum(t["cache_miss_reuse_after_completion"] for t in trials),
                            release_states=sorted(set(state for t in trials for state in t["released_states"])),
                            synchronized_states=sorted(set(state for t in trials for state in t["synchronized_states"])),
                            all_copies_correct=True, trials=trials))
    return dict(schema=1, analysis_status="passed", cases=results,
                trial_count=sum(case["repeats"] for case in results),
                observation="No candidates were read or written on the NPU; address reuse is not proof of a conflicting device access.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    result = analyze(args.run)
    (args.run / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print("case | reuse | pending done | pending copy end | pending copy start | after completion")
    for case in result["cases"]:
        print(case["case"], case["reuse_trials"], case["reuse_pending_done_trials"],
              case["reuse_pending_copy_end_trials"], case["reuse_pending_copy_start_trials"],
              case["post_completion_reuse_trials"], sep=" | ")


if __name__ == "__main__":
    main()
