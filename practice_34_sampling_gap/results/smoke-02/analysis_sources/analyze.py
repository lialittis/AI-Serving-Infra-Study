"""Offline gap-curve analysis; corruption is counted per iteration element check."""
import argparse
import hashlib
import json
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def summarize_row(row):
    require(row["outcome"] == ("intact" if row["all_elements_expected"] else "overwritten"),
            "outcome inconsistent with element check")
    if row["outcome"] == "overwritten":
        require(row["mismatch_count"] >= 1, "overwritten row without mismatch count")
    return dict(index=row["index"], expected=row["expected"], address=row["address"],
                outcome=row["outcome"],
                mismatch_count=row.get("mismatch_count"),
                foreign_values=row.get("foreign_values"),
                next_address_reused=row["next_address_reused"])


def summarize_interval(interval, iterations, elements):
    require([row["index"] for row in interval["rows"]] == list(range(iterations)), "iteration coverage")
    rows = [summarize_row(row) for row in interval["rows"]]
    corrupt = [row for row in rows if row["outcome"] == "overwritten"]
    reused = [row for row in rows if row["next_address_reused"]]
    return dict(gap_us=interval["gap_us"], iterations=len(rows),
                corrupt_iterations=len(corrupt),
                adjacent_address_reuse=len(reused),
                corrupt_indices=[row["index"] for row in corrupt],
                first_corrupt_index=corrupt[0]["index"] if corrupt else None,
                fully_overwritten_iterations=sum(1 for row in corrupt
                                                 if row["mismatch_count"] == elements),
                rows=rows)


def summarize_case(directory, metadata, plan):
    for relative, expected_hash in metadata["source_sha256"].items():
        require(hashlib.sha256((directory / relative).read_bytes()).hexdigest() == expected_hash,
                "source fingerprint mismatch: " + relative)
    require(metadata["allocator_backend"] == "native", "non-native backend")
    require(metadata["statistics_after"]["num_ooms"] == metadata["statistics_before"]["num_ooms"],
            "OOM confounds run")
    require(metadata["statistics_after"]["num_alloc_retries"]
            == metadata["statistics_before"]["num_alloc_retries"], "allocator retry confounds run")
    actions = [entry["action"] for entry in metadata["trace_entries_for_first_address"]]
    require("alloc" in actions and "free_requested" in actions,
            "first-address alloc/free not verified by allocator history")
    iterations = metadata["arguments"]["iterations"]
    elements = metadata["arguments"]["elements"]
    intervals = [summarize_interval(interval, iterations, elements) for interval in metadata["intervals"]]
    require([interval["gap_us"] for interval in intervals]
            == plan["arguments"]["gaps_us"], "gap coverage mismatch")
    mode = metadata["arguments"]["mode"]
    backlog = metadata["arguments"]["consumer_backlog"]
    if mode in ("record", "leading-wait"):
        require(all(interval["corrupt_iterations"] == 0 for interval in intervals),
                mode + " protection path still corrupted observed")
    return dict(case=f"{mode}-b{backlog}", mode=mode, consumer_backlog=backlog,
                environment=metadata["environment"], iterations=iterations,
                gaps_us=[interval["gap_us"] for interval in intervals],
                corrupt_by_gap={str(interval["gap_us"]): interval["corrupt_iterations"]
                                for interval in intervals},
                adjacent_reuse_by_gap={str(interval["gap_us"]): interval["adjacent_address_reuse"]
                                       for interval in intervals},
                total_corrupt=sum(interval["corrupt_iterations"] for interval in intervals),
                total_iterations=sum(interval["iterations"] for interval in intervals),
                first_fully_corrupt_gap=next((interval["gap_us"] for interval in intervals
                                              if interval["corrupt_iterations"]), None),
                intervals=intervals,
                allocator_actions_for_first_address=actions)


def analyze(run, configs=None):
    cases = json.loads((run / "cases.json").read_text())
    plan = json.loads((run / "plan.json").read_text())
    selected_configs = configs if configs is not None else plan["arguments"]["configs"]
    require(set(selected_configs) <= set(plan["arguments"]["configs"]), "unknown configuration subset")
    excluded = [case for case in cases if case["case"].split("-")[0] not in selected_configs]
    cases = [case for case in cases if case["case"].split("-")[0] in selected_configs]
    require(cases, "no completed cases")
    results = []
    for case in cases:
        require(case["returncode"] == 0, "failed case: " + case["case"])
        directory = run / case["case"]
        metadata = json.loads((directory / "run.json").read_text())
        results.append(summarize_case(directory, metadata, plan))
    return dict(schema=1, analysis_status="passed", cases=results, selected_configurations=selected_configs,
                excluded_case_statuses=excluded,
                iteration_count=sum(case["total_iterations"] for case in results),
                observation="Corruption counts come from full element checks after each interval's "
                            "synchronization. Gaps are host-side sleeps standing in for scheduler/forward "
                            "latency; they do not enqueue device work on the main stream.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--configs", nargs="+")
    args = parser.parse_args()
    result = analyze(args.run, args.configs)
    (args.run / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print("mode | gap: corrupt/iterations (adjacent reuse)")
    for case in result["cases"]:
        cells = [f"{case['gaps_us'][i]}us: {case['corrupt_by_gap'][str(case['gaps_us'][i])]}/{case['iterations']}"
                 f" ({case['adjacent_reuse_by_gap'][str(case['gaps_us'][i])]})"
                 for i in range(len(case["gaps_us"]))]
        print(case["case"], "|", " ".join(cells))


if __name__ == "__main__":
    main()
