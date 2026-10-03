"""Offline analysis of live sampling-reuse observations."""
import argparse
import json
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def summarize_case(directory, case, variant):
    records_files = sorted(directory.glob("records-*.json"))
    require(records_files, "no observer records: " + case)
    merged = [payload for path in records_files
              if (payload := json.loads(path.read_text()))["calls"]]
    generation = json.loads((directory / "generation.json").read_text())
    if not merged:
        # The protected variant never calls random_sample; that absence is the
        # expected observation, but only for that variant.
        require(variant == "protected", "no executor called random_sample: " + case)
        return dict(case=case, variant=variant, calls=0, address_reuse=0,
                    prev_div_pending_while_reused=0, precondition_indices=[],
                    unique_addresses=0, prev_gap_ms=None,
                    repeats_identical=generation["identical_across_repeats"],
                    interpretation="enable_async_exponential path never enters random_sample.")
    # Exactly one executor process should report calls; others import but never sample.
    require(len(merged) == 1, f"expected one executor with calls, got {len(merged)}")
    payload = merged[0]
    records = payload["records"]
    for entry in records:
        if entry["reused_prev"] is not None:
            require(isinstance(entry["reused_prev"], bool), "reused_prev must be boolean")
    with_prev = [entry for entry in records if entry["reused_prev"] is not None]
    reused = [entry for entry in with_prev if entry["reused_prev"]]
    precondition = [entry for entry in reused if entry["prev_div_pending"]]
    gaps = sorted(entry["prev_gap_ms"] for entry in with_prev)
    return dict(case=case, variant=variant, executor_pid=payload["pid"], calls=len(records),
                rows_first=records[0]["rows"] if records else 0,
                address_reuse=len(reused), prev_div_pending_while_reused=len(precondition),
                precondition_indices=[entry["index"] for entry in precondition],
                unique_addresses=len({entry["address"] for entry in records}),
                prev_gap_ms=dict(min=gaps[0], median=gaps[len(gaps) // 2], max=gaps[-1]) if gaps else None,
                repeats_identical=generation["identical_across_repeats"],
                original_source_sha256=payload["original_source"]["sha256"],
                rows_samples=[entry["rows"] for entry in records[:5]],
                interpretation=("Address reuse while the previous div_ was still pending is the "
                                "P34 race precondition; pending flags come from host-side event "
                                "queries, which do not reconstruct device instruction order."))


def analyze(run):
    cases = json.loads((run / "cases.json").read_text())
    results = []
    for case in cases:
        directory = run / case["case"]
        variant = case["case"].split("-", 1)[1]
        if case["returncode"] == 2:
            error = json.loads((directory / "engine_error.json").read_text())
            results.append(dict(case=case["case"], variant=variant, status="unsupported",
                                error=error["error"]))
            continue
        require(case["returncode"] == 0, "failed case: " + case["case"])
        results.append(summarize_case(directory, case["case"], variant))
    for result in results:
        if result.get("status") != "unsupported" and result["variant"] == "protected":
            require(result["calls"] == 0,
                    "protected variant unexpectedly used random_sample (flag not effective?)")
    return dict(schema=1, analysis_status="passed", cases=results,
                observation="Precondition counts are host-observed; they bound when the P34 hazard "
                            "could begin, and do not by themselves demonstrate corrupted tokens.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    result = analyze(args.run)
    (args.run / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print("case | calls | reuse | precondition | repeats identical")
    for case in result["cases"]:
        if case.get("status") == "unsupported":
            print(case["case"], "UNSUPPORTED", case["error"][:60], sep=" | ")
        else:
            print(case["case"], case["calls"], case["address_reuse"],
                  case["prev_div_pending_while_reused"], case["repeats_identical"], sep=" | ")


if __name__ == "__main__":
    main()
