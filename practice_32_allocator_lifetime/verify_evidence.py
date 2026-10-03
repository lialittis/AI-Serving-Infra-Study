"""Check archived bytes, recomputed summaries, snapshots and raw copy timings."""
import argparse
import csv
from decimal import Decimal
import gzip
import hashlib
import json
from pathlib import Path

from analyze import analyze
from analyze_profile import analyze as profile_analyze
from probe import locate


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = snapshots = trials = 0
    checks = []
    for manifest_path in sorted(args.results.glob("*/archive_manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        for entry in manifest["retained_files"] + manifest["raw_profiler_files"]:
            path = manifest_path.parent / entry["path"]
            data = path.read_bytes()
            require(len(data) == entry["bytes"], "size: " + str(path))
            require(hashlib.sha256(data).hexdigest() == entry["sha256"], "hash: " + str(path))
            files += 1
    for name in ("formal-02", "lazy-03", "custom-owner-01", "profile-01"):
        root = args.results / name
        expected = json.loads((root / "summary.json").read_text())
        require(analyze(root, expected["selected_configurations"]) == expected, "summary: " + name)
        for case in expected["cases"]:
            directory = root / case["case"]
            metadata = json.loads((directory / "run.json").read_text())
            for trial in metadata["trials"]:
                trials += 1
                stages = {"released": "block_after_release", "window": "block_after_window",
                          "synchronized": "block_after_synchronize"}
                for stage, key in stages.items():
                    path = directory / f"{trial['id']}-{stage}.json.gz"
                    with gzip.open(path, "rt") as stream:
                        snapshot = json.load(stream)
                    require(locate(snapshot, int(trial["address_a"])) == trial[key], "snapshot: " + str(path))
                    snapshots += 1
                trigger = trial.get("completion_cache_miss_trigger")
                if trigger and trigger["status"] == "executed":
                    require(trigger["requested_bytes"] > trigger["largest_cached_free_block"], "miss not forced")
                    require(trigger["requested_bytes"] <= 192 << 20, "miss bound exceeded")
                    for key in ("num_alloc_retries", "num_ooms"):
                        require(trigger["statistics_after_trigger"][key] == trial["statistics_before"][key], "miss: " + key)
            if name != "profile-01":
                continue
            evidence = json.loads((directory / "profile_evidence.json").read_text())
            recomputed = profile_analyze(directory)
            recomputed["source"]["trace"] = evidence["source"]["trace"]
            require(recomputed == evidence, "raw flow: " + case["case"])
            csv_path, = directory.rglob("kernel_details.csv")
            with csv_path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            for trial in evidence["trials"]:
                for task in trial["copy_tasks"]:
                    matched = [r for r in rows if r["Name"] == task["task_name"]
                               and r["Task ID"] == task["task_id"] and r["Stream ID"] == task["physical_stream"]
                               and Decimal(r["Start Time(us)"].strip()) == Decimal(task["device_start_us"])]
                    require(len(matched) == 1, "CSV task identity: " + case["case"])
                    row = matched[0]
                    require(Decimal(row["Duration(us)"]) == Decimal(task["device_end_us"]) - Decimal(task["device_start_us"]), "CSV duration")
                    checks.append(dict(case=case["case"], trial=trial["trial"], task_id=task["task_id"],
                                       stream=task["physical_stream"], task_name=task["task_name"],
                                       device_start_us=task["device_start_us"], duration_us=row["Duration(us)"],
                                       allocation_before_copy=trial["reused_before_copy_tasks_start"]))
    result = dict(status="passed", hashed_files=files, recomputed_trials=trials,
                  primary_stage_snapshots_checked=snapshots, exact_csv_copy_checks=checks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v if k != "exact_csv_copy_checks" else len(v) for k, v in result.items()}))


if __name__ == "__main__":
    main()
