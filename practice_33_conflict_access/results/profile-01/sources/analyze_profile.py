"""Exact flow associations for copy and candidate-write tasks; element counts stay separate."""
import argparse
from collections import defaultdict
from decimal import Decimal
import hashlib
import json
from pathlib import Path

MAPPED_OPERATIONS = ("consumer-copy", "write-candidate", "create-a")


def number(value):
    return Decimal(str(value))


def finish(event):
    return number(event["ts"]) + number(event.get("dur", 0))


def point(event):
    return str(event["pid"]), str(event["tid"]), number(event["ts"])


def inside(scope, event):
    return (str(scope["pid"]) == str(event["pid"]) and str(scope["tid"]) == str(event["tid"])
            and number(scope["ts"]) <= number(event["ts"]) and finish(event) <= finish(scope))


def only(rows, message):
    rows = list(rows)
    if len(rows) != 1:
        raise ValueError(f"{message}: {len(rows)} candidates")
    return rows[0]


def analyze(directory):
    metadata = json.loads((directory / "run.json").read_text())
    trace_path = only(directory.rglob("trace_view.json"), "trace file")
    events = json.loads(trace_path.read_text(), parse_float=Decimal)
    if isinstance(events, dict):
        events = events["traceEvents"]
    scopes = {}
    for record in metadata["records"]:
        scopes[record["label"]] = only((event for event in events if event.get("ph") == "X"
                                       and event.get("name") == record["label"]), "scope " + record["label"])
    complete, starts, ends = defaultdict(list), defaultdict(list), defaultdict(list)
    for index, event in enumerate(events):
        if event.get("ph") == "X":
            complete[point(event)].append((index, event))
        elif event.get("ph") == "s":
            starts[event.get("cat"), str(event["id"])].append((index, event))
        elif event.get("ph") == "f":
            ends[(event.get("cat"),) + point(event)].append((index, event))

    def source(task, category):
        end_index, endpoint = only(ends[(category,) + point(task)], category + " endpoint")
        start_index, origin = only(starts[category, str(endpoint["id"])], category + " origin")
        source_index, node = only(complete[point(origin)], category + " source node")
        return source_index, node, dict(id=str(endpoint["id"]), start_index=start_index, end_index=end_index)

    operations = {record["label"]: record for record in metadata["records"]
                  if record["operation"] in MAPPED_OPERATIONS}
    record_by_label = {record["label"]: record for record in metadata["records"]}
    mapped, unresolved = [], []
    for index, event in enumerate(events):
        if event.get("ph") != "X" or "Task Type" not in event.get("args", {}):
            continue
        if event.get("name") in ("PROFILING_ENABLE", "PROFILING_DISABLE"):
            continue
        try:
            host_index, host, torch_flow = source(event, "async_npu")
        except ValueError as exc:
            unresolved.append(dict(task_index=index, name=event["name"], error=str(exc)))
            continue
        labels = [label for label in operations if inside(scopes[label], host)]
        if not labels:
            continue
        label = only(labels, "scope association")
        cann_index, cann, cann_flow = source(event, "HostToDevice")
        if str(cann["args"]["connection_id"]) != str(event["args"]["connection_id"]):
            raise ValueError("CANN connection mismatch")
        task = dict(trial=record_by_label[label]["trial"], label=label,
                    operation=record_by_label[label]["operation"],
                    task_index=index, task_name=event["name"], task_type=event["args"]["Task Type"],
                    physical_stream=str(event["args"]["Physic Stream Id"]), task_id=str(event["args"]["Task Id"]),
                    device_start_us=str(event["ts"]), device_end_us=str(finish(event)),
                    host_index=host_index, host_name=host["name"], cann_index=cann_index, cann_name=cann["name"],
                    connection_id=str(event["args"]["connection_id"]),
                    torch_flow=torch_flow, cann_flow=cann_flow)
        mapped.append(task)
    results = []
    for trial in metadata["trials"]:
        per_trial = [task for task in mapped if task["trial"] == trial["id"]]
        copies = [task for task in per_trial if task["operation"] == "consumer-copy"]
        writes = [task for task in per_trial if task["operation"] == "write-candidate"]
        if not copies:
            raise ValueError("No device copy task associated with " + trial["id"])
        if not writes:
            raise ValueError("No device write task associated with " + trial["id"])
        copy_start = min(number(task["device_start_us"]) for task in copies)
        copy_end = max(number(task["device_end_us"]) for task in copies)
        write_start = min(number(task["device_start_us"]) for task in writes)
        write_end = max(number(task["device_end_us"]) for task in writes)
        if write_end <= copy_start:
            order = "write_completed_before_copy_started"
        elif copy_end <= write_start:
            order = "copy_completed_before_write_started"
        else:
            order = "intervals_overlap"
        results.append(dict(trial=trial["id"], copy_tasks=copies, write_tasks=writes,
                            device_copy_interval_us=[str(copy_start), str(copy_end)],
                            device_write_interval_us=[str(write_start), str(write_end)],
                            device_order=order,
                            write_address_reused=trial["write_address_reused"],
                            observed_outcome=trial["observed_outcome"],
                            observed_sentinel_count=trial["observed_sentinel_count"],
                            first_reuse_index=next((row["index"] for row in trial["observations"]
                                                    if row["reused"]), None)))
    return dict(schema=1, analysis_status="passed", mode=metadata["arguments"]["mode"],
                trials=results, mapped_tasks=mapped,
                host_scopes={label: dict(start_us=str(scope["ts"]), end_us=str(finish(scope)))
                             for label, scope in scopes.items()},
                unresolved_device_task_flows=unresolved,
                source=dict(trace=str(trace_path), trace_sha256=hashlib.sha256(trace_path.read_bytes()).hexdigest(),
                            metadata_sha256=hashlib.sha256((directory / "run.json").read_bytes()).hexdigest()),
                limit="Profiler-reported device intervals for the old copy and the candidate write. "
                      "They do not reconstruct instruction-level memory accesses, and the element "
                      "classification in run.json remains the corruption evidence.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--check-existing", action="store_true", help="Recompute raw evidence without rewriting archived results")
    args = parser.parse_args()
    outcomes = []
    for case in json.loads((args.run / "cases.json").read_text()):
        directory = args.run / case["case"]
        result = analyze(directory)
        if args.check_existing:
            expected = json.loads((directory / "profile_evidence.json").read_text())
            # Restoring a trace changes its absolute filesystem path, not its
            # fingerprint, indices, flow associations or intervals.
            result["source"]["trace"] = expected["source"]["trace"]
            if result != expected:
                raise ValueError("Recomputed raw evidence differs from archive: " + case["case"])
        else:
            (directory / "profile_evidence.json").write_text(json.dumps(result, indent=2) + "\n")
        outcomes.append(dict(case=case["case"], evidence=result))
        print(case["case"], [(t["trial"], t["device_order"], t["observed_outcome"]) for t in result["trials"]])
    if not args.check_existing:
        (args.run / "profile_summary.json").write_text(json.dumps(outcomes, indent=2) + "\n")


if __name__ == "__main__":
    main()
