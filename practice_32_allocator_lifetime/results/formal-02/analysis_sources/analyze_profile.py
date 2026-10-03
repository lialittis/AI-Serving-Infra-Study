"""Exact flow associations for copy tasks; do not infer device access from query."""
import argparse
from collections import defaultdict
from decimal import Decimal
import hashlib
import json
from pathlib import Path


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

    copy_records = {record["label"]: record for record in metadata["records"] if record["operation"] == "consumer-copy"}
    copies, mapped, unresolved = [], [], []
    record_by_label = {record["label"]: record for record in metadata["records"]}
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
        labels = [label for label in record_by_label if inside(scopes[label], host)]
        if not labels:
            continue
        label = only(labels, "copy scope association")
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
        if label in copy_records:
            copies.append(task)
    results = []
    for trial in metadata["trials"]:
        tasks = [task for task in copies if task["trial"] == trial["id"]]
        if not tasks:
            raise ValueError("No device copy task associated with " + trial["id"])
        task_start = min(number(task["device_start_us"]) for task in tasks)
        task_end = max(number(task["device_end_us"]) for task in tasks)
        matches = [row for row in trial["observations"] if row["reused"]]
        allocated_before = []
        for allocation in matches:
            scope = scopes[allocation["allocation_label"]]
            if finish(scope) < task_start:
                allocated_before.append(dict(index=allocation["index"], allocation_scope_end_us=str(finish(scope)),
                                             device_copy_start_us=str(task_start), gap_us=str(task_start - finish(scope))))
        results.append(dict(trial=trial["id"], copy_tasks=tasks,
                            device_copy_start_us=str(task_start), device_copy_end_us=str(task_end),
                            reused_before_copy_tasks_start=allocated_before,
                            copy_end_marker_progress_separate=True))
    return dict(schema=1, analysis_status="passed", mode=metadata["arguments"]["mode"],
                trials=results, mapped_tasks=mapped,
                host_scopes={label: dict(start_us=str(scope["ts"]), end_us=str(finish(scope)))
                             for label, scope in scopes.items()},
                unresolved_device_task_flows=unresolved,
                source=dict(trace=str(trace_path), trace_sha256=hashlib.sha256(trace_path.read_bytes()).hexdigest(),
                            metadata_sha256=hashlib.sha256((directory / "run.json").read_bytes()).hexdigest()),
                limit="Profiler-reported host/device order of address assignment and old copy; no candidate device accesses, instruction-level read intervals or conflicting access demonstrated.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    outcomes = []
    for case in json.loads((args.run / "cases.json").read_text()):
        directory = args.run / case["case"]
        result = analyze(directory)
        (directory / "profile_evidence.json").write_text(json.dumps(result, indent=2) + "\n")
        outcomes.append(dict(case=case["case"], evidence=result))
        print(case["case"], [(t["trial"], len(t["reused_before_copy_tasks_start"])) for t in result["trials"]])
    (args.run / "profile_summary.json").write_text(json.dumps(outcomes, indent=2) + "\n")


if __name__ == "__main__":
    main()
