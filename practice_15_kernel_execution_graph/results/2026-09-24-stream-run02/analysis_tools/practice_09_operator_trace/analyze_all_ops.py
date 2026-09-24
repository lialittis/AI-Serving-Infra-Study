"""Audit every host operator and device task in the captured single-request trace.

Device attribution requires two real flow IDs, never a nearest-time guess.
CPU nesting is derived only within the same profiler PID/TID lane. Source
snapshots explain implementations; they are not fabricated Python stack frames.
"""
import argparse
from collections import Counter, defaultdict
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path

from summarize_profile import number, only, read_trace, require


TRITON = {
    "_triton_rope": "RoPE",
    "_compute_slot_mapping_kernel": "KV slot mapping",
    "token_bin_counts_and_mask_kernel": "token histogram / mask",
    "apply_all_penalties_kernel": "sampling penalties",
}
CONTROL = {"PROFILING_ENABLE", "PROFILING_DISABLE"}


def point(event):
    return event["pid"], event["tid"], number(event["ts"])


def route(task, ancestry):
    name = task["name"]
    if name in CONTROL:
        return "profiler_control"
    if name == "MEMCPY_ASYNC":
        return "runtime_copy"
    if name == "EVENT_RECORD":
        return "runtime_event"
    if name in TRITON:
        return "triton"
    if any(n.startswith("_C_ascend::") for n in ancestry):
        return "ascend_custom_cpp"
    if any(n.startswith("atb::") for n in ancestry):
        return "atb"
    if any(n.startswith("npu::") for n in ancestry):
        return "torch_npu"
    if any(n.startswith("aten::") for n in ancestry):
        return "aten_to_aclnn"
    raise ValueError("unclassified device task: " + name)


def cpu_parents(events, indices):
    lanes = defaultdict(list)
    for i in indices:
        e = events[i]
        lanes[e["pid"], e["tid"]].append(i)
    parents = {}
    for ids in lanes.values():
        ids.sort(key=lambda i: (number(events[i]["ts"]), -number(events[i]["dur"])))
        stack = []
        for i in ids:
            e = events[i]
            begin, end = number(e["ts"]), number(e["ts"]) + number(e["dur"])
            while stack:
                parent = events[stack[-1]]
                if (number(parent["ts"]) <= begin and
                        end <= number(parent["ts"]) + number(parent["dur"])):
                    break
                stack.pop()
            parents[i] = stack[-1] if stack else None
            stack.append(i)
    return parents


def audit(events, kernels):
    cpu_ids = [i for i, e in enumerate(events) if e.get("ph") == "X" and e.get("cat") == "cpu_op"]
    device_ids = [i for i, e in enumerate(events)
                  if e.get("ph") == "X" and "Task Type" in e.get("args", {})]
    require(cpu_ids and device_ids, "missing host/device events")
    parents = cpu_parents(events, cpu_ids)
    complete = defaultdict(list)
    flows = defaultdict(list)
    endpoints = defaultdict(list)
    for i, e in enumerate(events):
        if e.get("ph") == "X":
            complete[point(e)].append(i)
        elif e.get("ph") in ("s", "f"):
            flows[e["cat"], str(e["id"]), e["ph"]].append(i)
            if e["ph"] == "f":
                endpoints[(e["cat"],) + point(e)].append(i)
    kernel_rows = defaultdict(list)
    for row in kernels:
        kernel_rows[(row["Name"], row["Task ID"], row["Stream ID"],
                     number(row["Start Time(us)"]))].append(row)
    direct, descendants = defaultdict(set), defaultdict(set)
    rows, examples, used_csv = [], {}, set()
    for i in sorted(device_ids, key=lambda i: number(events[i]["ts"])):
        task = events[i]
        args = task["args"]
        row = {"trace_index": i, "kernel": task["name"], "task_type": args["Task Type"],
               "task_id": args["Task Id"], "stream_id": args["Physic Stream Id"],
               "device_start_us": str(task["ts"]), "device_duration_us": task["dur"],
               "phase": "profiler_control", "host_operator": "", "host_ancestry": [],
               "host_start_us": "", "host_duration_us": "",
               "torch_flow_id": "", "cann_flow_id": "", "cann_launch": "",
               "cann_connection_verified": False, "kernel_csv_verified": False}
        selected = [i]
        if task["name"] not in CONTROL:
            host = None
            for cat in ("async_npu", "HostToDevice"):
                end_id = only(endpoints[(cat,) + point(task)], cat + " device endpoint")
                end = events[end_id]
                start_id = only(flows[cat, str(end["id"]), "s"], cat + " flow start")
                require(len(flows[cat, str(end["id"]), "f"]) == 1, "ambiguous flow end")
                start = events[start_id]
                source_id = only(complete[point(start)], cat + " host event")
                source = events[source_id]
                selected.extend([end_id, start_id, source_id])
                if cat == "async_npu":
                    require(source.get("cat") == "cpu_op", "torch source is not a cpu_op")
                    host = source_id
                    row["torch_flow_id"] = str(start["id"])
                else:
                    row["cann_flow_id"] = str(start["id"])
                    row["cann_launch"] = source["name"]
                    row["cann_connection_verified"] = (
                        source.get("args", {}).get("connection_id") == args.get("connection_id"))
                    require(row["cann_connection_verified"], "CANN connection ID mismatch")
            ancestry_ids, cursor = [], host
            while cursor is not None:
                ancestry_ids.append(cursor)
                descendants[cursor].add(i)
                cursor = parents[cursor]
            ancestry_ids.reverse()
            ancestry = [events[n]["name"] for n in ancestry_ids]
            row["host_operator"] = events[host]["name"]
            row["host_ancestry"] = ancestry
            row["host_start_us"] = str(events[host]["ts"])
            row["host_duration_us"] = events[host]["dur"]
            # Do not guess a request phase merely from its time. Sampling outside
            # execute_model remains explicitly outside annotated model scopes.
            scope = next((n for n in ancestry if n.startswith("P09/step=")), None)
            row["phase"] = scope.split("/")[1] if scope else "outside_annotated_model"
            direct[host].add(i)
            selected.extend(ancestry_ids)
        row["route"] = route(task, row["host_ancestry"])
        if task["name"] not in CONTROL | {"MEMCPY_ASYNC", "EVENT_RECORD"}:
            key = (task["name"], str(args["Task Id"]), str(args["Physic Stream Id"]), number(task["ts"]))
            match = only(kernel_rows[key], "kernel CSV match")
            require(abs(number(match["Duration(us)"]) - number(task["dur"])) <= Decimal("0.001"),
                    "kernel duration mismatch")
            require(key not in used_csv, "duplicate device task")
            used_csv.add(key)
            row["kernel_csv_verified"] = True
        rows.append(row)
        # One representative of EVERY kernel name, including copies and controls.
        if task["name"] not in examples:
            examples[task["name"]] = {"task": row,
                                      "trace_events": [events[n] for n in dict.fromkeys(selected)]}
    require(len(used_csv) == len(kernels), "unmatched kernel CSV rows")
    require(sum(e["kernel"] in CONTROL for e in rows) == 2, "expected profiler enable/disable")
    host_names = defaultdict(list)
    for i in cpu_ids:
        host_names[events[i]["name"]].append(i)
    host_inventory = []
    for name, ids in sorted(host_names.items()):
        host_inventory.append({"name": name, "calls": len(ids),
            "calls_with_direct_flow": sum(bool(direct[i]) for i in ids),
            "calls_with_descendant_flow": sum(bool(descendants[i]) for i in ids),
            "calls_without_correlated_device_task": sum(not descendants[i] for i in ids),
            "unique_descendant_device_tasks": len(set().union(*(descendants[i] for i in ids)))})
    kernel_inventory = []
    for name in sorted(examples):
        group = [e for e in rows if e["kernel"] == name]
        kernel_inventory.append({"kernel": name, "calls": len(group),
            "routes": sorted({e["route"] for e in group}),
            "host_operators": sorted({e["host_operator"] for e in group if e["host_operator"]}),
            "task_types": sorted({e["task_type"] for e in group}),
            "kernel_csv_matches": sum(e["kernel_csv_verified"] for e in group)})
    queues = []
    queue_starts = [(key, ids) for key, ids in flows.items()
                    if key[0] == "async_task_queue" and key[2] == "s"]
    require(len(queue_starts) == sum(key[0] == "async_task_queue" and key[2] == "f" for key in flows),
            "unpaired queue flows")
    for key, ids in queue_starts:
        start = events[only(ids, "queue flow start")]
        end = events[only(flows[key[0], key[1], "f"], "queue flow end")]
        enq = events[only(complete[point(start)], "enqueue event")]
        deq = events[only(complete[point(end)], "dequeue event")]
        require(enq.get("cat") == "enqueue" and deq.get("cat") == "dequeue", "wrong queue endpoints")
        require(str(enq["args"]["correlation_id"]) == str(deq["args"]["correlation_id"]) == key[1],
                "queue correlation ID mismatch")
        queues.append({"correlation_id": key[1], "enqueue": enq, "dequeue": deq})
    summary = {"host_event_count": len(cpu_ids), "host_name_count": len(host_names),
               "device_task_count": len(rows), "kernel_name_count": len(examples),
               "correlated_device_tasks": sum(bool(e["torch_flow_id"]) for e in rows),
               "kernel_csv_rows_verified": len(used_csv),
               "cann_connection_ids_verified": sum(e["cann_connection_verified"] for e in rows),
               "queue_pairs_verified": len(queues),
               "route_counts": dict(Counter(e["route"] for e in rows)),
               "unattributed": [e["kernel"] for e in rows if not e["torch_flow_id"]]}
    return summary, rows, host_inventory, kernel_inventory, list(examples.values()), queues


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    out = run / "full_analysis"
    out.mkdir(exist_ok=True)
    manifest = json.loads((out / "source_manifest.json").read_text())
    for entry in manifest["files"]:
        require(hashlib.sha256((run / entry["snapshot"]).read_bytes()).hexdigest() == entry["sha256"],
                "source snapshot checksum mismatch: " + entry["snapshot"])
    trace_path = only(list((run / "profiler").rglob("trace_view.json")), "raw trace")
    events = read_trace(trace_path)
    kernel_path = only(list((run / "profiler").rglob("kernel_details.csv")), "kernel CSV")
    with kernel_path.open() as stream:
        kernels = list(csv.DictReader(stream))
    summary, rows, hosts, device_names, examples, queues = audit(events, kernels)
    summary["raw_trace_sha256"] = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    summary["source_manifest_sha256"] = hashlib.sha256((out / "source_manifest.json").read_bytes()).hexdigest()
    summary["analysis_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out / "coverage.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_csv(out / "device_tasks.csv", rows)
    write_csv(out / "host_operators.csv", hosts)
    write_csv(out / "kernel_inventory.csv", device_names)
    (out / "kernel_examples.json").write_text(json.dumps(examples, ensure_ascii=False, indent=2) + "\n")
    (out / "queue_pairs.json").write_text(json.dumps(queues, ensure_ascii=False, indent=2) + "\n")
    lines = ["# Complete operator audit", "", "Only the recorded Qwen2.5 workload is covered.", "",
             "- Host events: {} ({} distinct names, including diagnostic labels).".format(
                 summary["host_event_count"], summary["host_name_count"]),
             "- Device tasks: {} ({} distinct names).".format(
                 summary["device_task_count"], summary["kernel_name_count"]),
             "- Both torch and CANN flow IDs resolved: {}.".format(summary["correlated_device_tasks"]),
             "- Kernel CSV rows verified: {}.".format(summary["kernel_csv_rows_verified"]),
             "- Enqueue/dequeue pairs verified: {}.".format(summary["queue_pairs_verified"]),
             "- Unattributed: {} (profiler control, not model kernels).".format(", ".join(summary["unattributed"])), "",
             "Route counts partition device tasks; wrapper/host events are separate dimensions.", "",
             "| Route | Device tasks |", "|---|---:|"]
    lines += ["| {} | {} |".format(k, v) for k, v in sorted(summary["route_counts"].items())]
    lines += ["", "| Device kernel/task name | Count | Route |", "|---|---:|---|"]
    lines += ["| `{}` | {} | {} |".format(e["kernel"], e["calls"], ", ".join(e["routes"])) for e in device_names]
    lines += ["", "CPU events without a correlated device task are not automatically no-ops: they may",
              "change metadata, allocate memory, synchronize, or do host work. Zero device work is",
              "a statement about this trace, not every possible input or implementation.", "",
              "outside_annotated_model includes sampling and supporting operations. No unobserved Python",
              "scope or layer number is assigned to those events. Nested CPU scopes use same-lane interval",
              "containment; device attribution uses explicit flow IDs, not temporal proximity.", ""]
    (out / "summary.md").write_text("\n".join(lines))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
