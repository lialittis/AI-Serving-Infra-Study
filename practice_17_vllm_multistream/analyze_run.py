"""Reconstruct one real vLLM run's multi-stream kernel execution graph."""
import argparse
from collections import Counter, defaultdict
import csv
from decimal import Decimal
import gzip
import hashlib
import json
from pathlib import Path


RUNTIME_TASKS = {"EVENT_RECORD", "EVENT_WAIT", "MEMCPY_ASYNC"}
CONTROL_TASKS = {"PROFILING_ENABLE", "PROFILING_DISABLE"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def only(items, message):
    values = list(items)
    require(len(values) == 1, "%s: got %d" % (message, len(values)))
    return values[0]


def number(value):
    return value if isinstance(value, Decimal) else Decimal(str(value))


def end(event):
    return number(event["ts"]) + number(event.get("dur", 0))


def inside(outer, inner):
    return (outer["pid"] == inner["pid"] and outer["tid"] == inner["tid"] and
            number(outer["ts"]) <= number(inner["ts"]) and end(inner) <= end(outer))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_trace(path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as stream:
            data = json.load(stream, parse_float=Decimal)
    else:
        data = json.loads(path.read_text(), parse_float=Decimal)
    return data["traceEvents"] if isinstance(data, dict) else data


def union(intervals):
    merged = []
    for start, finish in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, finish])
        elif finish > merged[-1][1]:
            merged[-1][1] = finish
    return merged


def intersection(left, right):
    result = []
    for a, b in union(left):
        for c, d in union(right):
            start, finish = max(a, c), min(b, d)
            if start < finish:
                result.append((start, finish))
    return union(result)


def resolve_colliding_host(candidates, cann, queue_links):
    """Disambiguate timestamp-derived async_npu IDs using real task-queue flows.

    CANN uses a synthetic pid, so link by its worker tid and containment inside
    a dequeue range. The queue's real pid and enqueue flow must in turn match the
    candidate CPU operator. Never select by nearest timestamp or operator name.
    """
    matches = []
    for index, host in candidates:
        for dequeue, enqueue, correlation in queue_links:
            if (dequeue["pid"] == host["pid"] and dequeue["tid"] == cann["tid"]
                    and number(dequeue["ts"]) <= number(cann["ts"])
                    and end(cann) <= end(dequeue)
                    and enqueue["pid"] == host["pid"] and enqueue["tid"] == host["tid"]
                    and number(host["ts"]) <= number(enqueue["ts"]) <= end(host)):
                matches.append((index, host, correlation))
    return only(matches, "task-queue resolution of colliding async_npu ID")


def placeholder_task(index, task):
    """Preserve an uncorrelated runtime placeholder without inventing ownership."""
    args = task["args"]
    require(task["name"] == args["Task Type"] == "PLACE_HOLDER_SQE", "unexpected uncorrelated task")
    require(str(args["connection_id"]) == str(2**64 - 1), "placeholder connection must be sentinel")
    return {"id": "k:%d" % index, "trace_index": index, "name": task["name"],
            "task_type": args["Task Type"], "task_id": str(args["Task Id"]),
            "stream": str(args["Physic Stream Id"]), "device_lane": str(task["pid"]),
            "start_us": str(task["ts"]), "end_us": str(end(task)), "duration_us": str(task["dur"]),
            "step": None, "scope": None, "scope_kind": "unattributed_runtime",
            "host_operator": None, "host_trace_index": None, "cann_api": None,
            "cann_trace_index": None, "torch_flow": None, "cann_flow": None,
            "connection_id": str(args["connection_id"]), "is_compute": False,
            "unresolved_reason": "No host flow; sentinel connection ID. Retained in physical-stream order only."}


def analyze(run, events=None, records=None):
    command = json.loads((run / "command.json").read_text())
    mode = command["mode"]
    enabled = command["enable_async_exponential"]
    require(enabled == (mode == "enabled"), "mode/config mismatch")
    require("--enforce-eager" in command["argv"], "expected eager mode")
    require("--no-async-scheduling" in command["argv"], "async scheduler confounds stream trigger")
    require(json.loads((run / "shutdown.json").read_text())["server_exit_code"] == 0, "unclean shutdown")
    controls = json.loads((run / "profile_control.json").read_text())
    require([(x["endpoint"], x["status"]) for x in controls] ==
            [("/start_profile", 200), ("/stop_profile", 200)], "profiler controls")
    response = json.loads((run / "response.json").read_text())
    request = json.loads((run / "request.json").read_text())
    prompt_info = json.loads((run / "prompt_info.json").read_text())
    batch_size = prompt_info["batch_size"]
    require(len(response["choices"]) == batch_size, "response choice count mismatch")
    require(response["usage"]["completion_tokens"] == request["max_tokens"] * batch_size,
            "completion token mismatch")
    require(request["temperature"] > 0 and request["top_p"] < 1, "request must use random sampling")
    for name, expected in json.loads((run / "instrumentation_hashes.json").read_text()).items():
        require(digest(run / name) == expected, "instrumentation mismatch " + name)
    manifest = json.loads((run / "source_manifest.json").read_text())
    for name, metadata in manifest.items():
        if isinstance(metadata, dict):
            require(digest(run / name) == metadata["sha256"], "source mismatch " + name)
    if records is None:
        records = [json.loads(line) for path in (run / "events").glob("*.jsonl")
                   for line in path.read_text().splitlines()]
    require(not any(r["event"] == "trace_error" for r in records), "instrumentation errors")
    entries = {r["label"]: r for r in records if r["event"] == "enter"}
    exits = {r["label"]: r for r in records if r["event"] == "exit"}
    require(set(entries) == set(exits), "unbalanced scopes")
    runners = sorted((r for r in entries.values() if r["kind"] == "runner"), key=lambda r: r["step"])
    step_ids = [r["step"] for r in runners]
    require(step_ids == list(range(1, len(runners) + 1)), "non-contiguous inference steps")
    require(len(runners) >= request["max_tokens"], "too few inference steps")
    sampling_runners = sorted((r for r in entries.values() if r["kind"] == "sampling_runner"),
                              key=lambda r: r["step"])
    require([r["step"] for r in sampling_runners] == step_ids, "sampling runner step mismatch")
    expected_branch = "async_exponential" if enabled else "inline_exponential"
    require(Counter(r["kind"] for r in entries.values())[expected_branch] == len(runners),
            "missing random branch scopes")
    trace_path = only(list((run / "profiler").rglob("trace_view.json")) +
                      list((run / "profiler").rglob("trace_view.json.gz")), "trace")
    events = events if events is not None else read_trace(trace_path)
    csv_path = only((run / "profiler").rglob("kernel_details.csv"), "kernel CSV")
    with csv_path.open() as stream:
        kernels = list(csv.DictReader(stream))
    kernel_index = defaultdict(list)
    for row_index, row in enumerate(kernels):
        kernel_index[row["Name"], row["Stream ID"].strip(), row["Task ID"].strip(),
                     number(row["Start Time(us)"])].append(row_index)
    scopes = {e["name"]: e for e in events if e.get("ph") == "X" and e.get("name", "").startswith("P17/")}
    require(set(scopes) == set(entries), "event/profiler scope mismatch")
    complete, starts, finishes = defaultdict(list), defaultdict(list), defaultdict(list)
    def point(event):
        return event["pid"], event["tid"], number(event["ts"])
    for index, event in enumerate(events):
        if event.get("ph") == "X":
            complete[point(event)].append((index, event))
        elif event.get("ph") == "s":
            starts[event.get("cat"), str(event["id"])].append(event)
        elif event.get("ph") == "f":
            finishes[(event.get("cat"),) + point(event)].append(event)
    queue_links = None
    resolutions = {}
    def source(task, category, cann=None):
        nonlocal queue_links
        finish = only(finishes[(category,) + point(task)], category + " endpoint")
        candidates = starts[category, str(finish["id"])]
        if len(candidates) > 1 and category == "async_npu" and cann is not None:
            if queue_links is None:
                queue_links = []
                for event in events:
                    if event.get("ph") != "X" or event.get("cat") != "dequeue":
                        continue
                    for queue_end in finishes[("async_task_queue",) + point(event)]:
                        correlation = str(queue_end["id"])
                        require(str(event["args"]["correlation_id"]) == correlation,
                                "dequeue correlation mismatch")
                        enqueue = only(starts["async_task_queue", correlation], "enqueue flow")
                        queue_links.append((event, enqueue, correlation))
            host_candidates = [only(complete[point(s)], category + " source") for s in candidates]
            index, event, queue_flow = resolve_colliding_host(host_candidates, cann, queue_links)
            resolutions[point(task)] = queue_flow
            return index, event, str(finish["id"])
        start = only(candidates, category + " start")
        index, event = only(complete[point(start)], category + " source")
        return index, event, str(finish["id"])
    used_csv = set()
    tasks = []
    for index, task in enumerate(events):
        args = task.get("args", {})
        if task.get("ph") != "X" or "Task Type" not in args or task["name"] in CONTROL_TASKS:
            continue
        if task["name"] == "PLACE_HOLDER_SQE":
            require(not finishes[("HostToDevice",) + point(task)]
                    and not finishes[("async_npu",) + point(task)], "unexpected correlated placeholder")
            tasks.append(placeholder_task(index, task))
            continue
        ci, cann, cann_flow = source(task, "HostToDevice")
        require(cann["args"]["connection_id"] == args["connection_id"], "connection mismatch")
        hi, host, torch_flow = source(task, "async_npu", cann)
        containing = [(label, scope) for label, scope in scopes.items() if inside(scope, host)]
        containing.sort(key=lambda pair: number(pair[1]["dur"]))
        label = containing[0][0] if containing else None
        observation = entries[label] if label is not None else None
        item = {
            "id": "k:%d" % index, "trace_index": index, "name": task["name"],
            "task_type": args["Task Type"], "task_id": str(args["Task Id"]),
            "stream": str(args["Physic Stream Id"]), "device_lane": str(task["pid"]),
            "start_us": str(task["ts"]), "end_us": str(end(task)),
            "duration_us": str(task["dur"]), "step": observation["step"] if observation else None,
            "scope": label, "scope_kind": observation["kind"] if observation else "outside_observed_runner",
            "host_operator": host["name"], "host_trace_index": hi,
            "cann_api": cann["name"], "cann_trace_index": ci,
            "torch_flow": torch_flow, "cann_flow": cann_flow,
            "connection_id": str(args["connection_id"]),
            "is_compute": task["name"] not in RUNTIME_TASKS,
        }
        if point(task) in resolutions:
            item["async_task_queue_flow"] = resolutions[point(task)]
            item["torch_flow_resolution"] = "CANN connection -> dequeue range -> enqueue flow -> CPU operator"
        if item["is_compute"]:
            matches = [row_index for row_index in kernel_index[
                       item["name"], item["stream"], item["task_id"], number(task["ts"])]
                       if abs(number(kernels[row_index]["Duration(us)"]) - number(task["dur"])) <= Decimal(".001")]
            match = only(matches, "kernel CSV identity")
            require(match not in used_csv, "duplicate CSV match")
            used_csv.add(match)
            item["kernel_csv_row"] = match
        tasks.append(item)
    require(len(used_csv) == len(kernels), "unmatched kernel CSV rows")
    steps = []
    graph_edges = []
    for step in step_ids:
        current = [t for t in tasks if t["step"] == step]
        branch = [t for t in current if t["scope_kind"] == expected_branch]
        forward = [t for t in current if t["scope_kind"] == "forward"]
        sampling = [t for t in current if t["scope_kind"] == "sampler"]
        all_branch_compute = [t for t in branch if t["is_compute"]]
        forward_compute = [t for t in forward if t["is_compute"]]
        sampling_compute = [t for t in sampling if t["is_compute"]]
        if enabled:
            branch_compute = all_branch_compute
        else:
            event_record = only((t for t in branch if t["name"] == "EVENT_RECORD"),
                                "inline event record")
            event_wait = only((t for t in branch if t["name"] == "EVENT_WAIT"),
                              "inline event wait")
            branch_compute = [t for t in all_branch_compute if t["stream"] == event_record["stream"]]
            sampling_compute += [t for t in all_branch_compute if t["stream"] == event_wait["stream"]]
        require(branch_compute and forward_compute and sampling_compute, "missing step compute tasks")
        branch_streams = sorted({t["stream"] for t in branch_compute})
        forward_streams = sorted({t["stream"] for t in forward_compute})
        require(len(branch_streams) == len(forward_streams) == 1, "ambiguous branch stream")
        require(branch_streams != forward_streams, "random and model compute share a stream")
        overlaps = intersection([(number(t["start_us"]), number(t["end_us"])) for t in branch_compute],
                                [(number(t["start_us"]), number(t["end_us"])) for t in forward_compute])
        overlap_us = sum(finish - start for start, finish in overlaps)
        branch_start = min(number(t["start_us"]) for t in branch_compute)
        branch_end = max(number(t["end_us"]) for t in branch_compute)
        forward_start = min(number(t["start_us"]) for t in forward_compute)
        forward_end = max(number(t["end_us"]) for t in forward_compute)
        if enabled:
            records_in_branch = sorted((t for t in branch if t["name"] == "EVENT_RECORD"),
                                       key=lambda t: number(t["start_us"]))
            waits_in_branch = [t for t in branch if t["name"] == "EVENT_WAIT"]
            require(len(records_in_branch) == 2 and len(waits_in_branch) == 0,
                    "enabled async scope event shape")
            wait_scope = scopes[only((r["label"] for r in entries.values()
                                      if r["step"] == step and r["kind"] == "sampler"), "sampler scope")]
            native_waits = [e for e in events if e.get("name") == "AscendCL@aclrtSynchronizeEvent"
                            and e.get("args", {}).get("Thread Id", e.get("tid")) == wait_scope["tid"]
                            and number(wait_scope["ts"]) <= number(e["ts"]) and end(e) <= end(wait_scope)]
            native_wait = only(native_waits, "async exponential CPU wait")
            require(number(records_in_branch[-1]["end_us"]) <= end(native_wait), "CPU wait before event record")
            wait_id = "cpu-wait:%d" % step
            graph_edges.append({"source": records_in_branch[-1]["id"], "target": wait_id,
                                "kind": "event_sync", "step": step,
                                "wait_end_us": str(end(native_wait)),
                                "semantics": "sampler CPU waits for precomputed random tensor"})
            first_sample = min(sampling_compute, key=lambda t: number(t["start_us"]))
            graph_edges.append({"source": wait_id, "target": first_sample["id"],
                                "kind": "host_after_wait", "step": step,
                                "semantics": "sampling compute is submitted after the host wait returns"})
            producer = max(branch_compute, key=lambda t: number(t["end_us"]))
            graph_edges.append({"source": producer["id"], "target": first_sample["id"],
                                "kind": "data_contract", "step": step, "resource": "precomputed_q",
                                "semantics": "async exponential produces q consumed by random sampling"})
        else:
            event_records = [t for t in branch if t["name"] == "EVENT_RECORD"]
            event_waits = [t for t in branch if t["name"] == "EVENT_WAIT"]
            require(len(event_records) == len(event_waits) == 1, "inline scope event shape")
            require(number(event_records[0]["end_us"]) <= number(event_waits[0]["end_us"]),
                    "inline wait completes before record")
            graph_edges.append({"source": event_records[0]["id"], "target": event_waits[0]["id"],
                                "kind": "event_wait", "step": step,
                                "semantics": "default stream waits for inline random generation"})
            producer = max(branch_compute, key=lambda t: number(t["end_us"]))
            first_sample = min(sampling_compute, key=lambda t: number(t["start_us"]))
            graph_edges.append({"source": producer["id"], "target": first_sample["id"],
                                "kind": "data_contract", "step": step, "resource": "q",
                                "semantics": "inline exponential produces q consumed by random sampling"})
        runner = only((r for r in runners if r["step"] == step), "runner observation")
        request_count = len(runner["request_ids"])
        steps.append({
            "step": step,
            "phase": "decode" if runner["total_tokens"] == request_count else "prefill-or-mixed",
            "request_count": request_count, "scheduled_tokens": runner["total_tokens"],
            "branch_kind": expected_branch, "branch_stream": branch_streams[0],
            "forward_stream": forward_streams[0], "branch_compute_tasks": len(branch_compute),
            "forward_compute_tasks": len(forward_compute), "sampling_compute_tasks": len(sampling_compute),
            "branch_start_us": str(branch_start), "branch_end_us": str(branch_end),
            "forward_start_us": str(forward_start), "forward_end_us": str(forward_end),
            "branch_to_forward_gap_us": str(forward_start - branch_end),
            "forward_to_branch_gap_us": str(branch_start - forward_end),
            "overlap_us": str(overlap_us),
            "overlap_intervals": [{"start_us": str(a), "end_us": str(b), "duration_us": str(b-a)}
                                  for a, b in overlaps],
        })
    relevant_tasks = [t for t in tasks if t["step"] is not None
                      or t["scope_kind"] == "unattributed_runtime"]
    lanes = defaultdict(list)
    for task in relevant_tasks:
        lanes[task["device_lane"], task["stream"]].append(task)
    for (device_lane, stream), lane in lanes.items():
        lane.sort(key=lambda t: number(t["start_us"]))
        require(all(number(a["end_us"]) <= number(b["start_us"]) for a, b in zip(lane, lane[1:])),
                "overlap within physical stream")
        for a, b in zip(lane, lane[1:]):
            graph_edges.append({"source": a["id"], "target": b["id"], "kind": "stream_order",
                                "stream": stream, "device_lane": device_lane,
                                "semantics": "adjacent observed tasks on one physical stream"})
    nodes = [{"id": t["id"], "kind": "device_task", **t} for t in relevant_tasks]
    if enabled:
        for step in step_ids:
            edge = only((e for e in graph_edges if e["kind"] == "event_sync" and e["step"] == step), "wait edge")
            nodes.append({"id": edge["target"], "kind": "cpu_wait_return", "step": step,
                          "name": "aclrtSynchronizeEvent returned", "timestamp_us": edge["wait_end_us"]})
        for step in step_ids:
            async_exit = only((r for r in exits.values() if r["step"] == step and r["kind"] == "async_exponential"),
                              "async result metadata")
            sample_entry = only((r for r in entries.values() if r["step"] == step and r["kind"] == "sampler"),
                                "sample input metadata")
            require(async_exit["q"] == sample_entry["precomputed_q"], "precomputed q identity mismatch")
            require(async_exit["event_handle"] == sample_entry["async_event_handle"], "async event identity mismatch")
    summary = {
        "mode": mode, "model": Path(command["argv"][command["argv"].index("serve") + 1]).name,
        "batch_size": batch_size,
        "steps": len(steps), "device_tasks": len(relevant_tasks),
        "compute_tasks": sum(t["is_compute"] for t in relevant_tasks),
        "outside_scope_device_tasks": sum(t["step"] is None for t in tasks),
        "physical_streams": sorted({t["stream"] for t in relevant_tasks}),
        "overlap_steps": sum(number(s["overlap_us"]) > 0 for s in steps),
        "total_model_random_overlap_us": str(sum(number(s["overlap_us"]) for s in steps)),
        "kernel_csv_rows_verified": len(used_csv),
        "graph_nodes": len(nodes), "graph_edges": len(graph_edges),
        "edges_by_kind": dict(Counter(e["kind"] for e in graph_edges)),
    }
    if resolutions:
        summary["disambiguated_async_npu_tasks"] = len(resolutions)
    unattributed = [t for t in relevant_tasks if t["scope_kind"] == "unattributed_runtime"]
    if unattributed:
        summary["unattributed_runtime_tasks"] = len(unattributed)
        summary["unattributed_runtime_duration_us"] = str(sum(number(t["duration_us"]) for t in unattributed))
    return {"summary": summary, "steps": steps, "nodes": nodes, "edges": graph_edges,
            "tasks": tasks, "observations": list(entries.values()), "returns": list(exits.values()),
            "provenance": {"trace_sha256": digest(trace_path), "kernel_csv_sha256": digest(csv_path),
                           "analyzer_sha256": digest(Path(__file__))},
            "limits": ["One batched HTTP request, one card, eager mode; not a throughput benchmark.",
                       "Overlap is device-task interval intersection, not instruction-level occupancy.",
                       "Data edge covers the precomputed random tensor; full model dataflow remains in Practice 15.",
                       "Profiler overhead can change durations but not the observed stream/task identities."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    result = analyze(args.run.resolve())
    output = args.run.resolve() / "analysis"
    output.mkdir(exist_ok=True)
    (output / "execution_graph.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (output / "summary.json").write_text(json.dumps({"summary": result["summary"], "steps": result["steps"]},
                                                       ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
