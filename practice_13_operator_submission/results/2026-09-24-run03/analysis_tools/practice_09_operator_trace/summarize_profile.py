"""Validate and join real Python scopes, torch-to-NPU flows and CANN flows.

Only standard library; works locally without torch or an NPU. Fail closed when
this exercise's captured schema or expected execution path changes.
"""
import argparse
from collections import Counter, defaultdict
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value):
    # Trace timestamps are epoch microseconds. Float loses sub-microsecond bits.
    return Decimal(str(value).strip())


def inside(outer, inner):
    return (outer["pid"] == inner["pid"] and outer["tid"] == inner["tid"]
            and number(outer["ts"]) <= number(inner["ts"])
            and number(inner["ts"]) + number(inner.get("dur", 0))
            <= number(outer["ts"]) + number(outer["dur"]))


def only(items, context):
    require(len(items) == 1, "{}: expected one event, found {}".format(context, len(items)))
    return items[0]


def read_trace(path):
    obj = json.loads(path.read_text())
    return obj if isinstance(obj, list) else obj["traceEvents"]


def analyze(run, trace=None):
    trace_path = only(list((run / "profiler").rglob("trace_view.json")), "trace file")
    events = read_trace(trace_path) if trace is None else trace
    records = [json.loads(line) for p in (run / "events").glob("*.jsonl")
               for line in p.read_text().splitlines()]
    require(not any(e["event"] == "trace_error" for e in records), "annotation error")
    response = json.loads((run / "response.json").read_text())
    require(response["usage"]["prompt_tokens"] == 126 and
            response["usage"]["completion_tokens"] == 2, "unexpected response token counts")
    require(response["choices"][0]["finish_reason"] == "length", "unexpected finish reason")
    require(json.loads((run / "shutdown.json").read_text())["server_exit_code"] == 0,
            "service did not exit cleanly")
    controls = json.loads((run / "profile_control.json").read_text())
    require([(e["endpoint"], e["status"]) for e in controls] ==
            [("/start_profile", 200), ("/stop_profile", 200)], "profiler control failed")
    enters = {e["label"]: e for e in records if e["event"] == "scope_enter"}
    exits = Counter(e["label"] for e in records if e["event"] == "scope_exit")
    scopes = {e["name"]: e for e in events if e.get("name", "").startswith("P09/")}
    require(len(enters) == len(scopes) == 14 and set(enters) == set(scopes),
            "expected 14 matching Python/profiler scopes")
    require(exits == Counter(enters.keys()), "unbalanced Python scopes")
    rid = only(list({rid for e in enters.values() for rid in e["request_ids"]}), "request ID")
    require(rid.startswith(response["id"] + "-"), "scope/HTTP request mismatch")
    schedule = sorted([e for e in records if e["event"] == "schedule"
                       and rid in e["scheduled_tokens"]], key=lambda e: e["step"])
    require([e["scheduled_tokens"][rid] for e in schedule] == [126, 1], "expected prefill/decode")
    step_phases = {e["step"]: phase for e, phase in zip(schedule, ("prefill", "decode"))}
    finished = only([e for e in records if e["event"] == "request_finish"
                     and e["request_id"] == rid], "finished request")
    require(finished["computed_tokens"] == 127 and finished["output_tokens"] == 2,
            "unexpected final request state")

    xs = [e for e in events if e.get("ph") == "X"]
    cpu = [e for e in xs if e.get("cat") == "cpu_op"]
    device = [e for e in xs if "Task Type" in e.get("args", {})]
    require(device, "missing device events")
    flows = defaultdict(list)
    for e in events:
        if e.get("ph") in ("s", "f"):
            flows[(e.get("cat"), str(e["id"]), e["ph"])].append(e)
    kernel_path = only(list((run / "profiler").rglob("kernel_details.csv")), "kernel CSV")
    with kernel_path.open() as stream:
        kernels = list(csv.DictReader(stream))

    examples, selected = [], []
    for step, phase in step_phases.items():
        prefix = "P09/step={}/".format(step)
        attention = enters[prefix + "AscendAttentionBackendImpl.forward"]
        count = 126 if phase == "prefill" else 1
        require(attention["tensors"]["query"]["shape"] == [count, 14, 64], "wrong Q shape")
        require(attention["tensors"]["key"]["shape"] == [count, 2, 64], "wrong K shape")
        for function, torch_name, kernel_name in (
            ("BaseDeviceAdaptor.reshape_and_cache", "atb::_npu_reshape_and_cache", "ReshapeAndCacheNdKernel"),
            ("AscendAttentionBackendImpl.forward_fused_infer_attention",
             "npu::npu_fused_infer_attention_score", "FusedInferAttentionScore"),
        ):
            scope = scopes[prefix + function]
            op = only([e for e in cpu if e["name"] == torch_name and inside(scope, e)], torch_name)
            # Start/end IDs are the profiler's correlation, not nearest timestamps.
            starts = [e for e in events if e.get("cat") == "async_npu"
                      and e.get("ph") == "s" and inside(op, e)]
            require(starts, "missing torch-to-NPU flow")
            links = []
            for start in starts:
                end = only(flows[("async_npu", str(start["id"]), "f")], "torch flow end")
                task = only([e for e in device if e["pid"] == end["pid"]
                             and e["tid"] == end["tid"] and number(e["ts"]) == number(end["ts"])],
                            "flow device endpoint")
                launch_op = only([e for e in cpu if e["pid"] == start["pid"]
                                  and e["tid"] == start["tid"]
                                  and number(e["ts"]) == number(start["ts"])], "flow host endpoint")
                links.append((start, end, task, launch_op))
                selected.extend([start, end, task, launch_op])
            start, end, task, launch_op = only([link for link in links
                                               if link[2]["name"] == kernel_name], kernel_name)
            cann_end = only([e for e in events if e.get("cat") == "HostToDevice"
                             and e.get("ph") == "f" and e["pid"] == task["pid"]
                             and e["tid"] == task["tid"] and number(e["ts"]) == number(task["ts"])],
                            "CANN device endpoint")
            cann_start = only(flows[("HostToDevice", str(cann_end["id"]), "s")], "CANN flow start")
            cann_scopes = [e for e in xs if inside(e, cann_start) and e["pid"] == cann_start["pid"]]
            require(cann_scopes, "missing CANN launch scope")
            require(any(e.get("args", {}).get("connection_id") == task["args"].get("connection_id")
                        for e in cann_scopes), "CANN/device connection ID mismatch")
            row = only([r for r in kernels if r["Name"] == kernel_name
                        and r["Task ID"] == str(task["args"]["Task Id"])
                        and r["Stream ID"] == str(task["args"]["Physic Stream Id"])
                        and number(r["Start Time(us)"]) == number(task["ts"])], "kernel CSV match")
            require(abs(number(row["Duration(us)"]) - number(task["dur"])) <= Decimal("0.001"),
                    "kernel duration mismatch")
            selected.extend([scope, op, cann_start, cann_end] + cann_scopes)
            selected.extend(e for e in cpu if inside(scope, e))
            examples.append({
                "phase": phase, "step": step, "request_id": rid,
                "python_scope": scope, "python_source": enters[scope["name"]]["source"],
                "torch_operator": op, "host_launch_operator": launch_op,
                "torch_flow_id": start["id"], "cann_flow_id": cann_start["id"],
                "cann_launch_scopes": cann_scopes, "device_kernel": task, "kernel_csv": row,
                "associated_tasks": [link[2] for link in links],
                "kernel_start_after_torch_return_us": str(number(task["ts"]) - number(op["ts"]) - number(op["dur"])),
                "kernel_start_after_python_return_us": str(number(task["ts"]) - number(scope["ts"]) - number(scope["dur"])),
            })
    counts = Counter(e["name"] for e in device)
    require(counts["ReshapeAndCacheNdKernel"] == counts["FusedInferAttentionScore"] == 48,
            "expected 24 layers x 2 forwards; collection may be incomplete")
    # Keep lane metadata and genuine events; no timestamp rewriting or inferred flows.
    selected.extend(e for e in events if e.get("ph") == "M")
    selected.extend(scopes.values())
    unique = {json.dumps(e, sort_keys=True): e for e in selected}
    result = {"request_id": rid, "profiled_steps": list(step_phases),
              "cpu_events": len(cpu), "device_tasks": len(device), "kernel_csv_rows": len(kernels),
              "device_task_counts": dict(counts.most_common()), "examples": examples,
              "trace_file": str(trace_path.relative_to(run)),
              "trace_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest()}
    return result, list(unique.values())


def write_results(run, result, excerpt):
    (run / "operator_links.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (run / "first_layer_trace.json").write_text(json.dumps(excerpt, ensure_ascii=False) + "\n")
    lines = ["# Real operator trace validation", "", "Request: `{}`".format(result["request_id"]), "",
             "126 input tokens; 2 output tokens; 127 computed tokens. Warmup excluded from profiler.", "",
             "{} host cpu_op events; {} device tasks; {} kernel CSV rows.".format(
                 result["cpu_events"], result["device_tasks"], result["kernel_csv_rows"]), "",
             "Device task count includes copies/events; it is not the kernel CSV row count.", "",
             "| Phase | PyTorch operator | Device kernel | Task / stream | Device duration (us) | Kernel start minus Python scope end (us) |",
             "|---|---|---|---|---|---|"]
    for e in result["examples"]:
        task = e["device_kernel"]
        lines.append("| {} | `{}` | `{}` | {} / {} | {} | {} |".format(
            e["phase"], e["torch_operator"]["name"], task["name"], task["args"]["Task Id"],
            task["args"]["Physic Stream Id"], task["dur"], e["kernel_start_after_python_return_us"]))
    lines += ["", "Positive last column means device execution started after the annotated Python function returned.",
              "Negative means the device started before that host scope ended; it does not establish completion.", "",
              "Each example follows an async_npu flow ID and a HostToDevice flow ID, and matches the kernel CSV",
              "by task ID, stream ID, exact start timestamp, name and rounded duration.", "",
              "Open first_layer_trace.json in a compatible trace viewer; operator_links.json contains exact raw evidence.",
              "These instrumented durations are diagnostic observations, not benchmark numbers.", ""]
    (run / "summary.md").write_text("\n".join(lines))
    with (run / "operator_links.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["phase", "python_function", "torch_operator", "host_launch_operator", "kernel",
                         "task_id", "stream_id", "kernel_duration_us", "torch_flow_id", "cann_flow_id"])
        for e in result["examples"]:
            task = e["device_kernel"]
            writer.writerow([e["phase"], e["python_source"]["function"], e["torch_operator"]["name"],
                             e["host_launch_operator"]["name"], task["name"], task["args"]["Task Id"],
                             task["args"]["Physic Stream Id"], task["dur"], e["torch_flow_id"], e["cann_flow_id"]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    result, excerpt = analyze(args.run)
    write_results(args.run, result, excerpt)
    print("Verified: 2 forwards, 14 scopes, 4 first-layer operator/kernel chains")


if __name__ == "__main__":
    main()
