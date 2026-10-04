"""Exact flow/CSV joins and request-set attribution; no nearest-time ownership."""
import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import csv
from decimal import Decimal
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "practice_17_vllm_multistream"))
from analyze_run import number, end, only, union, resolve_colliding_host
sys.path.pop(0)


def read(path):
    return json.loads(path.read_text())


def compute_core(core, is_copy=False):
    # DSARandomUniform is an accelerator compute task in kernel_details.csv;
    # its DSA_SQE core label is distinct from both AI cores and SDMA copies.
    return (core.startswith(("AI_", "MIX_")) or core == "DSA_SQE") and not is_copy


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n")


def length(intervals):
    return sum((b - a for a, b in union(intervals)), Decimal(0))


def intersect_length(left, right):
    a, b = union(left), union(right)
    i = j = 0
    result = Decimal(0)
    while i < len(a) and j < len(b):
        result += max(Decimal(0), min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return result


def overlap(tasks):
    """Half-open intervals; union of time with >=2 distinct compute streams."""
    points = defaultdict(list)
    by_id = {}
    for t in tasks:
        if not t["is_compute"] or number(t["end_us"]) <= number(t["start_us"]):
            continue
        by_id[t["id"]] = t
        points[number(t["start_us"])].append((1, t["id"]))
        points[number(t["end_us"])].append((-1, t["id"]))
    active = set()
    last = None
    total = Decimal(0)
    busy = Decimal(0)
    peak = 0
    evidence = []
    for now, changes in sorted(points.items()):
        streams = {by_id[i]["stream"] for i in active}
        if last is not None and active:
            busy += now - last
        if last is not None and len(streams) >= 2:
            total += now - last
            evidence.append(dict(start_us=str(last), end_us=str(now), tasks=sorted(active), streams=sorted(streams)))
        for delta, ident in sorted(changes):
            if delta < 0:
                active.remove(ident)
            else:
                active.add(ident)
        peak = max(peak, len({by_id[i]["stream"] for i in active}))
        last = now
    return dict(compute_union_us=str(busy), compute_overlap_us=str(total),
                overlap_fraction=float(total / busy) if busy else None,
                peak_compute_streams=peak, evidence=evidence)


def request_metrics(records):
    groups = defaultdict(list)
    for r in records:
        groups[r["repetition"]].append(r)
    trials = []
    for rep, rs in sorted(groups.items()):
        points = [(r["start_mono_ns"], 1) for r in rs] + [(r["end_mono_ns"], -1) for r in rs]
        active = peak = 0
        for _, delta in sorted(points):
            active += delta
            peak = max(peak, active)
        span = (max(r["end_mono_ns"] for r in rs) - min(r["start_mono_ns"] for r in rs)) / 1e9
        trials.append(dict(repetition=rep, requests=len(rs), peak_http_inflight=peak,
                           generated_tokens=sum(len(r["token_ids"]) for r in rs), window_s=span,
                           output_tokens_per_s=sum(len(r["token_ids"]) for r in rs) / span))
    def summary(key):
        values = sorted(r[key] for r in records if key in r)
        return dict(n=len(values), median=statistics.median(values), min=min(values), max=max(values)) if values else None
    return dict(requests=len(records), failed=sum(r["status"] != "passed" for r in records), trials=trials,
                ttft_ms=summary("ttft_ms"), latency_ms=summary("latency_ms"))


def attach_requests(requests, records):
    front = defaultdict(list)
    engines = defaultdict(list)
    admits = defaultdict(list)
    for r in records:
        if r["kind"] == "frontend_map":
            front[r["client_request_id"]].append(r)
        elif r["kind"] == "engine_map":
            engines[r["external_id"]].append(r)
        elif r["kind"] == "admitted":
            admits[r["internal_id"]].append(r)
    mapping = {}
    issues = []
    for r in requests:
        try:
            f = only(front[r["client_request_id"]], "frontend map")
            if r["response_ids"] != [f["response_id"]]:
                raise ValueError("response ID mismatch")
            e = only(engines[f["external_id"]], "engine map")
            if e["internal_id"] in mapping:
                raise ValueError("internal request reused")
            mapping[e["internal_id"]] = r["client_request_id"]
            r.update(internal_id=e["internal_id"], external_id=f["external_id"])
            admitted = only(admits[e["internal_id"]], "scheduler admission")
            r["admitted_ns"] = admitted["wall_ns"]
            receives = [x for x in records if x["kind"] == "received" and x["client_request_id"] == r["client_request_id"]]
            r["received_ns"] = only(receives, "service reception")["wall_ns"]
        except ValueError as exc:
            issues.append(dict(request=r["client_request_id"], error=str(exc)))
    return mapping, issues


def analyze_case(case):
    command = read(case / "command.json")
    requests = read(case / "requests.json")
    result = dict(schema=1, case=case.name, phase=command["phase"], concurrency=command["concurrency"],
                  requests=requests, metrics=request_metrics(requests), validation=read(case / "status.json"))
    if command.get("sampling", {}).get("temperature", 0) > 0:
        result.update(sampling=command["sampling"], precompute=command["precompute"])
    if command["phase"] == "benchmark":
        return result
    records = [json.loads(line) for p in sorted((case / "observer").glob("*.jsonl")) for line in p.read_text().splitlines()]
    mapping, issues = attach_requests(requests, records)
    schedules = {r["key"]: r for r in records if r["kind"] == "schedule"}
    executions = [r for r in records if r["kind"] == "execute"]
    scope_records = {r["label"]: r for r in records if r["kind"] == "scope"}
    trace = only((case / "profiler").rglob("trace_view.json"), "one trace")
    csv_path = only((case / "profiler").rglob("kernel_details.csv"), "one kernel CSV")
    events = json.loads(trace.read_text(), parse_float=Decimal)
    if isinstance(events, dict):
        events = events["traceEvents"]
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    scopes = {e["name"]: e for e in events if e.get("ph") == "X" and e.get("name") in scope_records}
    by_thread = defaultdict(list)
    for label, s in scopes.items():
        by_thread[s["tid"]].append((label, s))
    for tid in by_thread:
        by_thread[tid].sort(key=lambda x: number(x[1]["ts"]))
    times = {tid: [number(s["ts"]) for _, s in ss] for tid, ss in by_thread.items()}
    def containing(e, native=False):
        tid = e.get("args", {}).get("Thread Id", e["tid"]) if native else e["tid"]
        if tid not in times:
            return None
        ss = by_thread[tid]
        for i in range(bisect_right(times[tid], number(e["ts"])) - 1, -1, -1):
            label, s = ss[i]
            if end(e) <= end(s) and (native or e["pid"] == s["pid"]):
                return label
        return None
    points, starts, finishes = defaultdict(list), defaultdict(list), defaultdict(list)
    def point(e):
        return e["pid"], e["tid"], number(e["ts"])
    for i, e in enumerate(events):
        if e.get("ph") == "X":
            points[point(e)].append((i, e))
        elif e.get("ph") == "s":
            starts[e.get("cat"), str(e["id"])].append(e)
        elif e.get("ph") == "f":
            finishes[(e.get("cat"),) + point(e)].append(e)
    queues = []
    for e in events:
        if e.get("ph") == "X" and e.get("cat") == "dequeue":
            for f in finishes[("async_task_queue",) + point(e)]:
                candidates = starts["async_task_queue", str(f["id"])]
                if len(candidates) == 1:
                    queues.append((e, candidates[0], str(f["id"])))
    qi = {}
    for tid in {q[0]["tid"] for q in queues}:
        qs = sorted((q for q in queues if q[0]["tid"] == tid), key=lambda q: number(q[0]["ts"]))
        qi[tid] = ([number(q[0]["ts"]) for q in qs], qs, max(number(q[0]["dur"]) for q in qs))
    flow_issues = []
    def source(e, category, cann=None):
        fs = finishes[(category,) + point(e)]
        if not fs:
            return None
        try:
            f = only(fs, "flow finish")
            ss = starts[category, str(f["id"])]
            if category == "async_npu" and len(ss) > 1 and cann:
                ticks, qs, longest = qi.get(cann["tid"], ([], [], 0))
                t = number(cann["ts"])
                candidates = qs[bisect_left(ticks, t-longest):bisect_right(ticks, t)]
                i, h, _ = resolve_colliding_host([only(points[point(s)], "host source") for s in ss], cann, candidates)
            else:
                i, h = only(points[point(only(ss, "flow source"))], "source range")
            return i, h, str(f["id"])
        except ValueError as exc:
            flow_issues.append(dict(category=category, device_point=list(map(str, point(e))), error=str(exc)))
            return None
    csv_index = defaultdict(list)
    for j, r in enumerate(rows):
        csv_index[r["Name"], r["Stream ID"].strip(), r["Task ID"].strip(), number(r["Start Time(us)"])].append(j)
    tasks, hosts, used = [], {}, set()
    for i, e in enumerate(events):
        a = e.get("args", {})
        if e.get("ph") != "X" or "Task Type" not in a or e["name"] in ("PROFILING_ENABLE", "PROFILING_DISABLE"):
            continue
        cann = source(e, "HostToDevice")
        host = source(e, "async_npu", cann[1] if cann else None)
        label = containing(host[1]) if host else None
        if label is None and cann:
            label = containing(cann[1], True)
        record = scope_records.get(label, {})
        t = dict(id="k:%d" % i, trace_index=i, name=e["name"], stream=str(a["Physic Stream Id"]),
                 task_id=str(a["Task Id"]), task_type=a["Task Type"], start_us=str(e["ts"]), end_us=str(end(e)),
                 duration_us=str(e["dur"]), scope=label, step=record.get("key"), stage=record.get("stage"),
                 csv_row=None, is_compute=False, is_copy="MEMCPY" in e["name"] or a["Task Type"] in ("SDMA_SQE", "PCIE_DMA_SQE"),
                 host_index=host[0] if host else None, cann_index=cann[0] if cann else None,
                 host_flow=host[2] if host else None, cann_flow=cann[2] if cann else None,
                 connection_id=str(a.get("connection_id")))
        if cann and t["connection_id"] != str(cann[1]["args"].get("connection_id")):
            raise ValueError("CANN connection mismatch")
        matches = csv_index[e["name"], t["stream"], t["task_id"], number(e["ts"])]
        if matches:
            j = only(matches, "CSV identity")
            if j in used or abs(number(rows[j]["Duration(us)"]) - number(e["dur"])) > Decimal(".001"):
                raise ValueError("CSV duplicated or duration mismatch")
            used.add(j)
            core = rows[j]["Accelerator Core"].strip()
            t.update(csv_row=j, core_type=core, is_compute=compute_core(core, t["is_copy"]))
        t["requests"] = [mapping.get(r, r) for r in record.get("scheduled", {})]
        tasks.append(t)
        for pair in (host, cann):
            if pair:
                index, h, _ = pair
                hosts[str(index)] = dict(name=h["name"], start_us=str(h["ts"]), end_us=str(end(h)),
                                         pid=h["pid"], tid=h["tid"], args=h.get("args", {}))
    steps = []
    tasks_by_step = defaultdict(list)
    for t in tasks:
        tasks_by_step[t["step"]].append(t)
    for x in executions:
        key = x["key"]
        sched = schedules.get(key)
        if not sched or sched["scheduled"] != x["scheduled"]:
            issues.append(dict(step=key, error="Missing/mismatched scheduler-worker exact signature occurrence"))
        group = tasks_by_step[key]
        steps.append(dict(key=key, requests=[mapping.get(r, r) for r in x["scheduled"]], scheduled=x["scheduled"],
                          scheduler=sched, tasks=[t["id"] for t in group],
                          start_us=min((number(t["start_us"]) for t in group), default=None),
                          end_us=max((number(t["end_us"]) for t in group), default=None)))
    for r in requests:
        ss = [s for s in steps if r["client_request_id"] in s["requests"]]
        r["steps"] = [s["key"] for s in ss]
        r["first_scheduled_ns"] = min((s["scheduler"]["start_ns"] for s in ss if s["scheduler"]), default=None)
        internal = r.get("internal_id")
        r["scheduled_token_total"] = sum(s["scheduled"].get(internal, 0) for s in ss)
        usage = r.get("usage", {})
        expected = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0) - 1
        r["expected_scheduled_tokens"] = expected
        if expected < 1 or r["scheduled_token_total"] != expected:
            issues.append(dict(request=r["client_request_id"], error="Incomplete per-request scheduled-token coverage",
                               expected=expected, observed=r["scheduled_token_total"]))
    scheduler_order = [r["key"] for r in records if r["kind"] == "schedule"]
    worker_order = [r["key"] for r in executions]
    if scheduler_order != worker_order:
        issues.append(dict(error="Scheduler/worker sequence mismatch for synchronous TP1 execution"))
    stats = overlap(tasks)
    evidence = stats.pop("evidence")
    comp = [(number(t["start_us"]), number(t["end_us"])) for t in tasks if t["is_compute"]]
    copy = [(number(t["start_us"]), number(t["end_us"])) for t in tasks if t["is_copy"]]
    inventory = []
    worker_pids = {r.get("pid") for r in executions}
    identities = [r for r in records if r["kind"] == "stream_identity" and r.get("pid") in worker_pids]
    for sid in sorted({t["stream"] for t in tasks}, key=int):
        group = [t for t in tasks if t["stream"] == sid]
        existing = [r for r in identities if str(r["runtime_stream_id"]) == sid]
        inventory.append(dict(stream=sid, tasks=len(group), compute_tasks=sum(t["is_compute"] for t in group),
                              types=dict(Counter(t["task_type"] for t in group)),
                              compute_busy_us=str(length([(number(t["start_us"]), number(t["end_us"])) for t in group if t["is_compute"]])),
                              observed_python_objects=existing, identity_basis="runtime ID equality (TP1, one device)" if existing else "unresolved",
                              creation_proven=False))
    computes = [t for t in tasks if t["is_compute"]]
    coverage = dict(requests_mapped=len(mapping), requests_expected=len(requests), csv_rows=len(rows), csv_matched=len(used),
                    compute_tasks=len(computes), compute_attributed=sum(bool(t["step"]) for t in computes),
                    scopes_expected=len(scope_records), scopes_found=len(scopes), scheduler_steps=len(schedules), worker_steps=len(steps),
                    observer_errors=[r for r in records if r["kind"] == "observer_error"], identity_issues=issues,
                    flow_issues=flow_issues, unclassified_csv_rows=[t["csv_row"] for t in tasks if t["csv_row"] is not None and not t["is_compute"]])
    # All CSV rows in this export must be compute, not just a matched opaque task.
    coverage["compute_csv_classified"] = len(computes) == len(rows)
    complete = (result["validation"]["status"] == "passed" and result["metrics"]["failed"] == 0
                and bool(computes) and len(mapping) == len(requests) and len(used) == len(rows)
                and coverage["compute_attributed"] == len(computes) and len(scopes) == len(scope_records)
                and len(steps) == len(schedules) and not issues and not coverage["observer_errors"]
                and coverage["compute_csv_classified"])
    # Same-stream physical execution must be serial for this eager workload.
    same_stream_overlaps = []
    for sid in {t["stream"] for t in computes}:
        group = sorted((t for t in computes if t["stream"] == sid), key=lambda t: number(t["start_us"]))
        for a, b in zip(group, group[1:]):
            if number(a["end_us"]) > number(b["start_us"]):
                same_stream_overlaps.append([a["id"], b["id"]])
    coverage["same_stream_compute_overlaps"] = same_stream_overlaps
    result.update(tasks=tasks, hosts=hosts, steps=steps, streams=inventory, overlap_evidence=evidence,
                  scopes=[dict(label=label, step=scope_records[label]["key"], stage=scope_records[label]["stage"],
                               start_us=str(e["ts"]), end_us=str(end(e)), pid=e["pid"], tid=e["tid"]) for label, e in scopes.items()],
                  stream_observations=[r for r in records if r["kind"].startswith("stream_")], coverage=coverage,
                  analysis_status="passed" if complete and not same_stream_overlaps else "incomplete",
                  device_metrics=dict(**stats, copy_compute_overlap_us=str(intersect_length(comp, copy)),
                                      batch_histogram=dict(Counter(len(s["requests"]) for s in steps)),
                                      physical_streams=len(inventory), compute_streams=sum(s["compute_tasks"] > 0 for s in inventory)),
                  raw_sources=[dict(path=str(p.relative_to(case)), sha256=hashlib.sha256(p.read_bytes()).hexdigest(), bytes=p.stat().st_size)
                               for p in (trace, csv_path)])
    if "sampling" in result:
        from sampling_analysis import analyze_sampling
        detail = analyze_sampling(result, records, events, result["precompute"])
        result.update(sampling_analysis=detail["summary"], sampling_steps=detail["steps"], sampling_examples=detail["examples"])
        if detail["summary"]["issues"] or detail["summary"]["validated_steps"] != len(steps):
            result["analysis_status"] = "incomplete"
    return result


def export(case, result):
    output = case / "analysis"
    output.mkdir(exist_ok=True)
    raw = json.dumps(result, ensure_ascii=False, separators=(",", ":"), default=str).encode()
    (output / "analysis.json.gz").write_bytes(gzip.compress(raw, mtime=0))
    summary = {k: v for k, v in result.items() if k not in ("requests", "tasks", "hosts", "steps", "scopes", "stream_observations", "overlap_evidence", "sampling_steps", "sampling_examples")}
    dump(output / "summary.json", summary)
    if "sampling_analysis" in result:
        dump(output / "sampling_examples.json", result["sampling_examples"])
        with gzip.open(output / "sampling_steps.json.gz", "wt") as f:
            json.dump(result["sampling_steps"], f)
    if "tasks" in result:
        columns = ["id", "name", "stream", "task_id", "task_type", "start_us", "end_us", "is_compute", "is_copy", "step", "stage", "requests", "trace_index", "csv_row", "host_flow", "cann_flow"]
        with gzip.open(output / "tasks.csv.gz", "wt") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(result["tasks"])
    from render import render
    render(result, output / "report.html")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    args = p.parse_args()
    cases = [args.run] if (args.run / "command.json").exists() else sorted(args.run.glob("c*-*"))
    summaries = []
    for case in cases:
        result = analyze_case(case)
        summaries.append(export(case, result))
        print(case.name, result.get("analysis_status", "benchmark"), result.get("device_metrics", {}), flush=True)
    if len(cases) > 1:
        dump(args.run / "summary.json", summaries)
        from render import render_index
        render_index(args.run, cases, summaries)
    if any(s.get("analysis_status") == "incomplete" for s in summaries):
        raise SystemExit("Incomplete attribution; inspect coverage before drawing conclusions")


if __name__ == "__main__":
    main()
