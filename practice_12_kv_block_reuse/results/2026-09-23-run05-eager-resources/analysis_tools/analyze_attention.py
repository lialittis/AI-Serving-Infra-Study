"""Validate real graph-mode attention evidence without torch or an NPU."""
import argparse
from collections import Counter, defaultdict
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "practice_09_operator_trace"))
sys.path.insert(0, str(ROOT.parent / "practice_10_model_graph"))
from summarize_profile import inside, number, only, read_trace, require
from analyze_graph import validate_run as validate_graph_run, validate_nodes


def load(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def same_address(a, b, message, same_view=False):
    fields = ["device", "dtype", "data_ptr", "storage_ptr", "storage_offset", "element_size"]
    if same_view:
        fields += ["shape", "stride"]
    require(a is not None and b is not None and all(a[k] == b[k] for k in fields), message)
    for value in (a, b):
        require(value["data_ptr"] == value["storage_ptr"] + value["storage_offset"] * value["element_size"],
                "inconsistent storage offset")


class DeviceLinks:
    """Require actual correlation endpoints, not nearest-time associations."""
    def __init__(self, events, kernels):
        self.events = events
        self.cpu = [e for e in events if e.get("ph") == "X" and e.get("cat") == "cpu_op"]
        self.device = [e for e in events if e.get("ph") == "X" and "Task Type" in e.get("args", {})]
        require(self.device, "missing device events")
        self.flows, self.points = defaultdict(list), defaultdict(list)
        self.ends = defaultdict(list)
        self.rows = defaultdict(list)
        for e in events:
            if e.get("ph") in ("s", "f"):
                self.flows[e.get("cat"), str(e["id"]), e["ph"]].append(e)
                if e["ph"] == "f":
                    self.ends[e.get("cat"), e["pid"], e["tid"], number(e["ts"])].append(e)
            if e.get("ph") == "X":
                self.points[e["pid"], e["tid"], number(e["ts"])].append(e)
        for row in kernels:
            self.rows[row["Name"], row["Task ID"], row["Stream ID"], number(row["Start Time(us)"])].append(row)

    def follow(self, start, scope, op=None):
        end = only(self.flows["async_npu", str(start["id"]), "f"], "torch flow end")
        candidates = self.points[end["pid"], end["tid"], number(end["ts"])]
        task = only([e for e in candidates if "Task Type" in e.get("args", {})], "device endpoint")
        host = only([e for e in self.points[start["pid"], start["tid"], number(start["ts"])]
                     if e.get("cat") == "cpu_op"], "torch host endpoint")
        cann_end = only(self.ends["HostToDevice", task["pid"], task["tid"], number(task["ts"])],
                        "CANN device endpoint")
        cann_start = only(self.flows["HostToDevice", str(cann_end["id"]), "s"], "CANN flow start")
        cann = only(self.points[cann_start["pid"], cann_start["tid"], number(cann_start["ts"])],
                    "CANN launch endpoint")
        require(cann.get("args", {}).get("connection_id") == task["args"].get("connection_id"),
                "CANN/device connection mismatch")
        row = None
        if task["name"] not in ("MEMCPY_ASYNC", "EVENT_RECORD"):
            row = only(self.rows[task["name"], str(task["args"]["Task Id"]),
                                 str(task["args"]["Physic Stream Id"]), number(task["ts"])], "kernel CSV match")
            require(abs(number(row["Duration(us)"]) - number(task["dur"])) <= Decimal("0.001"),
                    "kernel duration mismatch")
        op = host if op is None else op
        return {"python_scope": scope, "torch_operator": op, "host_operator": host,
                "kernel": task, "kernel_csv": row, "torch_flow_id": str(start["id"]),
                "cann_flow_id": str(cann_start["id"]), "cann_launch": cann,
                "trace_events": [scope, op, host, start, end, cann_start, cann_end, cann, task]}

    def in_scope(self, scope):
        starts = [e for e in self.events if e.get("cat") == "async_npu"
                  and e.get("ph") == "s" and inside(scope, e)]
        require(starts, "missing torch-to-NPU flow")
        return [self.follow(start, scope) for start in starts]

    def core(self, scope, torch_name, kernel_name):
        op = only([e for e in self.cpu if e["name"] == torch_name and inside(scope, e)], torch_name)
        candidates = self.in_scope(op)
        result = only([c for c in candidates if c["kernel"]["name"] == kernel_name], kernel_name)
        result["python_scope"] = scope
        result["torch_operator"] = op
        result["trace_events"].extend([scope, op])
        return result

    def replay_coverage(self):
        """Report missing replay correlations explicitly, with no fabricated joins."""
        tasks = [e for e in self.device if e["args"].get("Model Id") not in (None, 4294967295)]
        inventory = []
        for e in tasks:
            ends = self.ends["HostToDevice", e["pid"], e["tid"], number(e["ts"])]
            starts = [s for f in ends for s in self.flows["HostToDevice", str(f["id"]), "s"]]
            torch_ends = self.ends["async_npu", e["pid"], e["tid"], number(e["ts"])]
            inventory.append({"name": e["name"], "model_id": e["args"]["Model Id"],
                              "stream_id": e["args"]["Physic Stream Id"], "task_id": e["args"]["Task Id"],
                              "start_us": e["ts"], "has_cann_flow_start": bool(starts),
                              "has_torch_flow_end": bool(torch_ends),
                              "fx_node_attribution": "not established"})
        return inventory


def analyze(run, trace=None, records=None):
    graph_summary, nodes, effects, parts = validate_graph_run(run)
    graph = {n["name"]: n for n in nodes}
    fx = only([n for n in nodes if n["target"] == "vllm.unified_attention_with_output"
               and n["args"][4] == "model.layers.0.self_attn.attn"], "first FX attention")
    children = {}
    for name in ("submod_0", "submod_1", "submod_2"):
        children[name] = load(run / "graphs" / ("first_layer_" + name + ".json"))["nodes"]
        validate_nodes(children[name])
    post_ops = [n for n in children["submod_2"] if n["op"].startswith("call_")]
    require(post_ops[0]["target"] == "view" and post_ops[0]["args"][0] == {"node": "output_2"},
            "post-attention partition no longer starts with output view")
    require(post_ops[1]["target"] == "vllm.unquantized_gemm" and
            "o_proj" in post_ops[1]["args"][1]["node"], "O projection boundary missing")
    trace_path = only(list((run / "profiler").rglob("trace_view.json")), "trace file")
    events = read_trace(trace_path) if trace is None else trace
    if records is None:
        records = [json.loads(line) for p in (run / "events").glob("*.jsonl") for line in p.read_text().splitlines()]
    require(not any(e["event"] == "trace_error" for e in records), "annotation error")
    enters = {e["label"]: e for e in records if e["event"] == "scope_enter"}
    exits = {e["label"]: e for e in records if e["event"] == "scope_exit"}
    scopes = {e["name"]: e for e in events if e.get("name", "").startswith("P11/") and e.get("ph") == "X"}
    require(set(enters) == set(exits) == set(scopes) and len(enters) == 23,
            "expected 23 matching Python/profiler scopes")
    require(Counter(e["label"] for e in records if e["event"] == "scope_enter") == Counter(enters.keys()) and
            Counter(e["label"] for e in records if e["event"] == "scope_exit") == Counter(exits.keys()),
            "duplicate or unbalanced Python scopes")
    response = load(run / "response.json")
    rid = only(list({r for e in enters.values() for r in e["request_ids"]}), "request ID")
    require(rid.startswith(response["id"] + "-"), "scope/HTTP request mismatch")
    schedule = sorted([e for e in records if e["event"] == "schedule" and rid in e["scheduled_tokens"]],
                      key=lambda e: e["step"])
    require([e["scheduled_tokens"][rid] for e in schedule] == [126, 1], "expected prefill/decode")
    require(load(run / "shutdown.json")["server_exit_code"] == 0, "service did not exit cleanly")
    controls = load(run / "profile_control.json")
    require([(e["endpoint"], e["status"]) for e in controls] ==
            [("/start_profile", 200), ("/stop_profile", 200)], "profiler control failed")
    kernel_path = only(list((run / "profiler").rglob("kernel_details.csv")), "kernel CSV")
    with kernel_path.open() as stream:
        kernels = list(csv.DictReader(stream))
    links = DeviceLinks(events, kernels)
    phases, chains, attention_tasks = [], [], []
    for sched, phase in zip(schedule, ("prefill", "decode")):
        step = sched["step"]
        prefix = "P11/step=%s/" % step
        select = lambda kind: only([e for e in records if e["event"] == kind and e.get("step") == step], kind)
        fx_call, fx_return = enters[prefix + "fx_attention"], exits[prefix + "fx_attention"]
        tensors = fx_call["tensors"]
        count = sched["scheduled_tokens"][rid]
        require(tensors["query"]["shape"] == [count, 14, 64] and
                tensors["key"]["shape"] == tensors["value"]["shape"] == [count, 2, 64],
                "unexpected runtime Q/K/V shapes")
        producer = exits[prefix + "acl/submod_0"]["returned"]
        require(len(producer) == 5, "unexpected pre-attention partition outputs")
        for index, name in enumerate(("query", "key", "value", "output")):
            same_address(producer[index], tensors[name], "producer/attention storage mismatch: " + name, True)
        context = select("attention_context")
        require(context["layer"] == fx_call["layer"] == fx["args"][4], "FX/runtime layer mismatch")
        require(context["metadata"]["num_actual_tokens"] == count, "metadata token count mismatch")
        require(all(cache["shape"][1:] == [128, 2, 64] for cache in context["kv_cache"]),
                "unexpected KV cache layout")
        same_address(tensors["output"], fx_return["output"], "attention changed output storage", True)
        require(fx_return["returned"] is None, "expected mutation-only attention return")
        consumer = enters[prefix + "acl/submod_2"]
        same_address(tensors["output"], consumer["argument_mapping"]["output_2"],
                     "output/post-partition storage mismatch", True)
        cache_write = enters[prefix + "BaseDeviceAdaptor.reshape_and_cache"]["tensors"]
        for index, name in enumerate(("key", "value")):
            same_address(tensors[name], cache_write[name], "KV write input mismatch: " + name, True)
            same_address(context["kv_cache"][index], cache_write[name + "_cache"],
                         "context/cache destination mismatch: " + name, True)
        same_address(context["metadata"]["slot_mapping"], cache_write["slot_mapping"],
                     "metadata/KV write slot mapping mismatch", True)
        fia = select("fia_inputs")
        require(fia["actual_seq_lengths_kv"] == [sched["before"][rid]["computed_tokens"] + count],
                "FIA sequence length mismatch")
        if phase == "prefill":
            require(context["metadata"]["attn_state"].endswith("PrefillNoCache") and
                    fia["block_table"] is None, "expected uncached prefill")
            for name in ("key", "value"):
                same_address(tensors[name], fia[name], "prefill FIA must use current " + name, True)
            linear = enters[prefix + "first_linear_after_attention"]["tensors"]
            same_address(tensors["output"], linear["x"], "O projection input storage mismatch")
            require(linear["x"]["shape"] == [126, 896], "unexpected O projection input shape")
            require(consumer["runtime_mode"] == "NONE", "expected uncaptured prefill size")
        else:
            require(context["metadata"]["attn_state"].endswith("DecodeOnly"), "expected decode")
            for index, name in enumerate(("key", "value")):
                same_address(context["kv_cache"][index], fia[name], "decode FIA must read cached " + name)
            same_address(context["metadata"]["block_tables"], fia["block_table"], "decode block table mismatch", True)
            require(consumer["runtime_mode"] == "PIECEWISE" and consumer["has_captured_graph"],
                    "missing decode graph replay evidence")
            require(prefix + "first_linear_after_attention" not in enters and prefix + "submod_2" not in enters,
                    "decode unexpectedly traversed Python partition computation")
        first_pos = sched["before"][rid]["computed_tokens"]
        count = sched["scheduled_tokens"][rid]
        blocks = select("host_block_table")["groups"][0]
        require(blocks["block_size"] == 128, "unexpected block size")
        slots = [blocks["rows"][0][p // 128] * 128 + p % 128 for p in range(first_pos, first_pos + count)]
        phase_tasks = links.in_scope(scopes[prefix + "fx_attention"])
        require(len(phase_tasks) == (5 if phase == "prefill" else 3), "unexpected attention task coverage")
        for task in phase_tasks:
            task.update(phase=phase, step=step)
        attention_tasks.extend(phase_tasks)
        phase_chains = []
        for suffix, op, kernel in (
            ("BaseDeviceAdaptor.reshape_and_cache", "atb::_npu_reshape_and_cache", "ReshapeAndCacheNdKernel"),
            ("AscendAttentionBackendImpl.forward_fused_infer_attention", "npu::npu_fused_infer_attention_score", "FusedInferAttentionScore"),
        ):
            chain = links.core(scopes[prefix + suffix], op, kernel)
            chain.update(phase=phase, step=step)
            chains.append(chain)
            phase_chains.append(chain)
        write, attention = [c["kernel"] for c in phase_chains]
        require(write["args"]["Physic Stream Id"] == attention["args"]["Physic Stream Id"] and
                number(write["ts"]) + number(write["dur"]) <= number(attention["ts"]),
                "KV write / attention device order not established")
        if phase == "prefill":
            chain = links.core(scopes[prefix + "first_linear_after_attention"], "aten::linear",
                               "aclnnMatmul_MatMulCommon_MatMulV2")
            chain.update(phase=phase, step=step)
            chains.append(chain)
            require(chain["kernel"]["args"]["Physic Stream Id"] == attention["args"]["Physic Stream Id"] and
                    number(attention["ts"]) + number(attention["dur"]) <= number(chain["kernel"]["ts"]),
                    "attention / O projection device order not established")
        phases.append({"phase": phase, "step": step, "tokens": count, "attention_inputs": tensors,
                       "context": context, "fia_inputs": fia,
                       "consumer_output_argument": consumer["argument_mapping"]["output_2"],
                       "consumer_mode": consumer["runtime_mode"], "producer_return_matches": True,
                       "output_alias_verified": True, "host_blocks": blocks,
                       "expected_slots_from_host_table": [slots[0], slots[-1]],
                       "npu_slot_values_read": False, "storage_values_compared": False,
                       "o_projection_python_observed": phase == "prefill"})
    counts = Counter(e["name"] for e in links.device)
    require(counts["ReshapeAndCacheNdKernel"] == counts["FusedInferAttentionScore"] == 48,
            "expected 24 layers x 2 forwards for KV/attention")
    replay = links.replay_coverage()
    selected = [e for c in chains + attention_tasks for e in c["trace_events"]] + list(scopes.values())
    selected += [e for e in events if e.get("ph") == "M"]
    excerpt = list({json.dumps(e, sort_keys=True): e for e in selected}.values())
    result = {"graph_summary": graph_summary, "fx_node": fx, "request_id": rid,
              "phases": phases, "operator_chains": chains, "attention_tasks": attention_tasks, "scopes": len(scopes),
              "device_task_counts": dict(counts.most_common()), "device_tasks": len(links.device),
              "kernel_csv_rows": len(kernels), "replayed_tasks": len(replay),
              "replayed_tasks_without_torch_flow": sum(not e["has_torch_flow_end"] for e in replay),
              "replayed_tasks_without_cann_start": sum(not e["has_cann_flow_start"] for e in replay),
              "replay_fx_kernel_mapping": "not established; no per-node attribution inferred",
              "trace_file": str(trace_path.relative_to(run)),
              "trace_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest()}
    return result, excerpt, replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    result, excerpt, replay = analyze(args.run)
    output = args.run / "analysis"
    output.mkdir(exist_ok=True)
    save(output / "attention_evidence.json", result)
    save(output / "first_layer_trace.json", excerpt)
    save(output / "replay_coverage.json", replay)
    with (output / "operator_links.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["phase", "python_scope", "torch_operator", "kernel", "stream", "task", "duration_us", "torch_flow", "cann_flow"])
        for c in result["operator_chains"]:
            k = c["kernel"]
            writer.writerow([c["phase"], c["python_scope"]["name"], c["torch_operator"]["name"], k["name"],
                             k["args"]["Physic Stream Id"], k["args"]["Task Id"], k["dur"],
                             c["torch_flow_id"], c["cann_flow_id"]])
    print("Verified: %s scopes; 2 attention storage chains; %s exact operator/kernel links" %
          (result["scopes"], len(result["operator_chains"])))
    print("Replay boundary: %s device tasks, %s without torch flow, %s without CANN flow start" %
          (result["replayed_tasks"], result["replayed_tasks_without_torch_flow"], result["replayed_tasks_without_cann_start"]))


if __name__ == "__main__":
    main()
