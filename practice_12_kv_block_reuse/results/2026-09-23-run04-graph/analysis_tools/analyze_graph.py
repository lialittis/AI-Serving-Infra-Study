"""Validate and visualize archived FX graphs. Standard library; no torch needed."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess


def load(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def references(value):
    if isinstance(value, dict):
        if set(value) == {"node"}:
            return {value["node"]}
        return set().union(*(references(x) for x in value.values()))
    if isinstance(value, list):
        return set().union(*(references(x) for x in value))
    return set()


def validate_nodes(nodes):
    """Check both representations of every edge and their topological order."""
    by_name = {n["name"]: n for n in nodes}
    require(len(by_name) == len(nodes), "duplicate node name")
    seen = set()
    for node in nodes:
        inputs = set(node["inputs"])
        require(inputs <= seen, "missing producer or invalid order: " + node["name"])
        require(inputs == references([node["args"], node["kwargs"]]),
                "argument/edge mismatch: " + node["name"])
        for source in inputs:
            require(node["name"] in by_name[source]["users"], "missing reciprocal user")
        for user in node["users"]:
            require(user in by_name and node["name"] in by_name[user]["inputs"],
                    "invalid user: " + node["name"])
        seen.add(node["name"])
    require(sum(n["op"] == "output" for n in nodes) == 1, "expected one graph output")
    return by_name


def layer(node):
    for value in node["module_stack"].values():
        match = re.search(r"\.layers\.(\d+)(?:\.|$)", value[0])
        if match:
            return int(match.group(1))
    return None


def attention_effects(nodes, by_name):
    """Derived mutation annotations, deliberately separate from raw FX edges."""
    edges = []
    for node in nodes:
        if node["target"] != "vllm.unified_attention_with_output":
            continue
        schemas = node["schema"] + " ".join(node.get("schemas", {}).values())
        require(re.search(r"Tensor\([^)]*!\) output\b", schemas) is not None,
                "attention output mutation schema missing")
        buffer = node["args"][3]["node"]
        for user in by_name[buffer]["users"]:
            if user != node["name"]:
                edges.append({"writer": node["name"], "reader": user, "buffer": buffer,
                              "kind": "derived_mutation_dependency",
                              "evidence": "operator schema marks output mutable; reader uses that buffer"})
    return edges


def validate_run(run):
    paths = [p for p in (run / "graphs").glob("model_graph_*.json")
             if not p.name.endswith(".capture.json")]
    require(len(paths) == 1, "expected exactly one whole-model FX graph")
    graph = load(paths[0])
    nodes = graph["nodes"]
    by_name = validate_nodes(nodes)
    counts = Counter(n["target"] for n in nodes if n["op"].startswith("call_"))
    require({layer(n) for n in nodes if layer(n) is not None} == set(range(24)),
            "expected all 24 Qwen decoder layers")
    require(counts["vllm.unified_attention_with_output"] == 24, "attention coverage mismatch")
    require(counts["vllm.unquantized_gemm"] == 96, "linear coverage mismatch")
    require(counts["vllm.npu_rotary_embedding"] == 24, "RoPE coverage mismatch")
    effects = attention_effects(nodes, by_name)
    require(len(effects) == 24, "expected one mutable output dependency per attention")
    split_path = run / "graphs" / ("split_graph_%s.json" % graph["pid"])
    split = load(split_path)
    validate_nodes(split["nodes"])
    parts = load(split_path.with_suffix(".partitions.json"))
    calls = [n["target"] for n in split["nodes"] if n["op"] == "call_module"]
    require(set(calls) == {p["name"] for p in parts} and len(calls) == 49,
            "split graph/partition coverage mismatch")
    require(sum(p["is_splitting_graph"] for p in parts) == 24, "expected 24 split boundaries")
    for part in parts:
        if part["is_splitting_graph"]:
            require(part["call_targets"] == ["vllm.unified_attention_with_output"],
                    "unexpected split-boundary operator")
    events = [json.loads(line) for p in sorted((run / "graphs").glob("events-*.jsonl"))
              for line in p.read_text().splitlines()]
    require(not any(e["event"] == "capture_error" for e in events), "capture_error in events")
    window = load(run / "request_window.json")
    request_events = [e for e in events if window["request_start_ns"] <= e["time_ns"]
                      <= window["request_end_ns"]]
    forwards = [e for e in request_events if e["event"] == "model_forward"]
    require([e["tokens"] for e in forwards] == [126, 1], "request forward coverage mismatch")
    wrappers = [e for e in request_events if e["event"] == "compiled_wrapper_call"]
    require(len(wrappers) == 2 and all(e["has_compiled_bytecode"] and e["inside_model_forward"]
                                     for e in wrappers), "compiled execution evidence missing")
    captures = [e for e in events if e["event"] == "whole_graph_captured"]
    require(len(captures) == 1 and captures[0]["time_ns"] < window["request_start_ns"],
            "expected startup capture and request-time reuse")
    response = load(run / "response.json")
    require(response["usage"]["prompt_tokens"] == 126 and
            response["usage"]["completion_tokens"] == 2 and
            response["choices"][0]["finish_reason"] == "length", "request did not complete")
    output_node = next(n for n in nodes if n["op"] == "output")
    output_values = [by_name[n]["value"] for n in output_node["inputs"]]
    require(len(output_values) == 1 and output_values[0]["shape"][-1] == "896",
            "expected model hidden-state output")
    summary = {"graph_file": str(paths[0].relative_to(run)),
               "graph_sha256": hashlib.sha256(paths[0].read_bytes()).hexdigest(),
               "nodes": len(nodes), "edges": sum(len(n["inputs"]) for n in nodes),
               "node_kinds": dict(Counter(n["op"] for n in nodes)),
               "operator_counts": dict(sorted(counts.items())), "layers": 24,
               "partitions": len(parts), "attention_boundaries": 24,
               "derived_mutation_edges": len(effects), "output_values": output_values,
               "request_forward_tokens": [126, 1], "request_compiled_wrapper_calls": len(wrappers),
               "response_text": response["choices"][0]["text"],
               "checks": "graph edges, schema, layer coverage, partitions and actual request passed"}
    return summary, nodes, effects, parts


def short_target(node):
    return re.sub(r" at 0x[0-9a-f]+", "", node["target"])


def dot_graph(nodes, effects, selected=None):
    by_name = {n["name"]: n for n in nodes}
    core = set(by_name) if selected is None else set(selected)
    included = core | {name for n in nodes if n["name"] in core for name in n["inputs"]}
    lines = ['digraph FX {', 'rankdir=TB; graph [bgcolor="white", nodesep=0.2, ranksep=0.4];',
             'node [shape=box, style="rounded,filled", fontname="sans-serif", fontsize=10];']
    for node in nodes:
        if node["name"] not in included:
            continue
        value = node["value"]
        shape = "[" + ", ".join(value["shape"]) + "]" if isinstance(value, dict) and "shape" in value else ""
        label = node["name"] + "\n" + short_target(node) + "\n" + shape
        if node["op"] == "placeholder":
            label = (node["name"].replace("l_self_modules_", "").replace("_modules_", ".")
                     .replace("_parameters_", ".").replace("_buffers_", ".").rstrip("_")
                     + "\n" + shape)
        color = "#fff0ce" if "unified_attention" in node["target"] else (
            "#eeeeee" if node["op"] == "placeholder" or node["name"] not in core else "#e2efff")
        lines.append('%s [label=%s, fillcolor="%s", URL=%s, target="_top"];' % (
            json.dumps(node["name"]), json.dumps(label), color,
            json.dumps("graph_viewer.html#" + node["name"])))
        if node["name"] in core:
            for name in node["inputs"]:
                lines.append('%s -> %s;' % (json.dumps(name), json.dumps(node["name"])))
    for edge in effects:
        if edge["writer"] in included and edge["reader"] in included:
            lines.append('%s -> %s [style=dashed,color="#c04c00",label="writes shared output",fontsize=9];' % (
                json.dumps(edge["writer"]), json.dumps(edge["reader"])))
    return "\n".join(lines + ['}']) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    summary, nodes, effects, parts = validate_run(args.run)
    output = args.run / "analysis"
    output.mkdir(exist_ok=True)
    save(output / "summary.json", summary)
    save(output / "mutation_edges.json", effects)
    selected = {n["name"] for n in nodes if layer(n) == 0 and n["op"] != "placeholder"}
    (output / "first_layer.dot").write_text(dot_graph(nodes, effects, selected))
    (output / "whole_model.dot").write_text(dot_graph(nodes, effects))
    if shutil.which("dot"):
        subprocess.run(["dot", "-Tsvg", str(output / "first_layer.dot"),
                        "-o", str(output / "first_layer.svg")], check=True)
    template = Path(__file__).with_name("viewer_template.html").read_text()
    data = json.dumps({"summary": summary, "nodes": nodes, "effects": effects,
                       "layers": {n["name"]: layer(n) for n in nodes}, "partitions": parts},
                      ensure_ascii=False).replace("<", "\\u003c")
    (output / "graph_viewer.html").write_text(template.replace("__GRAPH_DATA__", data))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
