"""Observe vLLM's native Dynamo/FX boundary without replacing its compiler.

sys.setprofile sees the graph argument on entry to VllmBackend.__call__.
Only metadata is saved: never read weights, KV tensors or device memory.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time

ROOT = Path(os.environ["P10_GRAPH_DIR"])
_seen = set()
_local = threading.local()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def describe(value):
    """Do not turn symbolic sizes into integers or materialize tensor values."""
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return {"shape": [str(x) for x in value.shape], "dtype": str(value.dtype),
                "device": str(value.device), "stride": [str(x) for x in value.stride()]}
    if isinstance(value, (tuple, list)):
        return [describe(x) for x in value]
    if isinstance(value, dict):
        return {str(k): describe(v) for k, v in value.items()}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


def arguments(value):
    from torch.fx import Node

    if isinstance(value, Node):
        return {"node": value.name}
    if isinstance(value, (tuple, list)):
        return [arguments(x) for x in value]
    if isinstance(value, dict):
        return {str(k): arguments(v) for k, v in value.items()}
    return describe(value)


def dump_graph(graph, stem):
    nodes = []
    for node in graph.graph.nodes:
        nodes.append({"name": node.name, "op": node.op, "target": str(node.target),
                      "args": arguments(node.args), "kwargs": arguments(node.kwargs),
                      "inputs": [n.name for n in node.all_input_nodes],
                      "users": [n.name for n in node.users],
                      "value": describe(node.meta.get("example_value", node.meta.get("val"))),
                      "module_stack": describe(node.meta.get("nn_module_stack", {})),
                      "source_fn_stack": describe(node.meta.get("source_fn_stack", [])),
                      "stack_trace": node.meta.get("stack_trace", ""),
                      "schema": str(getattr(node.target, "_schema", "")),
                      "schemas": {str(k): str(v) for k, v in
                                  getattr(node.target, "_schemas", {}).items()}})
    save(ROOT / (stem + ".json"), {"stage": stem, "pid": os.getpid(), "nodes": nodes})
    (ROOT / (stem + ".py.txt")).write_text(graph.code)
    (ROOT / (stem + ".readable.txt")).write_text(graph.print_readable(print_output=False))


def emit(kind, **values):
    event = {"event": kind, "pid": os.getpid(), "tid": threading.get_ident(),
             "time_ns": time.time_ns(), **values}
    with (ROOT / ("events-%s.jsonl" % os.getpid())).open("a") as out:
        out.write(json.dumps(event, ensure_ascii=False) + "\n")


def profile(frame, event, result):
    if event not in ("call", "return"):
        return
    name = frame.f_code.co_name
    if name not in ("__call__", "split_graph", "_model_forward"):
        return
    module = frame.f_globals.get("__name__", "")
    qualname = frame.f_code.co_qualname
    values = frame.f_locals
    try:
        if (module == "vllm.compilation.backends" and
                qualname == "VllmBackend.__call__" and event == "call"):
            graph = values["graph"]
            key = (os.getpid(), id(graph))
            if key in _seen:
                return
            _seen.add(key)
            stem = "model_graph_%s_%s" % (os.getpid(), len(_seen))
            dump_graph(graph, stem)
            config = values["self"].compilation_config
            save(ROOT / (stem + ".capture.json"), {
                "source": frame.f_code.co_filename, "line": frame.f_code.co_firstlineno,
                "compilation_config": str(config),
                "example_inputs": [describe(x) for x in values["example_inputs"]],
                "capture_phase": "startup compilation; not a per-request graph",
                "graph_file": stem + ".json"})
            emit("whole_graph_captured", graph=stem)
        elif module == "vllm.compilation.backends" and name == "split_graph" and event == "return":
            if result is None:
                return
            graph, items = result
            stem = "split_graph_%s" % os.getpid()
            dump_graph(graph, stem)
            children = []
            for item in items:
                child = graph.get_submodule(item.submod_name)
                children.append({"name": item.submod_name,
                                 "is_splitting_graph": item.is_splitting_graph,
                                 "nodes": len(list(child.graph.nodes)),
                                 "call_targets": [str(n.target) for n in child.graph.nodes
                                                  if n.op.startswith("call_")]})
            save(ROOT / (stem + ".partitions.json"), children)
            emit("split_graph_captured", graph=stem, partitions=len(children))
        elif module == "vllm_ascend.worker.model_runner_v1" and name == "_model_forward":
            if event == "call":
                _local.in_forward = True
                emit("model_forward", tokens=values.get("num_tokens_padded"),
                     input_ids=describe(values.get("input_ids")),
                     positions=describe(values.get("positions")))
            else:
                _local.in_forward = False
        elif (module == "vllm.compilation.wrapper" and
              qualname == "TorchCompileWithNoGuardsWrapper.__call__" and event == "call"):
            obj = values["self"]
            emit("compiled_wrapper_call", inside_model_forward=getattr(_local, "in_forward", False),
                 has_compiled_bytecode=bool(obj._compiled_bytecode))
    except Exception as error:
        emit("capture_error", source=module + "." + qualname, error=repr(error))


def install():
    ROOT.mkdir(parents=True, exist_ok=True)
    emit("capture_installed", file_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    sys.setprofile(profile)
    threading.setprofile(profile)
