"""Join native FX boundaries, tensor storage metadata and profiler scopes.

No tensor value copies, device synchronization, tensor mutation or compiler
replacement. Only the first attention layer of the measured request is labeled.
"""
import os
from pathlib import Path
import sys
import threading

import trace_hooks as base  # Practice 07: scheduler/request correlation.
import graph_capture as graphs  # Practice 10: native FX export.

LAYER = "model.layers.0.self_attn.attn"
TARGETS = dict(base.TARGETS)
TARGETS.update({
    ("vllm.model_executor.layers.attention.attention", "unified_attention_with_output"): "custom_attention",
    ("vllm.model_executor.layers.attention.attention", "get_attention_context"): "context",
    ("vllm_ascend.worker.model_runner_v1", "_prepare_inputs"): "prepare",
    ("vllm_ascend.device.device_op", "reshape_and_cache"): "cache_write",
    ("vllm_ascend.ops.linear", "unquantized_gemm"): "linear",
    ("vllm.compilation.piecewise_backend", "__call__"): "partition",
    ("vllm_ascend.compilation.acl_graph", "__call__"): "aclgraph",
})
for name in ("reshape_and_cache", "forward_impl", "forward_fused_infer_attention",
             "forward_paged_attention", "_get_fia_params"):
    TARGETS[("vllm_ascend.attention.attention_v1", name)] = "attention_child"
NAMES = {name for _, name in TARGETS} | {"split_graph"}
_codes = {}
_local = threading.local()


def layout(value):
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        return [layout(x) for x in value]
    if not hasattr(value, "shape"):
        return str(value)
    return {**base.tensor_info(value), "stride": list(value.stride()),
            "data_ptr": value.data_ptr(), "storage_ptr": value.untyped_storage().data_ptr(),
            "storage_offset": value.storage_offset(), "element_size": value.element_size(),
            "object_id": id(value)}


def metadata(value):
    if value is None:
        return None
    return {"type": base.typename(value), "object_id": id(value),
            "attn_state": str(value.attn_state), "num_actual_tokens": int(value.num_actual_tokens),
            "seq_lens_list": list(value.seq_lens_list),
            "actual_seq_lengths_q": list(value.actual_seq_lengths_q),
            "block_tables": layout(value.block_tables), "slot_mapping": layout(value.slot_mapping)}


def measured():
    return (getattr(base._local, "active", False) and
            any("practice11-profile" in rid for rid in getattr(base._local, "request_ids", [])))


def emit(kind, frame, **fields):
    base.emit(kind, frame, step=getattr(base._local, "step", None),
              request_ids=getattr(base._local, "request_ids", []), **fields)


def enter_scope(frame, suffix, **fields):
    import torch

    label = "P11/step={}/{}".format(base._local.step, suffix)
    scope = torch.profiler.record_function(label)
    scope.__enter__()
    _local.scopes[id(frame)] = (scope, label)
    emit("scope_enter", frame, label=label, **fields)


def handle(kind, frame, event, result):
    values = frame.f_locals
    obj = values.get("self")
    if not hasattr(_local, "scopes"):
        _local.scopes = {}
        _local.attention_frame = None
        _local.partition = None
        _local.partition_frames = {}

    if kind in base.TARGETS.values():
        base.handle(kind, frame, event, result)

    if event == "return":
        entry = _local.scopes.pop(id(frame), None)
        if entry:
            scope, label = entry
            scope.__exit__(None, None, None)
            fields = {}
            if kind in ("custom_attention", "attention", "attention_child"):
                fields["output"] = layout(values.get("output"))
                fields["returned"] = layout(result)
            elif kind in ("partition", "aclgraph"):
                fields["returned"] = layout(result)
            emit("scope_exit", frame, label=label, **fields)
        if id(frame) == _local.attention_frame:
            _local.attention_frame = None
        if id(frame) in _local.partition_frames:
            _local.partition = _local.partition_frames.pop(id(frame))

    if not measured():
        return

    if kind == "prepare" and event == "return":
        tables = []
        for table in obj.input_batch.block_table.block_tables:
            tables.append({"block_size": table.block_size,
                           "rows": [table.block_table.np[i, :int(table.num_blocks_per_row[i])].tolist()
                                    for i in range(obj.input_batch.num_reqs)]})
        emit("host_block_table", frame, groups=tables, batch_request_ids=list(obj.input_batch.req_ids))

    if kind == "context" and event == "return" and _local.attention_frame is not None:
        md, layer, kv, slots = result
        emit("attention_context", frame, layer=layer.layer_name, metadata=metadata(md),
             kv_cache=layout(kv), context_slot_mapping=layout(slots))

    if (kind == "attention_child" and frame.f_code.co_name == "_get_fia_params"
            and event == "return" and _local.attention_frame is not None):
        key, value, block_size, table, lengths = result
        emit("fia_inputs", frame, key=layout(key), value=layout(value),
             block_size=block_size, block_table=layout(table), actual_seq_lengths_kv=list(lengths))

    if event != "call":
        return
    if kind == "runner":
        _local.attention_frame = None
        _local.linear_seen = False
        enter_scope(frame, "runner")
    elif kind == "forward":
        enter_scope(frame, "model_forward", input_ids=layout(values.get("input_ids")),
                    positions=layout(values.get("positions")))
    elif kind == "custom_attention" and str(values["layer_name"]) == LAYER:
        _local.attention_frame = id(frame)
        enter_scope(frame, "fx_attention", layer=LAYER,
                    tensors={k: layout(values[k]) for k in ("query", "key", "value", "output")})
    elif _local.attention_frame is not None and kind in ("attention", "attention_child", "cache_write"):
        if frame.f_code.co_name == "_get_fia_params":
            return
        tensors = {k: layout(values[k]) for k in (
            "query", "key", "value", "output", "kv_cache", "key_cache", "value_cache", "slot_mapping")
                   if k in values}
        enter_scope(frame, frame.f_code.co_qualname, tensors=tensors,
                    metadata=metadata(values.get("attn_metadata")))
    elif kind == "partition" and frame.f_code.co_qualname == "PiecewiseBackend.__call__":
        name = obj.submod_name
        if name not in ("submod_0", "submod_2"):
            return
        _local.partition_frames[id(frame)] = _local.partition
        _local.partition = name
        placeholders = [n.name for n in obj.graph.graph.nodes if n.op == "placeholder"]
        args = values["args"]
        if len(placeholders) != len(args):
            raise ValueError("partition placeholder/argument count mismatch")
        enter_scope(frame, name, argument_mapping={n: layout(a) for n, a in zip(placeholders, args)})
    elif kind == "linear" and _local.partition == "submod_2" and not _local.linear_seen:
        _local.linear_seen = True
        enter_scope(frame, "first_linear_after_attention",
                    tensors={k: layout(values[k]) for k in ("x", "weight", "bias")})
    elif kind == "aclgraph" and frame.f_code.co_qualname == "ACLGraphWrapper.__call__":
        from vllm.forward_context import get_forward_context

        ctx = get_forward_context()
        entry = obj.concrete_aclgraph_entries.get(ctx.batch_descriptor)
        backend = obj.runnable
        name = getattr(backend, "submod_name", None)
        emit("aclgraph_dispatch", frame, runtime_mode=str(ctx.cudagraph_runtime_mode),
             wrapper_mode=str(obj.runtime_mode), partition=name,
             has_captured_graph=bool(entry is not None and entry.aclgraph is not None))
        if name in ("submod_0", "submod_2"):
            _local.partition_frames[id(frame)] = _local.partition
            _local.partition = name
            placeholders = [n.name for n in backend.graph.graph.nodes if n.op == "placeholder"]
            args = values["args"]
            if len(placeholders) != len(args):
                raise ValueError("ACL wrapper placeholder/argument count mismatch")
            enter_scope(frame, "acl/" + name,
                        argument_mapping={n: layout(a) for n, a in zip(placeholders, args)},
                        runtime_mode=str(ctx.cudagraph_runtime_mode),
                        has_captured_graph=bool(entry is not None and entry.aclgraph is not None))


def profile(frame, event, result):
    graphs.profile(frame, event, result)
    if event not in ("call", "return") or frame.f_code.co_name not in NAMES:
        return
    module = frame.f_globals.get("__name__", "")
    try:
        # Keep the exact first pre-attention, attention and post-attention graphs.
        if module == "vllm.compilation.backends" and frame.f_code.co_name == "split_graph" and event == "return":
            if result is not None:
                gm, items = result
                for item in items:
                    if item.submod_name in ("submod_0", "submod_1", "submod_2"):
                        graphs.dump_graph(gm.get_submodule(item.submod_name), "first_layer_" + item.submod_name)
            return
        code = frame.f_code
        if code not in _codes:
            _codes[code] = TARGETS.get((module, code.co_name))
        kind = _codes[code]
        if kind is not None:
            handle(kind, frame, event, result)
    except Exception as error:
        base.emit("trace_error", frame, error=repr(error))


def install():
    base._run_dir = Path(os.environ["P11_TRACE_DIR"])
    base._run_dir.mkdir(parents=True, exist_ok=True)
    graphs.ROOT.mkdir(parents=True, exist_ok=True)
    base.emit("trace_installed", diagnostic_device_copies=False, explicit_device_sync=False)
    sys.setprofile(profile)
    threading.setprofile(profile)
