"""Label real Python scopes inside the Ascend profiler timeline.

No tensor data reads and no explicit device synchronization. The labels measure
host scopes, not device completion. torch-npu supplies the device evidence.
"""
import os
from pathlib import Path
import sys
import threading

import trace_hooks as base


TARGETS = dict(base.TARGETS)
for name in ("reshape_and_cache", "forward_impl", "forward_fused_infer_attention",
             "forward_paged_attention"):
    TARGETS[("vllm_ascend.attention.attention_v1", name)] = "attention_path"
TARGETS[("vllm_ascend.device.device_op", "reshape_and_cache")] = "device_adapter"
NAMES = {name for _, name in TARGETS}
_codes = {}
_scopes = threading.local()


def handle(kind, frame, event, result):
    if not hasattr(_scopes, "open"):
        _scopes.open = {}
        _scopes.first_attention = None
    first_attention = (kind == "attention" and event == "call"
                       and getattr(base._local, "active", False)
                       and not base._local.attention_seen)
    if kind in base.TARGETS.values():
        base.handle(kind, frame, event, result)
    if event == "return":
        entry = _scopes.open.pop(id(frame), None)
        if entry:
            scope, label = entry
            scope.__exit__(None, None, None)
            base.emit("scope_exit", frame, label=label)
        if id(frame) == _scopes.first_attention:
            _scopes.first_attention = None
        return

    # One warmup request is deliberately outside the collection window.
    rids = getattr(base._local, "request_ids", [])
    if not any("practice09-profile" in rid for rid in rids):
        return
    if first_attention:
        _scopes.first_attention = id(frame)
    selected = (kind in ("runner", "forward") or first_attention
                or (kind in ("attention_path", "device_adapter")
                    and _scopes.first_attention is not None))
    if not selected or not getattr(base._local, "active", False):
        return
    import torch

    step = getattr(base._local, "step", None)
    label = "P09/step={}/{}".format(step, frame.f_code.co_qualname)
    scope = torch.profiler.record_function(label)
    scope.__enter__()
    _scopes.open[id(frame)] = (scope, label)
    values = frame.f_locals
    metadata = values.get("attn_metadata")
    tensors = {key: base.tensor_info(values[key]) for key in ("query", "key", "value")
               if values.get(key) is not None}
    base.emit("scope_enter", frame, label=label, step=step, request_ids=rids,
              tensors=tensors, attn_state=str(metadata.attn_state) if metadata else None)


def profile(frame, event, result):
    if event not in ("call", "return") or frame.f_code.co_name not in NAMES:
        return
    code = frame.f_code
    if code not in _codes:
        _codes[code] = TARGETS.get((frame.f_globals.get("__name__"), code.co_name))
    kind = _codes[code]
    if kind is not None:
        try:
            handle(kind, frame, event, result)
        except Exception as error:
            base.emit("trace_error", frame, kind=kind,
                      error="{}: {}".format(type(error).__name__, error))


def install():
    base._run_dir = Path(os.environ["P09_TRACE_DIR"])
    base._run_dir.mkdir(parents=True, exist_ok=True)
    base.emit("trace_installed", python=sys.executable, diagnostic_device_copies=False)
    sys.setprofile(profile)
    threading.setprofile(profile)
