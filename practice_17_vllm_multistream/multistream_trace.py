"""Add profiler scopes around the real vLLM async-sampling stream path.

The observer reads host metadata only. It does not read tensor values, create
streams/events, or add NPU synchronization.
"""
import json
import os
from pathlib import Path
import sys
import threading
import time


TARGETS = {
    ("vllm_ascend.worker.model_runner_v1", "execute_model"): "runner",
    ("vllm_ascend.worker.model_runner_v1", "sample_tokens"): "sampling_runner",
    ("vllm_ascend.worker.model_runner_v1", "_model_forward"): "forward",
    ("vllm_ascend.worker.model_runner_v1", "_sample"): "sampler",
    ("vllm_ascend.sample.sampler", "do_async_exponential"): "async_exponential",
    ("vllm_ascend.sample.sampler", "random_sample"): "inline_exponential",
}
NAMES = {name for _, name in TARGETS}
_codes = {}
_local = threading.local()
_run_dir = None


def emit(event, frame=None, **fields):
    record = {
        "event": event,
        "time_ns": time.time_ns(),
        "monotonic_ns": time.monotonic_ns(),
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "tid": threading.get_native_id(),
        **fields,
    }
    if frame is not None:
        record["source"] = {
            "file": frame.f_code.co_filename,
            "line": frame.f_code.co_firstlineno,
            "function": frame.f_code.co_qualname,
        }
    path = _run_dir / ("events-%d-%d.jsonl" % (os.getpid(), threading.get_native_id()))
    with path.open("a") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def tensor(value):
    if value is None:
        return None
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "data_ptr": value.data_ptr(),
        "storage_ptr": value.untyped_storage().data_ptr(),
    }


def begin(frame, kind, **fields):
    import torch

    step = _local.step
    label = "P17/step=%d/%s" % (step, kind)
    scope = torch.profiler.record_function(label)
    scope.__enter__()
    _local.open[id(frame)] = (scope, label, kind)
    emit("enter", frame, label=label, kind=kind, step=step,
         request_ids=_local.request_ids, **fields)


def handle(kind, frame, event, result):
    values = frame.f_locals
    if kind == "runner":
        output = values.get("scheduler_output")
        request_ids = list(output.num_scheduled_tokens) if output is not None else []
        measured = any("p17-profile" in rid for rid in request_ids)
        if event == "call":
            if not measured:
                return
            _local.step = getattr(_local, "step", 0) + 1
            _local.request_ids = request_ids
            _local.active = True
            begin(frame, kind, scheduled_tokens=output.num_scheduled_tokens,
                  total_tokens=output.total_num_scheduled_tokens)
            return
        opened = _local.open.pop(id(frame), None)
        if opened:
            scope, label, _ = opened
            scope.__exit__(None, None, None)
            emit("exit", frame, label=label, kind=kind, step=_local.step,
                 return_type=type(result).__module__ + "." + type(result).__qualname__)
            _local.active = False
        return
    if kind == "sampling_runner":
        if event == "call":
            obj = values.get("self")
            state = getattr(obj, "execute_model_state", None)
            output = state[0] if state is not None else None
            request_ids = list(output.num_scheduled_tokens) if output is not None else []
            measured = any("p17-profile" in rid for rid in request_ids)
            if not measured:
                return
            _local.request_ids = request_ids
            _local.active = True
            begin(frame, kind, scheduled_tokens=output.num_scheduled_tokens,
                  total_tokens=output.total_num_scheduled_tokens)
            return
        opened = _local.open.pop(id(frame), None)
        if opened:
            scope, label, _ = opened
            scope.__exit__(None, None, None)
            emit("exit", frame, label=label, kind=kind, step=_local.step,
                 return_type=type(result).__module__ + "." + type(result).__qualname__)
            _local.active = False
        return
    if not getattr(_local, "active", False):
        return
    if event == "call":
        fields = {}
        if kind == "async_exponential":
            fields = {"batch_size": values.get("b_s"), "vocab_size": values.get("head_dim"),
                      "generator_count": len(values.get("generators", {}))}
        elif kind == "inline_exponential":
            fields = {"probabilities": tensor(values.get("probs")),
                      "generator_count": len(values.get("generators", {}))}
        elif kind == "sampler":
            obj = values.get("self")
            topk = getattr(getattr(obj, "sampler", None), "topk_topp_sampler", None)
            fields = {"logits": tensor(values.get("logits")),
                      "precomputed_q": tensor(getattr(topk, "q", None)),
                      "async_event_handle": str(topk.async_event.npu_event)
                      if getattr(topk, "async_event", None) is not None else None}
        begin(frame, kind, **fields)
        return
    opened = _local.open.pop(id(frame), None)
    if not opened:
        return
    scope, label, opened_kind = opened
    fields = {}
    if opened_kind == "async_exponential":
        obj = values.get("self")
        q = getattr(getattr(obj, "topk_topp_sampler", None), "q", None)
        event_obj = getattr(obj, "async_exponential_event", None)
        from vllm_ascend.utils import current_stream, global_stream
        fields = {"q": tensor(q),
                  "default_stream_handle": str(current_stream().npu_stream),
                  "global_stream_handle": str(global_stream().npu_stream),
                  "event_handle": str(event_obj.npu_event) if event_obj is not None else None}
    scope.__exit__(None, None, None)
    emit("exit", frame, label=label, kind=opened_kind, step=_local.step, **fields)


def profile(frame, event, result):
    if event not in ("call", "return") or frame.f_code.co_name not in NAMES:
        return
    code = frame.f_code
    if code not in _codes:
        _codes[code] = TARGETS.get((frame.f_globals.get("__name__"), code.co_name))
    kind = _codes[code]
    if kind is None:
        return
    try:
        handle(kind, frame, event, result)
    except Exception as error:
        emit("trace_error", frame, kind=kind,
             error=type(error).__name__ + ": " + str(error))


def install():
    global _run_dir
    _run_dir = Path(os.environ["P17_TRACE_DIR"])
    _run_dir.mkdir(parents=True, exist_ok=True)
    _local.open = {}
    _local.step = 0
    _local.active = False
    emit("trace_installed", schema=1, extra_device_waits=False,
         device_values_read=False, python=sys.executable)
    sys.setprofile(profile)
    threading.setprofile(profile)
