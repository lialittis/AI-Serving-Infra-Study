"""Buffered host-only observer for the pinned vLLM/Ascend eager service.

No device tensor values, extra waits, or stream creation. Python profiling
changes host overhead; use the separate uninstrumented benchmark for latency.
"""
import atexit
from collections import Counter
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time

_records = []
_lock = threading.RLock()
_local = threading.local()
_seen = set()
_occurrences = Counter()
_streams = {}
_query = None
_directory = None


def emit(kind, **fields):
    with _lock:
        _records.append(dict(kind=kind, pid=os.getpid(), tid=threading.get_native_id(),
                             wall_ns=time.time_ns(), mono_ns=time.monotonic_ns(), **fields))


def flush():
    with _lock:
        if not _records:
            return
        with (_directory / ("observer-%d.jsonl" % os.getpid())).open("a") as f:
            for r in _records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        _records.clear()


def measured(rid):
    # Selection only. All identity joins use captured exact IDs, not substring joins.
    return "p31-measure-" in str(rid)


def step_key(role, scheduled):
    signature = json.dumps(sorted(scheduled.items()), separators=(",", ":"))
    _occurrences[role, signature] += 1
    return hashlib.sha256(signature.encode()).hexdigest()[:20] + ":" + str(_occurrences[role, signature])


def describe_stream(obj):
    global _query
    if obj is None or not hasattr(obj, "npu_stream"):
        return None
    handle = int(obj.npu_stream)
    if handle not in _streams:
        if _query is None:
            _query = ctypes.CDLL("libascendcl.so").aclrtStreamGetId
            _query.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32)]
            _query.restype = ctypes.c_int
        sid = ctypes.c_int32(-1)
        rc = _query(ctypes.c_void_p(handle), ctypes.byref(sid))
        _streams[handle] = dict(handle=str(handle), runtime_stream_id=sid.value if rc == 0 else None,
                                query_return=rc, python_stream_id=str(obj.stream_id), device=str(obj.device))
        emit("stream_identity", **_streams[handle], origin="observed_existing_object",
             creation_proven=False)
    return dict(_streams[handle], object_id=str(id(obj)))


def open_scope(frame, stage):
    import torch
    label = "P31/%s/%s" % (_local.key, stage)
    ctx = torch.profiler.record_function(label)
    ctx.__enter__()
    _local.scopes[id(frame)] = (ctx, label, stage, time.time_ns())


def handle(frame, event, result):
    mod = frame.f_globals.get("__name__", "")
    name = frame.f_code.co_name
    v = frame.f_locals
    if not hasattr(_local, "scopes"):
        _local.scopes = {}
        _local.active = False
    if mod == "vllm.entrypoints.openai.completion.serving" and name in ("create_completion", "_create_completion"):
        body = v.get("request")
        rid = getattr(body, "request_id", None)
        if not measured(rid):
            return
        if event == "call" and ("receive", rid) not in _seen:
            _seen.add(("receive", rid))
            emit("received", client_request_id=rid)
        if event == "return" and v.get("request_id_item") and ("front", rid) not in _seen:
            _seen.add(("front", rid))
            emit("frontend_map", client_request_id=rid, response_id=v["request_id"],
                 external_id=v["request_id_item"])
    elif mod == "vllm.v1.engine.input_processor" and name == "assign_request_id" and event == "return":
        req = v["request"]
        if measured(req.external_req_id):
            emit("engine_map", external_id=req.external_req_id, internal_id=req.request_id)
    elif mod == "vllm.v1.core.sched.scheduler" and name == "add_request" and event == "call":
        req = v["request"]
        if measured(req.request_id):
            emit("admitted", internal_id=req.request_id, prompt_tokens=req.num_prompt_tokens)
    elif mod == "vllm.v1.core.sched.scheduler" and name == "schedule":
        if event == "call":
            _local.before = {rid: (r.num_computed_tokens, r.num_prompt_tokens)
                             for rid, r in v["self"].requests.items() if measured(rid)}
            _local.schedule_start = time.time_ns()
        elif result is not None:
            scheduled = dict(result.num_scheduled_tokens)
            if any(measured(r) for r in scheduled):
                phases = {}
                for rid in scheduled:
                    computed, prompt = _local.before.get(rid, (None, None))
                    phases[rid] = dict(computed_before=computed, prompt_tokens=prompt,
                                       phase="unresolved" if computed is None else
                                       "prefill" if computed < prompt else "decode")
                emit("schedule", key=step_key("scheduler", scheduled), scheduled=scheduled,
                     requests=phases, start_ns=_local.schedule_start, end_ns=time.time_ns())
    elif mod == "vllm_ascend.worker.model_runner_v1" and name in (
            "execute_model", "sample_tokens", "_model_forward", "_sample"):
        stage = {"execute_model": "execute", "sample_tokens": "sample",
                 "_model_forward": "forward", "_sample": "sampler"}[name]
        if event == "call":
            if name == "execute_model":
                scheduled = dict(v["scheduler_output"].num_scheduled_tokens)
                _local.active = any(measured(r) for r in scheduled)
                if not _local.active:
                    return
                _local.key = step_key("worker", scheduled)
                _local.scheduled = scheduled
                emit("execute", key=_local.key, scheduled=scheduled)
            elif name == "sample_tokens":
                state = getattr(v["self"], "execute_model_state", None)
                _local.active = bool(state and any(measured(r) for r in state[0].num_scheduled_tokens))
            if _local.active:
                open_scope(frame, stage)
        else:
            opened = _local.scopes.pop(id(frame), None)
            if opened:
                ctx, label, stage, start = opened
                ctx.__exit__(None, None, None)
                emit("scope", label=label, key=_local.key, stage=stage, start_ns=start,
                     end_ns=time.time_ns(), scheduled=_local.scheduled)
            if name in ("execute_model", "sample_tokens"):
                _local.active = False
    elif mod in ("vllm_ascend.utils", "torch_npu.npu", "torch_npu.npu.streams"):
        if event == "return" and name in ("current_stream", "global_stream", "prefetch_stream", "__new__"):
            desc = describe_stream(result)
            if desc and (_local.active or name != "current_stream"):
                emit("stream_use", operation=name, key=getattr(_local, "key", None), stream=desc)
        elif event == "call" and _local.active and name in ("set_stream", "wait", "record", "wait_event", "wait_stream", "synchronize"):
            emit("stream_api", operation=frame.f_code.co_qualname, key=_local.key,
                 stream=describe_stream(v.get("stream")), self_stream=describe_stream(v.get("self")),
                 event_handle=str(getattr(v.get("event", v.get("self")), "npu_event", "")))
    # Both the engine process and the API process flush after profile shutdown.
    if event == "return" and ((name == "profile" and v.get("is_start") is False and mod.startswith("vllm"))
                               or (name == "stop_profile" and mod.startswith("vllm"))):
        flush()


NAMES = {"create_completion", "_create_completion", "assign_request_id", "add_request", "schedule", "execute_model",
         "sample_tokens", "_model_forward", "_sample", "current_stream", "global_stream", "prefetch_stream",
         "__new__", "set_stream", "wait", "record", "wait_event", "wait_stream", "synchronize", "profile", "stop_profile"}


def profile(frame, event, result):
    if event not in ("call", "return") or frame.f_code.co_name not in NAMES:
        return
    try:
        handle(frame, event, result)
    except Exception as exc:
        emit("observer_error", module=frame.f_globals.get("__name__"), function=frame.f_code.co_name,
             error=repr(exc))


def install():
    global _directory
    _directory = Path(os.environ["P31_OBSERVER_DIR"])
    _directory.mkdir(parents=True, exist_ok=True)
    emit("installed", extra_device_waits=False, device_values_read=False, creates_streams=False)
    atexit.register(flush)
    sys.setprofile(profile)
    threading.setprofile(profile)
