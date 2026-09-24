"""Observe installed Python functions without patching vLLM or reading NPU data.

This is a host control-flow trace, not an accelerator timing profiler.
Function targets are deliberately specific to the recorded v0.21 source tree.
"""

import json
import os
from pathlib import Path
import sys
import threading
import time


TARGETS = {
    ("vllm.entrypoints.openai.completion.serving", "_create_completion"): "api",
    ("vllm.v1.engine.core", "__init__"): "engine_init",
    ("vllm.v1.core.sched.scheduler", "__init__"): "scheduler_init",
    ("vllm.v1.core.sched.scheduler", "add_request"): "enqueue",
    ("vllm.v1.core.sched.scheduler", "schedule"): "schedule",
    ("vllm.v1.core.sched.scheduler", "update_from_output"): "update",
    ("vllm.v1.core.sched.scheduler", "_free_request"): "finish",
    ("vllm.v1.core.sched.scheduler", "_free_blocks"): "cleanup",
    ("vllm_ascend.worker.worker", "execute_model"): "worker",
    ("vllm_ascend.worker.model_runner_v1", "execute_model"): "runner",
    ("vllm_ascend.worker.model_runner_v1", "_model_forward"): "forward",
    ("vllm_ascend.attention.attention_v1", "forward"): "attention",
}
NAMES = {name for _, name in TARGETS}
_codes = {}
_local = threading.local()
_seen_api = set()
_step = 0
_run_dir = None


def typename(obj):
    return f"{type(obj).__module__}.{type(obj).__qualname__}"


def tensor_info(value):
    if value is None:
        return None
    # Shape, dtype and device are host metadata; no .cpu(), .item() or sync.
    return {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}


def emit(event, frame=None, **fields):
    record = {"event": event, "time_ns": time.time_ns(),
              "monotonic_ns": time.monotonic_ns(), "pid": os.getpid(),
              "ppid": os.getppid(), "tid": threading.get_native_id(), **fields}
    if frame is not None:
        record["source"] = {"file": frame.f_code.co_filename,
                            "line": frame.f_code.co_firstlineno,
                            "function": frame.f_code.co_qualname}
    # Each thread has its own file, so records cannot interleave across writers.
    path = _run_dir / f"events-{os.getpid()}-{threading.get_native_id()}.jsonl"
    with path.open("a") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def request_info(request):
    params = request.sampling_params
    return {"request_id": request.request_id,
            "prompt_tokens": request.num_prompt_tokens,
            "computed_tokens": request.num_computed_tokens,
            "output_tokens": request.num_output_tokens,
            "max_tokens": params.max_tokens if params else None,
            "status": str(request.status)}


def handle(kind, frame, event, result):
    global _step
    values = frame.f_locals
    obj = values.get("self")
    if kind == "api":
        request = values["request"]
        rid = request.request_id
        if event == "call" and rid not in _seen_api:
            _seen_api.add(rid)
            emit("api_receive", frame, request_id=rid, model=request.model,
                 max_tokens=request.max_tokens, temperature=request.temperature,
                 ignore_eos=request.ignore_eos, stream=request.stream)
        if event == "return" and type(result).__name__ == "CompletionResponse":
            emit("api_response", frame, request_id=rid, response=result.model_dump())
        return
    if kind in ("engine_init", "scheduler_init"):
        if event != "return":
            return
        if kind == "scheduler_init":
            config = obj.scheduler_config
            emit("scheduler_config", frame, scheduler=typename(obj),
                 async_scheduling=config.async_scheduling,
                 chunked_prefill=config.enable_chunked_prefill,
                 max_num_seqs=config.max_num_seqs,
                 max_num_batched_tokens=config.max_num_batched_tokens,
                 prefix_caching=obj.cache_config.enable_prefix_caching,
                 block_size=obj.cache_config.block_size)
        elif frame.f_code.co_qualname == "EngineCore.__init__" and hasattr(obj, "scheduler"):
            emit("engine_config", frame, engine=typename(obj),
                 executor=typename(obj.model_executor))
        return
    if kind in ("enqueue", "finish", "cleanup"):
        if (kind != "cleanup" and event == "call") or (kind == "cleanup" and event == "return"):
            emit({"enqueue": "scheduler_enqueue", "finish": "request_finish",
                  "cleanup": "request_cleanup_return"}[kind], frame,
                 **request_info(values["request"]))
        return
    if kind == "schedule":
        if event == "call":
            _local.before = {rid: request_info(req) for rid, req in obj.requests.items()}
        elif result is not None and result.total_num_scheduled_tokens:
            _step += 1
            _local.step = _step
            emit("schedule", frame, step=_step,
                 scheduled_tokens=result.num_scheduled_tokens,
                 total_tokens=result.total_num_scheduled_tokens,
                 before={rid: _local.before[rid] for rid in result.num_scheduled_tokens},
                 scheduler=typename(obj))
        return
    if kind in ("worker", "runner"):
        output = values["scheduler_output"]
        if not output.total_num_scheduled_tokens:
            return
        if kind == "runner" and event == "call":
            _local.active = True
            _local.attention_seen = False
            _local.request_ids = list(output.num_scheduled_tokens)
        emit(f"{kind}_{event}", frame, step=getattr(_local, "step", None),
             scheduled_tokens=output.num_scheduled_tokens, implementation=typename(obj),
             return_type=typename(result) if event == "return" else None)
        if kind == "runner" and event == "return":
            _local.active = False
        return
    if kind == "forward" and event == "call" and getattr(_local, "active", False):
        emit("model_forward", frame, step=getattr(_local, "step", None),
             request_ids=_local.request_ids, model=typename(obj.model),
             input_ids=tensor_info(values.get("input_ids")),
             positions=tensor_info(values.get("positions")),
             num_tokens_padded=values["num_tokens_padded"])
    if kind == "attention" and event == "call" and getattr(_local, "active", False):
        if not _local.attention_seen:
            _local.attention_seen = True
            metadata = values.get("attn_metadata")
            emit("attention_first_layer", frame, step=getattr(_local, "step", None),
                 request_ids=_local.request_ids, implementation=typename(obj),
                 layer=getattr(values.get("layer"), "layer_name", None),
                 query=tensor_info(values.get("query")),
                 key=tensor_info(values.get("key")), metadata_type=typename(metadata))
    if kind == "update" and event == "return" and result is not None:
        outputs = []
        for batch in result.values():
            for output in batch.outputs:
                outputs.append({"request_id": output.request_id,
                                "new_token_ids": list(output.new_token_ids),
                                "finish_reason": str(output.finish_reason)})
        if outputs:
            emit("engine_output", frame, step=getattr(_local, "step", None), outputs=outputs)


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
        # Keep inference semantics; the summary rejects any incomplete trace.
        emit("trace_error", frame, kind=kind, error=f"{type(error).__name__}: {error}")


def install():
    global _run_dir
    _run_dir = Path(os.environ["P07_TRACE_DIR"])
    _run_dir.mkdir(parents=True, exist_ok=True)
    emit("trace_installed", python=sys.executable)
    sys.setprofile(profile)
    threading.setprofile(profile)
