"""Instrument vllm-ascend random_sample without modifying installed files.

The wrapper reproduces the installed random_sample semantics line for line and
adds only host-side observation: q's address, whether it equals the previous
q's address, and whether the previous main-stream consumer (div_) has already
completed by the time the next fill is submitted.
"""
import atexit
import hashlib
import inspect
import json
import os
import time

_STATE = {"prev": None}
_RECORDS = []


def install():
    import torch
    import torch_npu  # noqa: F401  (initializes the device runtime)
    import vllm_ascend.sample.sampler as sampler_module
    if getattr(sampler_module, "_p35_instrumented", False):
        return
    sampler_module._p35_instrumented = True
    original = sampler_module.random_sample
    source_file = inspect.getfile(original)
    source_text = inspect.getsource(original)

    def random_sample(probs, generators):
        # Semantics identical to the installed vllm-ascend 0.21.0rc1
        # random_sample; instrumentation is host-side only.
        from vllm_ascend.utils import global_stream, npu_stream_switch
        prev = _STATE["prev"]
        entry = dict(index=len(_RECORDS), t_mono_ns=time.monotonic_ns(),
                     rows=int(probs.shape[0]), address=None,
                     reused_prev=None, prev_div_pending=None)
        if prev is not None:
            entry["prev_div_pending"] = not prev["event"].query()
            entry["prev_gap_ms"] = (entry["t_mono_ns"] - prev["t_mono_ns"]) / 1e6
        with npu_stream_switch(global_stream()):
            q = torch.empty_like(probs)
            entry["address"] = q.data_ptr()
            if prev is not None:
                entry["reused_prev"] = entry["address"] == prev["address"]
            if len(generators) != probs.shape[0]:
                q.exponential_()
            if generators:
                for i, generator in generators.items():
                    q[i].exponential_(generator=generator)
        torch.npu.current_stream().wait_stream(global_stream())
        result = probs.div_(q).argmax(dim=-1).view(-1)
        event = torch.npu.Event()
        event.record()  # main stream, queued after div_
        _STATE["prev"] = dict(address=entry["address"], event=event,
                              t_mono_ns=entry["t_mono_ns"])
        _RECORDS.append(entry)
        return result

    sampler_module.random_sample = random_sample
    directory = os.environ.get("P35_OBSERVER_DIR")
    if directory:
        atexit.register(_dump, directory, source_file, source_text)


def _dump(directory, source_file, source_text):
    try:
        os.makedirs(directory, exist_ok=True)
        payload = dict(pid=os.getpid(), calls=len(_RECORDS), records=_RECORDS,
                       original_source=dict(file=source_file, text=source_text,
                                            sha256=hashlib.sha256(
                                                source_text.encode()).hexdigest()))
        path = os.path.join(directory, f"records-{os.getpid()}.json")
        with open(path, "w") as stream:
            json.dump(payload, stream, indent=1)
    except BaseException as exc:  # never break the workload at exit
        print("P35 observer dump failed:", exc, flush=True)
