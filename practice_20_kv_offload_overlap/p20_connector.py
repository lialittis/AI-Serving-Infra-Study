"""Experiment-local wrapper around the installed Ascend SimpleCPUOffload path.

Native mode preserves its copy thread, queues, streams, polling and block manager.
Serialized mode adds explicit host barriers; it is not a pure stream-ID ablation.
"""
import contextlib
import itertools
import json
import os
from pathlib import Path
import threading
import time

import torch
from vllm_ascend.distributed.kv_transfer.kv_pool.simple_cpu_offload.simple_cpu_offload_connector import (
    AscendSimpleCPUOffloadConnector,
)

DIAGNOSTIC = os.environ.get("P20_PHASE") == "diagnostic"
ROOT = Path(os.environ["P20_OUTPUT"])
MODE = os.environ.get("P20_MODE", "native")
_sequence = itertools.count()
_state = threading.local()
_installed = False
_pools = {}


def emit(kind, **fields):
    record = dict(kind=kind, pid=os.getpid(), tid=threading.get_native_id(),
                  time_ns=time.time_ns(), monotonic_ns=time.monotonic_ns(), **fields)
    path = ROOT / ("events-%d-%d.jsonl" % (os.getpid(), threading.get_native_id()))
    with path.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


@contextlib.contextmanager
def scope(kind, **fields):
    if not DIAGNOSTIC:
        yield None
        return
    label = "P20/%d/%d/%s" % (os.getpid(), next(_sequence), kind)
    emit("scope_begin", label=label, step=getattr(_state, "step", None), **fields)
    with torch.profiler.record_function(label):
        try:
            yield label
        finally:
            emit("scope_end", label=label)


def block(b):
    h = b.block_hash
    return dict(id=b.block_id, ref=b.ref_cnt, null=b.is_null,
                hash=h.hex() if isinstance(h, bytes) else repr(h) if h is not None else None)


def metadata(m):
    return {key: getattr(m, key) for key in (
        "load_event", "load_gpu_blocks", "load_cpu_blocks", "load_event_to_reqs",
        "store_event", "store_gpu_blocks", "store_cpu_blocks", "need_flush")}


class PublishedEvents(list):
    """Preserve native event-list semantics and acknowledge *actual* publication."""
    def __init__(self, direction, owner):
        super().__init__()
        self.direction, self.owner = direction, owner

    def append(self, item):
        index, event = item
        super().append(item)
        emit("dma_published", direction=self.direction, event_index=index,
             event_handle=str(event.npu_event))
        with self.owner.condition:
            # Only the serial caller needs a strong reference until its wait.
            if MODE == "serialized":
                self.owner.published[self.direction, index] = event
                self.owner.condition.notify_all()


class P20Connector(AscendSimpleCPUOffloadConnector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        emit("connector", mode=MODE, diagnostic=DIAGNOSTIC,
             scheduler=self.scheduler_manager is not None,
             worker=self.worker_handler is not None,
             base=AscendSimpleCPUOffloadConnector.__module__)
        self.condition = threading.Condition()
        self.published = {}
        if self.scheduler_manager is not None:
            manager = self.scheduler_manager
            _pools[id(manager.cpu_block_pool)] = "CPU"
            emit("cpu_pool", blocks=manager.num_cpu_blocks)
        if self.worker_handler is not None:
            worker = self.worker_handler
            worker._load_events = PublishedEvents("H2D", self)
            worker._store_events = PublishedEvents("D2H", self)
            original = worker._backend.launch_copy

            def launch(src_blocks, dst_blocks, is_store, event_idx, events_list):
                direction = "D2H" if is_store else "H2D"
                params = worker._backend._store_params if is_store else worker._backend._load_params
                if MODE == "serialized":
                    with scope("serialize_compute", direction=direction, event_index=event_idx):
                        torch.npu.current_stream().synchronize()
                emit("dma_submit", direction=direction, event_index=event_idx,
                     src_blocks=list(src_blocks), dst_blocks=list(dst_blocks),
                     num_bytes=len(src_blocks) * int(params.bpb.sum()))
                original(src_blocks, dst_blocks, is_store, event_idx, events_list)
                if MODE == "serialized":
                    with self.condition:
                        ok = self.condition.wait_for(
                            lambda: (direction, event_idx) in self.published, timeout=120)
                        if not ok:
                            raise TimeoutError("copy thread did not publish event")
                        event = self.published.pop((direction, event_idx))
                    with scope("serialize_dma", direction=direction, event_index=event_idx):
                        event.synchronize()

            worker._backend.launch_copy = launch
        if DIAGNOSTIC:
            install_observer()

    def bind_gpu_block_pool(self, pool):
        _pools[id(pool)] = "NPU"
        emit("npu_pool", blocks=pool.num_gpu_blocks)
        return super().bind_gpu_block_pool(pool)

    def register_kv_caches(self, caches):
        super().register_kv_caches(caches)
        worker = self.worker_handler
        rows = []
        for name, npu in worker.gpu_kv_caches.items():
            cpu = worker.cpu_kv_caches[name]
            rows.append(dict(name=name, npu_ptr=npu.data_ptr(), cpu_ptr=cpu.data_ptr(),
                             page_bytes=npu.stride(0) * npu.element_size(),
                             npu_shape=list(npu.shape), cpu_shape=list(cpu.shape),
                             pinned=cpu.is_pinned(), npu_offset=npu.storage_offset(),
                             npu_storage_ptr=npu.untyped_storage().data_ptr()))
        if not all(r["pinned"] for r in rows):
            raise RuntimeError("P1 requires pinned CPU buffers")
        emit("cache_layout", tensors=rows, cpu_blocks=worker.num_cpu_blocks,
             load_stream=str(worker.load_stream.npu_stream),
             store_stream=str(worker.store_stream.npu_stream))

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        result = super().get_num_new_matched_tokens(request, num_computed_tokens)
        emit("lookup", request=request.request_id, npu_hit_tokens=num_computed_tokens,
             cpu_hit_tokens=result[0], asynchronous=result[1])
        return result

    def build_connector_meta(self, output):
        if output.preempted_req_ids:
            emit("unsupported_preemption", requests=list(output.preempted_req_ids))
            raise RuntimeError("P1 excludes active request preemption")
        result = super().build_connector_meta(output)
        if DIAGNOSTIC:
            manager = self.scheduler_manager
            states = {rid: dict(computed=s.request.num_computed_tokens,
                                blocks=[list(g) for g in s.block_ids])
                      for rid, s in manager._reqs_to_store.items()}
            emit("schedule", scheduled=dict(output.num_scheduled_tokens),
                 metadata=metadata(result), states=states)
        return result

    def bind_connector_metadata(self, m):
        if DIAGNOSTIC:
            emit("worker_metadata", step=getattr(_state, "step", None), metadata=metadata(m))
        return super().bind_connector_metadata(m)

    def get_finished(self, ids):
        result = super().get_finished(ids)
        worker = self.worker_handler
        if DIAGNOSTIC or result[1] or worker._completed_store_events:
            emit("worker_completed", step=getattr(_state, "step", None),
                 received=sorted(result[1] or []),
                 stored=dict(worker._completed_store_events),
                 load_hwm=worker._load_hwm, store_hwm=worker._store_hwm)
        return result

    def update_connector_output(self, output):
        if DIAGNOSTIC:
            m = output.kv_connector_worker_meta
            emit("scheduler_completed_begin", received=sorted(output.finished_recving or []),
                 stored=dict(getattr(m, "completed_store_events", {})))
        result = super().update_connector_output(output)
        if DIAGNOSTIC:
            emit("scheduler_completed_end")
        return result


def install_observer():
    global _installed
    if _installed:
        return
    _installed = True
    from vllm.v1.core.block_pool import BlockPool
    for name in ("get_new_blocks", "free_blocks", "touch"):
        original = getattr(BlockPool, name)

        def pool_call(self, *args, _name=name, _fn=original, **kwargs):
            medium = _pools.get(id(self))
            if medium is None:
                return _fn(self, *args, **kwargs)
            before = {b.block_id: block(b) for b in self.blocks}
            if _name != "get_new_blocks":
                args = (list(args[0]),) + args[1:]
            result = _fn(self, *args, **kwargs)
            affected = result if _name == "get_new_blocks" else args[0]
            emit("pool", medium=medium, operation=_name,
                 before=[before[b.block_id] for b in affected], after=[block(b) for b in affected])
            return result
        setattr(BlockPool, name, pool_call)

    # Keep the actual native copy loop; annotate the function it calls.
    from vllm_ascend.simple_kv_offload import copy_backend
    original_copy = copy_backend.copy_blocks

    def copy(src, dst, params):
        import sys
        index = sys._getframe(1).f_locals["event_idx"]
        direction = "H2D" if params.direction == 0 else "D2H"
        with scope("copy", direction=direction, event_index=index,
                   src_blocks=list(src), dst_blocks=list(dst),
                   stream=str(torch.npu.current_stream().npu_stream),
                   num_bytes=len(src) * int(params.bpb.sum())):
            return original_copy(src, dst, params)
    copy_backend.copy_blocks = copy

    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
    for name in ("execute_model", "_model_forward", "sample_tokens", "_to_list"):
        original = getattr(NPUModelRunner, name)

        def runner(self, *args, _name=name, _fn=original, **kwargs):
            fields = {}
            if _name == "execute_model":
                _state.step = getattr(_state, "step", 0) + 1
                output = args[0] if args else kwargs["scheduler_output"]
                _state.scheduled = dict(output.num_scheduled_tokens)
            fields["scheduled"] = getattr(_state, "scheduled", {})
            if _name == "_model_forward":
                batch = self.input_batch
                fields["request_ids"] = list(batch.req_ids)
                fields["block_tables"] = [
                    [t.block_table.np[i, :int(t.num_blocks_per_row[i])].tolist()
                     for i in range(len(batch.req_ids))]
                    for t in batch.block_table.block_tables]
            with scope(_name, **fields):
                return _fn(self, *args, **kwargs)
        setattr(NPUModelRunner, name, runner)

    # Capture real event operations, including host waits used for token results.
    for name in ("record", "query", "synchronize", "wait"):
        original = getattr(torch.npu.Event, name)

        def event_call(self, *args, _name=name, _fn=original, **kwargs):
            with scope("event_" + _name) as label:
                result = _fn(self, *args, **kwargs)
                emit("event_api", label=label, api=_name,
                     event_handle=str(self.npu_event), result=result if isinstance(result, bool) else None,
                     stream=str(torch.npu.current_stream().npu_stream))
                return result
        setattr(torch.npu.Event, name, event_call)
