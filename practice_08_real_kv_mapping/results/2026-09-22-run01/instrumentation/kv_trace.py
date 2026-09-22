"""Extend Practice 07 with real allocation, slot indices, and KV write checks.

Unlike Practice 07, this diagnostic deliberately copies small NPU tensors to
CPU. It changes synchronization/timing and must not be used as a race detector
or performance benchmark. Only the first attention layer is sampled each step.
"""

import os
from pathlib import Path
import sys
import threading

import trace_hooks as base  # Practice 07, explicitly supplied on PYTHONPATH.


TARGETS = dict(base.TARGETS)
TARGETS.update({
    ("vllm.v1.core.kv_cache_manager", "allocate_slots"): "allocate",
    ("vllm_ascend.worker.model_runner_v1", "_prepare_inputs"): "prepare",
    ("vllm_ascend.device.device_op", "reshape_and_cache"): "cache_write",
})
NAMES = {name for _, name in TARGETS}
_codes = {}


def host_values(tensor):
    """Intentional device-to-host copy of a bounded index tensor."""
    return tensor.detach().cpu().tolist()


def layout(tensor):
    return {**base.tensor_info(tensor), "stride": list(tensor.stride()),
            "storage_offset": tensor.storage_offset(), "data_ptr": tensor.data_ptr(),
            "element_size": tensor.element_size(), "contiguous": tensor.is_contiguous()}


def groups(blocks):
    return [list(group) for group in blocks]


def handle(kind, frame, event, result):
    values = frame.f_locals
    obj = values.get("self")
    active = getattr(base._local, "active", False)
    first_attention = (kind == "attention" and event == "call" and active
                       and not base._local.attention_seen)

    if kind in base.TARGETS.values():
        base.handle(kind, frame, event, result)

    if kind == "allocate" and event == "return":
        request = values["request"]
        base.emit("kv_allocate", frame, request_id=request.request_id,
                  computed_before=request.num_computed_tokens,
                  num_new_tokens=values["num_new_tokens"], success=result is not None,
                  new_block_ids=groups(result.get_block_ids()) if result is not None else None,
                  request_block_ids=groups(obj.get_block_ids(request.request_id)))

    if kind == "schedule" and event == "return" and result is not None and result.total_num_scheduled_tokens:
        # Read-only query of the real manager, after allocation has completed.
        blocks = {rid: groups(obj.kv_cache_manager.get_block_ids(rid))
                  for rid in result.num_scheduled_tokens}
        base._local.request_blocks = blocks
        base.emit("kv_manager_blocks", frame, step=base._local.step, blocks=blocks)

    if kind == "runner" and event == "call":
        base._local.cache_seen = False
        base._local.cache_frame = None
        base._local.first_key_ptr = None

    if kind == "prepare" and event == "return" and active:
        batch = obj.input_batch
        tables = []
        for table in batch.block_table.block_tables:
            tables.append({
                "block_size": table.block_size,
                "physical_block_size": table.physical_block_size,
                "blocks_per_phys_block": table.blocks_per_phys_block,
                "rows": [table.block_table.np[i, :int(table.num_blocks_per_row[i])].tolist()
                         for i in range(batch.num_reqs)],
            })
        base.emit("runner_block_table", frame, step=base._local.step,
                  request_ids=list(batch.req_ids), groups=tables)

    if kind == "forward" and event == "call" and active:
        base._local.positions = host_values(values["positions"])
        base.emit("token_positions", frame, step=base._local.step,
                  request_ids=base._local.request_ids, positions=base._local.positions)

    if first_attention:
        metadata = values["attn_metadata"]
        cache = values["kv_cache"]
        count = int(metadata.num_actual_tokens)
        positions = base._local.positions[:count]
        block_size = 128  # Fail validation if the observed runtime differs.
        used_blocks = (max(positions) + block_size) // block_size
        base._local.first_key_ptr = cache[0].data_ptr()
        slots = host_values(metadata.slot_mapping[:count])
        base._local.attention_slots = slots
        base.emit("attention_kv_metadata", frame, step=base._local.step,
                  request_ids=base._local.request_ids,
                  layer=getattr(values["layer"], "layer_name", None),
                  attn_state=str(metadata.attn_state), num_actual_tokens=count,
                  positions=positions, block_tables=host_values(metadata.block_tables[:1, :used_blocks]),
                  slot_mapping=slots, slot_tensor=base.tensor_info(metadata.slot_mapping),
                  key_cache=layout(cache[0]), value_cache=layout(cache[1]))

    if kind == "cache_write" and active:
        if event == "call" and not base._local.cache_seen:
            if values["key_cache"].data_ptr() != base._local.first_key_ptr:
                return
            base._local.cache_seen = True
            base._local.cache_frame = id(frame)
            base.emit("kv_write_call", frame, step=base._local.step,
                      adaptor=values["cls"].__name__, slot_mapping=host_values(values["slot_mapping"]),
                      key=base.tensor_info(values["key"]), value=base.tensor_info(values["value"]),
                      key_cache=layout(values["key_cache"]), value_cache=layout(values["value_cache"]))
        elif event == "return" and id(frame) == base._local.cache_frame:
            import torch

            slots = base._local.attention_slots
            checks = {}
            for name in ("key", "value"):
                cache = values[name + "_cache"]
                if list(cache.shape[1:]) != [128, 2, 64]:
                    raise ValueError(f"unsupported Qwen cache layout: {list(cache.shape)}")
                # Copy only touched blocks, never the whole preallocated KV pool.
                cpu_blocks = {block: cache[block].detach().cpu() for block in {s // 128 for s in slots}}
                actual = torch.stack([cpu_blocks[s // 128][s % 128] for s in slots])
                expected = values[name].detach().cpu()
                checks[name + "_equal"] = bool(torch.equal(actual, expected))
            base.emit("kv_write_check", frame, step=base._local.step,
                      tokens_checked=len(slots), **checks)
            base._local.cache_frame = None


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
            base.emit("trace_error", frame, kind=kind, error=f"{type(error).__name__}: {error}")


def install():
    base._run_dir = Path(os.environ["P08_TRACE_DIR"])
    base._run_dir.mkdir(parents=True, exist_ok=True)
    base.emit("trace_installed", python=sys.executable, diagnostic_device_copies=True)
    sys.setprofile(profile)
    threading.setprofile(profile)
