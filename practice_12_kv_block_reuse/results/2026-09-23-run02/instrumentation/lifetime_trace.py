"""Observe real block ownership and all attention layers without device reads.

No allocator changes, extra tensor copies, event queries, or device waits.
The native small-pool configuration makes reuse deterministic.
"""
import os
from pathlib import Path
import sys
import threading
import trace_hooks as base

TARGETS = dict(base.TARGETS)
TARGETS.update({
    ('vllm.v1.core.kv_cache_manager', 'allocate_slots'): 'allocate',
    ('vllm.v1.core.kv_cache_manager', 'free'): 'manager_free',
    ('vllm.v1.core.block_pool', '__init__'): 'pool_init',
    ('vllm.v1.core.block_pool', 'get_new_blocks'): 'pool_allocate',
    ('vllm.v1.core.block_pool', 'free_blocks'): 'pool_free',
    ('vllm_ascend.worker.model_runner_v1', '_prepare_inputs'): 'prepare',
    ('vllm_ascend.worker.model_runner_v1', 'sample_tokens'): 'sample',
    ('vllm.v1.worker.gpu_model_runner', '_to_list'): 'to_list',
    ('vllm_ascend.device.device_op', 'reshape_and_cache'): 'cache_write',
    ('vllm_ascend.attention.attention_v1', 'forward_fused_infer_attention'): 'fia',
    ('vllm_ascend.attention.attention_v1', '_get_fia_params'): 'fia_params',
})
NAMES = {name for _, name in TARGETS}
_codes = {}
_local = threading.local()


def layout(x):
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return [layout(t) for t in x]
    return dict(base.tensor_info(x), stride=list(x.stride()), data_ptr=x.data_ptr(),
                storage_ptr=x.untyped_storage().data_ptr(), storage_offset=x.storage_offset(),
                element_size=x.element_size())


def role(rid):
    for name in ('A', 'B'):
        if 'practice12-' + name in rid:
            return name
    return None


def pool_state(pool):
    if len(pool.blocks) != 2:
        raise ValueError('expected two physical blocks, including null block')
    return dict(pool_id=id(pool), num_blocks=pool.num_gpu_blocks,
                prefix_caching=pool.enable_caching,
                blocks=[dict(block_id=b.block_id, ref_cnt=b.ref_cnt, is_null=b.is_null,
                             object_id=id(b)) for b in pool.blocks],
                free_queue=[b.block_id for b in pool.free_block_queue.get_all_free_blocks()])


def emit(event_name, frame, **fields):
    base.emit(event_name, frame, role=_local.role, request_id=_local.rid,
              step=getattr(base._local, 'step', None), **fields)


def scope(frame, kind, **fields):
    import torch
    _local.serial += 1
    label = 'P12/{}/{:04d}/{}'.format(_local.role, _local.serial, kind)
    span = torch.profiler.record_function(label)
    span.__enter__()
    _local.open[id(frame)] = (span, label, kind)
    emit('scope_enter', frame, label=label, kind=kind, **fields)


def handle(kind, frame, event, result):
    if not hasattr(_local, 'open'):
        _local.open = {}
        _local.owners = {}
        _local.serial = 0
        _local.role = _local.rid = _local.layer = None
    values = frame.f_locals
    obj = values.get('self')
    if kind in base.TARGETS.values():
        base.handle(kind, frame, event, result)

    if kind in ('allocate', 'manager_free') and event == 'call':
        _local.owners[id(frame)] = (_local.rid, _local.role)
        _local.rid = values['request'].request_id
        _local.role = role(_local.rid)
    elif kind == 'runner' and event == 'call':
        rids = list(values['scheduler_output'].num_scheduled_tokens)
        _local.rid = rids[0] if len(rids) == 1 else ''
        _local.role = role(_local.rid)
    if kind == 'pool_init' and event == 'return' and frame.f_code.co_qualname == 'BlockPool.__init__':
        base.emit('pool_init', frame, state=pool_state(obj))

    if _local.role is not None:
        if event == 'call':
            if kind in ('pool_allocate', 'pool_free'):
                # Do not consume free_blocks(ordered_blocks), which may be an iterator.
                scope(frame, kind, before=pool_state(obj))
            elif kind in ('allocate', 'manager_free'):
                scope(frame, kind, blocks=[list(g) for g in obj.get_block_ids(_local.rid)],
                      computed=values['request'].num_computed_tokens)
            elif kind in ('runner', 'sample'):
                scope(frame, kind)
            elif kind == 'to_list':
                scope(frame, kind, sampled_tensor=layout(values['sampled_token_ids']),
                      transfer_event_type=base.typename(obj.transfer_event),
                      transfer_event_object_id=id(obj.transfer_event))
            elif kind == 'attention' and getattr(base._local, 'active', False):
                _local.layer = values['layer'].layer_name
                md = values['attn_metadata']
                scope(frame, kind, layer=_local.layer, kv_cache=layout(values['kv_cache']),
                      state=str(md.attn_state), seq_lens=list(md.seq_lens_list),
                      num_actual_tokens=int(md.num_actual_tokens),
                      block_tables=layout(md.block_tables), slot_mapping=layout(md.slot_mapping))
            elif kind in ('cache_write', 'fia') and _local.layer is not None:
                names = ('key', 'value', 'key_cache', 'value_cache', 'slot_mapping') if kind == 'cache_write' else ('query', 'key', 'value', 'output')
                scope(frame, kind, layer=_local.layer,
                      tensors={k: layout(values[k]) for k in names if k in values})
        elif event == 'return':
            if kind == 'prepare':
                batch = obj.input_batch
                groups = [dict(block_size=t.block_size,
                               rows=[t.block_table.np[i, :int(t.num_blocks_per_row[i])].tolist()
                                     for i in range(batch.num_reqs)])
                          for t in batch.block_table.block_tables]
                emit('host_block_table', frame, groups=groups, batch_request_ids=list(batch.req_ids))
            elif kind == 'fia_params' and _local.layer is not None:
                key, value, block_size, table, lengths = result
                emit('fia_inputs', frame, layer=_local.layer, key=layout(key), value=layout(value),
                      block_size=block_size, block_table=layout(table), lengths=list(lengths))
            entry = _local.open.pop(id(frame), None)
            if entry:
                span, label, scope_kind = entry
                fields = {}
                if kind in ('pool_allocate', 'pool_free'):
                    fields['after'] = pool_state(obj)
                    blocks = result if kind == 'pool_allocate' else values['blocks_list']
                    fields['affected_blocks'] = [b.block_id for b in blocks]
                elif kind == 'allocate':
                    fields['blocks'] = [list(g) for g in obj.get_block_ids(_local.rid)]
                    fields['success'] = result is not None
                elif kind == 'to_list':
                    fields['token_ids'] = result  # Already a CPU list from the native implementation.
                span.__exit__(None, None, None)
                emit('scope_exit', frame, label=label, kind=scope_kind, **fields)
            if kind == 'attention':
                _local.layer = None
    if event == 'return' and id(frame) in _local.owners:
        _local.rid, _local.role = _local.owners.pop(id(frame))


def profile(frame, event, result):
    if event not in ('call', 'return') or frame.f_code.co_name not in NAMES:
        return
    code = frame.f_code
    if code not in _codes:
        _codes[code] = TARGETS.get((frame.f_globals.get('__name__'), code.co_name))
    kind = _codes[code]
    if kind is not None:
        try:
            handle(kind, frame, event, result)
        except Exception as error:
            base.emit('trace_error', frame, kind=kind, error=repr(error))


def install():
    base._run_dir = Path(os.environ['P12_TRACE_DIR'])
    base._run_dir.mkdir(parents=True, exist_ok=True)
    base.emit('trace_installed', diagnostic_device_copies=False, explicit_device_sync=False,
              allocator_mutation=False)
    sys.setprofile(profile)
    threading.setprofile(profile)
