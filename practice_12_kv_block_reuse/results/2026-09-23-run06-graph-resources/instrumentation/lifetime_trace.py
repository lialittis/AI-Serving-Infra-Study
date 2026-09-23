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
    ('vllm_ascend.compilation.acl_graph', '__call__'): 'acl_dispatch',
    ('vllm.compilation.piecewise_backend', '__call__'): 'partition_body',
    ('vllm.v1.utils', 'copy_to_gpu'): 'buffer_copy',
    ('vllm_ascend.worker.block_table', 'compute_slot_mapping'): 'slot_prepare',
    ('torch_npu.npu.graphs', 'replay'): 'graph_replay',
    ('torch_npu.npu.streams', 'record'): 'event_record',
    ('torch_npu.npu.streams', 'wait'): 'event_wait',
    ('torch_npu.npu.streams', 'synchronize'): 'native_synchronize',
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
    from resource_observer import stream_info
    parent = list(_local.open.values())[-1][1] if _local.open else None
    fields['parent_label'] = parent
    if kind not in ('pool_allocate','pool_free','allocate','manager_free'):
        fields['current_stream'] = stream_info()
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
    from resource_observer import describe, buffers, graph_state, stream_info
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

    # Record capture baselines after native capture has finished, including warmup.
    # Snapshots contain only JSON metadata, never strong tensor references.
    if kind == 'acl_dispatch' and event == 'return' and values.get('aclgraph') is not None:
        if getattr(obj.runnable, 'submod_name', None) is not None:
            base.emit('graph_capture_resources', frame,
                      resources=graph_state(obj, values['args'], values['kwargs']))

    if _local.role is not None:
        if event == 'call':
            if kind == 'acl_dispatch' and getattr(base._local, 'active', False):
                from vllm.forward_context import get_forward_context

                ctx = get_forward_context()
                name = getattr(obj.runnable, 'submod_name', None)
                if name is not None:
                    entry = obj.concrete_aclgraph_entries.get(ctx.batch_descriptor)
                    graph = entry.aclgraph if entry is not None else None
                    scope(frame, kind, partition=name,
                          runtime_mode=str(ctx.cudagraph_runtime_mode),
                          wrapper_mode=str(obj.runtime_mode), has_captured_graph=graph is not None,
                          graph_object_id=id(graph) if graph is not None else None,
                          resources=graph_state(obj, values['args'], values['kwargs']))
            elif kind == 'partition_body' and getattr(base._local, 'active', False):
                emit('partition_body_call', frame, partition=obj.submod_name)
            elif kind in ('pool_allocate', 'pool_free'):
                # Do not consume free_blocks(ordered_blocks), which may be an iterator.
                scope(frame, kind, before=pool_state(obj))
            elif kind in ('allocate', 'manager_free'):
                scope(frame, kind, blocks=[list(g) for g in obj.get_block_ids(_local.rid)],
                      computed=values['request'].num_computed_tokens)
            elif kind in ('runner', 'sample'):
                scope(frame, kind)
            elif kind == 'prepare':
                scope(frame, kind, buffers=buffers(obj))
            elif kind == 'forward':
                scope(frame, kind, tensors={k:describe(values.get(k)) for k in
                      ('input_ids','positions','inputs_embeds')},
                      num_tokens_padded=values['num_tokens_padded'],
                      model_object_id=id(obj.model),
                      weights={name:describe(t) for name,t in obj.model.named_parameters()})
            elif kind == 'buffer_copy':
                scope(frame, kind, buffer_object_id=id(obj), copied_rows=values.get('n'),
                      tensors={'cpu_source':describe(obj.cpu),'device_destination':describe(obj.gpu)},
                      non_blocking=True)
            elif kind == 'slot_prepare' and frame.f_code.co_qualname == 'BlockTable.compute_slot_mapping':
                scope(frame, kind, tensors={k:describe(values[k]) for k in ('query_start_loc','positions')},
                      block_table=describe(obj.block_table.gpu), slot_mapping=describe(obj.slot_mapping.gpu))
            elif kind == 'graph_replay':
                scope(frame, kind, graph_object_id=id(obj),
                      completion='host return only; device completion must be established separately')
            elif kind in ('event_record','event_wait','native_synchronize') and _local.open:
                scope(frame, kind, object_id=id(obj), object_type=base.typename(obj),
                      explicit_stream=stream_info(values['stream']) if values.get('stream') is not None else None)
            elif kind == 'to_list':
                scope(frame, kind, sampled_tensor=layout(values['sampled_token_ids']),
                      destination=describe(obj.sampled_token_ids_pinned_cpu),
                      transfer_event_type=base.typename(obj.transfer_event),
                      transfer_event_object_id=id(obj.transfer_event))
            elif kind == 'attention' and getattr(base._local, 'active', False):
                _local.layer = values['layer'].layer_name
                md = values['attn_metadata']
                scope(frame, kind, layer=_local.layer, kv_cache=layout(values['kv_cache']),
                      state=str(md.attn_state), seq_lens=list(md.seq_lens_list),
                      num_actual_tokens=int(md.num_actual_tokens),
                      block_tables=layout(md.block_tables), slot_mapping=layout(md.slot_mapping),
                      tensors={k:describe(values.get(k)) for k in ('query','key','value','output')})
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
                elif kind in ('acl_dispatch','forward','cache_write','fia','buffer_copy'):
                    fields['returned_resources'] = describe(result)
                    if kind == 'acl_dispatch':
                        fields['resources_after'] = graph_state(obj, values['args'], values['kwargs'])
                elif kind == 'prepare':
                    fields['buffers_after'] = buffers(obj)
                elif kind == 'attention':
                    fields['output_resources'] = describe(values.get('output'))
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
              allocator_mutation=False, resource_schema=1, mode=os.environ.get('P12_MODE'))
    sys.setprofile(profile)
    threading.setprofile(profile)
