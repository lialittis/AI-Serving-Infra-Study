"""Record KV initialization metadata; never read tensor values or synchronize."""
import dataclasses
import json
import os
from pathlib import Path
import sys
import threading
import time

TARGETS = {
    ('vllm_ascend.worker.worker', 'determine_available_memory'): 'budget',
    ('vllm.v1.core.kv_cache_utils', 'get_kv_cache_configs'): 'config',
    ('vllm_ascend.worker.worker', 'initialize_from_config'): 'worker_initialize',
    ('vllm_ascend.worker.model_runner_v1', 'initialize_kv_cache_tensors'): 'pool',
    ('vllm_ascend.worker.model_runner_v1', '_allocate_kv_cache_tensors'): 'raw_pool',
    ('vllm_ascend.worker.model_runner_v1', '_allocate_int8_cache_tensor'): 'raw_tensor',
    ('vllm_ascend.worker.model_runner_v1', '_reshape_kv_cache_tensors'): 'views',
    ('vllm.v1.worker.utils', 'bind_kv_cache'): 'bind',
    ('vllm_ascend.patch.worker.patch_qwen3_next_mtp', 'bind_kv_cache'): 'bind',
    ('vllm.v1.core.block_pool', '__init__'): 'block_pool',
}
NAMES = {name for _, name in TARGETS}
_codes = {}
_local = threading.local()
_output = None


def describe(value):
    import torch
    if isinstance(value, torch.Tensor):
        return dict(shape=list(value.shape), dtype=str(value.dtype), device=str(value.device),
                    data_ptr=value.data_ptr(), storage_ptr=value.untyped_storage().data_ptr(),
                    storage_bytes=value.untyped_storage().nbytes(), nbytes=value.numel()*value.element_size(),
                    stride=list(value.stride()), storage_offset=value.storage_offset())
    if dataclasses.is_dataclass(value):
        return {f.name: describe(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): describe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [describe(v) for v in value]
    if value is None or isinstance(value, (int, float, bool, str)):
        return value
    return str(value)


def emit(event, frame=None, **fields):
    item = dict(event=event, pid=os.getpid(), tid=threading.get_native_id(),
                monotonic_ns=time.monotonic_ns(), **fields)
    if frame is not None:
        item['source'] = dict(file=frame.f_code.co_filename, line=frame.f_code.co_firstlineno,
                              function=frame.f_code.co_qualname)
    with (_output/'events'/('events-%s-%s.jsonl' % (item['pid'], item['tid']))).open('a') as stream:
        stream.write(json.dumps(item, ensure_ascii=False)+'\n')


def snapshot(name):
    import torch_npu
    memory = torch_npu.npu.memory
    data = memory._snapshot()
    (_output/(name+'.json')).write_text(json.dumps(data, ensure_ascii=False)+'\n')
    emit('snapshot', name=name, statistics=dict(memory.memory_stats()))


def handle(kind, frame, event, result):
    if not hasattr(_local, 'opened'):
        _local.opened = {}
        _local.serial = 0
    if kind == 'block_pool' and frame.f_code.co_qualname != 'BlockPool.__init__':
        return
    values = frame.f_locals
    obj = values.get('self')
    if event == 'call':
        fields = {}
        if kind == 'budget':
            fields = dict(requested_memory=obj.requested_memory,
                          gpu_memory_utilization=obj.cache_config.gpu_memory_utilization,
                          init_snapshot=describe(obj.init_snapshot))
        elif kind == 'config':
            fields = dict(available_memory=describe(values['available_memory']))
        elif kind == 'worker_initialize':
            fields = dict(config=describe(values['kv_cache_config']),
                          sleep_mode=obj.vllm_config.model_config.enable_sleep_mode)
        elif kind == 'pool':
            import torch_npu
            torch_npu.npu.memory._record_memory_history(
                enabled='all', context='all', stacks='python', max_entries=100000)
            snapshot('allocator_before')
            fields = dict(config=describe(values['kv_cache_config']),
                          effective_allocation_config=os.environ.get('PYTORCH_NPU_ALLOC_CONF'))
        elif kind == 'raw_tensor':
            fields = dict(requested_bytes=values['numel'], alignment=values['alignment'],
                          layer=frame.f_back.f_locals.get('layer_name'))
        elif kind == 'views':
            fields = dict(raw_tensors=describe(values['kv_cache_raw_tensors']))
        _local.serial += 1
        label = 'P14/%s/%03d/%s' % (os.getpid(), _local.serial, kind)
        parent = list(_local.opened.values())[-1][0] if _local.opened else None
        _local.opened[id(frame)] = (label, kind)
        emit('enter', frame, label=label, kind=kind, parent=parent, **fields)
    else:
        opened = _local.opened.pop(id(frame), None)
        if opened is None:
            return
        label, kind = opened
        fields = {}
        if kind == 'budget':
            fields = dict(available_bytes=result, requested_memory=obj.requested_memory,
                          profile_result=describe(values.get('profile_result')),
                          graph_estimate_applied=values.get('npugraph_memory_estimate_applied'))
        elif kind in ('config', 'raw_tensor', 'raw_pool', 'views', 'pool'):
            fields = dict(returned=describe(result))
            if kind == 'pool' and result is not None:
                fields['bound'] = {name: describe(obj.compilation_config.static_forward_context[name].kv_cache)
                                   for name in result}
        elif kind == 'bind':
            # Record the actual model context after binding, without copying any KV data.
            fields['kv_caches'] = describe(values.get('kv_caches'))
            context = values.get('forward_context')
            if context is not None:
                fields['bound'] = {name: describe(context[name].kv_cache)
                                   for name in values['kv_caches']}
        elif kind == 'block_pool':
            fields = dict(num_blocks=obj.num_gpu_blocks, null_block=obj.null_block.block_id,
                          free_blocks=obj.free_block_queue.num_free_blocks)
        emit('exit', frame, label=label, kind=kind, **fields)
        if kind == 'pool':
            snapshot('allocator_after')
            import torch_npu
            torch_npu.npu.memory._record_memory_history(enabled=None)
            (_output/'process_maps.txt').write_text(Path('/proc/self/maps').read_text())


def profile(frame, event, result):
    if event not in ('call', 'return') or frame.f_code.co_name not in NAMES:
        return
    code = frame.f_code
    if code not in _codes:
        _codes[code] = TARGETS.get((frame.f_globals.get('__name__'), code.co_name))
    kind = _codes[code]
    if kind:
        try:
            handle(kind, frame, event, result)
        except Exception as error:
            emit('trace_error', frame, kind=kind, error=repr(error))


def install():
    global _output
    _output = Path(os.environ['P14_OUTPUT'])
    emit('installed', allocation_config=os.environ.get('PYTORCH_NPU_ALLOC_CONF'),
         device_synchronization_added=False, tensor_values_read=False)
    sys.setprofile(profile)
    threading.setprofile(profile)
