"""Metadata-only resource snapshots; never retain tensors, query events or wait.

Object IDs and pointers are process-local observations, not allocation generations.
"""
import dataclasses
import torch


def describe(value):
    if isinstance(value, torch.Tensor):
        return dict(type='tensor', object_id=id(value), shape=list(value.shape),
                    dtype=str(value.dtype), device=str(value.device), stride=list(value.stride()),
                    data_ptr=value.data_ptr(), storage_ptr=value.untyped_storage().data_ptr(),
                    storage_offset=value.storage_offset(), element_size=value.element_size())
    if isinstance(value, (list, tuple)):
        return [describe(x) for x in value]
    if isinstance(value, dict):
        return {str(k):describe(v) for k,v in value.items()}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    # Do not invoke arbitrary repr, conversion, iterator or properties.
    return dict(type=type(value).__module__+'.'+type(value).__qualname__, object_id=id(value),
                contents='not inspected')


def stream_info(stream=None):
    if stream is None:
        stream = torch.npu.current_stream()
    return dict(device=str(stream.device), stream_id=stream.npu_stream,
                note='runtime stream handle; not assumed equal to profiler Physic Stream Id')


def buffers(runner):
    values={}
    for name,value in vars(runner).items():
        if type(value).__name__ == 'CpuGpuBuffer':
            values[name] = dict(buffer_object_id=id(value), cpu=describe(value.cpu), gpu=describe(value.gpu))
    for i,table in enumerate(runner.input_batch.block_table.block_tables):
        for name in ('block_table','slot_mapping'):
            value=getattr(table,name)
            values['block_group_{}.{}'.format(i,name)] = dict(buffer_object_id=id(value),cpu=describe(value.cpu),gpu=describe(value.gpu))
    return values


def graph_state(wrapper, args, kwargs):
    from vllm.forward_context import get_forward_context
    ctx=get_forward_context()
    descriptor=ctx.batch_descriptor
    entry=wrapper.concrete_aclgraph_entries.get(descriptor)
    graph=entry.aclgraph if entry is not None else None
    backend=wrapper.runnable
    names=[n.name for n in backend.graph.graph.nodes if n.op=='placeholder']
    return dict(wrapper_object_id=id(wrapper), graph_object_id=id(graph) if graph is not None else None,
                graph_pool=describe(wrapper.graph_pool), batch_descriptor=dataclasses.asdict(descriptor) if dataclasses.is_dataclass(descriptor) else describe(descriptor),
                partition=backend.submod_name, runtime_mode=str(ctx.cudagraph_runtime_mode),
                wrapper_mode=str(wrapper.runtime_mode), captured_input_addresses=entry.input_addresses if entry else None,
                arguments={name:describe(arg) for name,arg in zip(names,args)},
                argument_count=len(args), placeholder_count=len(names), keyword_arguments=describe(kwargs),
                input_addresses=[x.data_ptr() for x in args if isinstance(x,torch.Tensor)],
                persistent_output=describe(entry.output) if graph is not None else None,
                hidden_workspace='not exposed by this wrapper; no allocation lifetime proof')
