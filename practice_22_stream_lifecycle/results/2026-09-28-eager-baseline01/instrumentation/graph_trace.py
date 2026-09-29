"""Extend P13's metadata observer with ACL dispatch, capture and replay scopes.

No extra device wait, tensor readback or changes to installed execution code.
Graph IDs are qualified by capture occurrence; addresses are not lifetimes.
"""
import sys
import threading
import submission_trace as base

TARGETS = {
    ('vllm_ascend.compilation.acl_graph', '__call__'): 'acl_dispatch',
    ('vllm.compilation.piecewise_backend', '__call__'): 'partition_body',
    ('torch_npu.npu.graphs', 'replay'): 'graph_replay',
}


def state(wrapper, args):
    import torch
    from vllm.forward_context import get_forward_context
    ctx = get_forward_context()
    entry = wrapper.concrete_aclgraph_entries.get(ctx.batch_descriptor)
    graph = entry.aclgraph if entry else None
    return dict(wrapper_id=id(wrapper), partition=wrapper.runnable.submod_name,
                graph_id=id(graph) if graph is not None else None,
                runtime_mode=str(ctx.cudagraph_runtime_mode), wrapper_mode=str(wrapper.runtime_mode),
                graph_pool=base.describe(wrapper.graph_pool),
                inputs=base.describe(args), output=base.describe(entry.output) if graph else None,
                input_addresses=[x.data_ptr() for x in args if isinstance(x, torch.Tensor)],
                captured_input_addresses=entry.input_addresses if entry else None)


def profile(frame, event, result):
    base.profile(frame, event, result)
    if event not in ('call', 'return'):
        return
    kind = TARGETS.get((frame.f_globals.get('__name__'), frame.f_code.co_name))
    if kind is None:
        return
    try:
        v = frame.f_locals
        obj = v.get('self')
        if kind == 'acl_dispatch' and not getattr(obj.runnable, 'submod_name', None):
            return
        if event == 'return':
            if kind == 'acl_dispatch' and v.get('aclgraph') is not None:
                base.emit('graph_capture', frame, resources=state(obj, v['args']))
            item = getattr(base._local, 'open', {}).pop(id(frame), None)
            if item:
                span, label, _ = item
                fields = dict(returned=base.describe(result)) if kind != 'graph_replay' else {}
                if span:
                    span.__exit__(None, None, None)
                base.emit('exit', frame, label=label, kind=kind, **fields)
            return
        if not getattr(base._local, 'measured', False):
            return
        if kind == 'acl_dispatch':
            base.start(frame, kind, resources=state(obj, v['args']))
        elif kind == 'partition_body':
            base.start(frame, kind, partition=obj.submod_name)
        else:
            import torch
            base.start(frame, kind, graph_id=id(obj), runtime_stream=torch.npu.current_stream().npu_stream)
    except Exception as error:
        base.base.emit('trace_error', frame, kind=kind, error=repr(error))


def install():
    base.install()
    sys.setprofile(profile)
    threading.setprofile(profile)
