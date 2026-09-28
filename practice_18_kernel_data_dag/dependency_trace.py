"""Record dispatcher arguments/results, storage generations and direct Triton calls.

No tensor values, extra copies or device waits. Dispatcher schemas describe the
public operator boundary, not private native workspace or per-kernel accesses.
"""
import itertools
import os
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT.parent / "practice_13_operator_submission"),
               str(ROOT.parent / "practice_07_real_request_trace")]
import submission_trace as submission

_local = threading.local()
_serial = itertools.count(1)
_storages = {}
_generation = itertools.count(1)
_original_describe = submission.describe


def describe(value):
    import torch
    if isinstance(value, torch.Tensor):
        result = _original_describe(value)
        storage = value.untyped_storage()
        key = (str(value.device), storage._cdata)
        previous = _storages.get(key)
        if previous is None or torch.UntypedStorage._expired(previous[0]):
            if previous is not None:
                torch.UntypedStorage._free_weak_ref(previous[0])
            previous = (storage._weak_ref(), next(_generation))
            _storages[key] = previous
        result.update(storage_generation=previous[1], storage_bytes=storage.nbytes())
        return result
    if isinstance(value, (tuple, list)):
        return [describe(x) for x in value]
    if isinstance(value, dict):
        return {str(k): describe(v) for k, v in value.items()}
    return _original_describe(value)


def make_mode():
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode

    class AccessMode(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            label = f"P18/{os.getpid()}/{next(_serial):06d}/dispatch"
            arguments = []
            for index, parameter in enumerate(func._schema.arguments):
                value = args[index] if index < len(args) else kwargs.get(parameter.name, parameter.default_value)
                alias = parameter.alias_info
                arguments.append(dict(name=parameter.name, value=describe(value),
                                      write=bool(alias and alias.is_write)))
            common = dict(label=label, operator=str(func), schema=str(func._schema),
                          step=getattr(submission.base._local, "step", None),
                          request_id=getattr(submission._local, "rid", None))
            submission.base.emit("dispatch_enter", **common, arguments=arguments)
            try:
                with torch.profiler.record_function(label):
                    result = func(*args, **kwargs)
            except BaseException as error:
                submission.base.emit("dispatch_error", **common, error=repr(error))
                raise
            submission.base.emit("dispatch_exit", **common, returned=describe(result))
            return result
    return AccessMode()


def install():
    submission.describe = describe
    original_handle = submission.handle
    submission.TARGETS[('vllm_ascend.ops.triton.rope', 'rope_forward_triton')] = 'rope_impl'
    submission.NAMES.add('rope_forward_triton')

    def handle(kind, frame, event, result):
        if event == 'return':
            mode = getattr(_local, 'modes', {}).pop(id(frame), None)
            if mode is not None:
                mode.__exit__(None, None, None)
        original_handle(kind, frame, event, result)
        if (event == "call" and kind == "buffer_copy"
                and getattr(submission._local, "measured", False)):
            obj = frame.f_locals['self']
            # Observe small *existing CPU* integer staging buffers before their
            # original H2D copy. No device reads or additional synchronization.
            source = obj.cpu
            if (source.device.type == 'cpu' and source.numel() <= 2048
                    and str(source.dtype) in ('torch.int32', 'torch.int64')):
                rows = frame.f_locals.get('n')
                values = source.numpy()
                if rows is not None:
                    values = values[:rows]
                label = submission._local.open[id(frame)][1]
                submission.base.emit('host_copy_values', label=label,
                                     destination=describe(obj.gpu), rows=rows,
                                     values=values.tolist())
        if event == 'call' and kind == 'launch' and getattr(submission._local, 'measured', False):
            bound = frame.f_back.f_locals.get('bound_args') or {}
            if 'slot_mapping_ptr' in bound:
                parent = frame.f_back
                while parent is not None and parent.f_code.co_name != '_prepare_inputs':
                    parent = parent.f_back
                if parent is not None:
                    runner = parent.f_locals['self']
                    if (runner.pcp_size == 1 and not runner.use_async_spec_decode
                            and not runner.uses_mrope and not runner.uses_xdrope_dim
                            and not runner.use_compress):
                        # In this audited synchronous path the CPU positions_np
                        # and device num_computed_tokens + query_pos are equal by
                        # source contract. Do not misread query_pos as positions.
                        values = parent.f_locals['positions_np'].tolist()
                        if len(values) != bound['num_tokens']:
                            raise ValueError('CPU position contract size mismatch')
                        submission.base.emit('slot_host_contract',
                            label=submission._local.open[id(frame)][1],
                            positions_tensor=describe(bound['positions_ptr']), positions=values,
                            contract='single-card synchronous positions_np equivalence; no device readback')
        if (event == 'call' and kind in ('runner', 'sample', 'attention', 'linear', 'rope_impl')
                and getattr(submission._local, 'measured', False)):
            from torch.utils._python_dispatch import _get_current_dispatch_mode
            # PyTorch pops a mode while redispatching a custom operator. Reenter
            # only after its Python implementation has started, so inner ops
            # (especially FIA temporary -> output.copy_) become visible without
            # recursively redispatching the same outer custom operator.
            if _get_current_dispatch_mode() is None:
                mode = make_mode()
                mode.__enter__()
                if not hasattr(_local, 'modes'):
                    _local.modes = {}
                _local.modes[id(frame)] = mode
                submission.base.emit('dispatch_mode_scope', kind=kind,
                                     step=getattr(submission.base._local, 'step', None))

    submission.handle = handle
    os.environ["P13_TRACE_DIR"] = os.environ["P18_TRACE_DIR"]
    submission.install()
    submission.base.emit("dependency_trace_installed", schema=1,
                         access_precision="operator schema + tensor view bounding ranges",
                         native_workspace_observed=False)
