"""Small tolist diagnostic, plus a separate run without hooks for wall timings.

Run on the existing Ascend environment; never patches installed packages.
Input is a synthetic 20-element int32 array, not a new full-model measurement.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import threading
import time


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--instrument', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, args.output/'probe_source.py')
    shutil.copy2(Path(__file__).with_name('acl_probe.c'), args.output/'acl_probe.c')
    status = subprocess.check_output(['npu-smi', 'info'], text=True)
    (args.output/'device_before.txt').write_text(status)
    assert 'No running processes' in status, 'Device occupied; abort before initializing NPU'

    import torch
    import torch_npu
    from torch.utils._python_dispatch import TorchDispatchMode
    torch.set_num_threads(1)
    package = Path(torch.__file__).parent
    npu_package = Path(torch_npu.__file__).parent
    tracked = [Path(torch.__file__), Path(torch_npu.__file__), Path(torch._C.__file__),
               npu_package/'lib/libtorch_npu.so', package/'lib/libtorch_cpu.so',
               package/'lib/libtorch_python.so', package/'include/ATen/core/TensorBody.h',
               npu_package/'csrc/aten/npu_native_functions.yaml']
    before = {str(p):sha(p) for p in tracked}
    headers = args.output/'installed'; headers.mkdir()
    for p in tracked[-2:]: shutil.copy2(p, headers/p.name)
    dispatch = {op:torch._C._dispatch_dump_table(op) for op in
                ('aten::to.dtype_layout', 'aten::_to_copy', 'aten::copy_', 'aten::empty.memory_format')}
    save(args.output/'environment.json', dict(torch=torch.__version__,torch_git=torch.version.git_version,
         torch_npu=torch_npu.__version__,npu_git=torch_npu.version.git_version,
         python=os.sys.version, instrument=args.instrument, preload=os.environ.get('LD_PRELOAD'),
         sha256=before, dispatcher=dispatch))

    stream = torch.npu.Stream()
    expected = list(range(1,21))
    cpu = torch.tensor(expected, dtype=torch.int32)
    with torch.npu.stream(stream):
        x = cpu.to('npu'); torch.npu.synchronize()
        for _ in range(10): assert x.tolist() == expected

        if args.instrument:
            hook = ctypes.CDLL(None)
            hook.probe_scope.argtypes = [ctypes.c_uint]
            hook.probe_count.restype = ctypes.c_uint
            class Row(ctypes.Structure):
                _fields_ = [(k,ctypes.c_uint64) for k in
                    ('scope','op','begin_ns','end_ns','tid','src','dst','bytes','kind','stream')] + [('result',ctypes.c_int64)]
            hook.probe_rows.restype = ctypes.POINTER(Row)
            hook.probe_stack.argtypes = [ctypes.c_char_p]
            observed=[]; retained=[]; scope_rows=[]
            def layout(t):
                return dict(device=str(t.device),dtype=str(t.dtype),shape=list(t.shape),stride=list(t.stride()),
                            ptr=t.data_ptr(),storage_ptr=t.untyped_storage().data_ptr(),offset=t.storage_offset(),
                            bytes=t.numel()*t.element_size(),pinned=t.is_pinned() if t.device.type=='cpu' else None)
            class Observe(TorchDispatchMode):
                def __torch_dispatch__(self,func,types,values=(),kwargs=None):
                    kw=kwargs or {}; start=time.monotonic_ns(); out=func(*values,**kw); end=time.monotonic_ns()
                    if func == torch.ops.aten._to_copy.default:
                        observed.append(dict(op=str(func),kwargs={k:str(v) for k,v in kw.items()},
                            input=layout(values[0]),output=layout(out),begin_ns=start,end_ns=end))
                        retained.append(out)
                    return out
            def call(number, name, fn, observer=False):
                hook.probe_scope(number)
                begin=time.monotonic_ns()
                if observer:
                    with Observe(): result=fn()
                else: result=fn()
                end=time.monotonic_ns();hook.probe_scope(0)
                scope_rows.append(dict(id=number,name=name,begin_ns=begin,end_ns=end))
                return result
            assert call(1,'observed_tolist',lambda:x.tolist(),True)==expected
            assert call(2,'triple_tolist',lambda:[x.tolist() for _ in range(3)])==[expected]*3
            def reuse():
                values=x.tolist();return [values]*3
            assert call(3,'single_reused',reuse)==[expected]*3
            host=call(4,'explicit_cpu_copy',lambda:x.to('cpu'),True)
            assert call(5,'cpu_list_construction',host.tolist)==expected
            hook.probe_stack(os.fsencode(args.output/'memcpy_stack.txt'))
            rows=[{k:getattr(hook.probe_rows()[i],k) for k,_ in Row._fields_} for i in range(hook.probe_count())]
            save(args.output/'diagnostic.json',dict(input=layout(x),cpu_input=layout(cpu),stream_handle=stream.npu_stream,
                main_tid=threading.get_native_id(),scopes=scope_rows,dispatch=observed,acl=rows,
                outputs_equal=True,host_copy_values=host.tolist()))
            assert [r['op'] for r in rows if r['scope']==1]==[1,2]
            assert len([r for r in rows if r['scope']==2 and r['op']==2])==3
            assert len([r for r in rows if r['scope']==3 and r['op']==2])==1
            assert not [r for r in rows if r['scope']==5]
            memcpy=next(r for r in rows if r['scope']==1 and r['op']==2)
            assert memcpy['src']==x.data_ptr() and memcpy['dst']==retained[0].data_ptr()
            assert memcpy['bytes']==80 and memcpy['kind']==2
            assert not retained[0].is_pinned() and all(r['result']==0 for r in rows)
        else:
            assert not hasattr(ctypes.CDLL(None), 'probe_scope'), 'Do not benchmark under diagnostic hooks'
            def triple(): return [x.tolist() for _ in range(3)]
            def reuse():
                v=x.tolist();return [v]*3
            funcs={'triple':triple,'reuse':reuse,'cpu_list':cpu.tolist,'explicit_cpu_copy':lambda:x.to('cpu')}
            samples={k:[] for k in funcs}
            for i in range(1000):
                keys=list(funcs);keys=keys[i%len(keys):]+keys[:i%len(keys)]
                for k in keys:
                    begin=time.perf_counter_ns();v=funcs[k]();stop=time.perf_counter_ns()
                    samples[k].append((stop-begin)/1000)
                    assert (v.tolist() if isinstance(v,torch.Tensor) else v)==([expected]*3 if k in ('triple','reuse') else expected)
            save(args.output/'timings.json',dict(samples_us=samples,summary={k:dict(n=len(v),median_us=statistics.median(v),
                 min_us=min(v),max_us=max(v)) for k,v in samples.items()},outputs_equal=True,
                 boundary='ready 20-int32 tensor; no queued model work; no profiler or diagnostic hooks'))
    torch.npu.synchronize()
    after={str(p):sha(p) for p in tracked};assert before==after
    save(args.output/'integrity.json',dict(unchanged=True,sha256_after=after))
    save(args.output/'completed.json',dict(status='passed'))


if __name__=='__main__': main()
