"""Bounded eager Sub probe; profiling, API instrumentation and timings are separate."""
import argparse
from contextlib import nullcontext
import ctypes
import gc
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time


def save(p,x): p.write_text(json.dumps(x,indent=2)+'\n')


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--kind',choices=['plain','profile','api'],required=True)
    ap.add_argument('--mode',choices=['sub','tolist','sync'],required=True)
    args=ap.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    polls=[]
    for attempt in range(5):
        status=subprocess.check_output(['npu-smi','info'],text=True);polls.append(status)
        if 'No running processes' in status:break
        time.sleep(1)
    save(out/'device_polls.json',polls);(out/'device_before.txt').write_text(status)
    assert 'No running processes' in status,'Device occupied; abort before NPU initialization'
    shutil.copy2(__file__,out/'probe_source.py')
    shutil.copy2(Path(__file__).with_name('api_probe.c'),out/'api_probe.c')
    import torch
    import torch_npu
    torch.set_num_threads(1)
    tracked=[Path(torch.__file__),Path(torch_npu.__file__),Path(torch_npu.__file__).parent/'lib/libtorch_npu.so']
    before={str(p):sha(p) for p in tracked}
    hook=ctypes.CDLL(None)
    if args.kind=='api':
        hook.sub_probe_scope.argtypes=[ctypes.c_uint];hook.sub_probe_dump.argtypes=[ctypes.c_char_p]
    else:assert not hasattr(hook,'sub_probe_scope'),'Instrumentation cannot enter normal timings'
    stream=torch.npu.Stream()
    def layout(t):
        return dict(ptr=t.data_ptr(),storage_ptr=t.untyped_storage().data_ptr(),offset=t.storage_offset(),
                    shape=list(t.shape),stride=list(t.stride()),dtype=str(t.dtype),device=str(t.device),
                    itemsize=t.element_size(),bytes=t.numel()*t.element_size(),storage_bytes=t.untyped_storage().nbytes())
    with torch.npu.stream(stream):
        parent=torch.tensor([i*(i+1)//2 for i in range(21)],dtype=torch.int32).to('npu')
        left=parent[1:];right=parent[:-1];stream.synchronize()
        handle=stream.npu_stream;initial_alloc=torch.npu.memory_allocated()
        inputs={'parent':layout(parent),'left':layout(left),'right':layout(right)}
        assert inputs['left']['ptr']==inputs['parent']['ptr']+4
        assert inputs['right']['ptr']==inputs['parent']['ptr']
        prof=(torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,
              torch_npu.profiler.ProfilerActivity.NPU],record_shapes=True,profile_memory=False,
              schedule=torch_npu.profiler.schedule(wait=0,warmup=0,active=1,repeat=1),
              on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(out/'profiler')))
              if args.kind=='profile' else nullcontext())
        rows=[];retained=[]
        with prof:
            for i in range(101 if args.kind=='plain' else 6):
                label=f'SUB/{args.mode}/{i}'
                scope=torch.profiler.record_function(label) if args.kind=='profile' else nullcontext()
                if args.kind=='api':hook.sub_probe_scope(i+1)
                begin=time.monotonic_ns()
                with scope:
                    start=time.monotonic_ns();y=left-right;returned=time.monotonic_ns()
                    result=y.tolist() if args.mode=='tolist' else None
                    if args.mode=='sync':stream.synchronize()
                    consumed=time.monotonic_ns()
                submit_done=time.monotonic_ns()
                stream.synchronize();joined=time.monotonic_ns()
                if args.kind=='api':hook.sub_probe_scope(0)
                current=layout(y)
                assert current['ptr'] not in (inputs['left']['ptr'],inputs['right']['ptr'])
                assert y.cpu().tolist()==list(range(1,21))
                if result is not None:assert result==list(range(1,21))
                retained.append(y)
                rows.append(dict(index=i,label=label,scope_id=i+1,begin_ns=begin,sub_begin_ns=start,
                    sub_return_ns=returned,consume_return_ns=consumed,submit_done_ns=submit_done,joined_ns=joined,
                    output=current,exact=True))
            if args.kind=='profile':prof.step()
        allocated_retained=torch.npu.memory_allocated()
        last_reference_drop_begin=time.monotonic_ns();retained.clear();del y;gc.collect();stream.synchronize()
        released=time.monotonic_ns();allocated_after=torch.npu.memory_allocated()
        if args.kind=='api':hook.sub_probe_dump(os.fsencode(out/'api_calls.json'))
    env=dict(torch=torch.__version__,torch_git=torch.version.git_version,torch_npu=torch_npu.__version__,
             npu_git=torch_npu.version.git_version,python=os.sys.version,kind=args.kind,mode=args.mode,
             main_tid=threading.get_native_id(),stream_handle=handle,sha256_before=before,
             controls={k:os.environ.get(k) for k in ('TASK_QUEUE_ENABLE','ASCEND_LAUNCH_BLOCKING','LD_PRELOAD')},
             sub_dispatch=torch._C._dispatch_dump_table('aten::sub.Tensor'))
    save(out/'environment.json',env)
    save(out/'observations.json',dict(inputs=inputs,rows=rows,
         lifetime=dict(input_and_outputs_retained_until_join=True,initial_allocated=initial_alloc,
           allocated_with_outputs=allocated_retained,allocated_after_drop=allocated_after,
           last_reference_drop_begin_ns=last_reference_drop_begin,release_joined_ns=released),
         boundary='process-first Sub after NPU init/H2D; caches not cleared; no model workload'))
    if args.kind=='api':
        calls=json.loads((out/'api_calls.json').read_text());paths={r['path'] for r in calls if r['kind']==4 and r['result']>=0}
        artifacts=[]
        for p in sorted(paths):
            f=Path(p)
            if f.is_file():
                row=dict(path=p,size=f.stat().st_size,sha256=sha(f))
                if f.suffix=='.json':row['metadata']=json.loads(f.read_text())
                if f.suffix=='.o':row['file_type']=subprocess.check_output(['file',p],text=True).strip()
                artifacts.append(row)
        save(out/'opened_sub_artifacts.json',artifacts)
    after={str(p):sha(p) for p in tracked};assert before==after
    save(out/'integrity.json',dict(unchanged=True,sha256_after=after))
    save(out/'completed.json',dict(status='passed'))


if __name__=='__main__': main()
