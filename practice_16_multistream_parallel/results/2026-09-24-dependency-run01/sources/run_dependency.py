"""Three real NPU streams: A produces Y, B consumes Y, C is independent.

Only B.wait_event(A_done) is ablated. Every trial joins A/B/C before reading
results, so an incorrect Z is a device dependency error, not an early CPU read.
"""
import argparse
from contextlib import contextmanager, nullcontext
import datetime
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=7)
    parser.add_argument('--profile-repeats',type=int,default=3)
    parser.add_argument('--a-steps',type=int,default=6)
    parser.add_argument('--c-steps',type=int,default=6)
    args=parser.parse_args()
    if not (1<=args.repeats<=20 and 1<=args.profile_repeats<=5 and 2<=args.a_steps<=16 and 1<=args.c_steps<=16):
        parser.error('repeats 1..20, profile-repeats 1..5, a-steps 2..16, c-steps 1..16')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);(out/'sources').mkdir()
    info=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=30)
    save(out/'device_before.json',dict(exit_code=info.returncode,stdout=info.stdout,stderr=info.stderr))
    import torch
    import torch_npu
    torch.npu.set_device(0)
    torch_npu.npu.config.allow_internal_format=False
    streams={b:torch.npu.Stream() for b in ['A','B','C']}
    done={b:torch.npu.Event() for b in streams}
    resources={
        'X':torch.full((4096,4096),1/64,device='npu',dtype=torch.float16),
        'scratch':torch.empty((4096,4096),device='npu',dtype=torch.float16),
        'Y':torch.empty((4096,4096),device='npu',dtype=torch.float16),
        'Z':torch.empty((4096,4096),device='npu',dtype=torch.float16),
        'V':torch.full((64*1024*1024,),.25,device='npu',dtype=torch.float32),
        'W':torch.empty((64*1024*1024,),device='npu',dtype=torch.float32),
    }
    records=[]
    @contextmanager
    def recorded(trial,branch,name,kind,**extra):
        label='P16Dep/'+trial+'/'+name
        with torch.profiler.record_function(label):
            s=streams[branch]
            r=dict(label=label,trial=trial,branch=branch,name=name,kind=kind,
                   raw_stream_handle=str(s.npu_stream),host_stream_id=str(s.stream_id),
                   host_start_ns=time.monotonic_ns(),**extra)
            records.append(r)
            yield r
            r['host_end_ns']=time.monotonic_ns()

    def trial(mode,ident,profiled=False):
        # Deliberately initialize the shared input to a recognizable OLD value.
        # This and all allocations are outside the measured interval.
        torch.npu.synchronize()
        resources['Y'].fill_(-7);resources['Z'].fill_(-99)
        torch.npu.synchronize()
        def scope(b,name,kind,**extra):
            return recorded(ident,b,name,kind,**extra) if profiled else nullcontext()
        start=time.monotonic_ns()
        with torch.npu.stream(streams['A']):
            # A has real queued work before publishing Y. This enlarges the
            # observable missing-dependency window; it is not a model workload.
            for i in range(args.a_steps-1):
                with scope('A','backlog-%02d'%i,'compute',reads=['X'],writes=['scratch']):
                    torch.mm(resources['X'],resources['X'],out=resources['scratch'])
            with scope('A','produce-Y','compute',reads=['X'],writes=['Y']):
                torch.mm(resources['X'],resources['X'],out=resources['Y'])
            with scope('A','record-A','record') as r:
                done['A'].record(streams['A'])
                if r is not None:r['event_handle']=str(done['A'].npu_event)
        with torch.npu.stream(streams['B']):
            if mode=='event_wait':
                with scope('B','wait-A','device_wait',event_handle=str(done['A'].npu_event)):
                    streams['B'].wait_event(done['A'])
            with scope('B','consume-Y','compute',reads=['Y'],writes=['Z']):
                torch.mul(resources['Y'],2,out=resources['Z'])
            with scope('B','record-B','record') as r:
                done['B'].record(streams['B'])
                if r is not None:r['event_handle']=str(done['B'].npu_event)
        # wait_event returns on the host; C can be submitted while B is waiting.
        with torch.npu.stream(streams['C']):
            for i in range(args.c_steps):
                with scope('C','independent-%02d'%i,'compute',reads=['V'],writes=['W']):
                    torch.mul(resources['V'],1.5,out=resources['W'])
            with scope('C','record-C','record') as r:
                done['C'].record(streams['C'])
                if r is not None:r['event_handle']=str(done['C'].npu_event)
        submitted=time.monotonic_ns()
        for b in streams:
            with scope(b,'join-'+b,'host_wait',event_handle=str(done[b].npu_event)):
                done[b].synchronize()
        finished=time.monotonic_ns()
        # ALL streams have completed. Z will not be repaired by this join.
        with (torch.profiler.record_function('P16Dep/check/'+ident) if profiled else nullcontext()):
            y_ok=bool(torch.all(resources['Y']==1).item())
            c_ok=bool(torch.all(resources['W']==.375).item())
            correct=int(torch.count_nonzero(resources['Z']==2).item())
            old=int(torch.count_nonzero(resources['Z']==-14).item())
            total=resources['Z'].numel()
            samples=resources['Z'].view(-1)[:8].cpu().tolist()
            require_good=y_ok and c_ok and (mode=='no_wait' or correct==total)
            if not require_good:raise RuntimeError('unexpected producer/independent/synchronized output failure')
            # A later re-execution of B can repair Z; merely waiting cannot.
            repaired=None
            if mode=='no_wait':
                with torch.npu.stream(streams['B']):
                    torch.mul(resources['Y'],2,out=resources['Z'])
                    done['B'].record(streams['B'])
                done['B'].synchronize()
                repaired=bool(torch.all(resources['Z']==2).item())
                if not repaired:raise RuntimeError('post-join recomputation failed')
        return dict(id=ident,mode=mode,profiled=profiled,start_ns=start,submitted_ns=submitted,completed_ns=finished,
                    submission_ns=submitted-start,elapsed_ns=finished-start,host_join_calls=3,
                    cross_stream_wait_calls=1 if mode=='event_wait' else 0,
                    Y_correct=y_ok,W_correct=c_ok,Z_elements=total,Z_correct_elements=correct,
                    Z_old_value_elements=old,Z_other_elements=total-correct-old,Z_first_eight=samples,
                    recompute_after_join_correct=repaired)

    for mode in ['event_wait','no_wait']:trial(mode,'warmup-'+mode)
    measurements=[]
    for i in range(args.repeats):
        for mode in (['event_wait','no_wait'] if i%2==0 else ['no_wait','event_wait']):
            measurements.append(trial(mode,'bench-%02d-%s'%(i,mode)))
    profiles=[]
    config=torch_npu.profiler._ExperimentalConfig(profiler_level=torch_npu.profiler.ProfilerLevel.Level1)
    with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],
            schedule=torch_npu.profiler.schedule(wait=0,warmup=1,active=1,repeat=1),record_shapes=True,
            experimental_config=config,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(out/'profiler'))) as p:
        p.step()
        for i in range(args.profile_repeats):
            for mode in (['event_wait','no_wait'] if i%2==0 else ['no_wait','event_wait']):
                profiles.append(trial(mode,'trace-%02d-%s'%(i,mode),True))
        p.step()
    def tensor(t):
        return dict(address=str(t.data_ptr()),bytes=t.numel()*t.element_size(),shape=list(t.shape),
                    stride=list(t.stride()),dtype=str(t.dtype),device=str(t.device))
    sources={}
    for name,path in [('run_dependency.py',Path(__file__)),('streams.py',Path(inspect.getfile(torch_npu.npu.Stream)))]:
        content=path.read_bytes();(out/'sources'/name).write_bytes(content)
        sources['sources/'+name]=hashlib.sha256(content).hexdigest()
    save(out/'dependency_run.json',dict(
        captured_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),command=[sys.executable]+sys.argv,
        torch_version=torch.__version__,torch_npu_version=torch_npu.__version__,device=torch.npu.get_device_name(0),
        python=sys.version,pid=os.getpid(),a_steps=args.a_steps,c_steps=args.c_steps,repeats=args.repeats,
        profile_repeats=args.profile_repeats,resources={k:tensor(v) for k,v in resources.items()},
        environment={k:os.environ.get(k) for k in ['ASCEND_HOME_PATH','TASK_QUEUE_ENABLE','ASCEND_LAUNCH_BLOCKING']},
        measurements=measurements,profiles=profiles,records=records,source_sha256=sources,
        limits=['A has explicit queued matrix work to expose the missing-dependency window.',
                'Only B wait_event(A_done) removed; initialization and final joins always retained.',
                'No early CPU reads, no deallocation/reuse while kernels are pending.',
                'Incorrect-path elapsed time is not a valid optimization result.']))
    print('Captured:',out,flush=True)


if __name__=='__main__':main()
