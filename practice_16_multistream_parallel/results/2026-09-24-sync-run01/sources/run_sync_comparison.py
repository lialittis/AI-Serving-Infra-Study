"""Compare per-round waits, one final join, and deliberately premature CPU reads.

Performance runs have no profiler or early reads. A separate diagnostic keeps
all memory alive, reads pending pinned buffers, then always drains both streams.
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


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--rounds',type=int,default=4)
    parser.add_argument('--pairs',type=int,default=6)
    parser.add_argument('--repeats',type=int,default=7)
    args=parser.parse_args()
    if not (1<=args.rounds<=8 and 1<=args.pairs<=16 and 1<=args.repeats<=20):
        parser.error('rounds 1..8, pairs 1..16, repeats 1..20')
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    (output/'sources').mkdir()
    result=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=30)
    save(output/'device_before.json',dict(exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr))
    import torch
    import torch_npu
    torch.npu.set_device(0)
    torch_npu.npu.config.allow_internal_format=False
    streams={b:torch.npu.Stream() for b in ['mm','mul']}
    inputs={'mm':torch.full((4096,4096),1/64,device='npu',dtype=torch.float16),
            'mul':torch.full((64*1024*1024,),.25,device='npu',dtype=torch.float32)}
    # Each round owns its output and pinned sample: no round can overwrite a
    # previous round's result, even if the host submits all rounds at once.
    outputs=[{b:torch.empty_like(t) for b,t in inputs.items()} for _ in range(args.rounds)]
    samples=[{b:torch.empty(4,dtype=t.dtype,pin_memory=True) for b,t in inputs.items()}
             for _ in range(args.rounds)]
    events=[{b:torch.npu.Event() for b in streams} for _ in range(args.rounds)]
    torch.npu.synchronize()
    records=[]
    @contextmanager
    def recorded(label,branch,kind,**extra):
        with torch.profiler.record_function(label):
            s=streams[branch]
            row=dict(label=label,branch=branch,kind=kind,raw_stream_handle=str(s.npu_stream),
                     host_stream_id=str(s.stream_id),host_start_ns=time.monotonic_ns(),**extra)
            records.append(row)
            yield row
            row['host_end_ns']=time.monotonic_ns()

    def batch(mode,ident,profiled=False):
        # The global fence is before the timer, not between rounds. All prior
        # batches/validation finish before buffers are initialized/reused.
        torch.npu.synchronize()
        for pair in samples:
            for sample in pair.values(): sample.fill_(-7)
        early=[]
        api_waits=0
        def scope(round_id,branch,kind,index=None):
            label='P16Sync/%s/r%d/%s-%s'%(ident,round_id,kind,branch)
            if index is not None: label+='-%02d'%index
            return recorded(label,branch,kind,batch=ident,round=round_id) if profiled else nullcontext()
        def wait_round(r):
            nonlocal api_waits
            for b in streams:
                with scope(r,b,'host_wait'):
                    events[r][b].synchronize()
                api_waits+=1
        begin=time.monotonic_ns()
        for r in range(args.rounds):
            for k in range(args.pairs):
                with torch.npu.stream(streams['mm']):
                    with scope(r,'mm','compute',k):
                        torch.mm(inputs['mm'],inputs['mm'],out=outputs[r]['mm'])
                with torch.npu.stream(streams['mul']):
                    with scope(r,'mul','compute',k):
                        torch.mul(inputs['mul'],1.5,out=outputs[r]['mul'])
            # The D2H copy follows its producer on the SAME stream in all modes.
            # Removing a host wait must not accidentally remove this dependency.
            for b,s in streams.items():
                with torch.npu.stream(s):
                    with scope(r,b,'copy'):
                        samples[r][b].copy_(outputs[r][b].view(-1)[:4],non_blocking=True)
                    with scope(r,b,'record') as rec:
                        events[r][b].record(s)
                        if rec is not None: rec['event_handle']=str(events[r][b].npu_event)
            if mode=='per_round':
                wait_round(r)
            elif mode=='early_read':
                # Intentional missing CPU-consumer wait. Reading a pinned CPU
                # tensor does NOT implicitly wait for the outstanding NPU copy.
                start=time.monotonic_ns()
                observed={b:samples[r][b].tolist() for b in streams}
                stop=time.monotonic_ns()
                ready_after={b:events[r][b].query() for b in streams}
                early.append(dict(round=r,observed=observed,event_ready_after_read=ready_after,
                                  read_start_ns=start,read_end_ns=stop))
        returned=time.monotonic_ns()
        if mode!='per_round':
            wait_round(args.rounds-1)  # Includes all earlier rounds on each stream.
        finished=time.monotonic_ns()
        # No .cpu(), .item(), output comparison or profiler stop before this join.
        expected={'mm':1.0,'mul':.375}
        checks=[]
        for r in range(args.rounds):
            check=dict(round=r,samples_after_join={b:samples[r][b].tolist() for b in streams},
                       full_output_correct={b:bool(torch.all(outputs[r][b]==expected[b]).item()) for b in streams})
            checks.append(check)
            if not all(check['full_output_correct'].values()) or any(
                    check['samples_after_join'][b]!=[expected[b]]*4 for b in streams):
                raise RuntimeError('final result incorrect: '+str(check))
        return dict(id=ident,mode=mode,profiled=profiled,rounds=args.rounds,pairs=args.pairs,
                    begin_ns=begin,return_ns=returned,complete_ns=finished,
                    issue_phase_ns=returned-begin,completed_ns=finished-begin,
                    terminal_wait_ns=finished-returned,host_wait_calls=api_waits,
                    early_reads=early,checks=checks)

    # Warm exact ops, pin-copy path, event objects and full-output checks.
    batch('per_round','warmup-per-round')
    batch('final_only','warmup-final-only')
    performance=[]
    for i in range(args.repeats):
        for mode in (['per_round','final_only'] if i%2==0 else ['final_only','per_round']):
            performance.append(batch(mode,'bench-%02d-%s'%(i,mode)))
    diagnostics=[batch('early_read','diagnostic-%02d'%i) for i in range(args.repeats)]
    config=torch_npu.profiler._ExperimentalConfig(profiler_level=torch_npu.profiler.ProfilerLevel.Level1)
    profiles=[]
    with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],
            schedule=torch_npu.profiler.schedule(wait=0,warmup=1,active=1,repeat=1),record_shapes=True,
            experimental_config=config,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(output/'profiler'))) as prof:
        prof.step()
        for mode in ['per_round','final_only','early_read']:
            profiles.append(batch(mode,'trace-'+mode,True))
        prof.step()
    sources={}
    for name,path in [('run_sync_comparison.py',Path(__file__)),('streams.py',Path(inspect.getfile(torch_npu.npu.Stream)))]:
        data=path.read_bytes();(output/'sources'/name).write_bytes(data)
        sources['sources/'+name]=hashlib.sha256(data).hexdigest()
    def tensor(t):
        return dict(address=str(t.data_ptr()),bytes=t.numel()*t.element_size(),shape=list(t.shape),dtype=str(t.dtype),
                    device=str(t.device),pinned=t.is_pinned() if t.device.type=='cpu' else False)
    save(output/'sync_run.json',dict(
        captured_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),command=[sys.executable]+sys.argv,
        torch_version=torch.__version__,torch_npu_version=torch_npu.__version__,python=sys.version,
        device=torch.npu.get_device_name(0),pid=os.getpid(),rounds=args.rounds,pairs=args.pairs,repeats=args.repeats,
        environment={k:os.environ.get(k) for k in ['ASCEND_HOME_PATH','TASK_QUEUE_ENABLE','ASCEND_LAUNCH_BLOCKING','PYTORCH_NPU_ALLOC_CONF']},
        inputs={b:tensor(t) for b,t in inputs.items()},outputs=[{b:tensor(t) for b,t in pair.items()} for pair in outputs],
        samples=[{b:tensor(t) for b,t in pair.items()} for pair in samples],
        source_sha256=sources,performance=performance,diagnostics=diagnostics,profiles=profiles,records=records,
        limits=['Only per-round host waits removed; initialization fence and stream ordering retained.',
                'Every batch eventually waits for both streams before validation or storage reuse.',
                'Early reads are deliberately invalid CPU consumption, not a claim of NPU arithmetic error.',
                'Performance includes same per-round D2H samples/events in both modes, excludes validation.']))
    print('Captured:',output,flush=True)


if __name__=='__main__':
    main()
