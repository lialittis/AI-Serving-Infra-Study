"""P3a: full pretrained Qwen, explicit independent KV, one CPU submission thread."""
import argparse,datetime,hashlib,inspect,itertools,json,os,platform,subprocess,time,traceback
from contextlib import contextmanager,nullcontext
from pathlib import Path
from model import setup,make_case,prepare,cache_layout,state_hash,describe,MODEL


def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


class Capture:
    def __init__(self,torch):self.torch=torch;self.records=[];self.trial=None
    @contextmanager
    def scope(self,kind,**fields):
        s=self.torch.npu.current_stream()
        r=dict(label=f'P23/{self.trial}/{len(self.records):05}/{kind}',trial=self.trial,
               kind=kind,stream_handle=str(s.npu_stream),logical_stream=s.stream_id,**fields)
        self.records.append(r)
        with self.torch.profiler.record_function(r['label']):yield r
    def event(self,event,action,role):
        with self.scope('event_'+action,role=role) as r:
            getattr(event,action)();r['event_handle']=str(event.npu_event)


def execute(torch,model,jobs,mode,streams,order,capture=None):
    names=['AB'] if mode=='batch' else list(order)
    origin=torch.npu.Event(enable_timing=True)
    ends={n:torch.npu.Event(enable_timing=True) for n in names}
    result={};host_submit={}
    def event(e,action,role):
        if capture:capture.event(e,action,role)
        else:getattr(e,action)()
    torch.npu.reset_peak_memory_stats();allocated=torch.npu.memory_allocated()
    begin=time.perf_counter_ns();event(origin,'record','origin')
    for name in names:
        s=streams['B'] if mode=='parallel' and name=='B' else streams['A']
        with torch.npu.stream(s):
            event(origin,'wait',name+'_ready')
            with capture.scope('forward',task=name) if capture else nullcontext():
                result[name]=model(**jobs[name])
            event(ends[name],'record',name+'_done')
        host_submit[name]=(time.perf_counter_ns()-begin)/1000
    observed={}
    for name in names:
        event(ends[name],'synchronize',name+'_join')
        observed[name]=(time.perf_counter_ns()-begin)/1000
    wall=(time.perf_counter_ns()-begin)/1000
    ready={n:origin.elapsed_time(e)*1000 for n,e in ends.items()}
    metric=dict(wall_us=wall,ready_us={n:ready.get(n,ready.get('AB')) for n in ('A','B')},
        host_observed_us={n:observed.get(n,observed.get('AB')) for n in ('A','B')},
        host_submission_us=host_submit,allocated_before=allocated,
        allocated_peak=torch.npu.max_memory_allocated(),reserved_peak=torch.npu.max_memory_reserved())
    return result,metric


def snapshot(result,mode):
    def one(out,i=None):
        cut=lambda t:t[i:i+1].detach().cpu() if i is not None else t.detach().cpu()
        return dict(logits=cut(out.logits),kv=[(cut(l.keys),cut(l.values)) for l in out.past_key_values.layers])
    return {n:one(result['AB'],i) if mode=='batch' else one(result[n]) for i,n in enumerate(('A','B'))}


def compare(torch,reference,current,mode):
    # BF16 batch shape can change rounding; declared before formal measurement.
    atol,rtol=(.0625,.02) if mode=='batch' else (0,0)
    checks=[]
    for name in ('A','B'):
        a,b=reference[name],current[name];pairs=[(a['logits'],b['logits'])]+[(x,y) for aa,bb in zip(a['kv'],b['kv']) for x,y in zip(aa,bb)]
        checks.append(dict(task=name,tensors=len(pairs),exact=all(torch.equal(x,y) for x,y in pairs),
            max_abs=max((x.float()-y.float()).abs().max().item() for x,y in pairs),
            valid=all(bool(torch.isfinite(y).all()) and torch.allclose(x.float(),y.float(),atol=atol,rtol=rtol) for x,y in pairs),
            greedy_equal=torch.equal(a['logits'].argmax(-1),b['logits'].argmax(-1)),
            next_token=b['logits'].argmax(-1).tolist()))
    return checks


def isolation(torch,jobs):
    layouts={n:dict(input=describe(torch,j['input_ids']),kv=cache_layout(torch,j['past_key_values'])) for n,j in jobs.items()}
    if 'A' in layouts:
        def ranges(n):
            return [(int(v['ptr']),int(v['ptr'])+v['bytes']) for l in layouts[n]['kv'] for k,v in l.items() if k!='layer']
        assert all(a1<=b0 or b1<=a0 for a0,a1 in ranges('A') for b0,b1 in ranges('B')),'KV overlap'
    return layouts


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--rounds',type=int,default=12);p.add_argument('--warmup',type=int,default=3)
    p.add_argument('--profile-repeats',type=int,default=2);a=p.parse_args()
    assert 1<=a.rounds<=24 and 1<=a.warmup<=10 and 1<=a.profile_repeats<=3
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    cases=[('prefill',128),('prefill',1024),('decode',128),('decode',1024)]
    save(out/'plan.json',dict(cases=cases,rounds=a.rounds,warmup=a.warmup,profile_repeats=a.profile_repeats,
        model=MODEL,dtype='bfloat16',batch_atol=.0625,batch_rtol=.02,parallel_atol=0,parallel_rtol=0,
        known_invalid_comparison=['prefill-1024','batch'],
        invalid_policy='Retain failed numeric comparisons; never accept speedup claims for those cells; no tolerance relaxation',
        scope='Full checkpoint HF eager forward; one CPU submission thread; no vLLM scheduler',
        unit='two independent single forwards; decode prefill and all input/KV preparation excluded',
        ready_time='NPU event elapsed from common origin to terminal event; includes submission starvation',
        correctness='all vocabulary logits and all 24 layers KV, every formal and diagnostic pair'))
    def device(name):
        r=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=30,check=True);(out/name).write_text(r.stdout)
    device('device_before.txt')
    try:
        torch,torch_npu,transformers,model,tokenizer=setup()
        from transformers.models.qwen2 import modeling_qwen2
        from transformers import cache_utils,masking_utils
        paths=[Path(m.__file__) for m in (modeling_qwen2,cache_utils,masking_utils)]+[Path(inspect.getfile(torch.npu.Stream))]
        paths+=list(Path(__file__).parent.glob('*.py'))
        sources={}
        for path in paths:
            target=out/'sources'/path.name;target.parent.mkdir(exist_ok=True);target.write_bytes(path.read_bytes())
            sources[str(target.relative_to(out))]=dict(original=str(path),sha256=sha(target))
        save(out/'sources.json',sources)
        (out/'config.json').write_text(model.config.to_json_string())
        streams={n:torch.npu.Stream() for n in ('A','B')}
        env=dict(captured_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),torch=torch.__version__,torch_npu=torch_npu.__version__,
            transformers=transformers.__version__,python=platform.python_version(),device=str(torch.npu.get_device_properties(0)),
            streams={n:dict(id=s.stream_id,handle=str(s.npu_stream)) for n,s in {**streams,'origin':torch.npu.current_stream()}.items()},
            flags={k:os.environ.get(k) for k in ('ASCEND_LAUNCH_BLOCKING','TASK_QUEUE_ENABLE','PYTORCH_NPU_ALLOC_CONF')},
            shared_weight_bytes=sum(t.numel()*t.element_size() for t in model.parameters()),state_sha256=state_hash(torch,model),
            checkpoint={p.name:sha(p) for p in Path(MODEL).glob('*.safetensors')})
        save(out/'environment.json',env)
        checks=[];samples=[];trials=[];layouts=[];baselines={};inputs={}
        permutations=list(itertools.permutations(('serial','parallel','batch')))
        with torch.inference_mode():
            prepared={}
            for phase,length in cases:
                case=make_case(torch,model,tokenizer,phase,length);prepared[case['id']]=case
                inputs[case['id']]={n:dict(tokens=t.cpu().tolist(),sha256=hashlib.sha256(t.cpu().numpy().tobytes()).hexdigest()) for n,t in case['ids'].items()}
                jobs=prepare(torch,model,case,'serial');result,_=execute(torch,model,jobs,'serial',streams,'AB')
                baselines[case['id']]=snapshot(result,'serial');del result,jobs
                for mode in ('serial','parallel','batch'):
                    for _ in range(a.warmup):
                        jobs=prepare(torch,model,case,mode);result,_=execute(torch,model,jobs,mode,streams,'AB');del result,jobs
                for r in range(a.rounds):
                    for mode in permutations[r%6]:
                        jobs=prepare(torch,model,case,mode)
                        result,metric=execute(torch,model,jobs,mode,streams,'AB' if r%2==0 else 'BA')
                        check=compare(torch,baselines[case['id']],snapshot(result,mode),mode)
                        valid=all(c['valid'] and c['greedy_equal'] for c in check)
                        assert valid or (case['id'],mode)==('prefill-1024','batch'),(case['id'],mode,check)
                        checks.append(dict(case=case['id'],round=r,mode=mode,checks=check,stage='performance'))
                        samples.append(dict(case=case['id'],round=r,mode=mode,order='AB' if r%2==0 else 'BA',**metric))
                        if r==0:layouts.append(dict(case=case['id'],mode=mode,when='after_forward',jobs=isolation(torch,jobs)))
                        del result,jobs
                    save(out/'measurements.json',samples);save(out/'correctness.json',checks)
                print('BENCHMARK',case['id'],flush=True)
            save(out/'inputs.json',inputs);save(out/'layouts.json',layouts)
            capture=Capture(torch)
            experimental=torch_npu.profiler._ExperimentalConfig(profiler_level=torch_npu.profiler.ProfilerLevel.Level1)
            with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],
                schedule=torch_npu.profiler.schedule(wait=0,warmup=1,active=1,repeat=1),record_shapes=True,
                experimental_config=experimental,on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(out/'profiler'))) as prof:
                prof.step()
                for case in prepared.values():
                    for r in range(a.profile_repeats):
                        for mode in permutations[r%6]:
                            jobs=prepare(torch,model,case,mode)
                            ident=f"{case['id']}-r{r}-{mode}";capture.trial=ident
                            before=isolation(torch,jobs)
                            result,metric=execute(torch,model,jobs,mode,streams,'AB' if r%2==0 else 'BA',capture)
                            capture.trial=None
                            check=compare(torch,baselines[case['id']],snapshot(result,mode),mode)
                            valid=all(c['valid'] and c['greedy_equal'] for c in check)
                            assert valid or (case['id'],mode)==('prefill-1024','batch'),(case['id'],mode,check)
                            checks.append(dict(case=case['id'],round=r,mode=mode,checks=check,stage='diagnostic'))
                            trials.append(dict(id=ident,case=case['id'],repeat=r,mode=mode,order='AB' if r%2==0 else 'BA',
                                inputs_ready_before_origin=True,inputs=before,outputs=isolation(torch,jobs),correct=valid))
                            del result,jobs
                    print('PROFILE',case['id'],flush=True)
                prof.step()
            save(out/'observations.json',capture.records);save(out/'trials.json',trials);save(out/'correctness.json',checks)
            unchanged=state_hash(torch,model)==env['state_sha256'];save(out/'weight_check.json',dict(unchanged=unchanged));assert unchanged
        save(out/'completed.json',dict(status='captured',all_comparisons_passed=all(c['valid'] and c['greedy_equal'] for r in checks for c in r['checks']),performance_samples=len(samples),diagnostic_trials=len(trials),correctness_checks=len(checks)))
        device('device_after.txt');print('COMPLETED',out,flush=True)
    except Exception:
        (out/'failure.txt').write_text(traceback.format_exc());raise


if __name__=='__main__':main()
