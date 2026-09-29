"""Full-FP32 eager/replay factorial comparison with fixed-step live graph inputs."""
import argparse,datetime,hashlib,inspect,platform,subprocess,time,traceback
from pathlib import Path
from contextlib import contextmanager
from harness import setup,make_case,variant,Captured,run_pair,snapshot,compare,save,state_hash,ROOT


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


class Observer:
    def __init__(self,torch):self.torch=torch;self.records=[];self.trial=None
    @contextmanager
    def scope(self,kind,**kw):
        s=self.torch.npu.current_stream();r=dict(label=f'P24/{self.trial}/{len(self.records):05}/{kind}',trial=self.trial,
            kind=kind,stream_handle=str(s.npu_stream),logical_stream=s.stream_id,**kw);self.records.append(r)
        with self.torch.profiler.record_function(r['label']):yield r
    def event(self,event,action,role):
        with self.scope('event_'+action,role=role) as r:
            getattr(event,action)();r['event_handle']=str(event.npu_event)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    dumps=out/'dumps';dumps.mkdir();rounds=12;repeats=2;cases=[('prefill',128),('prefill',1024),('decode',128),('decode',1024)]
    configs=[(b,m) for m in ('serial','parallel','batch') for b in ('eager','graph')]
    save(out/'plan.json',dict(cases=cases,rounds=rounds,profile_repeats=repeats,warmup=3,precision='full_fp32',hf32=False,
        cross_mode_atol=.0625,cross_mode_rtol=.02,same_shape_atol=0,same_shape_rtol=0,
        modes=configs,order='six rotations; reverse second cycle; AB/BA alternate; payload swap every two rounds',
        unit='two independent fixed-step forwards; graph captures fixed inputs and initial KV addresses; not growing autoregressive generation',
        allocation='each graph has a distinct pool and keeps input KV tensors alive after DynamicCache replaces layer fields',
        timing='exclude loading/reset/capture/validation; include origin/waits/submission/terminal completion; capture and first replay separate',
        scope='HF checkpoint full forward harness; no vLLM scheduler or compilation partitions'))
    def device(name):(out/name).write_text(subprocess.check_output(['npu-smi','info'],text=True))
    device('device_before.txt')
    try:
        torch,tn,tr,model,tok=setup()
        from transformers.models.qwen2 import modeling_qwen2
        from transformers import cache_utils
        paths=[Path(m.__file__) for m in (modeling_qwen2,cache_utils)]+[Path(inspect.getfile(torch.npu.NPUGraph)),Path(inspect.getfile(torch.npu.Stream))]
        paths+=list(Path(__file__).parent.glob('*.py'))+[ROOT/'practice_23_independent_inference'/f for f in ('model.py','run_experiment.py')]
        sources={}
        for f in paths:
            prefix='p23-' if f.parent.name=='practice_23_independent_inference' else ''
            target=out/'sources'/(prefix+f.name);target.parent.mkdir(exist_ok=True);target.write_bytes(f.read_bytes())
            sources[target.name]=dict(original=str(f),sha256=sha(target))
        save(out/'sources.json',sources)
        from model import MODEL
        env=dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),torch=torch.__version__,torch_npu=tn.__version__,transformers=tr.__version__,
            python=platform.python_version(),device=str(torch.npu.get_device_properties(0)),parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}),
            hf32=torch.npu.matmul.allow_hf32,checkpoint={p.name:sha(p) for p in Path(MODEL).glob('*.safetensors')},state_sha256=state_hash(torch,model))
        streams={n:torch.npu.Stream() for n in ('A','B')}
        env['streams']={n:dict(id=s.stream_id,handle=str(s.npu_stream)) for n,s in {**streams,'origin':torch.npu.current_stream()}.items()}
        save(out/'environment.json',env)
        all_cases={};all_captures={};baselines={};captures_meta=[];checks=[];samples=[];trials=[];stress=[]
        with torch.inference_mode():
            for phase,length in cases:
                case=make_case(torch,model,tok,phase,length);all_cases[case['id']]=case
                for swap in (False,True):
                    result,_=run_pair(torch,model,variant(case,swap),'serial','eager',streams,{},'AB')
                    baselines[case['id'],swap]=snapshot(result,'serial');del result
                captures={n:Captured(torch,model,case,n,streams['B'] if n=='B' else streams['A'],dumps) for n in ('A','B','AB')}
                all_captures[case['id']]=captures;captures_meta.extend(c.metadata for c in captures.values());save(out/'captures.json',captures_meta)
                assert len({tuple(m['pool']) for m in captures_meta})==len(captures_meta),'shared graph pools'
                # Alternate input contents and initial KV twice without recapture.
                for i in range(6):
                    swap=bool(i%2);mode=('serial','parallel','batch')[i%3]
                    result,_=run_pair(torch,model,variant(case,swap),mode,'graph',streams,captures,'AB' if i%2==0 else 'BA')
                    cs=compare(torch,baselines[case['id'],swap],snapshot(result,mode),mode)
                    stress.append(dict(case=case['id'],iteration=i,mode=mode,swap=swap,checks=cs))
                    assert all(c['valid'] and c['greedy_equal'] for c in cs),stress[-1]
                save(out/'lifetime_checks.json',stress)
                for backend,mode in configs:
                    for _ in range(3):result,_=run_pair(torch,model,case,mode,backend,streams,captures)
                for r in range(rounds):
                    offset=r%6;order_configs=configs[offset:]+configs[:offset]
                    if r>=6:order_configs=list(reversed(order_configs))
                    swap=bool((r//2)%2);current=variant(case,swap);order='AB' if r%2==0 else 'BA'
                    for backend,mode in order_configs:
                        result,metric=run_pair(torch,model,current,mode,backend,streams,captures,order)
                        cs=compare(torch,baselines[case['id'],swap],snapshot(result,mode),mode)
                        checks.append(dict(case=case['id'],round=r,backend=backend,mode=mode,swap=swap,checks=cs,stage='performance'))
                        assert all(c['valid'] and c['greedy_equal'] for c in cs),checks[-1]
                        samples.append(dict(case=case['id'],round=r,backend=backend,mode=mode,swap=swap,order=order,**metric));del result
                    save(out/'measurements.json',samples);save(out/'correctness.json',checks)
                print('BENCHMARK',case['id'],flush=True)
            observer=Observer(torch)
            exp=tn.profiler._ExperimentalConfig(profiler_level=tn.profiler.ProfilerLevel.Level1)
            with tn.profiler.profile(activities=[tn.profiler.ProfilerActivity.CPU,tn.profiler.ProfilerActivity.NPU],
                schedule=tn.profiler.schedule(wait=0,warmup=1,active=1,repeat=1),record_shapes=True,experimental_config=exp,
                on_trace_ready=tn.profiler.tensorboard_trace_handler(str(out/'profiler'))) as prof:
                prof.step()
                for case in all_cases.values():
                    for r in range(repeats):
                        for backend,mode in (configs if r==0 else list(reversed(configs))):
                            ident=f'{case["id"]}-r{r}-{backend}-{mode}';observer.trial=ident;swap=bool(r%2);order='AB' if r%2==0 else 'BA'
                            result,_=run_pair(torch,model,variant(case,swap),mode,backend,streams,all_captures[case['id']],order,observer)
                            observer.trial=None;cs=compare(torch,baselines[case['id'],swap],snapshot(result,mode),mode)
                            checks.append(dict(case=case['id'],round=r,backend=backend,mode=mode,swap=swap,checks=cs,stage='diagnostic'))
                            assert all(c['valid'] and c['greedy_equal'] for c in cs),checks[-1]
                            trials.append(dict(id=ident,case=case['id'],repeat=r,backend=backend,mode=mode,swap=swap,order=order))
                    print('PROFILE',case['id'],flush=True)
                prof.step()
            save(out/'observations.json',observer.records);save(out/'trials.json',trials);save(out/'correctness.json',checks)
            unchanged=state_hash(torch,model)==env['state_sha256'];assert unchanged
            save(out/'weight_check.json',dict(unchanged=unchanged))
            # The same output tensor addresses and cache lengths survive all replays.
            for case_id,captures in all_captures.items():
                for c in captures.values():
                    assert c.output.logits.data_ptr()==int(c.metadata['output_logits']['ptr'])
                    expected=all_cases[case_id]['length']+(1 if all_cases[case_id]['phase']=='decode' else 0)
                    assert all(layer.keys.shape[-2]==expected for layer in c.output.past_key_values.layers)
        save(out/'completed.json',dict(status='passed',samples=len(samples),diagnostic_trials=len(trials),captures=len(captures_meta),lifetime_checks=len(stress)))
        device('device_after.txt');print('COMPLETED',out,flush=True)
    except Exception:
        (out/'failure.txt').write_text(traceback.format_exc());raise


if __name__=='__main__':main()
