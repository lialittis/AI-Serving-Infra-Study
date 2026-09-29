"""P27 controlled metadata ablation, using unchanged P25 request/stream harness."""
import argparse
import datetime
import importlib.util
import inspect
import itertools
import json
import platform
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from metadata import VARIANTS, checked_attention, prepare, use_variant

HERE = Path(__file__).resolve().parent
P25 = HERE.parent/'practice_25_multimodal_overlap'
sys.path.insert(0, str(P25))
import model as base
spec = importlib.util.spec_from_file_location('p25_run', P25/'run.py')
p25 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p25)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    save = lambda name, data: base.save(out/name, data)
    cases = list(itertools.product(['prefill','decode'], ['beach.jpeg','beijing.jpeg'], [64,256]))
    if args.smoke: cases = cases[:1]
    rounds = 12
    save('plan.json', dict(cases=cases, variants=VARIANTS, rounds=rounds, warmup=3, smoke=args.smoke,
        order='LV/VL alternate; six variant/mode combinations rotate one position every two rounds, balancing all six positions within each order',
        timing='P25 pair boundary; metadata preparation excluded and reported separately',
        precision='bf16', atol=0, rtol=0, cpu_submitters=1))
    (out/'device_before.txt').write_text(subprocess.check_output(['npu-smi','info'],text=True))
    sources={}
    def source(path, name):
        dest=out/'sources'/name; dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
        sources[name]=dict(sha256=base.digest(dest),original=str(path))
    for folder,label in [(HERE,'p27'),(P25,'p25')]:
        for f in sorted(folder.glob('*.py')): source(f,label+'/'+f.name)
    try:
        torch,tn,tr,model,processor=base.setup()
        from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as module
        from transformers import cache_utils
        from transformers.generation import utils as generation_utils
        for m in [module,cache_utils,generation_utils]:source(Path(inspect.getfile(m)),'installed/'+m.__name__+'.py')
        for cls in [type(processor),type(processor.image_processor),torch.npu.Stream,torch.npu.Event]:
            f=Path(inspect.getfile(cls));source(f,'installed/'+f.name)
        attention,patched_source=checked_attention(module)
        (out/'patched_attention.py').write_text(patched_source)
        save('sources.json',sources)
        shutil.copyfile(base.IMAGES/'manifest.json',out/'images.json')
        for item in json.loads((base.IMAGES/'manifest.json').read_text())['images']:
            assert base.digest(base.IMAGES/item['file'])==item['sha256']
        model.config.to_json_file(out/'resolved_config.json')
        state=base.state_hash(torch,model); streams={n:torch.npu.Stream() for n in ['L','V']}
        save('environment.json',dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            torch=torch.__version__,torch_npu=tn.__version__,transformers=tr.__version__,python=platform.python_version(),
            device=str(torch.npu.get_device_properties(0)),checkpoint={f.name:base.digest(f) for f in base.MODEL.glob('*.safetensors')},
            streams={n:dict(id=s.stream_id,handle=str(s.npu_stream)) for n,s in streams.items()},parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()})))
        measures=[];checks=[];equivalence=[];case_records=[];preparations=[]
        with torch.inference_mode():
            for phase,image,budget in cases:
                case=base.make_case(torch,model,processor,image,budget,phase);ident=case['id'];case_records.append(p25.metadata(case));save('cases.json',case_records)
                result,_=base.execute(torch,model,case,'serial',streams);reference=base.result_snapshot(result);del result
                native_checks={n:base.compare(torch,p25.native(torch,model,case[n],decode=(n=='A' and phase=='decode')),reference['outputs'][n]) for n in ['A','B']}
                assert all(c['exact'] and c['finite'] and c['greedy_equal'] for c in native_checks.values())
                equivalence.append(dict(case=ident,checks=native_checks));save('full_vs_split.json',equivalence)
                meta=prepare(torch,model,case['B']);preparations.append(dict(case=ident,grid=meta['grid'],**meta['preparation']));save('preparation.json',preparations)
                def execute(variant,mode,order,observer=None,stage='performance',round=0):
                    with use_variant(base,model,case['B'],variant,meta,attention):
                        result,metric=base.execute(torch,model,case,mode,streams,order,observer)
                    cs=base.check_result(torch,reference,result)
                    checks.append(dict(case=ident,variant=variant,mode=mode,order=order,round=round,stage=stage,checks=cs))
                    return result,metric
                combos=list(itertools.product(VARIANTS,['serial','parallel']))
                for variant,mode in combos:
                    for r in range(1 if args.smoke else 3):
                        result,_=execute(variant,mode,'VL',stage='qualification',round=r);del result
                save('correctness.json',checks)
                print('QUALIFIED',ident,flush=True)
                if args.smoke: continue
                for r in range(rounds):
                    order='LV' if r%2==0 else 'VL';shift=(r//2)%len(combos)
                    for variant,mode in combos[shift:]+combos[:shift]:
                        result,metric=execute(variant,mode,order,round=r);del result
                        measures.append(dict(case=ident,variant=variant,mode=mode,order=order,round=r,**metric))
                    save('measurements.json',measures);save('correctness.json',checks)
                print('BENCHMARK',ident,flush=True)
                for variant in VARIANTS:
                    folder=out/'diagnostic'/variant/ident;folder.mkdir(parents=True)
                    observer=p25.Observer(torch);trials=[]
                    exp=tn.profiler._ExperimentalConfig(profiler_level=tn.profiler.ProfilerLevel.Level1)
                    with tn.profiler.profile(activities=[tn.profiler.ProfilerActivity.CPU,tn.profiler.ProfilerActivity.NPU],
                        schedule=tn.profiler.schedule(wait=0,warmup=1,active=1,repeat=1),record_shapes=True,experimental_config=exp,
                        on_trace_ready=tn.profiler.tensorboard_trace_handler(str(folder/'profiler'))) as prof:
                        prof.step()
                        for r,order in enumerate(['LV','VL']):
                            for mode in (['serial','parallel'] if r==0 else ['parallel','serial']):
                                observer.trial=f'{variant}-{ident}-r{r}-{mode}'
                                result,_=execute(variant,mode,order,observer,stage='diagnostic',round=r)
                                trials.append(dict(id=observer.trial,case=ident,variant=variant,mode=mode,order=order,repeat=r,
                                    features=base.layout(result['features']),outputs={n:dict(logits=base.layout(o.logits),kv=base.cache_layout(o.past_key_values)) for n,o in result['outputs'].items()}))
                                observer.trial=None;del result
                        prof.step()
                    base.save(folder/'observations.json',observer.records);base.save(folder/'trials.json',trials);save('correctness.json',checks)
                    print('PROFILE',variant,ident,flush=True)
                del case,reference,meta;torch.npu.synchronize()
        assert base.state_hash(torch,model)==state and model.model.rope_deltas is None
        save('weight_check.json',dict(unchanged=True,shared_rope_deltas=None))
        save('completed.json',dict(status='passed',smoke=args.smoke,samples=len(measures),numerical_pairs=len(checks),native_cases=len(equivalence)))
        (out/'device_after.txt').write_text(subprocess.check_output(['npu-smi','info'],text=True))
        print('COMPLETED',flush=True)
    except Exception:
        (out/'failure.txt').write_text(traceback.format_exc());raise


if __name__=='__main__':main()
