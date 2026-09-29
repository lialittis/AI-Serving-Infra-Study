"""P3b: two independent multimodal request stages, BF16 eager, one submitter."""
import argparse,datetime,inspect,os,platform,shutil,subprocess,traceback
from contextlib import contextmanager
from model import *


class Observer:
    def __init__(self,torch):self.torch=torch;self.records=[];self.trial=None
    @contextmanager
    def scope(self,kind,**kw):
        s=self.torch.npu.current_stream();r=dict(label=f'P25/{self.trial}/{len(self.records):05}/{kind}',trial=self.trial,
            kind=kind,stream_handle=str(s.npu_stream),logical_stream=s.stream_id,**kw);self.records.append(r)
        with self.torch.profiler.record_function(r['label']):yield r
    def event(self,event,action,role):
        with self.scope('event_'+action,role=role) as r:getattr(event,action)();r['event_handle']=str(event.npu_event)


def native(torch,model,request,decode=False):
    model.model.rope_deltas=None
    y=model(input_ids=request['ids'],pixel_values=request['pixel_values'],image_grid_thw=request['grid'].to('npu'),
        mm_token_type_ids=request['mm_types'],attention_mask=request['attention'],use_cache=True,logits_to_keep=1)
    if decode:
        token=y.logits[:,-1].argmax(-1,keepdim=True)
        # generate supplies cached-input/position slicing required by this
        # installed version; a raw forward with the full mask is not equivalent.
        generated=model.generate(input_ids=torch.cat([request['ids'],token],1),past_key_values=y.past_key_values,
            attention_mask=torch.ones((1,request['ids'].shape[1]+1),dtype=torch.long,device='npu'),
            max_new_tokens=1,do_sample=False,use_cache=True,return_dict_in_generate=True,output_logits=True)
        y=SimpleNamespace(logits=generated.logits[0][:,None,:],past_key_values=generated.past_key_values)
    torch.npu.synchronize();result=snapshot(y);model.model.rope_deltas=None;return result


def metadata(case):
    def req(r):return dict(image=r['image'],budget=r['budget'],grid=r['grid'].tolist(),visual_tokens=int(r['grid'].prod()//4),
        language_tokens=r['ids'].shape[1],pixel_values=layout(r['pixel_values']),input_ids=r['ids'].cpu().tolist(),position_ids=r['position'].cpu().tolist(),rope_delta=r['delta'].tolist())
    return dict(id=case['id'],phase=case['phase'],A=req(case['A']),B=req(case['B']),
        A_embeddings=layout(case['A_embeddings']),A_base_kv=[dict(key=layout(k),value=layout(v)) for k,v in (case['A_base'] or [])])


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    rounds=12;warmup=3;repeats=2;cases=[(phase,image,budget) for phase in ['prefill','decode'] for image in ['beach.jpeg','beijing.jpeg'] for budget in [64,256]]
    save(out/'plan.json',dict(cases=cases,rounds=rounds,warmup=warmup,profile_repeats=repeats,precision='bf16',attention='eager',atol=0,rtol=0,
        A_context_tokens=512,A_image_budget=64,modes=['serial','parallel'],order='LV/VL alternate; mode order changes every two rounds; each combination repeats three times',
        timing='include A language + B vision + B merge/language and all terminal joins; exclude image preprocessing, A vision/prefix, input transfer and fresh KV preparation',
        scope='full pretrained HF stage harness; not vLLM scheduling, graph replay or full autoregressive requests',
        command=['python','practice_25_multimodal_overlap/run.py','--output',str(out)],device='container logical 0, physical NPU 5; no ASCEND_RT_VISIBLE_DEVICES override'))
    (out/'device_before.txt').write_text(subprocess.check_output(['npu-smi','info'],text=True))
    sources={};source_dir=out/'sources';source_dir.mkdir()
    for f in Path(__file__).parent.glob('*.py'):
        shutil.copyfile(f,source_dir/f.name);sources[f.name]=dict(original=str(f),sha256=digest(f))
    try:
        torch,tn,tr,model,processor=setup()
        from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl
        from transformers import cache_utils
        for m in [modeling_qwen2_5_vl,cache_utils]:
            f=Path(m.__file__);shutil.copyfile(f,source_dir/f.name);sources[f.name]=dict(original=str(f),sha256=digest(f))
        for cls in [type(processor),type(processor.image_processor),torch.npu.Stream,torch.npu.Event]:
            f=Path(inspect.getfile(cls));shutil.copyfile(f,source_dir/f.name);sources[f.name]=dict(original=str(f),sha256=digest(f))
        from transformers.generation import utils as generation_utils
        f=Path(generation_utils.__file__);shutil.copyfile(f,source_dir/'generation_utils.py');sources['generation_utils.py']=dict(original=str(f),sha256=digest(f))
        save(out/'sources.json',sources)
        config_dir=out/'model_config';config_dir.mkdir()
        for name in ['config.json','preprocessor_config.json','tokenizer_config.json','chat_template.json','generation_config.json','model.safetensors.index.json']:
            if (MODEL/name).exists():shutil.copyfile(MODEL/name,config_dir/name)
        model.config.to_json_file(out/'resolved_config.json')
        shutil.copyfile(IMAGES/'manifest.json',out/'images.json')
        for item in json.loads((IMAGES/'manifest.json').read_text())['images']:
            assert digest(IMAGES/item['file'])==item['sha256']
        before=state_hash(torch,model);streams={n:torch.npu.Stream() for n in ['L','V']}
        env=dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),torch=torch.__version__,torch_npu=tn.__version__,transformers=tr.__version__,python=platform.python_version(),
            device=str(torch.npu.get_device_properties(0)),checkpoint={f.name:digest(f) for f in MODEL.glob('*.safetensors')},state_sha256=before,
            streams={n:dict(id=s.stream_id,handle=str(s.npu_stream)) for n,s in {**streams,'origin':torch.npu.current_stream()}.items()},
            parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}),disk_free_bytes=shutil.disk_usage(out).free)
        save(out/'environment.json',env);samples=[];checks=[];equivalence=[];all_metadata=[]
        with torch.inference_mode():
            for phase,image,budget in cases:
                case=make_case(torch,model,processor,image,budget,phase);meta=metadata(case);all_metadata.append(meta);save(out/'cases.json',all_metadata)
                case_out=out/'diagnostic'/case['id'];case_out.mkdir(parents=True)
                result,_=execute(torch,model,case,'serial',streams);reference=result_snapshot(result);del result
                native_checks={n:compare(torch,native(torch,model,case[n],decode=(n=='A' and phase=='decode')),reference['outputs'][n]) for n in ['A','B']}
                assert all(c['exact'] and c['finite'] and c['greedy_equal'] for c in native_checks.values()),native_checks
                equivalence.append(dict(case=case['id'],checks=native_checks));save(out/'full_vs_split.json',equivalence)
                for mode in ['serial','parallel']:
                    for _ in range(warmup):result,_=execute(torch,model,case,mode,streams)
                del result
                for r in range(rounds):
                    order='LV' if r%2==0 else 'VL';modes=['serial','parallel'] if (r//2)%2==0 else ['parallel','serial']
                    for mode in modes:
                        result,metric=execute(torch,model,case,mode,streams,order);cs=check_result(torch,reference,result)
                        samples.append(dict(case=case['id'],round=r,mode=mode,order=order,**metric));checks.append(dict(case=case['id'],round=r,mode=mode,stage='performance',checks=cs));del result
                    save(out/'measurements.json',samples);save(out/'correctness.json',checks)
                print('BENCHMARK',case['id'],flush=True)
                observer=Observer(torch);trials=[]
                exp=tn.profiler._ExperimentalConfig(profiler_level=tn.profiler.ProfilerLevel.Level1)
                with tn.profiler.profile(activities=[tn.profiler.ProfilerActivity.CPU,tn.profiler.ProfilerActivity.NPU],
                    schedule=tn.profiler.schedule(wait=0,warmup=1,active=1,repeat=1),record_shapes=True,experimental_config=exp,
                    on_trace_ready=tn.profiler.tensorboard_trace_handler(str(case_out/'profiler'))) as prof:
                    prof.step()
                    for r in range(repeats):
                        for mode in (['serial','parallel'] if r==0 else ['parallel','serial']):
                            ident=f'{case["id"]}-r{r}-{mode}';observer.trial=ident;order='LV' if r==0 else 'VL'
                            result,_=execute(torch,model,case,mode,streams,order,observer);observer.trial=None
                            cs=check_result(torch,reference,result)
                            checks.append(dict(case=case['id'],round=r,mode=mode,stage='diagnostic',checks=cs))
                            trials.append(dict(id=ident,case=case['id'],mode=mode,order=order,repeat=r,features=layout(result['features']),
                                outputs={n:dict(logits=layout(o.logits),kv=cache_layout(o.past_key_values)) for n,o in result['outputs'].items()}));del result
                    prof.step()
                save(case_out/'observations.json',observer.records);save(case_out/'trials.json',trials);save(out/'correctness.json',checks)
                print('PROFILE',case['id'],flush=True)
                del case,reference;torch.npu.synchronize()
        assert state_hash(torch,model)==before;assert model.model.rope_deltas is None
        save(out/'weight_check.json',dict(unchanged=True,shared_rope_deltas=None))
        (out/'device_after.txt').write_text(subprocess.check_output(['npu-smi','info'],text=True))
        save(out/'completed.json',dict(status='passed',samples=len(samples),numerical_pairs=len(checks),diagnostic_trials=len(cases)*repeats*2,full_vs_split=len(equivalence)))
        print('COMPLETED',out,flush=True)
    except Exception:
        (out/'failure.txt').write_text(traceback.format_exc());raise


if __name__=='__main__':main()
