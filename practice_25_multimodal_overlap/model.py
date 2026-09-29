"""Real Qwen2.5-VL stages with request-local MRoPE, KV and explicit handoff."""
import hashlib,json,time
from pathlib import Path
from types import SimpleNamespace
MODEL=Path('/data/huggingface_home/hub/Qwen2.5-VL-3B-Instruct')
IMAGES=Path('/data/tianchi/datasets/p3b_images')


def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def setup():
    import torch,torch_npu,transformers
    from transformers import Qwen2_5_VLForConditionalGeneration,AutoProcessor
    torch.npu.set_device(0);torch.set_num_threads(4)
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL,dtype=torch.bfloat16,attn_implementation='eager',local_files_only=True).eval().to('npu')
    model.requires_grad_(False)
    assert model.config.text_config.rope_parameters['rope_type']=='default'
    assert not model.config.text_config.use_sliding_window
    processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True)
    torch.npu.synchronize();return torch,torch_npu,transformers,model,processor


def state_hash(torch,model):
    return {n:hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() for n,t in list(model.named_parameters())+list(model.named_buffers())}


def layout(t):
    return dict(ptr=str(t.data_ptr()),storage=str(t.untyped_storage().data_ptr()),shape=list(t.shape),stride=list(t.stride()),bytes=t.numel()*t.element_size(),dtype=str(t.dtype))


def cache_layout(cache):return [dict(key=layout(l.keys),value=layout(l.values)) for l in cache.layers]


def prepare_request(torch,model,processor,image,budget,language_length=None):
    from PIL import Image
    im=Image.open(IMAGES/image).convert('RGB')
    messages=[{'role':'user','content':[{'type':'image'},{'type':'text','text':'Describe the main subjects and the setting of this image.'}]}]
    text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    cpu=processor(text=[text],images=[im],min_pixels=budget*28*28,max_pixels=budget*28*28,return_tensors='pt')
    if 'mm_token_type_ids' not in cpu:cpu['mm_token_type_ids']=(cpu['input_ids']==model.config.image_token_id).long()
    if language_length:
        # Extend the textual user prompt before its closing role delimiter.
        ids=cpu['input_ids'];types=cpu['mm_token_type_ids'];need=language_length-ids.shape[1];assert need>=0
        words=processor.tokenizer.encode(' Describe colors, objects and their spatial relationships.',add_special_tokens=False)
        extra=torch.tensor((words*((need+len(words)-1)//len(words)))[:need],dtype=torch.long)[None]
        ending=processor.tokenizer.convert_tokens_to_ids('<|im_end|>');cut=int((ids[0]==ending).nonzero()[-1])
        cpu['input_ids']=torch.cat([ids[:,:cut],extra,ids[:,cut:]],1)
        cpu['mm_token_type_ids']=torch.cat([types[:,:cut],torch.zeros_like(extra),types[:,cut:]],1)
        cpu['attention_mask']=torch.ones_like(cpu['input_ids'])
    # Pure metadata work stays on CPU; no shared model.rope_deltas mutation.
    position,delta=model.model.get_rope_index(cpu['input_ids'],cpu['mm_token_type_ids'],image_grid_thw=cpu['image_grid_thw'],attention_mask=cpu['attention_mask'])
    n=cpu['input_ids'].shape[1];image_mask=(cpu['input_ids']==model.config.image_token_id).unsqueeze(-1).expand(-1,-1,model.config.text_config.hidden_size).to('npu')
    mask=torch.full((n,n),torch.finfo(torch.bfloat16).min,dtype=torch.bfloat16).triu(1)[None,None].to('npu')
    d=dict(image=image,budget=budget,ids=cpu['input_ids'].to('npu'),pixel_values=cpu['pixel_values'].to(device='npu',dtype=torch.bfloat16),
        grid=cpu['image_grid_thw'],position=position.to('npu'),delta=delta,mask=mask,image_mask=image_mask,
        mm_types=cpu['mm_token_type_ids'].to('npu'),attention=cpu['attention_mask'].to('npu'))
    torch.npu.synchronize();return d


def vision(model,request):return model.get_image_features(request['pixel_values'],request['grid']).pooler_output[0]


def merge(model,request,features):
    return model.get_input_embeddings()(request['ids']).masked_scatter(request['image_mask'],features)


def new_cache(model,base=None):
    from transformers import DynamicCache
    return DynamicCache(config=model.config.text_config) if base is None else DynamicCache(ddp_cache_data=[(k.clone(),v.clone()) for k,v in base],config=model.config.text_config)


def language(model,embeddings,position,mask,cache):
    out=model.model.language_model(inputs_embeds=embeddings,position_ids=position,attention_mask={'full_attention':mask},past_key_values=cache,use_cache=True)
    return SimpleNamespace(logits=model.lm_head(out.last_hidden_state[:,-1:]),past_key_values=out.past_key_values)


def snapshot(out):return dict(logits=out.logits.detach().cpu(),kv=[(l.keys.detach().cpu(),l.values.detach().cpu()) for l in out.past_key_values.layers])


def compare(torch,a,b):
    pairs=[(a['logits'],b['logits'])]+[(x,y) for aa,bb in zip(a['kv'],b['kv']) for x,y in zip(aa,bb)]
    assert len(a['kv'])==len(b['kv'])==36
    return dict(tensors=len(pairs),exact=all(torch.equal(x,y) for x,y in pairs),finite=all(bool(torch.isfinite(y).all()) for x,y in pairs),
        max_abs=max((x.float()-y.float()).abs().max().item() for x,y in pairs),greedy_equal=torch.equal(a['logits'].argmax(-1),b['logits'].argmax(-1)))


def make_case(torch,model,processor,image,budget,phase):
    other='beijing.jpeg' if image=='beach.jpeg' else 'beach.jpeg'
    a=prepare_request(torch,model,processor,other,64,512);b=prepare_request(torch,model,processor,image,budget)
    fa=vision(model,a);ea=merge(model,a,fa);torch.npu.synchronize();base=None
    if phase=='decode':
        out=language(model,ea,a['position'],a['mask'],new_cache(model));torch.npu.synchronize()
        base=[(l.keys,l.values) for l in out.past_key_values.layers]
        next_id=out.logits[:,-1].argmax(-1,keepdim=True);ea=model.get_input_embeddings()(next_id)
        position=(a['ids'].shape[1]+a['delta']).view(1,1,1).expand(3,1,1).to('npu')
        mask=torch.zeros((1,1,1,a['ids'].shape[1]+1),dtype=torch.bfloat16,device='npu')
    else:position=a['position'];mask=a['mask']
    torch.npu.synchronize()
    return dict(id=f'{phase}-{image.split(".")[0]}-v{budget}',phase=phase,A=a,B=b,A_embeddings=ea,A_position=position,A_mask=mask,A_base=base)


def execute(torch,model,case,mode,streams,order='LV',observer=None):
    from contextlib import nullcontext
    ca=new_cache(model,case['A_base']);cb=new_cache(model);torch.npu.synchronize()
    def scope(kind,**kw):return observer.scope(kind,**kw) if observer else nullcontext()
    def event(e,action,role):
        if observer:observer.event(e,action,role)
        else:getattr(e,action)()
    origin=torch.npu.Event(enable_timing=True);ends={n:torch.npu.Event(enable_timing=True) for n in ['A','V','B']}
    outputs={};features=None;embeddings=None;submission={};torch.npu.reset_peak_memory_stats();allocated=torch.npu.memory_allocated()
    begin=time.perf_counter_ns();event(origin,'record','origin')
    for stage in order:
        stream=streams['V'] if stage=='V' and mode=='parallel' else streams['L']
        with torch.npu.stream(stream):
            event(origin,'wait',stage+'_ready')
            if stage=='L':
                with scope('language_A',task='A'):outputs['A']=language(model,case['A_embeddings'],case['A_position'],case['A_mask'],ca)
                event(ends['A'],'record','A_done')
            else:
                with scope('vision_B',task='V') as r:
                    features=vision(model,case['B'])
                    if observer:r['feature_output']=layout(features)
                event(ends['V'],'record','V_done')
        submission[stage]=(time.perf_counter_ns()-begin)/1000
    with torch.npu.stream(streams['L']):
        event(ends['V'],'wait','B_features_ready')
        with scope('merge_B',task='B') as r:
            embeddings=merge(model,case['B'],features)
            if observer:r.update(feature_input=layout(features),embedding_output=layout(embeddings))
        with scope('language_B',task='B') as r:
            outputs['B']=language(model,embeddings,case['B']['position'],case['B']['mask'],cb)
            if observer:r['embedding_input']=layout(embeddings)
        event(ends['B'],'record','B_done')
    submission['B']=(time.perf_counter_ns()-begin)/1000
    for n in ['A','V','B']:event(ends[n],'synchronize',n+'_join')
    wall=(time.perf_counter_ns()-begin)/1000;ready={n:origin.elapsed_time(e)*1000 for n,e in ends.items()}
    assert model.model.rope_deltas is None,'shared MRoPE state mutated'
    # Producer tensor and consumer embeddings remain referenced through both joins.
    return dict(outputs=outputs,features=features,embeddings=embeddings),dict(wall_us=wall,ready_us=ready,host_submission_us=submission,
        allocated_before=allocated,allocated_peak=torch.npu.max_memory_allocated(),reserved_peak=torch.npu.max_memory_reserved())


def result_snapshot(result):return dict(outputs={n:snapshot(o) for n,o in result['outputs'].items()},features=result['features'].detach().cpu())

def check_result(torch,ref,result):
    cur=result_snapshot(result);checks={n:compare(torch,ref['outputs'][n],o) for n,o in cur['outputs'].items()}
    checks['V']=dict(exact=torch.equal(ref['features'],cur['features']),finite=bool(torch.isfinite(cur['features']).all()),max_abs=(ref['features'].float()-cur['features'].float()).abs().max().item())
    assert all(c['exact'] and c['finite'] and c.get('greedy_equal',True) for c in checks.values()),checks
    return checks
