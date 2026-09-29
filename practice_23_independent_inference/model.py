"""Full pretrained Qwen forward with explicit per-task KV, no serving scheduler."""
import hashlib
from pathlib import Path
MODEL='/data/huggingface_home/hub/Qwen2.5-0.5B-Instruct'


def setup():
    import torch,torch_npu,transformers
    from transformers import AutoModelForCausalLM,AutoTokenizer
    torch.npu.set_device(0);torch.set_num_threads(4)
    model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,
        attn_implementation='eager',local_files_only=True).eval().to('npu')
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    assert model.config.rope_parameters['rope_type']=='default'
    assert not model.config.use_sliding_window
    model.requires_grad_(False)
    torch.npu.synchronize()
    return torch,torch_npu,transformers,model,tokenizer


def make_case(torch,model,tokenizer,phase,length):
    from transformers.cache_utils import DynamicCache
    texts={'A':'Explain in simple words why the sky is blue. ',
           'B':'Write a short Python function to add two integers. '}
    prompts={}
    for name,text in texts.items():
        ids=tokenizer.encode(text,add_special_tokens=False)
        prompts[name]=torch.tensor([(ids*((length+len(ids)-1)//len(ids)))[:length]],dtype=torch.long,device='npu')
    mask=torch.full((length,length),torch.finfo(torch.bfloat16).min,dtype=torch.bfloat16,device='cpu').triu(1).to('npu')[None,None]
    positions=torch.arange(length,dtype=torch.long,device='npu')[None]
    base={};ids={};baselines={}
    for name,prompt in prompts.items():
        cache=DynamicCache(config=model.config)
        y=model(input_ids=prompt,position_ids=positions,attention_mask={'full_attention':mask},
                past_key_values=cache,use_cache=True,logits_to_keep=1)
        torch.npu.synchronize()
        if phase=='decode':
            base[name]=[(l.keys,l.values) for l in cache.layers]
            ids[name]=y.logits[:,-1].argmax(-1,keepdim=True)
        else:ids[name]=prompt
        baselines[name]=y.logits.detach().cpu()
    if phase=='decode':
        mask=torch.zeros((1,1,1,length+1),dtype=torch.bfloat16,device='npu')
        positions=torch.tensor([[length]],dtype=torch.long,device='npu')
    torch.npu.synchronize()
    return dict(id=f'{phase}-{length}',phase=phase,length=length,ids=ids,mask=mask,positions=positions,base=base)


def prepare(torch,model,case,mode):
    from transformers.cache_utils import DynamicCache
    names=['AB'] if mode=='batch' else ['A','B'];jobs={}
    for name in names:
        members=['A','B'] if name=='AB' else [name]
        ids=torch.cat([case['ids'][n] for n in members],0) if name=='AB' else case['ids'][name]
        if case['phase']=='decode':
            layers=[]
            for i in range(len(model.model.layers)):
                layers.append(tuple(torch.cat([case['base'][n][i][kv] for n in members],0)
                    if name=='AB' else case['base'][name][i][kv].clone() for kv in (0,1)))
            cache=DynamicCache(ddp_cache_data=layers,config=model.config)
        else:cache=DynamicCache(config=model.config)
        jobs[name]=dict(input_ids=ids,position_ids=case['positions'],attention_mask={'full_attention':case['mask']},
                       past_key_values=cache,use_cache=True,logits_to_keep=1)
    torch.npu.synchronize()  # all immutable inputs and independent KV ready before timing
    return jobs


def describe(torch,t):
    return dict(ptr=str(t.data_ptr()),storage=str(t.untyped_storage().data_ptr()),
        shape=list(t.shape),stride=list(t.stride()),offset=t.storage_offset(),bytes=t.numel()*t.element_size())


def cache_layout(torch,cache):
    return [dict(layer=i,key=describe(torch,l.keys),value=describe(torch,l.values))
            for i,l in enumerate(cache.layers) if l.keys is not None]


def state_hash(torch,model):
    return {name:hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
            for name,t in list(model.named_parameters())+list(model.named_buffers())}
