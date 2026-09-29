"""Locate the first divergent layer, then its first divergent module output."""
import argparse,traceback
from pathlib import Path
from common import setup,make_case,prepare,save,metrics,provenance,snapshot,compare


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=False)
    save(out/'plan.json',dict(length=1024,dtype='bfloat16',comparison='A/B batch rows against independent identical input forwards',
        numeric_threshold=dict(atol=.0625,rtol=.02),stages=['plain repeat','all decoder boundaries','first divergent decoder module outputs'],performance=False))
    try:
        torch,tn,tr,model,tokenizer=setup();provenance(torch,tn,tr,model,out)
        with torch.inference_mode():
            case=make_case(torch,model,tokenizer,'prefill',1024)
            def run(mode):
                jobs=prepare(torch,model,case,mode);result={n:model(**j) for n,j in jobs.items()};torch.npu.synchronize()
                return snapshot(result,mode)
            baseline=run('serial');plain=run('batch');save(out/'plain_checks.json',compare(torch,baseline,plain,'batch'))
            names=['model.embed_tokens']+[f'model.layers.{i}' for i in range(len(model.model.layers))]+['model.norm','lm_head']
            modules=dict(model.named_modules());reports={};saved={}
            def capture(selected,stage):
                records=[];handles=[];state={'mode':None,'name':None};reference={}
                def hook(label):
                    def apply(module,inputs,output):
                        tensor=output[0] if isinstance(output,tuple) else output
                        tensor=tensor.detach().cpu()
                        if state['mode']=='serial':reference[state['name'],label]=tensor
                        else:
                            for i,name in enumerate(('A','B')):
                                rec=dict(stage=stage,module=label,task=name,**metrics(torch,reference[name,label],tensor[i:i+1]));records.append(rec)
                    return apply
                for n in selected:handles.append(modules[n].register_forward_hook(hook(n)))
                observed={}
                try:
                    for mode in ('serial','batch'):
                        jobs=prepare(torch,model,case,mode);result={};state['mode']=mode
                        for name,job in jobs.items():state['name']=name;result[name]=model(**job)
                        torch.npu.synchronize();observed[mode]=snapshot(result,mode)
                finally:
                    for h in handles:h.remove()
                unperturbed={m:compare(torch,baseline if m=='serial' else plain,observed[m],'serial') for m in observed}
                assert all(c['exact'] for cs in unperturbed.values() for c in cs),'hooks changed outputs'
                save(out/(stage+'.json'),dict(records=records,unperturbed=unperturbed));return records
            layers=capture(names,'layers');first=next(r['module'] for r in layers if r['module'].startswith('model.layers.') and not r['exact'])
            print('FIRST_DIVERGENT_LAYER',first,flush=True)
            selected=[n for n in modules if n.startswith(first+'.') and not any(n.startswith(o+'.') for o in [first+'.self_attn.rotary_emb'])]
            detail=capture(selected,'modules')
            for r in detail:
                if not r['exact']:print('DIVERGENT_MODULE',r['task'],r['module'],r['max_abs'],r['different'],flush=True)
            save(out/'completed.json',dict(status='located',first_layer=first,first_module=next(r['module'] for r in detail if not r['exact'])))
    except Exception:
        (out/'failure.txt').write_text(traceback.format_exc());raise


if __name__=='__main__':main()
