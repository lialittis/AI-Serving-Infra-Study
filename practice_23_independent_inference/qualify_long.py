"""Investigate long-input BF16 mode differences without relaxing acceptance."""
import argparse,json
from pathlib import Path
from model import setup,make_case,prepare
from run_experiment import execute,snapshot
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
torch,_,_,model,tokenizer=setup();streams={n:torch.npu.Stream() for n in ('A','B')};checks=[]
with torch.inference_mode():
    for phase in ('prefill','decode'):
        case=make_case(torch,model,tokenizer,phase,1024);reference=None
        for mode in ('serial','serial','parallel','batch'):
            jobs=prepare(torch,model,case,mode);outputs,_=execute(torch,model,jobs,mode,streams,'AB');current=snapshot(outputs,mode)
            if reference is None:reference=current
            for name in ('A','B'):
                x,y=reference[name],current[name]
                pairs=[('logits',x['logits'],y['logits'])]+[(f'{i}-{kv}',aa[kv],bb[kv]) for i,(aa,bb) in enumerate(zip(x['kv'],y['kv'])) for kv in (0,1)]
                metrics=[]
                for label,xx,yy in pairs:
                    xx=xx.float();yy=yy.float();d=xx-yy
                    metrics.append(dict(tensor=label,exact=torch.equal(xx,yy),max_abs=d.abs().max().item(),ref_max=xx.abs().max().item(),
                        rmse=d.square().mean().sqrt().item(),relative_l2=(d.norm()/xx.norm()).item(),valid=torch.allclose(xx,yy,atol=.0625,rtol=.02)))
                checks.append(dict(phase=phase,mode=mode,task=name,metrics=metrics))
                print(phase,mode,name,'logits',metrics[0],'KVmax',max(m['max_abs'] for m in metrics[1:]),flush=True)
            del outputs,jobs
(a.output/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
