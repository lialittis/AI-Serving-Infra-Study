import argparse,json,traceback
from pathlib import Path
from model import setup,make_case,prepare
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
try:
    torch,torch_npu,transformers,model,tokenizer=setup();checks=[]
    streams={n:torch.npu.Stream() for n in ('A','B')}
    with torch.inference_mode():
        for phase,length in [('prefill',128),('decode',128)]:
            case=make_case(torch,model,tokenizer,phase,length);outputs={}
            for mode in ('serial','parallel','batch'):
                jobs=prepare(torch,model,case,mode);result={}
                for name,job in jobs.items():
                    stream=streams['B'] if mode=='parallel' and name=='B' else streams['A']
                    with torch.npu.stream(stream):result[name]=model(**job)
                torch.npu.synchronize()
                outputs[mode]=[result['AB'].logits[i:i+1].cpu() for i in range(2)] if mode=='batch' else [result[n].logits.cpu() for n in ('A','B')]
            r=dict(phase=phase,length=length,exact_parallel=all(torch.equal(x,y) for x,y in zip(outputs['serial'],outputs['parallel'])),
                batch_max_abs=max((x.float()-y.float()).abs().max().item() for x,y in zip(outputs['serial'],outputs['batch'])),
                tokens={m:[x.argmax(-1).tolist() for x in xs] for m,xs in outputs.items()})
            checks.append(r);print(r,flush=True)
    (a.output/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
except Exception:
    (a.output/'failure.txt').write_text(traceback.format_exc());raise
