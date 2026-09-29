"""Isolate the first divergent linear maps with identical BF16 inputs and FP64 reference."""
import argparse,inspect,traceback
from pathlib import Path
from common import setup,make_case,prepare,save,metrics,provenance


def rounding(torch,actual,reference):
    rounded=reference.bfloat16();actual=actual.cpu().double();nearest=rounded.double()
    above=torch.nextafter(rounded,torch.full_like(rounded,float('inf'))).double()
    below=torch.nextafter(rounded,torch.full_like(rounded,float('-inf'))).double()
    spacing=torch.where(actual>=nearest,above-nearest,nearest-below)
    different=actual!=nearest;steps=(actual-nearest).abs()/spacing.clamp_min(1e-300)
    # Margin to the rounding midpoint in the direction of the actual output.
    midpoint=(nearest+torch.where(actual>=nearest,above,below))/2
    distances=(reference-midpoint).abs()/spacing.clamp_min(1e-300)
    return dict(vs_fp64=metrics(torch,reference,actual),vs_correctly_rounded=metrics(torch,nearest,actual),
        max_bf16_steps=float(steps.max()),wrong_rounding_count=int(different.sum()),
        wrong_rounding_exact_midpoint=int((different & (reference==midpoint)).sum()),
        wrong_rounding_near_midpoint_1e4ulp=int((different & (distances<=1e-4)).sum()),
        worst_wrong_rounding_midpoint_distance_ulp=float(distances[different].max()) if different.any() else 0)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=False)
    try:
        torch,tn,tr,model,tokenizer=setup();provenance(torch,tn,tr,model,out)
        save(out/'plan.json',dict(operators=['model.layers.0.mlp.gate_proj','model.layers.0.mlp.up_proj'],
            reference='CPU FP64 matmul of identical exact BF16 operands; round once to BF16',performance=False))
        with torch.inference_mode():
            case=make_case(torch,model,tokenizer,'prefill',1024);inputs={};mode=None;task=None
            module=model.model.layers[0].mlp.gate_proj
            def hook(m,args):inputs[mode,task]=args[0].detach().cpu()
            h=module.register_forward_pre_hook(hook)
            for mode in ('serial','batch'):
                jobs=prepare(torch,model,case,mode)
                for task,job in jobs.items():result=model(**job)
                torch.npu.synchronize()
            h.remove()
            x=torch.cat([inputs['serial',n] for n in ('A','B')],0)
            assert torch.equal(x,inputs['batch','AB']),'upstream inputs differ'
            weights={n:getattr(model.model.layers[0].mlp,n).weight.detach().cpu() for n in ('gate_proj','up_proj')}
            torch.save(dict(inputs=x,weights=weights),out/'operands.pt')
            xn=x.npu();records=[];refs={}
            for name,w in weights.items():
                wn=w.npu();reference=x.double()@w.double().T;refs[name]=reference
                for precision in ('bf16','fp32'):
                    xx,ww=(xn,wn) if precision=='bf16' else (xn.float(),wn.float())
                    values={}
                    for strategy in ('single','batch','batch_repeat','flatten','transpose_copy'):
                        if strategy=='single':y=torch.cat([torch.nn.functional.linear(xx[i:i+1],ww) for i in range(2)],0)
                        elif strategy in ('batch','batch_repeat'):y=torch.nn.functional.linear(xx,ww)
                        elif strategy=='flatten':y=torch.nn.functional.linear(xx.reshape(-1,xx.shape[-1]),ww).reshape(2,1024,-1)
                        else:y=(xx@ww.T.contiguous())
                        y=y.bfloat16().cpu();values[strategy]=y
                        rec=dict(operator=name,precision=precision,strategy=strategy,**rounding(torch,y,reference))
                        records.append(rec);print('REFERENCE',name,precision,strategy,rec['wrong_rounding_count'],rec['max_bf16_steps'],rec['worst_wrong_rounding_midpoint_distance_ulp'],flush=True)
                    cross={k:metrics(torch,values['single'],v) for k,v in values.items()}
                    save(out/f'{name}-{precision}-cross.json',cross)
                save(out/'rounding.json',records)
            torch.save(refs,out/'references_fp64.pt')
            save(out/'completed.json',dict(status='passed',identical_input=True,operators=len(weights),variants=len(records)))
    except Exception:
        (out/'failure.txt').write_text(traceback.format_exc());raise


if __name__=='__main__':main()
