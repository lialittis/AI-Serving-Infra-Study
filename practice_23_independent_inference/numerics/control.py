"""Qualification for a shape-stable precision control; not a native BF16 fix."""
import argparse,types,traceback
from pathlib import Path
from common import setup,make_case,prepare,save,compare,snapshot,provenance


def install_fp32_linears(torch,model,scope='mlp'):
    originals=[];weights={}
    for name,module in model.named_modules():
        if not isinstance(module,torch.nn.Linear) or (scope=='mlp' and '.mlp.' not in name):continue
        w=module.weight.detach().float();b=module.bias.detach().float() if module.bias is not None else None
        weights[name]=(w,b);originals.append((module,module.forward))
        def forward(self,x,_w=w,_b=b):
            return torch.nn.functional.linear(x.float(),_w,_b).to(x.dtype)
        module.forward=types.MethodType(forward,module)
    return originals,weights


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--scope',choices=['mlp','all','model'],default='mlp');a=p.parse_args()
    out=a.output;out.mkdir(parents=True,exist_ok=False)
    try:
        torch,tn,tr,model,tokenizer=setup();provenance(torch,tn,tr,model,out)
        assert not torch.npu.matmul.allow_hf32
        with torch.inference_mode():
            if a.scope=='model':
                model.float();original=[];weights={n:(p,None) for n,p in model.named_parameters()}
            else:original,weights=install_fp32_linears(torch,model,a.scope)
            save(out/'plan.json',dict(scope=a.scope,matmul_allow_hf32=torch.npu.matmul.allow_hf32,linear_modules=len(weights),
                weights_fp32_bytes=sum(t.numel()*t.element_size() for pair in weights.values() for t in pair if t is not None),
                meaning='model: all parameters/activations FP32; other scopes: selected FP32 linears then BF16 cast. Same original weight values; separate precision configuration, not a native BF16 fix'))
            checks=[]
            for phase,length in [('prefill',128),('prefill',1024),('decode',128),('decode',1024)]:
                case=make_case(torch,model,tokenizer,phase,length);baseline=None
                for mode in ('serial','batch'):
                    jobs=prepare(torch,model,case,mode);result={n:model(**j) for n,j in jobs.items()};torch.npu.synchronize();current=snapshot(result,mode)
                    if baseline is None:baseline=current
                    c=compare(torch,baseline,current,mode);checks.append(dict(case=case['id'],mode=mode,checks=c))
                    print(case['id'],mode,c,flush=True)
                save(out/'checks.json',checks)
            save(out/'completed.json',dict(status='captured',all_exact=all(c['exact'] for r in checks for c in r['checks']),
                all_valid=all(c['valid'] and c['greedy_equal'] for r in checks for c in r['checks'])))
    except Exception:
        (out/'failure.txt').write_text(traceback.format_exc());raise


if __name__=='__main__':main()
