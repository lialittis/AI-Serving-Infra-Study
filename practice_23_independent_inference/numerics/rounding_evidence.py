"""Dot-product error bounds, exact-rational spot checks and isolated kernel trace."""
import argparse,datetime,json,traceback
from fractions import Fraction
from pathlib import Path
from common import save,metrics,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--operator-run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=False)
    try:
        import torch,torch_npu
        torch.set_num_threads(4);torch.npu.set_device(0)
        save(out/'environment.json',dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),torch=torch.__version__,torch_npu=torch_npu.__version__,hf32=torch.npu.matmul.allow_hf32))
        manifest={}
        for f in Path(__file__).parent.glob('*.py'):
            dest=out/'sources'/f.name;dest.parent.mkdir(exist_ok=True);dest.write_bytes(f.read_bytes());manifest[f.name]=dict(original=str(f),sha256=sha(f))
        save(out/'sources.json',manifest)
        data=torch.load(a.operator_run/'operands.pt',weights_only=True);refs=torch.load(a.operator_run/'references_fp64.pt',weights_only=True)
        x=data['inputs'];xn=x.npu();u=2**-24;gamma=x.shape[-1]*u/(1-x.shape[-1]*u);reports=[];goldens=[]
        save(out/'plan.json',dict(operands_sha256=sha(a.operator_run/'operands.pt'),reference_sha256=sha(a.operator_run/'references_fp64.pt'),
            fp32_unit_roundoff=u,k=x.shape[-1],gamma_k=gamma,
            bound='distance from FP64 dot to BF16 rounding interval <= gamma_K * sum(abs(x_i*w_i)); sufficient numerical consistency test, not proof of undocumented native implementation',
            hf32=torch.npu.matmul.allow_hf32))
        native={};weights={n:w.npu() for n,w in data['weights'].items()}
        with torch.inference_mode():
            for name,w in data['weights'].items():
                ref=refs[name];wn=weights[name]
                values={'single':torch.cat([torch.nn.functional.linear(xn[i:i+1],wn) for i in range(2)],0).cpu(),
                        'batch':torch.nn.functional.linear(xn,wn).cpu()}
                repeated=torch.nn.functional.linear(xn,wn).cpu()
                repeat_exact=torch.equal(repeated,values['batch']);assert repeat_exact
                native[name]=values
                different=values['single']!=values['batch'];sum_abs=x.double().abs()@w.double().abs().T
                bounds=[]
                for mode,y in values.items():
                    yd=y.double();direction=torch.where(ref>=yd,torch.full_like(y,float('inf')),torch.full_like(y,float('-inf')))
                    neighbor=torch.nextafter(y,direction).double();half_gap=(neighbor-yd).abs()/2
                    excess=((ref-yd).abs()-half_gap).clamp_min(0)
                    ratio=excess/(gamma*sum_abs).clamp_min(1e-300)
                    near_zero=(ref.abs()<1e-6)
                    bounds.append(dict(mode=mode,violations=int((ratio>1).sum()),max_bound_fraction=float(ratio.max()),
                        reference=metrics(torch,ref,yd),near_zero_elements=int(near_zero.sum()),
                        near_zero_max_abs=float((ref-yd)[near_zero].abs().max()) if near_zero.any() else 0))
                indices=different.flatten().nonzero().flatten()
                # Spread exact checks across the disagreement set, plus extrema.
                selected=indices[torch.linspace(0,len(indices)-1,12).long()].tolist()
                selected += [int((values['single'].double()-ref).abs().flatten().argmax()),int(ref.abs().flatten().argmin())]
                exact=[]
                for flat in sorted(set(selected)):
                    col=flat%w.shape[0];row=flat//w.shape[0];v=x.reshape(-1,x.shape[-1])[row]
                    exact_value=sum((Fraction(float(xx))*Fraction(float(ww)) for xx,ww in zip(v,w[col])),Fraction(0))
                    record=dict(flat_index=flat,reference_fp64=float(ref.flatten()[flat]),exact_dot_numerator=str(exact_value.numerator),
                        exact_dot_denominator=str(exact_value.denominator),fp64_matches_exact=float(exact_value)==float(ref.flatten()[flat]),
                        single=float(values['single'].flatten()[flat]),batch=float(values['batch'].flatten()[flat]))
                    exact.append(record)
                assert all(e['fp64_matches_exact'] for e in exact),'reference spotcheck'
                assert all(b['violations']==0 for b in bounds),'outside accumulation+rounding bound'
                r=dict(operator=name,repeat_exact=repeat_exact,disagreements=int(different.sum()),elements=different.numel(),
                    cross_mode=metrics(torch,values['single'],values['batch']),bounds=bounds,exact_spotchecks=exact)
                reports.append(r);print('BOUNDS',name,int(different.sum()),[b['max_bound_fraction'] for b in bounds],flush=True)
            save(out/'rounding_bounds.json',reports)
            exp=torch_npu.profiler._ExperimentalConfig(profiler_level=torch_npu.profiler.ProfilerLevel.Level1)
            cases={}
            for name,wn in weights.items():
                for precision in ('bf16','fp32'):
                    xx,ww=(xn,wn) if precision=='bf16' else (xn.float(),wn.float())
                    for mode in ('single','batch'):
                        cases[f'P23N/{name}/{precision}/{mode}']=(xx,ww,mode)
            torch.npu.synchronize();scopes=[]
            with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],
                schedule=torch_npu.profiler.schedule(wait=0,warmup=1,active=1,repeat=1),record_shapes=True,experimental_config=exp,
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(out/'profiler'))) as prof:
                prof.step()
                for label,(xx,ww,mode) in cases.items():
                    with torch.profiler.record_function(label):
                        if mode=='single':ys=[torch.nn.functional.linear(xx[i:i+1],ww) for i in range(2)]
                        else:ys=[torch.nn.functional.linear(xx,ww)]
                    torch.npu.synchronize();scopes.append(dict(label=label,shape=list(xx.shape),weight_shape=list(ww.shape),dtype=str(xx.dtype),mode=mode))
                prof.step()
            save(out/'scopes.json',scopes)
            save(out/'completed.json',dict(status='passed',exact_checks=sum(len(r['exact_spotchecks']) for r in reports),operators=2))
    except Exception:
        (out/'failure.txt').write_text(traceback.format_exc());raise


if __name__=='__main__':main()
