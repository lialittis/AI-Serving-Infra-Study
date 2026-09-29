"""Unprofiled three-way comparison in a separately declared full-FP32 model."""
import argparse,itertools,traceback
from pathlib import Path
from common import setup,make_case,prepare,save,snapshot,compare,provenance
from run_experiment import execute,isolation
from model import state_hash


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=False)
    cases=[('prefill',128),('prefill',1024),('decode',128),('decode',1024)];rounds=12
    save(out/'plan.json',dict(cases=cases,rounds=rounds,warmup=3,precision='full_fp32',hf32=False,
        weights='Same BF16 checkpoint values exactly promoted to FP32, no new checkpoint',
        tolerance=dict(serial=dict(atol=0,rtol=0),parallel=dict(atol=0,rtol=0),batch=dict(atol=.0625,rtol=.02)),
        scope='Separate precision control; native BF16 failing comparison remains invalid',
        timing='P23 execute: common origin to two completions; all inputs/cache preparation and validation excluded; no profiler/hooks'))
    try:
        torch,tn,tr,model,tokenizer=setup();model.float();assert not torch.npu.matmul.allow_hf32
        provenance(torch,tn,tr,model,out);before=state_hash(torch,model)
        save(out/'state_before.json',before)
        streams={n:torch.npu.Stream() for n in ('A','B')};perms=list(itertools.permutations(('serial','parallel','batch')))
        rows=[];checks=[];layouts=[]
        with torch.inference_mode():
            for phase,length in cases:
                case=make_case(torch,model,tokenizer,phase,length)
                jobs=prepare(torch,model,case,'serial');result,_=execute(torch,model,jobs,'serial',streams,'AB');baseline=snapshot(result,'serial');del result,jobs
                for mode in ('serial','parallel','batch'):
                    for _ in range(3):
                        jobs=prepare(torch,model,case,mode);result,_=execute(torch,model,jobs,mode,streams,'AB');del result,jobs
                for r in range(rounds):
                    for mode in perms[r%6]:
                        order='AB' if r%2==0 else 'BA';jobs=prepare(torch,model,case,mode)
                        result,times=execute(torch,model,jobs,mode,streams,order)
                        cs=compare(torch,baseline,snapshot(result,mode),mode)
                        assert all(c['valid'] and c['greedy_equal'] for c in cs),(case['id'],mode,cs)
                        rows.append(dict(case=case['id'],round=r,mode=mode,order=order,**times))
                        checks.append(dict(case=case['id'],round=r,mode=mode,checks=cs))
                        if r==0:layouts.append(dict(case=case['id'],mode=mode,jobs=isolation(torch,jobs)))
                        del result,jobs
                    save(out/'measurements.json',rows);save(out/'checks.json',checks)
                print('BENCHMARK',case['id'],flush=True)
            unchanged=state_hash(torch,model)==before;assert unchanged
            save(out/'layouts.json',layouts);save(out/'weight_check.json',dict(unchanged=unchanged))
            save(out/'completed.json',dict(status='passed',samples=len(rows),all_valid=True,
                max_abs=max(c['max_abs'] for r in checks for c in r['checks'])))
    except Exception:
        (out/'failure.txt').write_text(traceback.format_exc());raise


if __name__=='__main__':main()
