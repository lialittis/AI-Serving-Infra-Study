"""Capture/replay qualification, with changing fixed-shape inputs and initial KV."""
import argparse,traceback
from pathlib import Path
from harness import setup,make_case,variant,Captured,run_pair,snapshot,compare,save


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=False)
    (out/'dumps').mkdir()
    try:
        torch,tn,tr,model,tok=setup();streams={n:torch.npu.Stream() for n in ('A','B')};checks=[];metadata=[]
        with torch.inference_mode():
            for phase in ('prefill','decode'):
                case=make_case(torch,model,tok,phase,128)
                # Graph inputs must not alias the reusable case's prompt buffers.
                case['ids']={n:t.clone() for n,t in case['ids'].items()}
                captures={n:Captured(torch,model,case,n,streams['B'] if n=='B' else streams['A'],out/'dumps') for n in ('A','B','AB')}
                metadata.extend(c.metadata for c in captures.values())
                assert len({tuple(c.graph.pool()) for c in captures.values()})==3,'graph pools shared'
                for swap in (False,True,False):
                    current=variant(case,swap)
                    result,_=run_pair(torch,model,current,'serial','eager',streams,captures);reference=snapshot(result,'serial')
                    for mode in ('serial','parallel','batch'):
                        result,metric=run_pair(torch,model,current,mode,'graph',streams,captures)
                        cs=compare(torch,reference,snapshot(result,mode),mode)
                        record=dict(case=case['id'],swap=swap,mode=mode,checks=cs,metric=metric);checks.append(record)
                        save(out/'checks.json',checks);assert all(c['valid'] and c['greedy_equal'] for c in cs),record
                    print('QUALIFIED',case['id'],swap,flush=True)
                save(out/'captures.json',metadata)
            save(out/'completed.json',dict(status='passed',checks=len(checks)))
    except Exception:
        (out/'failure.txt').write_text(traceback.format_exc());raise


if __name__=='__main__':main()
