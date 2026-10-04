"""P31b: fixed eager/random sampling, off/on precomputation, independent HTTP users."""
import argparse
import json
from pathlib import Path
import shutil
import time
from run import collect_sources, run_case, save, snapshot, token_prompt


def case_plan(concurrency, phase):
    # Independent services; ABBA brackets elapsed-time effects in benchmarks.
    cases=[]
    for c in concurrency:
        if phase in ('both','benchmark'):
            for block, enabled in enumerate((False,True,True,False)):
                cases.append((c,'benchmark',enabled,f'c{c}-{"on" if enabled else "off"}-b{block}-benchmark'))
        if phase in ('both','diagnostic'):
            for enabled in ((False,True) if c<=8 else (True,False)):
                cases.append((c,'diagnostic',enabled,f'c{c}-{"on" if enabled else "off"}-diagnostic'))
    return cases


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',default='/data/huggingface_home/hub/Qwen2.5-0.5B-Instruct')
    p.add_argument('--concurrency',type=int,nargs='+',default=[8,32])
    p.add_argument('--phase',choices=['both','benchmark','diagnostic'],default='both')
    p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--output-tokens',type=int,default=64)
    p.add_argument('--input-tokens',type=int,default=128)
    p.add_argument('--warmup',type=int,default=2)
    p.add_argument('--port',type=int,default=8031)
    a=p.parse_args()
    if not set(a.concurrency)<={1,2,4,8,32} or min(a.rounds,a.warmup,a.input_tokens,a.output_tokens)<1:
        p.error('Unsupported concurrency or nonpositive workload')
    if a.input_tokens>128 or a.output_tokens>64 or a.rounds>3:
        p.error('Bounded P31b: input<=128, output<=64, rounds<=3')
    a.sampling=dict(temperature=.8,top_p=.9)
    a.max_num_seqs=32;a.max_num_batched_tokens=4096;a.repeats=1
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    device=snapshot(['npu-smi','info'],out/'preflight.json')
    if device.returncode or 'No running processes' not in device.stdout:
        raise RuntimeError('NPU is not idle; do not terminate other workloads')
    snapshot(['npu-smi','info','-t','health','-i','5','-c','0'],out/'device_health.json')
    if shutil.disk_usage(out).free<6*1024**3:raise RuntimeError('Need 6 GiB free for traces')
    cases=case_plan(a.concurrency,a.phase)
    save(out/'plan.json',dict(schema=2,experiment='P31b',start_ns=time.time_ns(),
                            arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
                            cases=cases,per_request_seed=False,benchmark_order='off,on,on,off',
                            output_equality_expected=False,only_treatment='enable_async_exponential'))
    collect_sources(out,a.model)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    inputs={str(u):[token_prompt(tokenizer,u,r,a.input_tokens) for r in range(max(a.rounds,a.warmup))]
            for u in range(max(a.concurrency))}
    save(out/'inputs.json',inputs)
    for c,phase,enabled,name in cases:
        a.precompute=enabled
        run_case(a,out,inputs,c,phase,name)
    print('Completed '+str(out),flush=True)


if __name__=='__main__':main()
