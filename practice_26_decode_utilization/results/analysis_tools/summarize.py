"""Validate same workload and exact greedy tokens before reporting performance."""
import argparse
import json
import math
from pathlib import Path
import re
from analyze import read,require,stats,digest

def normalize(argv):
    result=[];i=0
    while i<len(argv):
        if argv[i] in ('--port','--profiler-config','--compilation-config'):i+=2
        elif argv[i]=='--enforce-eager':i+=1
        else:result.append(argv[i]);i+=1
    return result

def summarize(root):
    runs=sorted(p for p in root.iterdir() if p.is_dir() and (p/'complete.json').exists())
    require(len(runs)==8,'expected four controls and four profiles')
    reference=None;work=None;model=None;source=None;config=None
    controls=[];profiles=[];checked=0
    for run in runs:
        cmd=read(run/'command.json');responses=read(run/'responses.json')
        require(read(run/'shutdown.json')['returncode'] in (0,-15),'unexpected shutdown')
        for name,sha in read(run/'source_hashes.json').items():require(digest(run/name)==sha,'changed source snapshot')
        current=(read(run/'request.json'),read(run/'model_hashes.json'),read(run/'source_manifest.json'),normalize(cmd['argv']))
        if work is None:work,model,source,config=current
        require(current==(work,model,source,config),'non-equivalent workload / model / source / config')
        require(len(responses)==cmd['requests'],'response count')
        require(cmd['tokens']==64 and cmd['warmups']==3,'workload bounds')
        all_responses=[r['response'] for r in responses]+[read(p) for p in sorted(run.glob('warmup-*.json'))]
        require(len(all_responses)==cmd['requests']+3,'warmup count')
        for response in all_responses:
            require(all(response['usage'][k]==v for k,v in dict(prompt_tokens=10,completion_tokens=64,total_tokens=74).items()),'token usage')
            choice=response['choices'];require(len(choice)==1,'one sequence required');c=choice[0]
            ids=c['logprobs']['tokens'];lp=c['logprobs']['token_logprobs']
            require(len(ids)==len(lp)==64 and all(re.fullmatch(r'token_id:\d+',t) for t in ids),'missing token IDs')
            require(c['finish_reason']=='length' and all(math.isfinite(x) for x in lp),'invalid output')
            if reference is None:reference=ids
            require(ids==reference,'exact greedy token mismatch in '+run.name);checked+=1
        record=dict(run=run.name,mode=cmd['mode'],profile=cmd['profile'],completed_ms=stats(r['completed_ms'] for r in responses),samples_ms=[r['completed_ms'] for r in responses])
        if cmd['profile']=='none':
            require(not list(run.glob('observer-*.json')) and not (run/'profiler').exists(),'instrumented performance control')
            controls.append(record)
        else:
            info=read(next((run/'profiler').rglob('profiler_info_0.json')))
            exp=info['config']['experimental_config'];expected='ACL_AICORE_PIPE_UTILIZATION' if cmd['profile']=='pipe' else 'ACL_AICORE_NONE'
            require(exp['_profiler_level']=='Level1' and exp['_aic_metrics']==expected,'profiler metric configuration')
            record['profiler_config_verified']=True
            record['diagnostic']=read(run/'analysis/summary.json');profiles.append(record)
    timings={mode:stats(t for r in controls if r['mode']==mode for t in r['samples_ms']) for mode in ('eager','graph')}
    require(all(v['n']==10 for v in timings.values()),'ten controls per mode')
    byname={r['run']:r for r in controls}
    pairs=[]
    for e,g in [('control-0-eager','control-1-graph'),('control-3-eager','control-2-graph')]:
        pairs.append(dict(eager=e,graph=g,graph_over_eager=byname[g]['completed_ms']['median']/byname[e]['completed_ms']['median']))
    return dict(status='passed',correctness=dict(responses_including_warmup=checked,formal_responses=24,exact_greedy_tokens_equal=True,token_ids=reference),controls=controls,profiles=profiles,timing=timings,pairs=pairs,graph_over_eager=timings['graph']['median']/timings['eager']['median'],limits=['Ten HTTP completion samples per mode in two independent server launches; descriptive statistics, not a stable tail-latency estimate.','HTTP latency includes prefill and response/logprobs overhead; it is not isolated decode latency.','Graph versus eager changes compilation and kernel fusion as well as submission; this is not a pure stream-count intervention.','Profiler timings and unprofiled HTTP timings belong to separate runs.'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();d=summarize(a.root)
    (a.root/'comparison.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:d[k] for k in ('status','timing','graph_over_eager','pairs')}))
