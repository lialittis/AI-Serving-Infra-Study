"""Compare actual outputs/configuration against separate uninstrumented runs."""
import argparse
import json
import math
from pathlib import Path
import re


def load(p):return json.loads(p.read_text())
def require(ok,message):
    if not ok:raise ValueError(message)


def normalized_command(command):
    args=command['argv'];out=[];i=0
    while i<len(args):
        if args[i] in ('--port','--profiler-config'):i+=2
        else:out.append(args[i]);i+=1
    return out


def outputs(run):
    responses=load(run/'responses.json');meta=load(run/'complete.json')
    require(len(responses)==2,'missing formal request')
    result=[]
    for record in responses:
        response=record['response'];choices=response['choices']
        require(len(choices)==meta['batch'] and response['usage']['completion_tokens']==4*meta['batch'],'response coverage')
        require(sorted(c['index'] for c in choices)==list(range(meta['batch'])),'choice indexes')
        tokens=[]
        for c in sorted(choices,key=lambda c:c['index']):
            ids=c['logprobs']['tokens']
            require(len(ids)==4 and all(re.fullmatch(r'token_id:\d+',v) for v in ids),'token IDs missing')
            require(c['finish_reason']=='length' and all(math.isfinite(v) for v in c['logprobs']['token_logprobs']),'invalid completion')
            tokens.append(ids)
        result.append(tokens)
    return result


def validate(root):
    records=[]
    for case in ('eager','graph','sampling-off','sampling-on'):
        observed=root/('2026-09-28-'+case+'-run01');baseline=root/('2026-09-28-'+case+'-baseline01')
        require(normalized_command(load(observed/'command.json'))==normalized_command(load(baseline/'command.json')),'changed model configuration')
        require(load(observed/'request.json')==load(baseline/'request.json'),'changed request')
        require(load(observed/'model_fingerprint.json')==load(baseline/'model_fingerprint.json'),'changed model files')
        require(load(observed/'source_manifest.json')==load(baseline/'source_manifest.json'),'changed installed sources')
        a,b=outputs(observed),outputs(baseline)
        deterministic=case in ('eager','graph')
        if deterministic:require(a==b,'greedy token mismatch with baseline')
        records.append(dict(case=case,configuration_and_model_match=True,requests=2,batch=len(a[0]),
                            greedy_exact_tokens_match=a==b if deterministic else None,
                            random_text_identity_required=False if not deterministic else None,valid_outputs=True))
    require(outputs(root/'2026-09-28-eager-run01')==outputs(root/'2026-09-28-graph-run01'),'eager / graph token mismatch')
    return dict(cases=records,eager_graph_token_ids_equal=True,
                limits='Correctness/control validation only; no performance claim; random texts need not match across processes.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('results',type=Path);a=p.parse_args()
    data=validate(a.results);(a.results/'baseline_validation.json').write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(data))
