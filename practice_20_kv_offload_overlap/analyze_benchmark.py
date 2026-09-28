"""Validate all samples and compare complete, matched workloads without profiling."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics

MODES=['native','serialized','recompute','recompute','serialized','native']


def read(path):
    return json.loads(path.read_text())


def require(ok,message):
    if not ok:
        raise ValueError(message)


def describe(values):
    quartiles=statistics.quantiles(values,n=4,method='inclusive')
    return dict(n=len(values),median=statistics.median(values),q1=quartiles[0],q3=quartiles[2],
                minimum=min(values),maximum=max(values))


def analyze(root):
    plan=read(root/'plan.json')
    require(plan['phase']=='benchmark' and plan['modes']==MODES,'unexpected formal plan')
    require(plan['warmup']==plan['repeats']==5 and plan['lengths']==[1024,3072],'sample plan changed')
    samples=[]; golden={}; source_golden=None; instrumentation_golden=None
    for i,mode in enumerate(MODES):
        run=root/('block-%02d-%s'%(i,mode))
        cmd=read(run/'command.json')
        require(cmd['mode']==mode and cmd['phase']=='benchmark','mode mismatch')
        require('--profiler-config' not in cmd['argv'],'profiler enabled in benchmark')
        require(read(run/'shutdown.json')['exit_code']==0,'unclean shutdown')
        sources=read(run/'source_manifest.json')
        fingerprints={k:v['sha256'] for k,v in sources.items()}
        for path,sha in fingerprints.items():
            require(hashlib.sha256((run/path).read_bytes()).hexdigest()==sha,'changed source')
        if source_golden is None:
            source_golden=fingerprints
        require(fingerprints==source_golden,'different installed sources')
        instruments={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (run/'instrumentation').glob('*.py')}
        if instrumentation_golden is None:
            instrumentation_golden=instruments
        require(instruments==instrumentation_golden,'instrumentation changed between service blocks')
        records=sorted((json.loads(line) for p in run.glob('events-*.jsonl')
                        for line in p.read_text().splitlines()),key=lambda r:r['monotonic_ns'])
        require(not any(r['kind'] in ('scope_begin','unsupported_preemption') for r in records),
                'heavy observer or unsupported preemption in performance run')
        for phase in ('warmup','measure'):
            for length in plan['lengths']:
                for j in range(5):
                    case=run/('%s-%d-%02d'%(phase,length,j))
                    summary=read(case/'summary.json')
                    inputs=read(case/'inputs.json')
                    responses={name:read(case/(name+'.json')) for name in ['A','B','reload']+['C%d'%c for c in range(5)]}
                    require(summary['a_reload_equal'],'A/reload output mismatch')
                    for name,r in responses.items():
                        expected=256 if name=='B' else 4 if name.startswith('C') else 64
                        require(len(r['token_ids'])==expected,'output length mismatch')
                    key=(cmd['block'],phase,length,j)
                    payload=dict(inputs=inputs,outputs={name:r['token_ids'] for name,r in responses.items()})
                    if key in golden:
                        require(payload==golden[key],'cross-mode input or greedy token mismatch '+str(key))
                    else:
                        golden[key]=payload
                    start=responses['A']['start_ns']; end=max(r['end_ns'] for r in responses.values())
                    current=[r for r in records if start<=r['monotonic_ns']<=end]
                    transfers=[r for r in current if r['kind']=='dma_submit']
                    amounts={direction:sum(r['num_bytes'] for r in transfers if r['direction']==direction)
                             for direction in ('D2H','H2D')}
                    lookup=[r for r in current if r['kind']=='lookup' and '-reload-' in r['request']]
                    if mode!='recompute':
                        require(len(lookup)==1,'missing reload lookup')
                        require(lookup[0]['npu_hit_tokens']==0 and lookup[0]['cpu_hit_tokens']==length,
                                'did not reload the expected CPU prefix')
                        require(amounts['H2D']==length*12288,'H2D bytes disagree with Qwen model contract')
                        require(amounts['D2H']>0,'no real store')
                    else:
                        require(not transfers,'recompute unexpectedly transferred KV')
                    samples.append(dict(service=i,mode=mode,phase=phase,**summary,**amounts,
                                        reload_prefill_tokens=32 if mode!='recompute' else length+32))
    formal=[s for s in samples if s['phase']=='measure']
    require(len(formal)==60,'formal sample count')
    cases=[]
    for length in plan['lengths']:
        groups={mode:[s for s in formal if s['prefix_tokens']==length and s['mode']==mode]
                for mode in set(MODES)}
        for mode in ('native','serialized'):
            counts=Counter((s['D2H'],s['H2D']) for s in groups[mode])
            require(len(counts)==1,'inconsistent transfer amount within mode')
        require({(s['D2H'],s['H2D']) for s in groups['native']}==
                {(s['D2H'],s['H2D']) for s in groups['serialized']},'transfer amounts differ across modes')
        stats={mode:{metric:describe([s[metric] for s in values]) for metric in
                     ('reload_ttft_ms','reload_latency_ms','b_latency_ms','pair_window_ms')}
               for mode,values in groups.items()}
        comparisons={}
        for baseline,pairs in [('serialized',[(0,1),(5,4)]),('recompute',[(0,2),(5,3)])]:
            comparisons[baseline]={}
            for metric in stats['native']:
                directions=[]
                for on,off in pairs:
                    left=statistics.median(s[metric] for s in groups['native'] if s['service']==on)
                    right=statistics.median(s[metric] for s in groups[baseline] if s['service']==off)
                    directions.append(100*(1-left/right))
                left,right=stats['native'][metric],stats[baseline][metric]
                separated=left['q3']<right['q1'] or right['q3']<left['q1']
                classification=('consistent_improvement' if all(v>0 for v in directions) else
                                'consistent_regression' if all(v<0 for v in directions) else 'uncertain')
                if not separated:
                    classification='uncertain'
                comparisons[baseline][metric]=dict(pair_reduction_pct=directions,
                    pooled_iqr_disjoint=separated,descriptive_classification=classification)
        cases.append(dict(prefix_tokens=length,modes=stats,
                          comparisons=comparisons,
                          native_vs_serialized_ttft_reduction_pct=100*(1-stats['native']['reload_ttft_ms']['median']/stats['serialized']['reload_ttft_ms']['median']),
                          native_vs_recompute_ttft_reduction_pct=100*(1-stats['native']['reload_ttft_ms']['median']/stats['recompute']['reload_ttft_ms']['median']),
                          bytes_per_cycle={m:dict(D2H=v[0]['D2H'],H2D=v[0]['H2D']) for m,v in groups.items()}))
    result=dict(status='passed',formal_samples=60,warmup_samples=len(samples)-60,
                exact_cross_mode_tokens=True,identical_transfer_bytes=True,cases=cases,
                samples=samples,limitations=['Descriptive medians/IQR, not a significance test.',
                'Serialized control includes host barriers; not a pure stream-ID comparison.',
                'No sustained-arrival service capacity or production p99 estimate.'])
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='samples'},indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path)
    analyze(p.parse_args().run)
