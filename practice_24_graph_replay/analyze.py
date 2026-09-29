"""Audit fixed-step replay evidence and compare equal-precision scheduling."""
import argparse,hashlib,json,statistics
from math import prod
from collections import Counter
from pathlib import Path
from trace_graph import extract,read,require,only,number,union,Graph


def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def duration(xs):return sum(b-a for a,b in union(xs))
def intervals(tasks):return [(number(t['start_us']),number(t['end_us'])) for t in tasks]
def overlap(a,b):
    a=union(a);b=union(b);i=j=0;result=number(0)
    while i<len(a) and j<len(b):
        result+=max(number(0),min(a[i][1],b[j][1])-max(a[i][0],b[j][0]))
        if a[i][1]<b[j][1]:i+=1
        else:j+=1
    return result

def stats(xs):
    q=statistics.quantiles(xs,n=4,method='inclusive') if len(xs)>1 else [xs[0]]*3
    return dict(n=len(xs),median=statistics.median(xs),q1=q[0],q3=q[2],min=min(xs),max=max(xs))


def validate(root):
    plan=read(root/'plan.json');completed=read(root/'completed.json');env=read(root/'environment.json')
    require(completed==dict(status='passed',samples=288,diagnostic_trials=48,captures=12,lifetime_checks=24),'incomplete run')
    manifest=read(root/'sources.json');contracts=read(Path(__file__).with_name('contracts.json'))
    require(set(manifest)==set(contracts['sources']),'source inventory')
    for f,v in manifest.items():
        require(hashlib.sha256((root/'sources'/f).read_bytes()).hexdigest()==v['sha256']==contracts['sources'][f],'source contract '+f)
    require(env['parameter_dtypes']==['torch.float32'] and env['hf32'] is False,'precision')
    require(read(root/'weight_check.json')['unchanged'],'weights mutated')
    require((plan['cross_mode_atol'],plan['cross_mode_rtol'],plan['same_shape_atol'],plan['same_shape_rtol'])==(.0625,.02,0,0),'numerical thresholds')
    rows=read(root/'measurements.json');checks=read(root/'correctness.json');trials=read(root/'trials.json');life=read(root/'lifetime_checks.json')
    cases=[f'{p}-{l}' for p,l in plan['cases']]
    expected={(c,r,b,m) for c in cases for r in range(12) for b,m in plan['modes']}
    require(len(rows)==len(expected) and {(r['case'],r['round'],r['backend'],r['mode']) for r in rows}==expected,'performance coverage')
    require(len(trials)==48 and len({t['id'] for t in trials})==48,'diagnostic coverage')
    require(len(checks)==336 and len(life)==24,'check coverage')
    require(Counter((c['case'],c['round'],c['backend'],c['mode']) for c in checks if c['stage']=='performance')==Counter(expected),'performance numerical identity')
    require(Counter((c['case'],c['round'],c['backend'],c['mode']) for c in checks if c['stage']=='diagnostic')==Counter((t['case'],t['repeat'],t['backend'],t['mode']) for t in trials),'diagnostic numerical identity')
    for r in checks+life:
        require(len(r['checks'])==2 and {c['task'] for c in r['checks']}=={'A','B'},'two task validation')
        require(all(c['tensors']==49 and c['valid'] and c['greedy_equal'] for c in r['checks']),'logits/KV numerical failure')
        if r['mode']!='batch':require(all(c['exact'] for c in r['checks']),'same-shape mismatch')
    for c in cases:
        for b,m in plan['modes']:
            require(Counter((r['order'],r['swap']) for r in rows if (r['case'],r['backend'],r['mode'])==(c,b,m))==Counter({(o,s):3 for o in ['AB','BA'] for s in [False,True]}),'payload/order balance')
    captures=read(root/'captures.json');require(len(captures)==12 and len({tuple(c['pool']) for c in captures})==12,'independent live pools')
    ranges={}
    for c in captures:
        require(len(c['input_kv'])==(24 if c['case'].startswith('decode') else 0),'retained input KV')
        require(len(c['output_kv'])==24,'output KV')
        ts=[c['input'],c['output_logits']]+[t for l in c['input_kv']+c['output_kv'] for k,t in l.items() if k in ('key','value')]
        for t in ts:
            require(all(s>=0 for s in t['stride']) and prod(t['shape'])>0,'unsupported tensor layout')
            extent=(1+sum((d-1)*s for d,s in zip(t['shape'],t['stride'])))*(t['bytes']//prod(t['shape']))
            require(extent==t['bytes'],'non-dense view requires strided byte-range analysis')
        ranges[c['case'],c['name']]=[(int(t['ptr']),int(t['ptr'])+t['bytes']) for t in ts]
    # All captures remain alive. Read-only weights/masks/positions are excluded.
    for key,rs in ranges.items():
        for other,ss in ranges.items():
            if key>=other:continue
            require(all(a1<=b0 or b1<=a0 for a0,a1 in rs for b0,b1 in ss),'cross-graph mutable storage alias')
    return plan,rows,checks,trials,captures


def analyze(root):
    plan,rows,checks,trials,captures=validate(root);performance=[]
    for phase,length in plan['cases']:
        case=f'{phase}-{length}'
        for backend in ('eager','graph'):
            subset=[r for r in rows if r['case']==case and r['backend']==backend];modes={};paired={}
            for mode in ('serial','parallel','batch'):
                xs=[r for r in subset if r['mode']==mode]
                modes[mode]=dict(numerically_valid=True,wall_us=stats([r['wall_us'] for r in xs]),tasks_per_second=stats([2e6/r['wall_us'] for r in xs]),
                    ready_us={n:stats([r['ready_us'][n] for r in xs]) for n in ('A','B')},first_ready_us=stats([min(r['ready_us'].values()) for r in xs]),
                    last_ready_us=stats([max(r['ready_us'].values()) for r in xs]),host_submit_us=stats([max(r['host_submission_us'].values()) for r in xs]),
                    allocated_peak=stats([r['allocated_peak'] for r in xs]),reserved_peak=stats([r['reserved_peak'] for r in xs]),
                    incremental_peak=stats([r['allocated_peak']-r['allocated_before'] for r in xs]))
                if mode!='serial':
                    changes=[100*(r['wall_us']/only((s['wall_us'] for s in subset if s['round']==r['round'] and s['mode']=='serial'),'paired serial')-1) for r in xs]
                    paired[mode]=dict(change_percent=stats(changes),faster_pairs=sum(x<0 for x in changes))
            performance.append(dict(case=case,backend=backend,modes=modes,paired=paired))
    replay_changes=[]
    for p in performance:
        if p['backend']!='graph':continue
        for mode in p['modes']:
            changes=[100*(r['wall_us']/only((s['wall_us'] for s in rows if s['case']==r['case'] and s['round']==r['round'] and s['backend']=='eager' and s['mode']==mode),'paired eager')-1) for r in rows if r['case']==p['case'] and r['backend']=='graph' and r['mode']==mode]
            replay_changes.append(dict(case=p['case'],mode=mode,change_percent=stats(changes),faster_pairs=sum(x<0 for x in changes)))
    evidence=extract(root);graph=evidence['graph'];tasks=evidence['tasks'];records=evidence['observations'];summaries=[];required=[];cross_queries=[]
    for trial in trials:
        ident=trial['id'];ts=[t for t in tasks if t['trial']==ident];compute=[t for t in ts if t['is_compute']]
        groups={n:sorted([t for t in compute if t.get('task')==n],key=lambda t:number(t['start_us'])) for n in ('A','B','AB')}
        names=['AB'] if trial['mode']=='batch' else ['A','B'];require(all(groups[n] for n in names),'missing compute')
        require(len(compute)==sum(len(groups[n]) for n in names),'unattributed compute')
        streams={t['stream'] for t in compute};expected=1 if trial['mode']=='batch' or (trial['backend']=='eager' and trial['mode']=='serial') else 2
        require(len(streams)==expected,'compute stream count')
        both=overlap(intervals(groups['A']),intervals(groups['B']))
        if trial['mode']=='serial':require(both==0,'serial overlap')
        origin=only((t for t in ts if t.get('role')=='origin' and t['name']=='EVENT_RECORD'),'origin')
        for name in names:
            forward=groups[name];join=only((r for r in records if r['trial']==ident and r.get('role')==name+'_join'),'host join')
            required.extend([dict(source=origin['id'],target=forward[0]['id'],kind='input_readiness',trial=ident,task=name),
                dict(source=forward[-1]['id'],target='h:'+join['label'],kind='output_completion',trial=ident,task=name)])
        if trial['mode']!='batch':
            first,second=trial['order'];cross_queries.append(dict(source=groups[first][-1]['id'],target=groups[second][0]['id'],kind='cross_task_order',trial=ident,expected=trial['mode']=='serial'))
        replays=[r for r in evidence['replays'] if r['trial']==ident]
        summaries.append(dict(**trial,streams=sorted(streams),caller_streams=sorted({r['caller_stream'] for r in replays}),tasks=len(ts),compute_tasks=len(compute),
            overlap_us=float(both),compute_busy_us={n:float(duration(intervals(groups[n]))) for n in names},
            span_us=float(max(number(t['end_us']) for t in compute)-min(number(t['start_us']) for t in compute)),node_ids=[t['id'] for t in ts]))
    verified=graph.verify(required);cross=graph.verify(cross_queries)
    require(all(r['satisfied'] for r in verified),'unprotected task boundary')
    require(all(r['satisfied']==r['expected'] for r in cross),'cross-task order')
    summary=dict(status='analysis_passed',all_comparisons_passed=True,invalid_cells=[],performance=performance,replay_vs_eager=replay_changes,trials=summaries,
        performance_samples=len(rows),diagnostic_trials=len(trials),numerical_pairs=len(checks),lifetime_pairs=24,
        all_exact=all(c['exact'] for r in checks for c in r['checks']),max_abs=max(c['max_abs'] for r in checks for c in r['checks']),
        device_tasks=len(tasks),compute_tasks=sum(t['is_compute'] for t in tasks),csv_compute_accounted=True,
        replay_invocations=len(evidence['replays']),internal_graph_tasks=sum(len(r['internal_tasks']) for r in evidence['replays']),
        stream_mapping=evidence['mapping'],collision_resolutions=evidence['collisions'],event_wait_edges=len(evidence['event_edges']),
        api_only_waits=len(evidence['api_waits']),same_stream_waits=evidence['same_stream_waits'],tasks_outside_scopes=len(evidence['outside']),
        boundary_requirements=len(verified),unsatisfied=0,cross_task_order_checks=len(cross),captures=captures,
        complete_exact_kernel_memory_dag=False,exact_notify_id_pairing=False,scope='HF full FP32 fixed-step forward; independent live graph pools; not native vLLM or growing-KV generation')
    output=root/'analysis';output.mkdir(exist_ok=True);save(output/'summary.json',summary)
    data=dict(summary=summary,nodes=list(graph.nodes.values()),edges=graph.edges,requirements=verified,cross_task_order=cross,records=records,replays=evidence['replays'])
    (output/'execution_graph.json').write_text(json.dumps(data,ensure_ascii=False,separators=(',',':'))+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('performance','trials','captures','replay_vs_eager')},indent=2));return data


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);analyze(p.parse_args().run)
