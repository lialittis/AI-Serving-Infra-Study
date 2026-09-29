"""Reconstruct observed execution DAG; compare equal-work unprofiled strategies."""
import argparse,hashlib,json,statistics
from pathlib import Path
from trace_graph import extract,read,require,only,number,union,intersection


def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def duration(xs):return sum(b-a for a,b in union(xs))
def intervals(tasks):return [(number(t['start_us']),number(t['end_us'])) for t in tasks]
def stats(xs):
    q=statistics.quantiles(xs,n=4,method='inclusive') if len(xs)>1 else [xs[0]]*3
    return dict(n=len(xs),median=statistics.median(xs),q1=q[0],q3=q[2],min=min(xs),max=max(xs))


def analyze(root):
    plan=read(root/'plan.json');completed=read(root/'completed.json')
    require(completed['status']=='captured','incomplete run')
    for f,v in read(root/'sources.json').items():require(hashlib.sha256((root/f).read_bytes()).hexdigest()==v['sha256'],'source hash '+f)
    manifest=read(root/'sources.json')
    for f,digest in read(Path(__file__).with_name('contracts.json'))['sources'].items():
        require(manifest[f]['sha256']==digest,'unaudited source '+f)
    require(read(root/'weight_check.json')['unchanged'],'mutated shared weights or buffers')
    rows=read(root/'measurements.json');checks=read(root/'correctness.json');trials=read(root/'trials.json')
    require(len(rows)==len(plan['cases'])*plan['rounds']*3,'performance coverage')
    require(len({(r['case'],r['round'],r['mode']) for r in rows})==len(rows),'duplicate sample')
    require(len(trials)==len(plan['cases'])*plan['profile_repeats']*3,'diagnostic coverage')
    require(len(checks)==len(rows)+len(trials),'numerical coverage')
    require(all(len(r['checks'])==2 and all(c['tensors']==49 for c in r['checks']) for r in checks),'logits/KV coverage')
    failed=[r for r in checks if not all(c['valid'] and c['greedy_equal'] for c in r['checks'])]
    require(all([r['case'],r['mode']]==plan['known_invalid_comparison'] for r in failed),'unexpected numerical failure')
    require(all(c['exact'] for r in checks if r['mode']!='batch' for c in r['checks']),'non-batch mismatch')
    performance=[]
    for phase,length in plan['cases']:
        case=f'{phase}-{length}';subset=[r for r in rows if r['case']==case];modes={}
        for mode in ('serial','parallel','batch'):
            xs=[r for r in subset if r['mode']==mode]
            modes[mode]=dict(numerically_valid=not any(r['case']==case and r['mode']==mode for r in failed),wall_us=stats([r['wall_us'] for r in xs]),pairs_per_second=stats([1e6/r['wall_us'] for r in xs]),
                tasks_per_second=stats([2e6/r['wall_us'] for r in xs]),
                ready_us={n:stats([r['ready_us'][n] for r in xs]) for n in ('A','B')},
                first_ready_us=stats([min(r['ready_us'].values()) for r in xs]),last_ready_us=stats([max(r['ready_us'].values()) for r in xs]),
                allocated_peak=stats([r['allocated_peak'] for r in xs]),
                incremental_peak=stats([r['allocated_peak']-r['allocated_before'] for r in xs]))
        changes={}
        for mode in ('parallel','batch'):
            ratios=[]
            for r in range(plan['rounds']):
                values={x['mode']:x['wall_us'] for x in subset if x['round']==r}
                ratios.append(100*(values[mode]/values['serial']-1))
            changes[mode]=dict(change_percent=stats(ratios),faster_pairs=sum(x<0 for x in ratios))
        performance.append(dict(case=case,modes=modes,paired=changes))
    evidence=extract(root);graph=evidence['graph'];tasks=evidence['tasks'];records=evidence['observations'];summaries=[];required=[];cross_queries=[]
    for trial in trials:
        ident=trial['id'];ts=[t for t in tasks if t['trial']==ident];compute=[t for t in ts if t['is_compute']]
        groups={n:sorted([t for t in compute if t.get('task')==n],key=lambda t:number(t['start_us'])) for n in ('A','B','AB')}
        names=['AB'] if trial['mode']=='batch' else ['A','B']
        require(all(groups[n] for n in names),'missing forward kernels '+ident)
        require(len(compute)==sum(len(groups[n]) for n in names),'unattributed compute')
        streams={t['stream'] for t in compute};require(len(streams)==(2 if trial['mode']=='parallel' else 1),'stream count')
        overlap=duration(intersection(intervals(groups['A']),intervals(groups['B'])))
        if trial['mode']=='serial':require(overlap==0,'serial overlap')
        origin=only((t for t in ts if t.get('role')=='origin' and t['name']=='EVENT_RECORD'),'origin')
        for name in names:
            forward=groups[name]
            join=only((r for r in records if r['trial']==ident and r.get('role')==name+'_join'),'join')
            required.extend([dict(source=origin['id'],target=forward[0]['id'],kind='input_readiness',trial=ident,task=name,
                                  evidence='explicit inputs/KV globally ready before common origin; native stream wait'),
                             dict(source=forward[-1]['id'],target='h:'+join['label'],kind='output_completion',trial=ident,task=name,
                                  evidence='terminal event follows forward; host joins before validation or storage release')])
        if trial['mode']!='batch':
            first,second=trial['order']
            cross_queries.append(dict(source=groups[first][-1]['id'],target=groups[second][0]['id'],kind='cross_task_order',trial=ident,expected=trial['mode']=='serial'))
            # All live cache tensors are distinct across tasks, not only logical IDs.
            ranges={n:[(int(l[k]['ptr']),int(l[k]['ptr'])+l[k]['bytes']) for l in trial['outputs'][n]['kv'] for k in ('key','value')] for n in ('A','B')}
            require(all(a1<=b0 or b1<=a0 for a0,a1 in ranges['A'] for b0,b1 in ranges['B']),'cache alias')
        summaries.append(dict(id=ident,case=trial['case'],mode=trial['mode'],repeat=trial['repeat'],order=trial['order'],
            streams=sorted(streams),tasks=len(ts),compute_tasks=len(compute),overlap_us=float(overlap),
            compute_busy_us={n:float(duration(intervals(groups[n]))) for n in names},
            span_us=float(max(number(t['end_us']) for t in compute)-min(number(t['start_us']) for t in compute)),
            node_ids=[t['id'] for t in ts]))
    verified=graph.verify(required);cross=graph.verify(cross_queries)
    require(all(r['satisfied'] for r in verified),'unprotected boundary')
    require(all(r['satisfied']==r['expected'] for r in cross),'unexpected cross-task HB')
    summary=dict(status='analysis_passed',all_comparisons_passed=not failed,failed_numeric_comparisons=len(failed),invalid_cells=sorted({r['case']+'/'+r['mode'] for r in failed}),performance=performance,trials=summaries,performance_samples=len(rows),diagnostic_trials=len(trials),
        all_exact=all(c['exact'] for r in checks for c in r['checks']),max_abs=max(c['max_abs'] for r in checks for c in r['checks']),
        device_tasks=len(tasks),compute_tasks=sum(t['is_compute'] for t in tasks),csv_compute_accounted=True,
        stream_mapping=evidence['mapping'],collision_resolutions=evidence['collisions'],
        event_wait_edges=len(evidence['event_edges']),api_only_waits=len(evidence['api_waits']),same_stream_waits=evidence['same_stream_waits'],
        tasks_outside_scopes=len(evidence['outside']),runtime_placeholders=len(evidence['unknown']),
        boundary_requirements=len(verified),unsatisfied=0,cross_task_order_checks=len(cross),
        complete_exact_kernel_memory_dag=False,scope='full-model HF eager execution harness; not vLLM scheduling or end-to-end serving')
    output=root/'analysis';output.mkdir(exist_ok=True);save(output/'summary.json',summary)
    data=dict(summary=summary,nodes=list(graph.nodes.values()),edges=graph.edges,requirements=verified,cross_task_order=cross,records=records)
    (output/'execution_graph.json').write_text(json.dumps(data,ensure_ascii=False,separators=(',',':'))+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('performance','trials')},indent=2));return data


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);analyze(p.parse_args().run)
