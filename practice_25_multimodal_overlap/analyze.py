"""Verify actual vision-to-language handoff and compare unprofiled pair completion."""
import argparse,gzip,hashlib,json,statistics
from collections import Counter
from math import prod
from pathlib import Path
from trace_graph import extract,read,require,only,number,union,Graph


def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def stats(xs):
    q=statistics.quantiles(xs,n=4,method='inclusive') if len(xs)>1 else [xs[0]]*3
    return dict(n=len(xs),median=statistics.median(xs),q1=q[0],q3=q[2],min=min(xs),max=max(xs))
def intervals(ts):return [(number(t['start_us']),number(t['end_us'])) for t in ts]
def duration(xs):return sum(b-a for a,b in union(xs))
def overlap(a,b):
    a=union(a);b=union(b);i=j=0;total=number(0)
    while i<len(a) and j<len(b):
        total+=max(number(0),min(a[i][1],b[j][1])-max(a[i][0],b[j][0]))
        if a[i][1]<b[j][1]:i+=1
        else:j+=1
    return total

def bounds(t):
    require(prod(t['shape'])>0 and all(s>=0 for s in t['stride']),'unsupported layout')
    element=t['bytes']//prod(t['shape']);start=int(t['ptr'])
    return start,start+(1+sum((d-1)*s for d,s in zip(t['shape'],t['stride'])))*element


def validate(root):
    complete=read(root/'completed.json');plan=read(root/'plan.json');env=read(root/'environment.json')
    require(complete==dict(status='passed',samples=192,numerical_pairs=224,diagnostic_trials=32,full_vs_split=8),'incomplete formal run')
    manifest=read(root/'sources.json');contracts=read(Path(__file__).with_name('contracts.json'))
    require(set(manifest)==set(contracts['sources']),'source inventory')
    for f,v in manifest.items():require(hashlib.sha256((root/'sources'/f).read_bytes()).hexdigest()==v['sha256']==contracts['sources'][f],'unaudited source '+f)
    require(env['parameter_dtypes']==['torch.bfloat16'] and (plan['atol'],plan['rtol'])==(0,0),'precision and exact tolerance')
    require(read(root/'weight_check.json')==dict(unchanged=True,shared_rope_deltas=None),'shared model state')
    rows=read(root/'measurements.json');checks=read(root/'correctness.json');cases=read(root/'cases.json')
    ids=[c['id'] for c in cases];require(len(set(ids))==8,'case coverage')
    expected={(c,r,m) for c in ids for r in range(12) for m in ['serial','parallel']}
    require(len(rows)==192 and {(r['case'],r['round'],r['mode']) for r in rows}==expected,'performance coverage')
    require(Counter((c['case'],c['round'],c['mode']) for c in checks if c['stage']=='performance')==Counter(expected),'performance numerical coverage')
    require(len(checks)==224,'total numerical coverage')
    for c in checks:
        require(set(c['checks'])=={'A','V','B'},'stage checks')
        for n,v in c['checks'].items():
            require(v['exact'] and v['finite'] and v['max_abs']==0 and v.get('greedy_equal',True),'numerical failure')
            if n!='V':require(v['tensors']==73,'all logits and 36-layer KV')
    eq=read(root/'full_vs_split.json');require({r['case'] for r in eq}==set(ids) and len(eq)==8,'native reference coverage')
    require(all(c['exact'] and c['finite'] and c['greedy_equal'] and c['tensors']==73 for r in eq for c in r['checks'].values()),'full vs split equivalence')
    for c in cases:
        require(c['A']['language_tokens']==512,'A context')
        for m in ['serial','parallel']:
            require(Counter(r['order'] for r in rows if r['case']==c['id'] and r['mode']==m)==Counter(LV=6,VL=6),'submission-order balance')
    return rows,checks,cases


def analyze_case(folder):
    evidence=extract(folder);g=evidence['graph'];tasks=evidence['tasks'];records=evidence['observations'];trials=read(folder/'trials.json')
    require(len(trials)==4 and len({(t['mode'],t['repeat']) for t in trials})==4,'diagnostic coverage')
    summary=[];required=[];cross=[]
    for trial in trials:
        ident=trial['id'];ts=[t for t in tasks if t['trial']==ident];compute=[t for t in ts if t['is_compute']]
        stages={n:sorted((t for t in ts if t['stage']==n),key=lambda t:number(t['start_us'])) for n in ['language_A','vision_B','merge_B','language_B']}
        require(all(stages.values()),'missing stage tasks')
        groups={n:[t for t in compute if t.get('task')==n] for n in ['A','V','B']}
        require(len(compute)==sum(map(len,groups.values())),'unattributed compute')
        for vs in groups.values():vs.sort(key=lambda t:number(t['start_us']))
        streams=sorted({t['stream'] for t in compute});require(len(streams)==(1 if trial['mode']=='serial' else 2),'compute stream mapping')
        origin=only((t for t in ts if t.get('role')=='origin' and t['name']=='EVENT_RECORD'),'origin event')
        for n,stage in [('A','language_A'),('V','vision_B')]:
            required.append(dict(source=origin['id'],target=stages[stage][0]['id'],kind='input_readiness',trial=ident,task=n))
        for n,stage in [('A','language_A'),('V','vision_B'),('B','language_B')]:
            join=only((r for r in records if r['trial']==ident and r.get('role')==n+'_join'),'stage host join')
            required.append(dict(source=stages[stage][-1]['id'],target='h:'+join['label'],kind='output_completion',trial=ident,task=n))
        vision=only((r for r in records if r['trial']==ident and r['kind']=='vision_B'),'vision observation')
        merge=only((r for r in records if r['trial']==ident and r['kind']=='merge_B'),'merge observation')
        language=only((r for r in records if r['trial']==ident and r['kind']=='language_B'),'language observation')
        require(vision['feature_output']==merge['feature_input']==trial['features'],'feature storage identity')
        require(merge['embedding_output']==language['embedding_input'],'embedding storage identity')
        consumers=[t for t in stages['merge_B'] if t['host_operator'] and 'maskedscatter' in t['host_operator'].lower().replace('_','')]
        require(bool(consumers),'actual feature consumer kernels')
        required.extend([dict(source=stages['vision_B'][-1]['id'],target=consumers[0]['id'],kind='visual_feature_ready',trial=ident,
            producer_scope=vision['label'],consumer_scope=merge['label'],storage=vision['feature_output'],evidence='same tensor address, source-pinned masked_scatter consumer; protected by V_done event'),
            dict(source=stages['merge_B'][-1]['id'],target=stages['language_B'][0]['id'],kind='merged_embedding_ready',trial=ident,storage=merge['embedding_output']),
            dict(source=stages['language_A'][-1]['id'],target=stages['language_B'][0]['id'],kind='language_stream_order',trial=ident)])
        first,second=('A','V') if trial['order']=='LV' else ('V','A')
        cross.append(dict(source=groups[first][-1]['id'],target=groups[second][0]['id'],kind='explicit_cross_stage_order',trial=ident,expected=trial['mode']=='serial'))
        # Check real live byte extents across request KV, feature and embeddings.
        layouts={n:[o['logits']]+[t for l in o['kv'] for t in l.values()] for n,o in trial['outputs'].items()}
        layouts['V']=[vision['feature_output']];layouts['embedding_B']=[merge['embedding_output']]
        for n,xs in layouts.items():
            for m,ys in layouts.items():
                if n>=m:continue
                require(all(a1<=b0 or b1<=a0 for a0,a1 in map(bounds,xs) for b0,b1 in map(bounds,ys)),'live mutable storage alias')
        both=overlap(intervals(groups['A']),intervals(groups['V']))
        if trial['mode']=='serial':require(both==0,'serial compute overlap')
        require(overlap(intervals(groups['B']),intervals(groups['V']))==0,'consumer before vision completes')
        summary.append(dict(id=ident,case=trial['case'],mode=trial['mode'],order=trial['order'],repeat=trial['repeat'],streams=streams,tasks=len(ts),compute_tasks=len(compute),
            overlap_us=float(both),compute_busy_us={n:float(duration(intervals(vs))) for n,vs in groups.items()},
            span_us=float(max(number(t['end_us']) for t in ts)-min(number(t['start_us']) for t in ts)),
            vision_host_scope_us=float(number(evidence['scopes'][vision['label']]['dur'])),
            vision_native_syncs=sum(x['scope']==vision['label'] for x in evidence['native_syncs']),
            node_ids=[t['id'] for t in ts]))
    verified=g.verify(required);queried=g.verify(cross)
    require(all(r['satisfied'] for r in verified),'unprotected data/readiness/completion')
    require(all(r['satisfied']==r['expected'] for r in queried),'explicit cross-stage ordering')
    data=dict(nodes=list(g.nodes.values()),edges=g.edges,requirements=verified,cross_task_order=queried,records=records,
        host_reads=evidence['host_reads'],native_syncs=evidence['native_syncs'])
    counts=dict(device_tasks=len(tasks),compute_tasks=sum(t['is_compute'] for t in tasks),event_wait_edges=len(evidence['event_edges']),api_only_waits=len(evidence['api_waits']),same_stream_waits=evidence['same_stream_waits'],outside_tasks=len(evidence['outside']),collision_resolutions=evidence['collisions'])
    return data,summary,counts,evidence['mapping']


def analyze(root):
    rows,checks,cases=validate(root);performance=[];summaries=[];aggregate={k:[] for k in ['nodes','edges','requirements','cross_task_order','records','host_reads','native_syncs']};counts=Counter();mappings={}
    for case in cases:
        ident=case['id'];xs=[r for r in rows if r['case']==ident];modes={}
        for mode in ['serial','parallel']:
            rs=[r for r in xs if r['mode']==mode]
            modes[mode]=dict(wall_us=stats([r['wall_us'] for r in rs]),pairs_per_second=stats([1e6/r['wall_us'] for r in rs]),
                ready_us={n:stats([r['ready_us'][n] for r in rs]) for n in ['A','V','B']},
                allocated_peak=stats([r['allocated_peak'] for r in rs]),incremental_peak=stats([r['allocated_peak']-r['allocated_before'] for r in rs]),
                host_submit_us={n:stats([r['host_submission_us'][n] for r in rs]) for n in ['L','V','B']})
        paired={}
        for order in ['all','LV','VL']:
            changes=[]
            for r in xs:
                if r['mode']!='parallel' or (order!='all' and r['order']!=order):continue
                serial=only((s for s in xs if s['round']==r['round'] and s['mode']=='serial'),'paired serial')
                changes.append(100*(r['wall_us']/serial['wall_us']-1))
            paired[order]=dict(change_percent=stats(changes),faster_pairs=sum(v<0 for v in changes))
        by_order={order:{mode:dict(wall_us=stats([r['wall_us'] for r in xs if r['order']==order and r['mode']==mode]),
            ready_us={n:stats([r['ready_us'][n] for r in xs if r['order']==order and r['mode']==mode]) for n in ['A','V','B']})
            for mode in ['serial','parallel']} for order in ['LV','VL']}
        performance.append(dict(case=ident,modes=modes,paired=paired,by_order=by_order,visual_tokens=case['B']['visual_tokens'],B_language_tokens=case['B']['language_tokens']))
        data,ts,cs,mapping=analyze_case(root/'diagnostic'/ident);counts.update(cs);mappings[ident]=mapping
        # Trace indices are local to a profiler capture; namespace graph IDs.
        def rename(x):return ident+':'+x
        for n in data['nodes']:n['id']=rename(n['id'])
        for kind in ['edges','requirements','cross_task_order']:
            for e in data[kind]:e['source']=rename(e['source']);e['target']=rename(e['target'])
        for t in ts:t['node_ids']=[rename(x) for x in t['node_ids']]
        summaries.extend(ts)
        for k in aggregate:aggregate[k].extend(data[k])
        print('ANALYZED',ident,cs,flush=True)
    expected_diag=Counter((t['case'],t['repeat'],t['mode']) for t in summaries)
    require(expected_diag==Counter((r['case'],r['round'],r['mode']) for r in checks if r['stage']=='diagnostic'),'diagnostic numerical identity')
    summary=dict(status='analysis_passed',performance_samples=len(rows),numerical_pairs=len(checks),diagnostic_trials=len(summaries),native_equivalence_pairs=8,
        all_exact=True,max_abs=0,boundary_requirements=len(aggregate['requirements']),unsatisfied=0,cross_stage_order_checks=len(aggregate['cross_task_order']),
        cases=cases,performance=performance,trials=summaries,stream_mapping=mappings,**counts,
        native_vision_sync_calls=len(aggregate['native_syncs']),complete_exact_kernel_memory_dag=False,
        scope='HF BF16 eager stage harness; A already encoded, B vision followed by B language; one CPU submitter, independent KV/MRoPE',
        hb_boundary='explicit stream/event/host joins; implicit native host blocking recorded but not a complete CPU happens-before graph')
    aggregate['summary']=summary;out=root/'analysis';out.mkdir(exist_ok=True);save(out/'summary.json',summary)
    (out/'execution_graph.json').write_text(json.dumps(aggregate,ensure_ascii=False,separators=(',',':'))+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['cases','performance','trials','stream_mapping']},indent=2));return aggregate


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);analyze(p.parse_args().run)
