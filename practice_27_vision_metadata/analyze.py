"""Rebuild P27 evidence with P25 exact flow and independent dependency checks."""
import argparse
from collections import Counter
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
P25=HERE.parent/'practice_25_multimodal_overlap'
sys.path.insert(0,str(P25))
spec=importlib.util.spec_from_file_location('p25_analyze',P25/'analyze.py')
p25=importlib.util.module_from_spec(spec);spec.loader.exec_module(p25)
from trace_graph import read,require
from sync_audit import audit
stats=p25.stats
VARIANTS=('native','lengths','cached')


def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def validate(root):
    complete=read(root/'completed.json')
    require(complete==dict(status='passed',smoke=False,samples=576,numerical_pairs=816,native_cases=8),'incomplete run')
    contracts=read(HERE/'contracts.json');sources=read(root/'sources.json')
    require(set(sources)==set(contracts['sources']),'source inventory')
    for name,item in sources.items():
        require(hashlib.sha256((root/'sources'/name).read_bytes()).hexdigest()==item['sha256']==contracts['sources'][name],'source '+name)
    require(read(root/'weight_check.json')==dict(unchanged=True,shared_rope_deltas=None),'shared state changed')
    require(read(root/'environment.json')['parameter_dtypes']==['torch.bfloat16'],'precision')
    plan=read(root/'plan.json')
    require((plan['rounds'],plan['warmup'],plan['smoke'],plan['atol'],plan['rtol'])==(12,3,False,0,0),'measurement plan')
    cases=read(root/'cases.json');rows=read(root/'measurements.json');checks=read(root/'correctness.json')
    require(len(cases)==8 and len({c['id'] for c in cases})==8,'cases')
    expected={(c['id'],v,m,r,'LV' if r%2==0 else 'VL') for c in cases for v in VARIANTS for m in ('serial','parallel') for r in range(12)}
    identity=lambda r:(r['case'],r['variant'],r['mode'],r['round'],r['order'])
    require(len(rows)==576 and {identity(r) for r in rows}==expected,'performance coverage')
    require(Counter(identity(r) for r in checks if r['stage']=='performance')==Counter(expected),'performance checks')
    expected_diag={(c['id'],v,m,r,o) for c in cases for v in VARIANTS for m in ('serial','parallel') for r,o in enumerate(('LV','VL'))}
    require(Counter(identity(r) for r in checks if r['stage']=='diagnostic')==Counter(expected_diag),'diagnostic checks')
    require(len(checks)==816,'all checks')
    captured=[]
    for case in cases:
        for variant in VARIANTS:
            ts=read(root/'diagnostic'/variant/case['id']/'trials.json')
            for t in ts:
                require(t['id']==f"{variant}-{case['id']}-r{t['repeat']}-{t['mode']}",'trial identity')
                require(t['variant']==variant and t['case']==case['id'],'trial folder identity')
                captured.append((t['case'],t['variant'],t['mode'],t['repeat'],t['order']))
    require(Counter(captured)==Counter(expected_diag),'captured diagnostics')
    for row in checks:
        require(set(row['checks'])=={'A','B','V'},'outputs')
        for n,c in row['checks'].items():
            require(c['exact'] and c['finite'] and c['max_abs']==0 and c.get('greedy_equal',True),'numerical failure')
            if n!='V':require(c['tensors']==73,'KV coverage')
    eq=read(root/'full_vs_split.json')
    require(len(eq)==8 and {r['case'] for r in eq}=={c['id'] for c in cases},'native reference')
    require(all(c['exact'] and c['finite'] and c['max_abs']==0 and c['greedy_equal'] and c['tensors']==73 for r in eq for c in r['checks'].values()),'native mismatch')
    return cases,rows


def performance(ident, rows):
    xs=[r for r in rows if r['case']==ident];modes={}
    for mode in ('serial','parallel'):
        rs=[r for r in xs if r['mode']==mode]
        modes[mode]=dict(wall_us=stats([r['wall_us'] for r in rs]),pairs_per_second=stats([1e6/r['wall_us'] for r in rs]),
            ready_us={n:stats([r['ready_us'][n] for r in rs]) for n in ('A','V','B')},
            allocated_peak=stats([r['allocated_peak'] for r in rs]),incremental_peak=stats([r['allocated_peak']-r['allocated_before'] for r in rs]),
            host_submit_us={n:stats([r['host_submission_us'][n] for r in rs]) for n in ('L','V','B')})
    paired={}
    for order in ('all','LV','VL'):
        changes=[]
        for r in xs:
            if r['mode']!='parallel' or (order!='all' and r['order']!=order):continue
            baseline=p25.only((s for s in xs if s['round']==r['round'] and s['mode']=='serial'),'serial baseline')
            changes.append(100*(r['wall_us']/baseline['wall_us']-1))
        paired[order]=dict(change_percent=stats(changes),faster_pairs=sum(v<0 for v in changes))
    by_order={o:{m:dict(wall_us=stats([r['wall_us'] for r in xs if r['order']==o and r['mode']==m]),
        ready_us={n:stats([r['ready_us'][n] for r in xs if r['order']==o and r['mode']==m]) for n in ('A','V','B')},
        host_submit_us={n:stats([r['host_submission_us'][n] for r in xs if r['order']==o and r['mode']==m]) for n in ('L','V','B')})
        for m in ('serial','parallel')} for o in ('LV','VL')}
    return dict(case=ident,modes=modes,paired=paired,by_order=by_order)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path);args=parser.parse_args();root=args.run
    cases,rows=validate(root);out=root/'analysis';out.mkdir(exist_ok=True)
    global_summary=dict(status='passed',performance_samples=576,numerical_pairs=816,diagnostic_trials=96,native_cases=8,variants={},
        preparation=read(root/'preparation.json'),complete_kernel_memory_dag=False)
    for variant in VARIANTS:
        aggregate={k:[] for k in ('nodes','edges','requirements','cross_task_order','records','host_reads','native_syncs')}
        counts=Counter();trials=[];perf=[]
        for case in cases:
            ident=case['id'];data,ts,cs,mapping=p25.analyze_case(root/'diagnostic'/variant/ident);counts.update(cs)
            rename=lambda x:ident+':'+x
            for n in data['nodes']:n['id']=rename(n['id'])
            for kind in ('edges','requirements','cross_task_order'):
                for e in data[kind]:e['source']=rename(e['source']);e['target']=rename(e['target'])
            for t in ts:t['node_ids']=[rename(x) for x in t['node_ids']]
            trials.extend(ts)
            for k in aggregate:aggregate[k].extend(data[k])
            perf.append(performance(ident,[r for r in rows if r['variant']==variant]))
            save(out/f'{variant}-{ident}-sync.json',audit(root/'diagnostic'/variant/ident,data['native_syncs']))
            print('ANALYZED',variant,ident,flush=True)
        summary=dict(status='passed',performance=perf,trials=trials,**counts,
            boundary_requirements=len(aggregate['requirements']),unsatisfied=sum(not r['satisfied'] for r in aggregate['requirements']),
            native_vision_sync_calls=len(aggregate['native_syncs']))
        aggregate['summary']=summary
        with gzip.GzipFile(filename=str(out/f'{variant}-graph.json.gz'),mode='wb',mtime=0) as f:
            f.write((json.dumps(aggregate,ensure_ascii=False,separators=(',',':'))+'\n').encode())
        compact=dict(summary);compact['trials']=[{k:v for k,v in t.items() if k!='node_ids'} for t in trials]
        global_summary['variants'][variant]=compact
        del aggregate,data,summary
    contrasts=[]
    for case in cases:
        for variant in ('lengths','cached'):
            for order in ('LV','VL'):
                for mode in ('serial','parallel'):
                    xs=[r for r in rows if (r['case'],r['variant'],r['order'],r['mode'])==(case['id'],variant,order,mode)]
                    changes=[]
                    for r in xs:
                        baseline=p25.only((s for s in rows if (s['case'],s['variant'],s['mode'],s['round'])==(r['case'],'native',mode,r['round'])),'native baseline')
                        changes.append(100*(r['wall_us']/baseline['wall_us']-1))
                    contrasts.append(dict(case=case['id'],variant=variant,order=order,mode=mode,change_percent=stats(changes)))
    global_summary['native_contrasts']=contrasts
    save(out/'summary.json',global_summary)
    print('COMPLETE',flush=True)


if __name__=='__main__':main()
