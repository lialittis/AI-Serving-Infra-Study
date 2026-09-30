"""Validate diagnostic identities and regenerate the small tolist report offline."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent


def validate(d):
    assert d['outputs_equal'] and d['host_copy_values']==list(range(1,21))
    rows=d['acl']; scopes={s['id']:s for s in d['scopes']}
    assert set(scopes)=={1,2,3,4,5}
    for r in rows:
        s=scopes[r['scope']]
        assert s['begin_ns']<=r['begin_ns']<=r['end_ns']<=s['end_ns']
        assert r['tid']==d['main_tid'] and r['result']==0
        if r['op']==1: assert r['stream']==d['stream_handle']
        else:
            assert r['op']==2 and r['kind']==2 and r['bytes']==80
            assert r['src']==d['input']['ptr']
    for sid,count in ((1,1),(2,3),(3,1),(4,1),(5,0)):
        rs=[r for r in rows if r['scope']==sid]
        assert [r['op'] for r in rs]==[1,2]*count
        assert all(a['end_ns']<=b['begin_ns'] for a,b in zip(rs,rs[1:]))
    assert len(d['dispatch'])==2
    for sid,record in zip((1,4),d['dispatch']):
        copy=next(r for r in rows if r['scope']==sid and r['op']==2)
        assert copy['src']==record['input']['ptr'] and copy['dst']==record['output']['ptr']
        assert record['output']['device']=='cpu' and record['output']['pinned'] is False
        assert record['begin_ns']<=copy['begin_ns']<=copy['end_ns']<=record['end_ns']
    return {'status':'passed','d2h_calls':6,'bytes_per_copy':80,'cpu_output_pinned':False}


def main():
    results=ROOT/'results'
    d=json.loads((results/'instrument-r02/diagnostic.json').read_text())
    summary=validate(d)
    for run in ('instrument-r02','timing-r02'):
        folder=results/run
        env=json.loads((folder/'environment.json').read_text())
        after=json.loads((folder/'integrity.json').read_text())
        assert env['sha256']==after['sha256_after'] and after['unchanged']
        assert json.loads((folder/'completed.json').read_text())['status']=='passed'
    for source in json.loads((ROOT/'sources/manifest.json').read_text()):
        assert hashlib.sha256((ROOT/'sources'/source['file']).read_bytes()).hexdigest()==source['sha256']
    timing=json.loads((results/'timing-r02/timings.json').read_text())
    assert timing['outputs_equal']
    summary['timings']={}
    for name,values in timing['samples_us'].items():
        assert len(values)==1000 and min(values)>=0
        median=statistics.median(values)
        assert median==timing['summary'][name]['median_us']
        summary['timings'][name]={'n':len(values),'median_us':median}
    summary['buffer']={'src':d['input']['ptr'],'dst':d['dispatch'][0]['output']['ptr'],
                       'stream_handle':d['stream_handle']}
    summary['scope_counts']={s['name']:{'sync':sum(r['scope']==s['id'] and r['op']==1 for r in d['acl']),
         'd2h':sum(r['scope']==s['id'] and r['op']==2 for r in d['acl'])} for s in d['scopes']}
    summary['instrumented_api_us']=[{'scope':r['scope'],'op':r['op'],
        'duration_us':(r['end_ns']-r['begin_ns'])/1000} for r in d['acl']]
    (results/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
