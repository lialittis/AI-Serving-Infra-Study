"""Offline evidence verification and coverage matrix; no NPU or torch needed."""
import hashlib
import json
from pathlib import Path
from spec import CHECKS,checks,make_cases
ROOT=Path(__file__).resolve().parent


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def memory_checks(row):
    ls=row['layout'];c=row['case'];p=ls['parent'];ok=True
    for name,offset in [('boundaries',c['pad']),('left',c['pad']+c['stride']),('right',c['pad'])]:
        l=ls[name];last=offset+(l['shape'][0]-1)*c['stride']
        ok &= (l['base']==p['base'] and l['offset']==offset and l['stride']==[c['stride']] and
               l['ptr']==l['base']+offset*4 and l['itemsize']==4 and
               0<=offset<=last and (last+1)*4<=l['storage_bytes'])
    out=ls['output']
    return dict(input_unchanged=row['parent_before']==row['parent_after'],layout=bool(ok),
        no_output_alias=out['base']+out['storage_bytes']<=p['base'] or out['base']>=p['base']+p['storage_bytes'])


def validate_row(row):
    c=row['case'];n=len(c['boundaries']);step=c['stride'];pad=c['pad']
    narrow=lambda x: ((x+(1<<31))%(1<<32))-(1<<31)
    expected_parent=[-777]*(pad+(n-1)*step+1+3)
    for i,value in enumerate(c['boundaries']):expected_parent[pad+i*step]=narrow(value)
    assert row['parent_before']==expected_parent,'initial parent/guard values inconsistent'
    assert row['device_boundaries']==expected_parent[pad:pad+n*step:step]
    # This models the observed int32 behavior, never the semantic reference oracle.
    device=row['device_boundaries'];delta=[b-a for a,b in zip(device,device[1:])]
    if c['operation']=='swap':delta=[-x for x in delta]
    assert row['computed_before_injection']==[narrow(x) for x in delta]
    expected_lengths={'parent':len(expected_parent),'boundaries':n,'left':n-1,'right':n-1,'output':len(row['observed'])}
    for name,l in row['layout'].items():
        assert l['shape']==[expected_lengths[name]] and l['itemsize']==4
        assert l['ptr']==l['base']+l['offset']*4,'reported effective pointer inconsistent'
    memory=memory_checks(row)
    assert memory==row['memory'],'memory evidence inconsistent'
    result=checks(row['case'],row['observed'],row['device_boundaries'],memory)
    assert result==row['checks'],'check verdicts inconsistent'
    assert all(memory.values()),'unexpected memory failure'
    assert all(result.values())==row['case']['expect_valid'],'unexpected semantic verdict'
    split=row['split']
    if split['status']=='accepted':
        assert split['sizes']==row['observed']
        assert [v for chunk in split['chunks'] for v in chunk]==list(range(row['case']['seq']))
        assert [len(chunk) for chunk in split['chunks']]==row['observed']
    return result


def main():
    run=ROOT/'results/run-r01'
    for name,digest in read(run/'checksums.json').items():assert sha(run/name)==digest,name
    for x in read(ROOT/'sources/manifest.json'):assert sha(ROOT/'sources'/x['file'])==x['sha256']
    assert read(run/'completed.json')=={'status':'passed','cases':26}
    env=read(run/'environment.json');integrity=read(run/'integrity.json')
    assert integrity['unchanged'] and env['sha256_before']==integrity['sha256_after']
    assert sha(run/'installed_model.py')==env['model_source_sha256']
    assert env['model_source_sha256']=='9d6d15040bdb985d9518117ed029f6a318a38abc06091462d96f16cf1d3d7820'
    rows=read(run/'cases.json');expected=make_cases(read(run/'preparation.json'))
    assert [r['case'] for r in rows]==expected
    for row in rows:validate_row(row)
    models=read(run/'model_reference.json');assert len(models)==7 and all(x['exact'] and x['unique_boundaries']==x['reference'] for x in models)
    # Specific observed outcomes protect the conclusions used in the report.
    by={r['case']['id']:r for r in rows}
    for name in ('same_sum','reverse_lengths','wrong_count','duplicate','changed_boundary'):
        assert by[name]['split']['status']=='accepted' and not by[name]['checks']['exact']
    assert by['overflow_cast']['observed']==[2147483647,2]
    assert by['overflow_sub']['observed']==[-1] and by['underflow_sub']['observed']==[1]
    assert by['wrap_origin']['observed']==[3,5,2] and not by['wrap_origin']['checks']['pre_int32']
    fw=read(run/'framework.json')
    assert [r['status'] for r in fw]==['rejected','rejected','rejected','accepted']
    assert fw[-1]['value']==[-2147483648]
    upstream=read(run/'upstream.json')
    assert upstream['cumsum_int32']==[2147483647,-2147483647]
    assert upstream['cumsum_int64']==upstream['cumulative_reference']
    assert upstream['product_int32'][0]!=upstream['product_reference']
    summary=dict(case_count=len(rows),valid=sum(r['case']['expect_valid'] for r in rows),
        injected=sum(not r['case']['expect_valid'] for r in rows),
        split_counts={k:sum(r['split']['status']==k for r in rows) for k in ('accepted','rejected','skipped')},
        invalid_split_accepted=[r['case']['id'] for r in rows if not r['case']['expect_valid'] and r['split']['status']=='accepted'],
        check_labels=CHECKS,rows=[dict(id=r['case']['id'],label=r['case']['label'],expect_valid=r['case']['expect_valid'],
            failed=[k for k,v in r['checks'].items() if not v],split=r['split']['status']) for r in rows],
        framework=fw,upstream=upstream,model_reference=models,
        boundary='Single-stream correctness, injected software faults; not performance, ECC, allocator stress or final device argument decoding.')
    (ROOT/'results/summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k in ['case_count','valid','injected','split_counts','invalid_split_accepted']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
