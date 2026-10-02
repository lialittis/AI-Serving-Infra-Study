"""Export bounded, independently reread raw trace/CSV examples for review."""
import argparse
import csv
from decimal import Decimal
import gzip
import json
from pathlib import Path


def audit(case):
    data=json.loads(gzip.decompress((case/'analysis/analysis.json.gz').read_bytes()))
    if data['analysis_status']!='passed':raise ValueError('Incomplete analysis')
    trace=next(p for p in data['raw_sources'] if p['path'].endswith('trace_view.json'))
    csv_source=next(p for p in data['raw_sources'] if p['path'].endswith('kernel_details.csv'))
    events=json.loads((case/trace['path']).read_text(),parse_float=Decimal)
    if isinstance(events,dict):events=events['traceEvents']
    with (case/csv_source['path']).open() as f:rows=list(csv.DictReader(f))
    steps={s['key']:s for s in data['steps']}
    examples=[]
    for phase in ('prefill','decode'):
        candidates=[]
        for t in data['tasks']:
            if not t['is_compute'] or t['stage']!='forward' or 'MatMul' not in t['name']:continue
            phases={r['phase'] for r in steps[t['step']]['scheduler']['requests'].values()}
            if (phase=='prefill' and 'prefill' in phases) or (phase=='decode' and phases=={'decode'}):candidates.append(t)
        if not candidates:raise ValueError('Missing representative '+phase+' MatMul')
        target=max(candidates,key=lambda t:len(t['requests']))
        e=events[target['trace_index']];row=rows[target['csv_row']]
        assert e['name']==row['Name']==target['name']
        assert str(e['args']['Physic Stream Id'])==row['Stream ID'].strip()==target['stream']
        assert str(e['args']['Task Id'])==row['Task ID'].strip()==target['task_id']
        assert Decimal(str(e['ts']))==Decimal(row['Start Time(us)'])==Decimal(target['start_us'])
        assert abs(Decimal(str(e['dur']))-Decimal(row['Duration(us)']))<=Decimal('.001')
        assert target['cann_index'] is not None and target['host_index'] is not None
        host=events[target['host_index']];native=events[target['cann_index']]
        assert str(native['args']['connection_id'])==str(e['args']['connection_id'])
        examples.append(dict(phase=phase,task=target,raw_device=e,csv_row=row,raw_host=host,raw_native=native,
                             scheduler=steps[target['step']]['scheduler'],
                             note='requests is batch membership, not proof of per-request tensor reads/writes'))
    output=case/'analysis/raw_spot_checks.json'
    output.write_text(json.dumps(dict(case=case.name,examples=examples),ensure_ascii=False,indent=2,default=str)+'\n')
    for example in examples:
        t=example['task']
        print(case.name,example['phase'],'requests='+str(len(t['requests'])),t['name'],
              'stream='+t['stream'],'task='+t['task_id'],'trace_index='+str(t['trace_index']),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);a=p.parse_args()
    for case in sorted(a.run.glob('c*-diagnostic')):audit(case)


if __name__=='__main__':main()
