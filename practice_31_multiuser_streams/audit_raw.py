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
    if 'sampling_analysis' in data:
        records=[]
        for path in (case/'observer').glob('*.jsonl'):
            records.extend(json.loads(line) for line in path.read_text().splitlines())
        sampling_steps={s['key']:s for s in data['sampling_steps']}
        waits=[r for r in records if r['kind']=='stream_api' and r['operation']=='Stream.wait_stream' and r.get('key') in sampling_steps]
        assert len(waits)==len(sampling_steps)
        assert len({r['key'] for r in waits})==len(sampling_steps)
        for call in waits:
            s=sampling_steps[call['key']]
            assert str(call['stream']['runtime_stream_id'])==s['random_stream']
            assert str(call['self_stream']['runtime_stream_id'])==(s['random_stream'] if data['precompute'] else s['model_stream'])
        checked=[]
        for example in data['sampling_examples']:
            for evidence in example['task_evidence']:
                t=evidence['task'];raw=events[t['trace_index']]
                assert raw['name']==t['name']
                assert str(raw['args']['Physic Stream Id'])==t['stream']
                assert str(raw['args']['Task Id'])==t['task_id']
                assert Decimal(str(raw['ts']))==Decimal(t['start_us'])
                if t['is_compute']:
                    row=rows[t['csv_row']]
                    assert row['Name']==t['name'] and row['Accelerator Core'].strip()==t['core_type']
                    assert Decimal(row['Start Time(us)'])==Decimal(t['start_us'])
                checked.append(t['id'])
            sync=example['synchronization']
            if 'native_trace_index' in sync:
                raw=events[sync['native_trace_index']]
                assert raw['name']==sync['name'] and Decimal(str(raw['ts']))==Decimal(sync['start_us'])
        selected={s['key'] for s in data['sampling_examples']}
        result=dict(case=case.name,checked_tasks=sorted(set(checked)),stream_wait_steps=len(waits),
                    stream_wait_relation='auxiliary waits on itself' if data['precompute'] else 'model waits on auxiliary',
                    representative_stream_api=[r for r in records if r['kind']=='stream_api' and r.get('key') in selected
                                               and (r['operation']=='Stream.wait_stream' or r.get('q_wait'))],
                    status='passed')
        (case/'analysis/sampling_spot_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    for example in examples:
        t=example['task']
        print(case.name,example['phase'],'requests='+str(len(t['requests'])),t['name'],
              'stream='+t['stream'],'task='+t['task_id'],'trace_index='+str(t['trace_index']),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);a=p.parse_args()
    for case in sorted(a.run.glob('c*-diagnostic')):audit(case)


if __name__=='__main__':main()
