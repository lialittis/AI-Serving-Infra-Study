"""Reuse P28 exact joins; independently check the stronger prequeue premise."""
import argparse
from collections import Counter
import csv
from decimal import Decimal as D
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
P28 = ROOT.parent / 'practice_28_native_decode_streams'
sys.path.insert(0, str(P28))
from common import save
spec = importlib.util.spec_from_file_location('p29_exact', ROOT / 'exact_join.py')
p28 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p28)


def end(event):
    return D(event['ts']) + D(str(event['dur']))


def gate_evidence(events, result):
    compute = [t for t in result['tasks'] if t['is_compute']]
    waits = [t for t in result['tasks'] if t['name'] == 'NOTIFY_WAIT']
    records = [t for t in result['tasks'] if t['name'] == 'NOTIFY_RECORD']
    calls = [e for e in events if e.get('ph') == 'X' and e.get('name') == 'AscendCL@aclrtRecordNotify']
    assert len(waits) == len(records) == len(calls) == (1 if result['trial']['strategy']=='serial' else 2)
    release = min(D(e['ts']) for e in calls)
    release_streams = {t['stream'] for t in records}
    compute_streams = {t['stream'] for t in compute}
    assert not (release_streams & compute_streams), 'release depends on blocked stream'
    assert {w['stream'] for w in waits} == compute_streams
    assert all(D(w['start_us']) <= release <= D(w['end_us']) for w in waits), 'wait did not hold until release'
    assert max(D(t['end_us']) for t in records) <= max(D(w['end_us']) for w in waits), 'waits finished before notifications'
    for task in compute:
        wait = next(w for w in waits if w['stream'] == task['stream'])
        assert D(wait['end_us']) <= D(task['start_us']), 'kernel escaped its gate'
    # CANN launch API RETURN, not merely Python return, must precede release.
    launches = [events[t['submission_index']] for t in compute]
    assert all(t['submission_index'] is not None for t in compute)
    pending = [t for t,e in zip(compute,launches) if end(e) > release]
    python_done = all(D(s['ts'])+D(s['dur']) <= release for label,s in result['cpu_scopes'].items()
                      if label in ('P28/forward/A','P28/forward/B'))
    rescued = result['trial']['timing']['rescued']
    return dict(release_native_us=str(release), gate_waits=len(waits),
        independent_release_streams=sorted(release_streams), all_compute_after_wait=True,
        kernel_launches_returned_before_release=len(compute)-len(pending),
        kernel_launches_total=len(compute), python_forward_returned_before_release=python_done,
        last_launch_return_minus_release_us=str(max(map(end,launches))-release),
        rescued=rescued, prequeue_proven=not rescued and python_done and not pending,
        pending_by_role=dict(Counter(t['role'] for t in pending)),
        evidence='frozen Notify ownership + exact CANN/device flows and CSV identities; no timestamp-only role assignment')


def analyze_probe(run):
    """Two tiny kernels: connection identity, CSV identity and native wait order."""
    root = run/'gate-probe'
    path, = root.rglob('trace_view.json')
    events = p28.read(path)
    if isinstance(events,dict): events=events['traceEvents']
    tasks = [e for e in events if e.get('ph')=='X' and 'Task Type' in e.get('args',{})]
    compute = [e for e in tasks if e['args']['Task Type']=='AI_VECTOR_CORE']
    waits = [e for e in tasks if e['name']=='NOTIFY_WAIT']
    assert len(compute)==len(waits)==2
    calls = [e for e in events if e.get('ph')=='X' and e.get('name')=='AscendCL@aclrtRecordNotify']
    assert len(calls)==2
    release = min(D(e['ts']) for e in calls)
    csv_path, = root.rglob('kernel_details.csv')
    with csv_path.open() as file: rows=list(csv.DictReader(file))
    assert len(rows)==2
    joins=[]
    for task in compute:
        args=task['args']
        launch, = [e for e in events if e.get('ph')=='X' and e.get('name')=='Node@launch'
                   and e.get('args',{}).get('connection_id')==args['connection_id']]
        row, = [r for r in rows if r['Name']==task['name'] and r['Task ID'].strip()==str(args['Task Id'])
                and r['Stream ID'].strip()==str(args['Physic Stream Id']) and D(r['Start Time(us)'])==D(task['ts'])]
        assert D(row['Duration(us)'])==D(str(task['dur']))
        wait, = [e for e in waits if e['args']['Physic Stream Id']==args['Physic Stream Id']]
        assert end(launch)<release and end(wait)<=D(task['ts']) and D(task['ts'])>release
        joins.append(dict(stream=args['Physic Stream Id'], connection=args['connection_id'],
            launch_return_minus_release_us=str(end(launch)-release),
            wait_duration_us=str(wait['dur']),kernel_duration_us=str(task['dur'])))
    checks = json.loads((root/'gate_checks.json').read_text())
    assert len(checks)==2 and all(c['passed'] for c in checks)
    result=dict(prequeue_proven=True, tiny_kernels=2, joins=joins,
        normal_release_and_watchdog_recovery_passed=True)
    save(run/'gate_analysis.json',result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    a = p.parse_args()
    analyze_probe(a.run)
    summaries, inventories = [], []
    for root in sorted((a.run/'model/trials').iterdir()):
        result = p28.analyze_trial(root, a.run/'model')
        trace = p28.read(root/result['provenance']['trace'])
        events = trace['traceEvents'] if isinstance(trace,dict) else trace
        if result['trial']['gated']:
            result['gate_evidence'] = gate_evidence(events, result)
        # Locate actual CPU-blocking runtime calls in the forwarding thread.
        sync = []
        for i,e in enumerate(events):
            if e.get('ph')!='X' or 'Synchronize' not in e.get('name','') or not e['name'].startswith('AscendCL@'):
                continue
            for r in ('A','B'):
                scope = result['cpu_scopes']['P28/forward/'+r]
                if str(e.get('args',{}).get('Thread Id',e['tid']))==scope['tid'] and D(scope['ts'])<=D(e['ts']) and end(e)<=D(scope['ts'])+D(scope['dur']):
                    containers = [x for x in events if x.get('ph')=='X' and x.get('cat')=='cpu_op' and
                        x['tid']==int(scope['tid']) and D(x['ts'])<=D(e['ts']) and end(e)<=end(x)]
                    sync.append(dict(role=r,index=i,name=e['name'],duration_us=str(e['dur']),
                        enclosing_ops=[x['name'] for x in sorted(containers,key=lambda x:D(str(x['dur'])))][:8]))
        result['forward_cpu_synchronizations'] = sync
        inventory = Counter((t['role'],t['name'],t['core_type'],t['block_num'],t['mix_block_num'])
                            for t in result['tasks'] if t['is_compute'])
        inventories.append(inventory)
        save(root/'analysis.json', result)
        summaries.append({k:result[k] for k in ('trial','compute_tasks','stream_ids','overlap_us',
            'device_compute_span_us','branch_timing','forward_cpu_synchronizations','provenance')}
            | dict(gate_evidence=result.get('gate_evidence')))
    assert len(summaries)==3
    assert all(x==inventories[0] for x in inventories), 'kernel inventory changed across schedules'
    save(a.run/'analysis_summary.json', summaries)
    save(a.run/'analysis_validation.json', dict(exact_flow_csv_joins=True,
        all_three_kernel_inventories_equal=True, traces=3, single_cpu_model_submission_thread=True))


if __name__ == '__main__':
    main()
