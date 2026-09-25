"""Verify profiler flows and measure device-kernel overlap (stdlib, Python 3.7+)."""
import argparse
from collections import defaultdict
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path


def number(value):
    return Decimal(str(value).strip())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def only(values, message):
    values = list(values)
    require(len(values)==1, message+' (found %d)'%len(values))
    return values[0]


def end(event):
    return number(event['ts'])+number(event.get('dur', 0))


def inside(outer, inner):
    return (outer['pid']==inner['pid'] and outer['tid']==inner['tid'] and
            number(outer['ts'])<=number(inner['ts']) and end(inner)<=end(outer))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def union_length(intervals):
    intervals = sorted((number(a),number(b)) for a,b in intervals)
    if not intervals:
        return Decimal(0)
    left, right = intervals[0]
    total = Decimal(0)
    for a,b in intervals[1:]:
        if a>right:
            total += right-left
            left,right = a,b
        else:
            right = max(right,b)
    return total+right-left


def overlap_metrics(tasks):
    """Count intersection union, not envelope overlap or summed double counts."""
    branches = {key:[t for t in tasks if t['branch']==key] for key in ['mm','mul']}
    require(all(branches.values()), 'missing compute branch')
    overlaps = []
    for a in branches['mm']:
        for b in branches['mul']:
            left=max(number(a['start_us']),number(b['start_us']))
            right=min(number(a['end_us']),number(b['end_us']))
            if right>left:
                overlaps.append(dict(mm=a['label'], mul=b['label'],
                                     start_us=str(left),end_us=str(right),duration_us=str(right-left)))
    busy = {k:union_length((t['start_us'],t['end_us']) for t in v) for k,v in branches.items()}
    overlap = union_length((t['start_us'],t['end_us']) for t in overlaps)
    start=min(number(t['start_us']) for t in tasks)
    finish=max(number(t['end_us']) for t in tasks)
    return dict(start_us=str(start),end_us=str(finish),device_span_us=str(finish-start),
                branch_busy_us={k:str(v) for k,v in busy.items()}, overlap_us=str(overlap),
                overlap_fraction_shorter_branch=str(overlap/min(busy.values())),
                overlapping_pairs=len(overlaps), overlaps=overlaps)


def analyze(run, metadata=None, events=None):
    meta = metadata if metadata is not None else json.loads((run/'run.json').read_text())
    for name, expected in meta['source_sha256'].items():
        require(digest(run/name)==expected, 'source fingerprint mismatch')
    trace = only((run/'profiler').rglob('trace_view.json'), 'trace')
    if events is None:
        events = json.loads(trace.read_text(),parse_float=Decimal)
        if isinstance(events,dict):
            events = events['traceEvents']
    kernel_csv = only((run/'profiler').rglob('kernel_details.csv'), 'kernel CSV')
    with kernel_csv.open() as stream:
        kernels=list(csv.DictReader(stream))
    trials=meta['trials']
    require(len(trials)==meta['rounds']*2, 'trial coverage')
    require(len({t['id'] for t in trials})==len(trials), 'duplicate trial')
    for mode in ['serial','parallel']:
        require(sum(t['mode']==mode for t in trials)==meta['rounds'], 'mode coverage')
    checks={c['trial']:c for c in meta['checks']}
    require(len(checks)==len(trials) and all(checks[t['id']]['matrix_all_one'] and
            checks[t['id']]['vector_all_point375'] for t in trials), 'numerical validation failed')
    resources=meta['resources']
    require(set(resources)=={'matrix','matrix_out','vector','vector_out'}, 'resource coverage')
    ranges=sorted((int(t['data_ptr']),int(t['data_ptr'])+t['bytes']) for t in resources.values())
    require(all(a[1]<=b[0] for a,b in zip(ranges,ranges[1:])), 'branch storage alias')
    records=meta['records']
    require([r['sequence'] for r in records]==list(range(len(records))), 'host sequence')
    require(all(a['host_end_ns']<=b['host_start_ns'] for a,b in zip(records,records[1:])), 'host order')
    scopes={}
    for r in records:
        scopes[r['label']]=only((e for e in events if e.get('ph')=='X' and e.get('name')==r['label']), 'host scope')
    complete,starts,finishes=defaultdict(list),defaultdict(list),defaultdict(list)
    def point(e):
        return e['pid'],e['tid'],number(e['ts'])
    for i,e in enumerate(events):
        if e.get('ph')=='X':
            complete[point(e)].append((i,e))
        elif e.get('ph')=='s':
            starts[e.get('cat'),str(e['id'])].append(e)
        elif e.get('ph')=='f':
            finishes[(e.get('cat'),)+point(e)].append(e)
    def source(task,cat):
        finish=only(finishes[(cat,)+point(task)], cat+' endpoint')
        start=only(starts[cat,str(finish['id'])], cat+' start')
        index,event=only(complete[point(start)], cat+' source')
        return index,event,str(finish['id'])
    tasks=[]
    excluded=[]
    used_csv=set()
    scope_tasks=defaultdict(list)
    mappings=defaultdict(set)
    for i,task in enumerate(events):
        if task.get('ph')!='X' or 'Task Type' not in task.get('args',{}):
            continue
        if task['name'] in {'PROFILING_ENABLE','PROFILING_DISABLE'}:
            excluded.append(dict(trace_index=i,name=task['name'],reason='profiler control'))
            continue
        hi,host,hflow=source(task,'async_npu')
        matches=[r for r in records if inside(scopes[r['label']],host)]
        if not matches:
            # Validation kernels are deliberately outside measured trial scopes.
            validation=only((e for e in events if e.get('ph')=='X' and
                             e.get('name','').startswith('P16/check/') and inside(e,host)),
                            'unaccounted device task outside trials')
            excluded.append(dict(trace_index=i,name=task['name'],reason=validation['name']))
            continue
        rec=only(matches,'operation association')
        ci,cann,cflow=source(task,'HostToDevice')
        require(cann['args']['connection_id']==task['args']['connection_id'], 'CANN connection mismatch')
        a=task['args']
        t=dict(label=rec['label'],trial=rec['trial'],kind=rec['kind'],branch=rec.get('branch'),
               name=task['name'],task_type=a['Task Type'],stream=str(a['Physic Stream Id']),
               device_lane=str(task['pid']),task_id=str(a['Task Id']),trace_index=i,
               start_us=str(task['ts']),end_us=str(end(task)),duration_us=str(task['dur']),
               host_operator=host['name'],host_trace_index=hi,cann_api=cann['name'],cann_trace_index=ci,
               torch_flow=hflow,cann_flow=cflow,connection_id=str(a['connection_id']),
               raw_stream_handle=rec['raw_stream_handle'],host_stream_id=rec['host_stream_id'])
        require(number(t['duration_us'])>0,'non-positive device duration')
        if rec['kind']=='compute':
            match=only((j for j,row in enumerate(kernels) if row['Name']==task['name'] and
                        row['Stream ID'].strip()==t['stream'] and row['Task ID'].strip()==t['task_id'] and
                        number(row['Start Time(us)'])==number(task['ts']) and
                        abs(number(row['Duration(us)'])-number(task['dur']))<=Decimal('.001')), 'kernel CSV identity')
            require(match not in used_csv,'duplicate kernel CSV')
            used_csv.add(match)
            t['kernel_csv_row']=match
        else:
            require(rec['kind']=='event_record' and task['name']=='EVENT_RECORD' and
                    cann['name']=='AscendCL@aclrtRecordEvent','unexpected device synchronization')
        mappings[rec['raw_stream_handle']].add((t['device_lane'],t['stream']))
        tasks.append(t)
        scope_tasks[rec['label']].append(t)
    require(len(mappings)==2 and all(len(v)==1 for v in mappings.values()),'stream handle mapping')
    require(len({next(iter(v)) for v in mappings.values()})==2, 'physical stream distinction')
    summaries=[]
    for trial in trials:
        ident,mode=trial['id'],trial['mode']
        rs=[r for r in records if r['trial']==ident]
        expected=[kind+'-%02d'%i for i in range(meta['pairs']) for kind in ['mm','mul']]
        names=['A'] if mode=='serial' else ['A','B']
        expected += ['record-'+n for n in names]+['wait-'+n for n in names]
        require([r['name'] for r in rs]==expected,'trial operation coverage/order')
        ts=[t for t in tasks if t['trial']==ident]
        compute=[t for t in ts if t['kind']=='compute']
        require(len(compute)==meta['pairs']*2,'compute coverage')
        lanes=defaultdict(list)
        for t in ts:
            lanes[t['stream']].append(t)
        require(len(lanes)==(1 if mode=='serial' else 2),'mode physical streams')
        for lane in lanes.values():
            lane.sort(key=lambda t:number(t['start_us']))
            require(all(number(a['end_us'])<=number(b['start_us']) for a,b in zip(lane,lane[1:])),
                    'overlap within same physical stream')
        boundaries=[]
        for rec in rs:
            if rec['kind']!='host_wait':
                t=only(scope_tasks[rec['label']],'one device task per operation')
                if rec['kind']=='compute':
                    branch=rec['branch']
                    require(rec['reads']==(['matrix'] if branch=='mm' else ['vector']) and
                            rec['writes']==(['matrix_out'] if branch=='mm' else ['vector_out']), 'resource contract')
                    require(rec['stream_name']==('A' if branch=='mm' or mode=='serial' else 'B'), 'stream contract')
                continue
            previous=only((r for r in rs if r['name']=='record-'+rec['stream_name']),'terminal record')
            require(previous['event_handle']==rec['event_handle'],'terminal event identity')
            terminal=only(scope_tasks[previous['label']],'terminal device task')
            require(all(number(t['end_us'])<=number(terminal['start_us']) for t in compute
                        if t['stream']==terminal['stream']),'terminal record before compute')
            scope=scopes[rec['label']]
            wait=only((e for e in events if e.get('name')=='AscendCL@aclrtSynchronizeEvent' and
                       e.get('args',{}).get('Thread Id',e.get('tid'))==scope['tid'] and
                       number(scope['ts'])<=number(e['ts']) and end(e)<=end(scope)), 'native host wait')
            require(number(terminal['end_us'])<=end(wait),'host wait before event completion')
            boundaries.append(dict(stream=terminal['stream'],event_handle=rec['event_handle'],
                                   record_label=previous['label'],wait_label=rec['label'],
                                   device_record_end_us=terminal['end_us'],host_wait_end_us=str(end(wait))))
        summary=dict(id=ident,mode=mode,physical_streams=sorted(lanes),compute_tasks=len(compute),
                     host_elapsed_ns=trial['host_elapsed_ns'],completion_boundaries=boundaries,
                     **overlap_metrics(compute))
        if mode=='serial':
            require(number(summary['overlap_us'])==0,'serial branch overlap')
        summaries.append(summary)
    require(len(used_csv)==meta['rounds']*meta['pairs']*4,'total compute coverage')
    require(len(records)==sum(len([r for r in records if r['trial']==t['id']]) for t in trials), 'orphan records')
    return dict(trials=summaries,tasks=tasks,excluded_device_tasks=excluded,
                stream_mappings=[dict(raw_handle=k,device_lane=next(iter(v))[0],physical_stream=next(iter(v))[1])
                                 for k,v in mappings.items()],
                summary=dict(compute_tasks=len(used_csv),trial_count=len(summaries),
                             parallel_trials_with_overlap=sum(t['mode']=='parallel' and number(t['overlap_us'])>0 for t in summaries),
                             all_outputs_correct=True,terminal_event_waits=sum(len(t['completion_boundaries']) for t in summaries)),
                provenance=dict(trace_sha256=digest(trace),csv_sha256=digest(kernel_csv),
                                metadata_sha256=digest(run/'run.json'),analyzer_sha256=digest(Path(__file__))),
                limits=['Device task interval overlap; not instruction-level occupancy.',
                        'Profiled measurements are not uninstrumented throughput benchmarks.',
                        'Independent fixed-shape operators, not vLLM/KV reuse/graph/concurrent requests.',
                        'Checks run outside measured trials; no nearest-time attribution.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    args=parser.parse_args()
    result=analyze(args.run)
    output=args.run/'analysis'
    output.mkdir(exist_ok=True)
    (output/'evidence.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (output/'summary.json').write_text(json.dumps(dict(summary=result['summary'],trials=result['trials']),indent=2)+'\n')
    from render_parallel import render
    render(result,output)
    print(json.dumps(result['summary']))
    for t in result['trials']:
        print(t['id'],'span_us='+t['device_span_us'],'overlap_us='+t['overlap_us'])


if __name__=='__main__':
    main()
