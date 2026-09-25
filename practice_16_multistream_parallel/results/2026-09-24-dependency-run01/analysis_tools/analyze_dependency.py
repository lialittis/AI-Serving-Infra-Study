"""Verify A -> B dependency versus independent C using actual NPU task flows."""
import argparse
from collections import Counter, defaultdict
import csv
from decimal import Decimal
import json
from pathlib import Path
import statistics

from analyze_parallel import digest,end,inside,number,only,require,union_length


def intersections(left,right):
    spans=[]
    for a in left:
        for b in right:
            start=max(number(a['start_us']),number(b['start_us']))
            stop=min(number(a['end_us']),number(b['end_us']))
            if stop>start:spans.append((start,stop))
    return union_length(spans)


def analyze(run,meta=None,events=None):
    meta=meta if meta is not None else json.loads((run/'dependency_run.json').read_text())
    for name,h in meta['source_sha256'].items():require(digest(run/name)==h,'source fingerprint')
    resources=meta['resources']
    require(set(resources)=={'X','scratch','Y','Z','V','W'},'resource coverage')
    ranges=sorted((int(t['address']),int(t['address'])+t['bytes']) for t in resources.values())
    require(all(a[1]<=b[0] for a,b in zip(ranges,ranges[1:])),'unexpected storage alias')
    for group,profiled,count in [('measurements',False,meta['repeats']),('profiles',True,meta['profile_repeats'])]:
        trials=meta[group]
        require(len(trials)==count*2 and len({t['id'] for t in trials})==len(trials),'trial coverage')
        require(Counter(t['mode'] for t in trials)==dict(event_wait=count,no_wait=count),'mode coverage')
        for t in trials:
            require(t['profiled'] is profiled,'profile/performance separation')
            require(t['start_ns']<=t['submitted_ns']<=t['completed_ns'],'timing order')
            require(t['elapsed_ns']==t['completed_ns']-t['start_ns'] and
                    t['submission_ns']==t['submitted_ns']-t['start_ns'],'completed timing arithmetic')
            require(t['host_join_calls']==3,'final joins missing')
            require(t['cross_stream_wait_calls']==(1 if t['mode']=='event_wait' else 0),'wait policy')
            require(t['Y_correct'] and t['W_correct'],'producer/independent arithmetic failure')
            require(t['Z_elements']==4096*4096 and all(t[k]>=0 for k in
                    ['Z_correct_elements','Z_old_value_elements','Z_other_elements']) and
                    t['Z_correct_elements']+t['Z_old_value_elements']+t['Z_other_elements']==t['Z_elements'],'consumer element accounting')
            require(len(t['Z_first_eight'])==8,'consumer sample coverage')
            if t['Z_correct_elements']==t['Z_elements']:
                require(t['Z_first_eight']==[2.0]*8,'consumer sample/count mismatch')
            if t['Z_old_value_elements']==t['Z_elements']:
                require(t['Z_first_eight']==[-14.0]*8,'consumer sample/count mismatch')
            if t['mode']=='event_wait':
                require(t['Z_correct_elements']==t['Z_elements'] and t['Z_first_eight']==[2.0]*8,'synchronized consumer incorrect')
                require(t['recompute_after_join_correct'] is None,'unexpected repair in safe mode')
            else:
                require(t['recompute_after_join_correct'] is True,'recomputation did not repair result')
    trace=only((run/'profiler').rglob('trace_view.json'),'trace')
    if events is None:
        events=json.loads(trace.read_text(),parse_float=Decimal)
        if isinstance(events,dict):events=events['traceEvents']
    kernel_csv=only((run/'profiler').rglob('kernel_details.csv'),'kernel CSV')
    with kernel_csv.open() as f:kernels=list(csv.DictReader(f))
    records=meta['records']
    require(len({r['label'] for r in records})==len(records),'duplicate scope')
    scopes={r['label']:only((e for e in events if e.get('ph')=='X' and e.get('name')==r['label']),'host scope') for r in records}
    complete,starts,finishes=defaultdict(list),defaultdict(list),defaultdict(list)
    def point(e):return e['pid'],e['tid'],number(e['ts'])
    for i,e in enumerate(events):
        if e.get('ph')=='X':complete[point(e)].append((i,e))
        elif e.get('ph')=='s':starts[e.get('cat'),str(e['id'])].append(e)
        elif e.get('ph')=='f':finishes[(e.get('cat'),)+point(e)].append(e)
    def source(task,cat):
        finish=only(finishes[(cat,)+point(task)],cat+' endpoint')
        start=only(starts[cat,str(finish['id'])],cat+' start')
        i,e=only(complete[point(start)],cat+' source')
        return i,e,str(finish['id'])
    tasks=[];outside=[];by_label=defaultdict(list);mappings=defaultdict(set);used_csv=set()
    for i,t in enumerate(events):
        if t.get('ph')!='X' or 'Task Type' not in t.get('args',{}):continue
        if t['name'].startswith('PROFILING_'):
            outside.append(dict(trace_index=i,name=t['name']));continue
        hi,h,hflow=source(t,'async_npu')
        matches=[r for r in records if inside(scopes[r['label']],h)]
        if not matches:
            outside.append(dict(trace_index=i,name=t['name'],host_operator=h['name']));continue
        r=only(matches,'device scope')
        ci,c,cflow=source(t,'HostToDevice')
        require(c['args']['connection_id']==t['args']['connection_id'],'CANN connection')
        a=t['args']
        row=dict(label=r['label'],trial=r['trial'],branch=r['branch'],operation=r['name'],kind=r['kind'],
                 name=t['name'],task_type=a['Task Type'],stream=str(a['Physic Stream Id']),device_lane=str(t['pid']),
                 task_id=str(a['Task Id']),start_us=str(t['ts']),end_us=str(end(t)),duration_us=str(t['dur']),
                 trace_index=i,host_trace_index=hi,cann_trace_index=ci,torch_flow=hflow,cann_flow=cflow,
                 connection_id=str(a['connection_id']),cann_api=c['name'],host_operator=h['name'])
        if r['kind']=='compute':
            j=only((j for j,k in enumerate(kernels) if k['Name']==t['name'] and k['Stream ID'].strip()==row['stream']
                    and k['Task ID'].strip()==row['task_id'] and number(k['Start Time(us)'])==number(t['ts'])
                    and abs(number(k['Duration(us)'])-number(t['dur']))<=Decimal('.001')), 'kernel CSV identity')
            require(j not in used_csv,'duplicate kernel CSV');used_csv.add(j);row['csv_row']=j
        elif r['kind']=='record':
            require(t['name']=='EVENT_RECORD' and c['name']=='AscendCL@aclrtRecordEvent','record API')
        else:
            require(r['kind']=='device_wait' and t['name']=='EVENT_WAIT' and c['name']=='AscendCL@aclrtStreamWaitEvent','cross-stream wait API')
        tasks.append(row);by_label[r['label']].append(row)
        mappings[r['branch'],r['raw_stream_handle']].add((row['device_lane'],row['stream']))
    require(len(mappings)==3 and all(len(v)==1 for v in mappings.values()) and
            len({next(iter(v)) for v in mappings.values()})==3,'three physical streams')
    trials=[]
    for t in meta['profiles']:
        ident=t['id'];rs=[r for r in records if r['trial']==ident];ts=[k for k in tasks if k['trial']==ident]
        expected=['backlog-%02d'%i for i in range(meta['a_steps']-1)]+['produce-Y','record-A']
        if t['mode']=='event_wait':expected+=['wait-A']
        expected+=['consume-Y','record-B']+['independent-%02d'%i for i in range(meta['c_steps'])]+['record-C','join-A','join-B','join-C']
        require([r['name'] for r in rs]==expected,'host operation order/coverage')
        require(all(a['host_end_ns']<=b['host_start_ns'] for a,b in zip(rs,rs[1:])),'host scope order')
        record_by_name={r['name']:r for r in rs};nodes={}
        for r in rs:
            if r['kind']=='host_wait':continue
            nodes[r['name']]=only(by_label[r['label']],'one device task per operation')
            if r['kind']=='compute':
                if r['name']=='produce-Y':role=(['X'],['Y'])
                elif r['name']=='consume-Y':role=(['Y'],['Z'])
                elif r['branch']=='A':role=(['X'],['scratch'])
                else:role=(['V'],['W'])
                require((r['reads'],r['writes'])==role,'data resource contract')
        branch_tasks={b:[k for k in ts if k['branch']==b] for b in ['A','B','C']}
        for b,lane in branch_tasks.items():
            require(len({k['stream'] for k in lane})==1,'branch stream changed')
            lane.sort(key=lambda k:number(scopes[k['label']]['ts']))
            require(all(number(a['end_us'])<=number(z['start_us']) for a,z in zip(lane,lane[1:])), 'same-stream order')
        producer,consumer=nodes['produce-Y'],nodes['consume-Y']
        wait_proof=None
        if t['mode']=='event_wait':
            wr=record_by_name['wait-A'];ar=record_by_name['record-A']
            require(wr['event_handle']==ar['event_handle'],'dependency event identity')
            wait=nodes['wait-A'];record=nodes['record-A']
            require(number(record['end_us'])<=number(wait['end_us']),'wait completed before producer record')
            require(number(producer['end_us'])<=number(consumer['start_us']),'consumer before producer completion')
            wait_proof=dict(record_label=record['label'],wait_label=wait['label'],event_handle=ar['event_handle'],
                            record_end_us=record['end_us'],wait_end_us=wait['end_us'])
        joins=[]
        for b in ['A','B','C']:
            r=record_by_name['join-'+b];rec=record_by_name['record-'+b]
            require(r['event_handle']==rec['event_handle'],'join event identity')
            scope=scopes[r['label']]
            call=only((e for e in events if e.get('name')=='AscendCL@aclrtSynchronizeEvent' and
                       e.get('args',{}).get('Thread Id',e.get('tid'))==scope['tid'] and
                       number(scope['ts'])<=number(e['ts']) and end(e)<=end(scope)), 'native terminal join')
            require(number(nodes['record-'+b]['end_us'])<=end(call),'join before terminal record completion')
            joins.append(dict(branch=b,record_label=rec['label'],event_handle=rec['event_handle'],host_wait_end_us=str(end(call))))
        compute={b:[k for k in branch_tasks[b] if k['kind']=='compute'] for b in ['A','B','C']}
        require({b:len(v) for b,v in compute.items()}==dict(A=meta['a_steps'],B=1,C=meta['c_steps']),'compute coverage')
        relation=('B_finished_before_A_producer_started' if number(consumer['end_us'])<=number(producer['start_us']) else
                  'B_started_after_A_producer_finished' if number(consumer['start_us'])>=number(producer['end_us']) else
                  'producer_consumer_intervals_overlap')
        if relation=='B_finished_before_A_producer_started':
            require(t['Z_old_value_elements']==t['Z_elements'],'early consumer/stale result inconsistency')
        trials.append(dict(id=ident,mode=t['mode'],nodes=nodes,physical_streams={b:branch_tasks[b][0]['stream'] for b in branch_tasks},
                           A_C_compute_overlap_us=str(intersections(compute['A'],compute['C'])),
                           C_while_B_waiting_us=str(intersections(compute['C'],[nodes['wait-A']])) if wait_proof else None,
                           producer_consumer_relation=relation,dependency_wait=wait_proof,joins=joins,
                           Z_correct_elements=t['Z_correct_elements'],Z_old_value_elements=t['Z_old_value_elements'],Z_elements=t['Z_elements'],
                           Z_first_eight=t['Z_first_eight'],recompute_after_join_correct=t['recompute_after_join_correct']))
    performance={}
    for mode in ['event_wait','no_wait']:
        group=[t for t in meta['measurements'] if t['mode']==mode]
        performance[mode]=dict(repeats=len(group),elapsed_median_ms=statistics.median(t['elapsed_ns']/1e6 for t in group),
            elapsed_min_ms=min(t['elapsed_ns']/1e6 for t in group),elapsed_max_ms=max(t['elapsed_ns']/1e6 for t in group),
            correct_trials=sum(t['Z_correct_elements']==t['Z_elements'] for t in group),
            stale_trials=sum(t['Z_old_value_elements']==t['Z_elements'] for t in group),
            samples=[dict(id=t['id'],elapsed_ms=t['elapsed_ns']/1e6,submission_ms=t['submission_ns']/1e6,
                          correct_elements=t['Z_correct_elements'],old_elements=t['Z_old_value_elements'],
                          other_elements=t['Z_other_elements'],first_eight=t['Z_first_eight']) for t in group])
    return dict(performance=performance,trials=trials,tasks=tasks,excluded_device_tasks=outside,resources=resources,
                stream_mappings=[dict(branch=b,raw_handle=h,device_lane=next(iter(v))[0],physical_stream=next(iter(v))[1]) for (b,h),v in mappings.items()],
                summary=dict(profiled_compute_tasks=len(used_csv),profiled_event_records=sum(k['kind']=='record' for k in tasks),
                             profiled_device_waits=sum(k['kind']=='device_wait' for k in tasks),profiled_host_joins=len(trials)*3,
                             profiled_trials=len(trials),all_producer_and_independent_outputs_correct=True),
                provenance=dict(metadata_sha256=digest(run/'dependency_run.json'),trace_sha256=digest(trace),
                                kernel_csv_sha256=digest(kernel_csv),analyzer_sha256=digest(Path(__file__))),limits=meta['limits'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);args=p.parse_args()
    data=analyze(args.run);out=args.run/'analysis';out.mkdir(exist_ok=True)
    (out/'dependency_evidence.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    from render_dependency import render
    render(data,out)
    print(json.dumps(data['summary']))
    for mode,x in data['performance'].items():print(mode,x['elapsed_median_ms'],x['correct_trials'],x['stale_trials'])
    for t in data['trials']:print(t['id'],t['producer_consumer_relation'],t['A_C_compute_overlap_us'],t['C_while_B_waiting_us'])


if __name__=='__main__':main()
