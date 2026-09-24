"""Verify synchronization ablation: completed timing, pending reads and real flows."""
import argparse
from collections import defaultdict, Counter
import csv
from decimal import Decimal
import json
from pathlib import Path
import statistics

from analyze_parallel import digest, end, inside, number, only, require


EXPECTED={'mm':1.0,'mul':.375}


def analyze(run,meta=None,events=None):
    meta=meta if meta is not None else json.loads((run/'sync_run.json').read_text())
    for name,h in meta['source_sha256'].items():
        require(digest(run/name)==h,'source fingerprint')
    for group in ['outputs','samples']:
        require(len(meta[group])==meta['rounds'],'resource rounds')
        ranges=sorted((int(t['address']),int(t['address'])+t['bytes'])
                      for pair in meta[group] for t in pair.values())
        require(all(a[1]<=b[0] for a,b in zip(ranges,ranges[1:])), 'round storage alias')
    require(all(t['pinned'] for pair in meta['samples'] for t in pair.values()),'unpinned CPU destination')
    perf,diagnostics,profiles=meta['performance'],meta['diagnostics'],meta['profiles']
    require(len(perf)==2*meta['repeats'] and len(diagnostics)==meta['repeats'],'repeat coverage')
    require([x['mode'] for x in profiles]==['per_round','final_only','early_read'],'profile mode coverage')
    final_output_checks=0
    for group,profiled in [(perf,False),(diagnostics,False),(profiles,True)]:
        for batch in group:
            require(batch['profiled'] is profiled,'profile/performance separation')
            require(batch['rounds']==meta['rounds'] and batch['pairs']==meta['pairs'],'workload equality')
            require(batch['begin_ns']<=batch['return_ns']<=batch['complete_ns'],'timing order')
            require(batch['completed_ns']==batch['complete_ns']-batch['begin_ns'] and
                    batch['issue_phase_ns']==batch['return_ns']-batch['begin_ns'] and
                    batch['terminal_wait_ns']==batch['complete_ns']-batch['return_ns'],'completed timing arithmetic')
            require(batch['host_wait_calls']==(2*meta['rounds'] if batch['mode']=='per_round' else 2),'wait policy count')
            require([x['round'] for x in batch['checks']]==list(range(meta['rounds'])),'final check coverage')
            for c in batch['checks']:
                require(set(c['full_output_correct'])==set(EXPECTED) and
                        all(c['full_output_correct'].values()),'final arithmetic failed')
                require(all(c['samples_after_join'][b]==[v]*4 for b,v in EXPECTED.items()),'final D2H sample mismatch')
                final_output_checks+=2
            if batch['mode']!='early_read':
                require(not batch['early_reads'],'early reads mixed into performance')
    performance={}
    for mode in ['per_round','final_only']:
        xs=[b for b in perf if b['mode']==mode]
        require(len(xs)==meta['repeats'],'performance mode coverage')
        performance[mode]=dict(repeats=len(xs),waits_per_batch=xs[0]['host_wait_calls'],
            samples=[dict(id=b['id'],issue_ms=b['issue_phase_ns']/1e6,completed_ms=b['completed_ns']/1e6,
                          terminal_wait_ms=b['terminal_wait_ns']/1e6) for b in xs],
            completed_median_ms=statistics.median(b['completed_ns']/1e6 for b in xs),
            completed_min_ms=min(b['completed_ns']/1e6 for b in xs),
            completed_max_ms=max(b['completed_ns']/1e6 for b in xs),
            issue_median_ms=statistics.median(b['issue_phase_ns']/1e6 for b in xs))
    rows=[]
    for batch in diagnostics:
        require(batch['mode']=='early_read','diagnostic mode')
        require([r['round'] for r in batch['early_reads']]==list(range(meta['rounds'])),'early read coverage')
        for r in batch['early_reads']:
            require(batch['begin_ns']<=r['read_start_ns']<=r['read_end_ns']<=batch['return_ns'],'early read timing')
            for b,v in EXPECTED.items():
                observed=r['observed'][b]
                require(len(observed)==4,'sample length')
                rows.append(dict(batch=batch['id'],round=r['round'],branch=b,expected=v,observed=observed,
                                 correct=observed==[v]*4,event_ready_after_read=r['event_ready_after_read'][b]))
    diagnostic=dict(branch_reads=len(rows),incorrect_reads=sum(not r['correct'] for r in rows),
                    not_ready_after_read=sum(not r['event_ready_after_read'] for r in rows),
                    sentinel_reads=sum(r['observed']==[-7.0]*4 for r in rows),rows=rows)
    trace=only((run/'profiler').rglob('trace_view.json'),'trace')
    if events is None:
        events=json.loads(trace.read_text(),parse_float=Decimal)
        if isinstance(events,dict):events=events['traceEvents']
    csv_path=only((run/'profiler').rglob('kernel_details.csv'),'kernel CSV')
    with csv_path.open() as f: kernels=list(csv.DictReader(f))
    recs=meta['records']
    require(len({r['label'] for r in recs})==len(recs),'duplicate scope')
    scopes={r['label']:only((e for e in events if e.get('ph')=='X' and e.get('name')==r['label']),'host scope') for r in recs}
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
        matches=[r for r in recs if inside(scopes[r['label']],h)]
        if not matches:
            outside.append(dict(trace_index=i,name=t['name'],host_operator=h['name']));continue
        r=only(matches,'device scope')
        ci,c,cflow=source(t,'HostToDevice')
        require(c['args']['connection_id']==t['args']['connection_id'],'CANN connection')
        a=t['args']
        row=dict(label=r['label'],batch=r['batch'],round=r['round'],branch=r['branch'],kind=r['kind'],
                 name=t['name'],stream=str(a['Physic Stream Id']),device_lane=str(t['pid']),task_id=str(a['Task Id']),
                 start_us=str(t['ts']),end_us=str(end(t)),duration_us=str(t['dur']),trace_index=i,
                 host_operator=h['name'],host_trace_index=hi,cann_api=c['name'],cann_trace_index=ci,
                 torch_flow=hflow,cann_flow=cflow,connection_id=str(a['connection_id']))
        if r['kind']=='compute':
            j=only((j for j,k in enumerate(kernels) if k['Name']==t['name'] and k['Stream ID'].strip()==row['stream']
                    and k['Task ID'].strip()==row['task_id'] and number(k['Start Time(us)'])==number(t['ts'])
                    and abs(number(k['Duration(us)'])-number(t['dur']))<=Decimal('.001')), 'kernel CSV identity')
            require(j not in used_csv,'duplicate kernel CSV');used_csv.add(j);row['csv_row']=j
        elif r['kind']=='record':
            require(t['name']=='EVENT_RECORD' and c['name']=='AscendCL@aclrtRecordEvent','terminal record API')
        else:
            require(r['kind']=='copy' and 'MEMCPY' in t['name'],'asynchronous sample copy')
        mappings[r['branch'],r['raw_stream_handle']].add((row['device_lane'],row['stream']))
        by_label[r['label']].append(row);tasks.append(row)
    require(len(mappings)==2 and all(len(v)==1 for v in mappings.values()) and
            len({next(iter(v)) for v in mappings.values()})==2,'independent physical streams')
    profile_summary=[]
    for batch in profiles:
        ident=batch['id'];rs=[r for r in recs if r['batch']==ident];ts=[t for t in tasks if t['batch']==ident]
        counts=Counter(t['kind'] for t in ts)
        require(counts==dict(compute=2*meta['rounds']*meta['pairs'],copy=2*meta['rounds'],record=2*meta['rounds']),
                'profile workload coverage')
        waits=[]
        for r in rs:
            if r['kind']!='host_wait':
                only(by_label[r['label']],'one device task per operation');continue
            scope=scopes[r['label']]
            call=only((e for e in events if e.get('name')=='AscendCL@aclrtSynchronizeEvent' and
                       e.get('args',{}).get('Thread Id',e.get('tid'))==scope['tid'] and
                       number(scope['ts'])<=number(e['ts']) and end(e)<=end(scope)), 'native event synchronize')
            record=only((t for t in ts if t['round']==r['round'] and t['branch']==r['branch'] and t['kind']=='record'),'wait terminal')
            require(number(record['end_us'])<=end(call),'wait returned before record')
            waits.append(dict(round=r['round'],branch=r['branch'],record_label=record['label'],
                              device_record_end_us=record['end_us'],host_wait_end_us=str(end(call))))
        require(len(waits)==batch['host_wait_calls'],'native host wait coverage')
        require(sorted((w['round'],w['branch']) for w in waits)==
                [(r,b) for r in (range(meta['rounds']) if batch['mode']=='per_round' else [meta['rounds']-1])
                 for b in ['mm','mul']], 'native wait placement')
        # Check submit order AND the device stream order, including every sample
        # copy after its own producer and before its own terminal marker.
        for b in ['mm','mul']:
            branch=[t for t in ts if t['branch']==b]
            branch.sort(key=lambda t:number(scopes[t['label']]['ts']))
            expected=[(r,k) for r in range(meta['rounds']) for k in
                      ['compute']*meta['pairs']+['copy','record']]
            require([(t['round'],t['kind']) for t in branch]==expected,'branch submission order')
            require(len({t['stream'] for t in branch})==1,'branch stream changed')
            require(all(number(a['end_us'])<=number(z['start_us']) for a,z in zip(branch,branch[1:])),
                    'copy/record/compute stream order')
        profile_summary.append(dict(id=ident,mode=batch['mode'],task_counts=dict(counts),host_waits=waits,
                                    device_span_us=str(max(number(t['end_us']) for t in ts)-min(number(t['start_us']) for t in ts))))
    reduction=(1-performance['final_only']['completed_median_ms']/performance['per_round']['completed_median_ms'])*100
    return dict(performance=performance,median_completed_reduction_percent=reduction,diagnostic=diagnostic,
                profile_summary=profile_summary,tasks=tasks,excluded_device_tasks=outside,
                summary=dict(all_final_outputs_correct=True,full_tensor_checks=final_output_checks,
                             profiled_compute_tasks=len(used_csv),profiled_sample_copies=sum(t['kind']=='copy' for t in tasks),
                             profiled_event_records=sum(t['kind']=='record' for t in tasks),
                             profiled_host_waits=sum(len(p['host_waits']) for p in profile_summary)),
                provenance=dict(metadata_sha256=digest(run/'sync_run.json'),trace_sha256=digest(trace),
                                kernel_csv_sha256=digest(csv_path),analyzer_sha256=digest(Path(__file__))),
                limits=meta['limits'])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path);args=parser.parse_args()
    data=analyze(args.run)
    out=args.run/'analysis';out.mkdir(exist_ok=True)
    (out/'sync_evidence.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    from render_sync import render
    render(data,out)
    print(json.dumps(dict(summary=data['summary'],performance=data['performance'],
                         reduction_percent=data['median_completed_reduction_percent'],
                         early_reads=data['diagnostic']['branch_reads'],incorrect_reads=data['diagnostic']['incorrect_reads']),indent=2))


if __name__=='__main__':main()
