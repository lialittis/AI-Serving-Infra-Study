"""P26 exact association reused; P30 input adapter and queue evidence added.

The original flow/CSV identity and same-stream checks are unchanged.
"""
import argparse
from bisect import bisect_left,bisect_right
from collections import Counter,defaultdict
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import statistics
import sys
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'practice_17_vllm_multistream'))
from analyze_run import number,end,inside,only,require,union,intersection,resolve_colliding_host

def read(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def stats(values):
    vs=[float(v) for v in values]
    return dict(n=len(vs),median=statistics.median(vs),min=min(vs),max=max(vs)) if vs else None

def length(intervals):return sum((b-a for a,b in union(intervals)),Decimal(0))
def difference(window,occupied):
    cursor=number(window[0]);last=number(window[1]);result=[]
    for a,b in union(occupied):
        a=max(a,cursor);b=min(b,last)
        if a>last:break
        if a>cursor:result.append((cursor,a))
        cursor=max(cursor,b)
    if cursor<last:result.append((cursor,last))
    return result

def overlap(tasks):
    points=[]
    for t in tasks:
        if t['is_compute']:
            points.extend([(number(t['start_us']),1,t['stream']),(number(t['end_us']),-1,t['stream'])])
    count=Counter();last=None;total=Decimal(0);peak=0
    for x,delta,s in sorted(points):
        if last is not None and sum(v>0 for v in count.values())>1:total+=x-last
        count[s]+=delta;peak=max(peak,sum(v>0 for v in count.values()));last=x
    return dict(overlap_us=str(total),peak_compute_streams=peak)

def core_count(value):
    try:n=int(value)
    except (ValueError,TypeError):return None
    return n if n>0 else None

def validate_graph_chunk(chunk,dump,launch,wait):
    require(len(chunk)==len(dump)+(dump[-1]['args']['Task Type']!='NOTIFY_RECORD'),'graph chunk size')
    for t,d in zip(chunk,dump):
        require((t['stream'],t['task_id'])==(str(d['args']['Stream Id']),str(d['args']['Task Id'])),'graph stream/task sequence')
    require(chunk[-1]['name']=='NOTIFY_RECORD','graph completion marker')
    require(number(launch['start_us'])<=number(chunk[0]['start_us']) and number(chunk[-1]['end_us'])<=number(wait['end_us']),'graph completion enclosure')

def validate_csv(t,row):
    require((t['name'],t['stream'],t['task_id'],number(t['start_us']))==(row['Name'],row['Stream ID'].strip(),row['Task ID'].strip(),number(row['Start Time(us)'])),'CSV task identity')
    require(abs(number(row['Duration(us)'])-number(t['duration_us']))<=Decimal('.001'),'CSV duration mismatch')

def validate_steps(executors,tokens=64):
    require(len(executors)==tokens and [r['index'] for r in executors]==list(range(tokens)),'complete scheduling steps')
    for r in executors:
        require(list(r['scheduled'].values())==[10 if r['index']==0 else 1],'unexpected scheduling shape')

def analyze(run):
    config=dict(mode='eager', profile='plain', tokens=64)
    require(read(run/'status.json')['status']=='passed','incomplete diagnostic')
    records=read(run/'observer.json')
    measured=[r for r in records if 'label' in r]
    record_by_label={r['label']:r for r in measured}
    require(len(record_by_label)==len(measured),'duplicate scope label')
    executors=[r for r in measured if r['kind']=='execute'];executors.sort(key=lambda r:r['index'])
    validate_steps(executors,config['tokens'])
    path=only((run/'profiler').rglob('trace_view.json'),'one trace')
    events=json.loads(path.read_text(),parse_float=Decimal)
    if isinstance(events,dict):events=events['traceEvents']
    csv_path=only((run/'profiler').rglob('kernel_details.csv'),'one CSV')
    with csv_path.open() as f:rows=list(csv.DictReader(f))
    scopes={e['name']:e for e in events if e.get('ph')=='X' and e.get('name') in record_by_label}
    require(set(scopes)==set(record_by_label),'missing record_function scope')
    scope_index=defaultdict(list)
    for label,s in scopes.items():scope_index[s['tid']].append((label,s))
    for tid in scope_index:scope_index[tid].sort(key=lambda x:number(x[1]['ts']))
    scope_times={tid:[number(s['ts']) for _,s in ss] for tid,ss in scope_index.items()}
    max_scope={tid:max(number(s['dur']) for _,s in ss) for tid,ss in scope_index.items()}
    def containing(e,native=False):
        tid=e.get('args',{}).get('Thread Id',e['tid']) if native else e['tid']
        if tid not in scope_index:return None
        ts=number(e['ts']);times=scope_times[tid];ss=scope_index[tid]
        # P30 scopes are nested on one thread. Search latest-start first;
        # a request-wide parent must not make every lookup scan all 64 steps.
        stop=end(e)
        for i in range(bisect_right(times,ts)-1,-1,-1):
            label,s=ss[i]
            if stop<=end(s) and (native or e['pid']==s['pid']):
                return label
        return None
    points,starts,finishes=defaultdict(list),defaultdict(list),defaultdict(list)
    def point(e):return e['pid'],e['tid'],number(e['ts'])
    for i,e in enumerate(events):
        if e.get('ph')=='X':points[point(e)].append((i,e))
        elif e.get('ph')=='s':starts[e.get('cat'),str(e['id'])].append(e)
        elif e.get('ph')=='f':finishes[(e.get('cat'),)+point(e)].append(e)
    queues=[]
    for e in events:
        if e.get('ph')=='X' and e.get('cat')=='dequeue':
            for f in finishes[('async_task_queue',)+point(e)]:queues.append((e,only(starts['async_task_queue',str(f['id'])],'queue source'),str(f['id'])))
    qi={}
    for tid in {q[0]['tid'] for q in queues}:
        qs=sorted((q for q in queues if q[0]['tid']==tid),key=lambda q:number(q[0]['ts']))
        qi[tid]=([number(q[0]['ts']) for q in qs],qs,max(number(q[0]['dur']) for q in qs))
    collisions=0
    def source(e,cat,cann=None):
        nonlocal collisions
        fs=finishes[(cat,)+point(e)]
        if not fs:return None
        f=only(fs,'flow finish');ss=starts[cat,str(f['id'])]
        if cat=='async_npu' and len(ss)>1 and cann:
            collisions+=1;times,qs,longest=qi.get(cann['tid'],([],[],0));t=number(cann['ts'])
            candidates=qs[bisect_left(times,t-longest):bisect_right(times,t)]
            i,h,_=resolve_colliding_host([only(points[point(s)],'host source') for s in ss],cann,candidates)
            return i,h,str(f['id'])
        i,h=only(points[point(only(ss,'flow source'))],'source range');return i,h,str(f['id'])
    replay_records=[r for r in measured if r['kind']=='replay']
    live_uids={r['uid'] for r in replay_records}
    captures={r['uid']:r for r in records if r['kind']=='capture' and r['uid'] in live_uids}
    models={};uid_model={}
    for uid,c in captures.items():
        dump=read(run/c['path']);mid=only({int(d['args']['Model Id']) for d in dump},'one model per graph')
        require(mid not in models,'reused model ID among replayed objects requires lifetime extension')
        models[mid]=dict(uid=uid,dump=dump);uid_model[uid]=mid
    csv_index=defaultdict(list)
    for j,r in enumerate(rows):csv_index[r['Name'],r['Stream ID'].strip(),r['Task ID'].strip(),number(r['Start Time(us)'])].append(j)
    replay_calls=defaultdict(list)
    for i,e in enumerate(events):
        if e.get('ph')=='X' and e.get('name')=='AscendCL@aclmdlRIExecuteAsync':replay_calls[str(e['args']['connection_id'])].append((i,e))
    tasks=[];used=set();by_scope=defaultdict(list)
    for i,e in enumerate(events):
        a=e.get('args',{})
        if e.get('ph')!='X' or 'Task Type' not in a or e['name'] in ('PROFILING_ENABLE','PROFILING_DISABLE'):continue
        internal=a.get('Model Id') in models
        cann=None if internal else source(e,'HostToDevice')
        if cann is None and e['name'] in ('MODEL_EXECUTE','NOTIFY_WAIT'):
            ci,c=only(replay_calls[str(a['connection_id'])],'replay connection');cann=(ci,c,'connection:'+str(a['connection_id']))
        host=None if internal else source(e,'async_npu',cann[1] if cann else None)
        label=containing(host[1]) if host else containing(cann[1],True) if cann else None
        if label is None and cann:label=containing(cann[1],True)
        r=record_by_label.get(label,{})
        t=dict(id='k:%d'%i,trace_index=i,name=e['name'],stream=str(a['Physic Stream Id']),task_id=str(a['Task Id']),model_id=a.get('Model Id'),task_type=a['Task Type'],connection_id=str(a.get('connection_id')),start_us=str(e['ts']),end_us=str(end(e)),duration_us=str(e['dur']),scope=label,step=r.get('step'),stage=r.get('kind'),csv_row=None,is_compute=False,
               host_index=host[0] if host else None,cann_index=cann[0] if cann else None,host_flow=host[2] if host else None,cann_flow=cann[2] if cann else None)
        if cann:require(t['connection_id']==str(cann[1]['args']['connection_id']),'connection mismatch')
        matches=csv_index[e['name'],t['stream'],t['task_id'],number(e['ts'])]
        if matches:
            j=only(matches,'CSV identity');require(j not in used,'duplicate CSV');used.add(j)
            validate_csv(t,rows[j])
            t.update(csv_row=j,is_compute=True,core_type=rows[j]['Accelerator Core'],block_num=core_count(rows[j]['Block Num']),mix_block_num=core_count(rows[j]['Mix Block Num']))
        tasks.append(t)
        if label:by_scope[label].append(t)
    require(len(used)==len(rows),'CSV coverage')
    byid={t['id']:t for t in tasks};replays=[]
    for r in replay_records:
        boundary=by_scope[r['label']]
        launch=only((t for t in boundary if t['name']=='MODEL_EXECUTE'),'replay launch')
        wait=only((t for t in boundary if t['name']=='NOTIFY_WAIT'),'replay completion')
        require(launch['cann_index']==wait['cann_index'],'boundary native connection')
        replays.append(dict(label=r['label'],step=r['step'],uid=r['uid'],model_id=uid_model[r['uid']],launch=launch['id'],wait=wait['id']))
    require(len(replays)==len(replay_calls),'replay API coverage')
    for mid,model in models.items():
        rs=sorted((r for r in replays if r['model_id']==mid),key=lambda r:number(scopes[r['label']]['ts']))
        group=sorted((t for t in tasks if t['model_id']==mid),key=lambda t:number(t['start_us']))
        dump=model['dump'];size=len(dump)+(dump[-1]['args']['Task Type']!='NOTIFY_RECORD')
        require(len(group)==len(rs)*size,'graph task repetition coverage')
        for index,r in enumerate(rs):
            chunk=group[index*size:(index+1)*size]
            launch=byid[r['launch']];wait=byid[r['wait']]
            validate_graph_chunk(chunk,dump,launch,wait)
            for t,d in zip(chunk,dump):t['dump_symbol']=d['name']
            for t in chunk:t.update(scope=r['label'],step=r['step'],stage='graph_internal',replay=r['label'])
            r['internal_tasks']=[t['id'] for t in chunk]
    require(all(t['step'] is not None for t in tasks if t['is_compute']),'unattributed compute')
    for sid in {t['stream'] for t in tasks}:
        ts=sorted((t for t in tasks if t['stream']==sid),key=lambda t:number(t['start_us']))
        require(all(number(a['end_us'])<=number(b['start_us']) for a,b in zip(ts,ts[1:])),'same stream overlap')
    steps=[]
    for r in executors:
        ts=[t for t in tasks if t['step']==r['step']];require(ts,'empty step')
        comp=[(number(t['start_us']),number(t['end_us'])) for t in ts if t['is_compute']]
        window=(min(number(t['start_us']) for t in ts),max(number(t['end_us']) for t in ts))
        compute=union(comp);no_compute=difference(window,compute)
        transfer=[(number(t['start_us']),number(t['end_us'])) for t in ts if 'MEMCPY' in t['name'] or t['task_type']=='SDMA_SQE']
        waits=[(number(t['start_us']),number(t['end_us'])) for t in ts if 'WAIT' in t['name']]
        all_task=[(number(t['start_us']),number(t['end_us'])) for t in ts]
        uncovered=difference(window,all_task)
        # Wait/copy coverage can overlap: report as intersections, never sum as slices.
        host=scopes[r['label']];sample=only((scopes[x['label']] for x in measured if x['kind']=='sample' and x['step']==r['step']),'sample scope')
        step=dict(index=r['index'],step=r['step'],phase='prefill' if r['index']==0 else 'decode',stable=16<=r['index']<=47,
                  start_us=str(window[0]),end_us=str(window[1]),device_span_us=str(window[1]-window[0]),compute_union_us=str(length(compute)),compute_coverage=float(length(compute)/(window[1]-window[0])),
                  no_compute_us=str(length(no_compute)),uncovered_us=str(length(uncovered)),wait_without_compute_us=str(length(intersection(no_compute,waits))),copy_without_compute_us=str(length(intersection(no_compute,transfer))),
                  host_execute_start_us=str(host['ts']),host_execute_end_us=str(end(host)),host_sample_start_us=str(sample['ts']),host_sample_end_us=str(end(sample)),
                  compute_tasks=sum(t['is_compute'] for t in ts),tasks=[t['id'] for t in ts],uncovered_intervals=[[str(a),str(b)] for a,b in uncovered],**overlap(ts))
        steps.append(step)
    for a,b in zip(steps,steps[1:]):
        require(number(a['end_us'])<=number(b['start_us']),'steps overlap needs extended boundary accounting')
        a['to_next_step_us']=str(number(b['start_us'])-number(a['end_us']))
    # Keep the measured native/CPU intervals. A late runtime launch is evidence of
    # dispatch timing, not proof of an idle device or absence of an earlier queue.
    host_indices=sorted({t[k] for t in tasks for k in ('host_index','cann_index') if t[k] is not None})
    host_events={str(i):events[i] for i in host_indices}
    queue_records=[]
    for dequeue,enqueue_flow,flow_id in queues:
        ei,enqueue=only(points[point(enqueue_flow)],'enqueue range')
        di,de=only(points[point(dequeue)],'dequeue range')
        label=containing(enqueue);r=record_by_label.get(label,{})
        queue_records.append(dict(id='q:'+flow_id,flow_id=flow_id,enqueue_index=ei,dequeue_index=di,
            enqueue=enqueue,dequeue=de,scope=label,step=r.get('step')))
    step_byid={s['step']:s for s in steps}
    for t in tasks:
        if t['stage']=='graph_internal':
            r=next(r for r in replays if r['label']==t['replay']);t['submission_kind']='graph_replay';t['submission_index']=byid[r['launch']]['cann_index']
        else:t['submission_kind']='direct';t['submission_index']=t['cann_index']
    # Classify each uncovered interval by the next recorded task's exact
    # submission. This describes timing; it does not prove a hardware stall cause.
    gap_records=[]
    for s in steps:
        ts=sorted((t for t in tasks if t['step']==s['step']),key=lambda t:number(t['start_us']))
        times=[number(t['start_us']) for t in ts]
        for start,finish in s['uncovered_intervals']:
            a,b=number(start),number(finish);pos=bisect_left(times,b)
            nexts=[t for t in ts[pos:] if number(t['start_us'])==b]
            record=dict(step=s['step'],start_us=start,end_us=finish,duration_us=str(b-a),next_tasks=[t['id'] for t in nexts],classification='unresolved')
            if len(nexts)==1:
                t=nexts[0];si=t['submission_index'];record['submission_kind']=t['submission_kind']
                if si is not None:
                    c=events[si];record.update(submission_index=si,native_start_us=str(c['ts']),native_end_us=str(end(c)),before_next_native_call_us=str(max(Decimal(0),min(b,number(c['ts']))-a)))
                    record['classification']='native_returned_before_gap' if end(c)<=a else 'native_call_begins_after_gap_start' if number(c['ts'])>a else 'native_call_spans_gap_start'
                hi=t['host_index']
                if hi is not None:record.update(host_start_us=str(events[hi]['ts']),host_end_us=str(end(events[hi])),before_next_host_call_us=str(max(Decimal(0),min(b,number(events[hi]['ts']))-a)))
            gap_records.append(record)
        s['uncovered_by_next_submission']={kind:str(sum(number(g['duration_us']) for g in gap_records if g['step']==s['step'] and g['classification']==kind)) for kind in sorted({g['classification'] for g in gap_records if g['step']==s['step']})}
        s['gap_before_next_native_call_us']=str(sum(number(g.get('before_next_native_call_us',0)) for g in gap_records if g['step']==s['step']))
        s['gap_before_next_host_call_us']=str(sum(number(g.get('before_next_host_call_us',0)) for g in gap_records if g['step']==s['step']))
        s['compute_type_union_us']={kind:str(length([(number(t['start_us']),number(t['end_us'])) for t in ts if t['is_compute'] and t['core_type']==kind])) for kind in sorted({t['core_type'] for t in ts if t['is_compute']})}
    stable=[s for s in steps if s['stable']]
    core_groups=defaultdict(list)
    for t in tasks:
        if t['is_compute'] and step_byid[t['step']]['stable']:core_groups[t['name']].append(t)
    cores=[]
    for name,ts in core_groups.items():
        metrics=defaultdict(list)
        for t in ts:
            for key,value in rows[t['csv_row']].items():
                if key.startswith(('aic_','aiv_','aicore_','cube_utilization')):
                    try:metrics[key].append(float(value))
                    except (ValueError,TypeError):pass
        cores.append(dict(name=name,count=len(ts),duration_us=stats(t['duration_us'] for t in ts),total_us=str(sum(number(t['duration_us']) for t in ts)),core_types=sorted({t['core_type'] for t in ts}),block_nums=sorted({t['block_num'] for t in ts if t['block_num'] is not None}),unknown_block_tasks=sum(t['block_num'] is None for t in ts),mix_block_nums=sorted({t['mix_block_num'] for t in ts if t['mix_block_num'] is not None}),pipeline={k:stats(v) for k,v in metrics.items()}))
    cores.sort(key=lambda c:number(c['total_us']),reverse=True)
    summary=dict(mode=config['mode'],profile=config['profile'],steps=len(steps),stable_steps=len(stable),device_tasks=len(tasks),compute_tasks=len(rows),physical_streams=len({t['stream'] for t in tasks}),graph_replays=len(replays),graph_objects=len(models),
                 stable_device_span_us=stats(s['device_span_us'] for s in stable),stable_compute_union_us=stats(s['compute_union_us'] for s in stable),stable_coverage=stats(s['compute_coverage'] for s in stable),stable_uncovered_us=stats(s['uncovered_us'] for s in stable),stable_interstep_us=stats(s['to_next_step_us'] for s in stable),**overlap(tasks),flow_collisions_resolved=collisions,stable_gap_before_next_native_call_us=stats(s['gap_before_next_native_call_us'] for s in stable),stable_gap_before_next_host_call_us=stats(s['gap_before_next_host_call_us'] for s in stable),stable_gap_by_next_submission={kind:str(sum(number(g['duration_us']) for g in gap_records if step_byid[g['step']]['stable'] and g['classification']==kind)) for kind in sorted({g['classification'] for g in gap_records})})
    return dict(config=config,summary=summary,steps=steps,tasks=tasks,gap_records=gap_records,host_events=host_events,queues=queue_records,scopes=scopes,replays=replays,kernel_rows=rows,cores=cores,unattributed_runtime=[t['id'] for t in tasks if t['step'] is None],provenance=dict(trace=str(path.relative_to(run)),trace_sha256=digest(path),csv=str(csv_path.relative_to(run)),csv_sha256=digest(csv_path)),limits=['Compute coverage includes AI_CPU where reported; types are separately recorded and it is not chip/core utilization.','No-compute wait and copy intersections may overlap; do not sum them.','Uncovered intervals are not proven hardware idle.','Reported zero or absent block counts are unknown. Mixed core fields are separate.','Hardware ratios are per-kernel reported metrics, not whole-chip occupancy or bandwidth saturation.','Graph tasks have replay submissions, not fresh per-kernel launches.','Notify membership/completion contract does not recover exact notify IDs.'])
