"""Measure branch overlap with only an outer profiler marker per real forward."""
import argparse,csv,json
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from analyze import read,save,only,require,number,end,inside,intersection,length,resolve_colliding_host


def analyze(root,formal):
    meta=read(root/'run.json');require(meta['status']=='passed' and not meta['observer_installed'],'invalid light run')
    environment=read(formal/'environment.json')
    require(meta['weights']=={k:v['sha256'] for k,v in environment['weights'].items()},'different weights')
    events=json.loads(only(root.rglob('trace_view.json'),'trace').read_text(),parse_float=Decimal)
    if isinstance(events,dict):events=events['traceEvents']
    with only(root.rglob('kernel_details.csv'),'kernel CSV').open() as f:csv_rows=list(csv.DictReader(f))
    points,starts,finishes=defaultdict(list),defaultdict(list),defaultdict(list)
    def point(e):return e['pid'],e['tid'],number(e['ts'])
    for i,e in enumerate(events):
        if e.get('ph')=='X':points[point(e)].append((i,e))
        elif e.get('ph')=='s':starts[e.get('cat'),str(e['id'])].append(e)
        elif e.get('ph')=='f':finishes[(e.get('cat'),)+point(e)].append(e)
    queues=[]
    for e in events:
        if e.get('ph')=='X' and e.get('cat')=='dequeue':
            for f in finishes[('async_task_queue',)+point(e)]:
                queues.append((e,only(starts['async_task_queue',str(f['id'])],'enqueue'),str(f['id'])))
    def source(t,cat,cann=None):
        f=only(finishes[(cat,)+point(t)],'flow endpoint');ss=starts[cat,str(f['id'])]
        if len(ss)>1 and cann:
            i,h,_=resolve_colliding_host([only(points[point(s)],'host') for s in ss],cann,queues)
            return i,h,str(f['id'])
        i,h=only(points[point(only(ss,'flow start'))],'source');return i,h,str(f['id'])
    scopes={r['label']:only((e for e in events if e.get('ph')=='X' and e.get('name')==r['label']),'trial marker') for r in meta['trials']}
    moe_scopes=[e for e in events if e.get('ph')=='X' and e.get('name')=='vllm::moe_forward_shared']
    tasks=[];used=set()
    for i,t in enumerate(events):
        a=t.get('args',{})
        if t.get('ph')!='X' or 'Task Type' not in a:continue
        candidates=[j for j,r in enumerate(csv_rows) if r['Name']==t['name'] and r['Stream ID'].strip()==str(a['Physic Stream Id']) and
                    r['Task ID'].strip()==str(a['Task Id']) and number(r['Start Time(us)'])==number(t['ts'])]
        if not candidates:continue
        j=only(candidates,'CSV identity');require(j not in used,'duplicate compute');used.add(j)
        require(abs(number(csv_rows[j]['Duration(us)'])-number(t['dur']))<=Decimal('.001'),'duration mismatch')
        ci,c,cf=source(t,'HostToDevice');hi,h,hf=source(t,'async_npu',c)
        require(str(c['args']['connection_id'])==str(a['connection_id']),'CANN connection')
        label=only((label for label,s in scopes.items() if inside(s,h)),'trial association')
        tasks.append(dict(id='k:'+str(i),trial=label,name=t['name'],stream=str(a['Physic Stream Id']),
            task_id=str(a['Task Id']),start_us=str(t['ts']),end_us=str(end(t)),duration_us=str(t['dur']),
            in_native_moe=any(inside(s,h) for s in moe_scopes),host_operator=h['name'],
            host_trace_index=hi,cann_trace_index=ci,torch_flow=hf,cann_flow=cf,csv_row=j,connection_id=str(a['connection_id'])))
    require(len(used)==len(csv_rows),'unmatched kernel CSV')
    summaries=[]
    for trial in meta['trials']:
        ts=[t for t in tasks if t['trial']==trial['label']]
        require(len(ts)==16,'changed compute kernel count')
        merge=only((t for t in ts if t['name']=='aclnnAdd_AddAiCore_Add'),'merge')
        main=merge['stream'];streams={t['stream'] for t in ts}
        require(len(streams)==(1 if trial['mode']=='serial' else 2),'stream count')
        if trial['mode']=='parallel':
            shared=[t for t in ts if t['stream']!=main];routed=[t for t in ts if t['stream']==main and t['in_native_moe']]
            require(len(shared)==6 and len(routed)==7,'branch kernel coverage')
            overlap=length(intersection([(number(t['start_us']),number(t['end_us'])) for t in shared],
                                        [(number(t['start_us']),number(t['end_us'])) for t in routed]))
        else:overlap=Decimal(0)
        summaries.append(dict(**trial,streams=sorted(streams),compute_tasks=len(ts),overlap_us=str(overlap)))
    result=dict(status='passed',observer_installed=False,torch_dispatch_mode=False,
        trials=summaries,compute_tasks=len(tasks),tasks=tasks)
    save(root/'summary.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='tasks'},indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('formal',type=Path);a=p.parse_args();analyze(a.run,a.formal)
