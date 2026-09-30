"""CPU phases, queue handoff and NPU gaps with explicit attribution limits."""
import argparse
from bisect import bisect_left, bisect_right
from collections import defaultdict
from decimal import Decimal as D
import hashlib
import json
from pathlib import Path
import statistics

import exact_join as exact


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,default=str,separators=(',',':'))+'\n')

def validate_queue(q):
    ids=[str(q[field]['args']['correlation_id']) for field in ('enqueue','dequeue')]
    if ids != [q['flow_id'],q['flow_id']]:raise ValueError('queue correlation/flow identity mismatch')


def phase_metrics(records,scopes,sources):
    byid={r['id']:r for r in records}
    if len(byid)!=len(records):raise ValueError('duplicate phase ID')
    children=defaultdict(list)
    for r in records:
        if r['parent'] is not None:
            p=byid[r['parent']]
            if p['tid']!=r['tid'] or not p['wall_start_ns']<=r['wall_start_ns']<=r['wall_end_ns']<=p['wall_end_ns']:
                raise ValueError('phase parent/thread enclosure')
            children[r['parent']].append(r)
    result=[]
    for r in records:
        intervals=sorted((c['wall_start_ns'],c['wall_end_ns']) for c in children[r['id']])
        if any(a[1]>b[0] for a,b in zip(intervals,intervals[1:])):
            raise ValueError('overlapping sibling phases')
        self_wall=r['wall_ns']-sum(b-a for a,b in intervals)
        self_cpu=r['thread_cpu_ns']-sum(c['thread_cpu_ns'] for c in children[r['id']])
        if min(self_wall,self_cpu)<0:raise ValueError('negative exclusive phase duration')
        s=scopes[r['label']]
        result.append(dict(r, self_wall_us=self_wall/1000,self_thread_cpu_us=self_cpu/1000,
            wall_us=r['wall_ns']/1000,thread_cpu_us=r['thread_cpu_ns']/1000,
            wall_minus_thread_cpu_us=(r['wall_ns']-r['thread_cpu_ns'])/1000,
            trace_start_us=str(s['ts']),trace_end_us=str(exact.end(s)),source=sources.get(r['kind'])))
    return sorted(result,key=lambda r:r['id'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);a=p.parse_args()
    root=a.run;run=root/'diagnostic'
    status=exact.read(root/'status.json')
    exact.require(status['status']=='passed','controller/recovery incomplete')
    for original,item in exact.read(root/'source_manifest.json').items():
        exact.require(exact.digest(root/item['path'])==item['sha256'],'source snapshot hash: '+original)
    for name,digest in exact.read(root/'collector_hashes.json').items():
        exact.require(exact.digest(root/'collector'/name)==digest,'collector hash: '+name)
    data=exact.analyze(run)
    path=run/data['provenance']['trace']
    events=json.loads(path.read_text(),parse_float=D)
    if isinstance(events,dict):events=events['traceEvents']
    records=exact.read(run/'observer.json');sources=exact.read(run/'phase_sources.json')
    phases=phase_metrics(records,data['scopes'],sources)
    phase_by_label={r['label']:r for r in phases}
    phase_by_id={r['id']:r for r in phases}
    phases_by_thread=defaultdict(list)
    for r in phases:phases_by_thread[r['tid']].append(r)
    def enclosing(event):
        tid=event.get('args',{}).get('Thread Id',event['tid'])
        candidates=[r for r in phases_by_thread[tid] if D(r['trace_start_us'])<=D(event['ts']) and exact.end(event)<=D(r['trace_end_us'])]
        return min(candidates,key=lambda r:D(r['trace_end_us'])-D(r['trace_start_us'])) if candidates else None

    # Index actual dequeue ranges by OS TID; a device task can bypass this queue.
    queue_by_tid=defaultdict(list)
    for q in data['queues']:
        validate_queue(q)
        en,de=q['enqueue'],q['dequeue']
        q.update(enqueue_to_dequeue_start_us=str(D(de['ts'])-D(en['ts'])),
            dequeue_start_minus_enqueue_end_us=str(D(de['ts'])-exact.end(en)))
        queue_by_tid[de['tid']].append(q)
    queue_index={}
    for tid,qs in queue_by_tid.items():
        qs.sort(key=lambda q:D(q['dequeue']['ts']))
        queue_index[tid]=([D(q['dequeue']['ts']) for q in qs],qs,max(D(str(q['dequeue']['dur'])) for q in qs))
    for t in data['tasks']:
        t['queue_id']=None
        if t['cann_index'] is not None:
            e=events[t['cann_index']]
            times,qs,longest=queue_index.get(e['tid'],([],[],D(0)))
            candidates=[q for q in qs[bisect_left(times,D(e['ts'])-longest):bisect_right(times,D(e['ts']))]
                        if exact.end(e)<=exact.end(q['dequeue'])]
            if len(candidates)>1:raise ValueError('ambiguous native dequeue range')
            if candidates:t['queue_id']=candidates[0]['id']

    sync=[]
    for index,e in enumerate(events):
        if e.get('ph')=='X' and e.get('name','').startswith('AscendCL@') and 'Synchronize' in e['name']:
            phase=enclosing(e)
            if phase:sync.append(dict(index=index,phase_id=phase['id'],step=phase['index'],
                kind=phase['kind'],event=e))
    # The observer names give method context; no full Python/native stack is inferred.
    op_events=[]
    for index,e in enumerate(events):
        if e.get('ph')=='X' and e.get('cat')=='cpu_op' and not e.get('name','').startswith('P30/'):
            op_events.append((index,e))

    tasks_by_id={t['id']:t for t in data['tasks']}
    queues_by_id={q['id']:q for q in data['queues']}
    focused=next(s for s in data['steps'] if s['index']==32)
    gaps=sorted((g for g in data['gap_records'] if g['step']==focused['step']),
                key=lambda g:D(g['duration_us']),reverse=True)[:3]
    for g in gaps:
        lo,hi=D(g['start_us']),D(g['end_us'])
        g['main_thread_phases']=[dict(phase_id=r['id'],kind=r['kind'],
            overlap_us=str(min(hi,D(r['trace_end_us']))-max(lo,D(r['trace_start_us'])))) for r in phases
            if r['index']==32 and max(lo,D(r['trace_start_us']))<min(hi,D(r['trace_end_us']))]
        g['next_queue']=None
        if len(g['next_tasks'])==1:
            task=tasks_by_id[g['next_tasks'][0]]
            if task['queue_id']:
                q=queues_by_id[task['queue_id']];g['next_queue']=q
                g['before_next_enqueue_us']=str(max(D(0),min(hi,D(q['enqueue']['ts']))-lo))
                g['queue_start_state']=('not_enqueued' if D(q['enqueue']['ts'])>lo else
                    'enqueued_dequeue_not_started' if D(q['dequeue']['ts'])>lo else 'dequeue_started')
        g['overlapping_cpu_ops']=[dict(index=i,name=e['name'],start_us=str(e['ts']),duration_us=str(e['dur']),
            overlap_us=str(min(hi,exact.end(e))-max(lo,D(e['ts'])))) for i,e in op_events
            if max(lo,D(e['ts']))<min(hi,exact.end(e))]
        g['overlapping_cpu_ops'].sort(key=lambda e:D(e['overlap_us']),reverse=True)

    # Four concrete paths in the selected step. Source locations are static
    # entrypoints plus observed phase membership, not fabricated call stacks.
    focused_tasks=[t for t in data['tasks'] if t['step']==focused['step']]
    examples={}
    selectors={'input_h2d':lambda t:'MEMCPY' in t['name'] and phase_by_label.get(t['scope'],{}).get('kind') in ('prepare_inputs','preprocess','state_update','execute'),
               'linear':lambda t:t['is_compute'] and 'MatMul' in t['name'],
               'rope':lambda t:t['is_compute'] and 'rope' in t['name'].lower(),
               'token_d2h':lambda t:'MEMCPY' in t['name'] and phase_by_label.get(t['scope'],{}).get('kind')=='token_to_list'}
    for name,selector in selectors.items():
        candidates=sorted((t for t in focused_tasks if selector(t)),key=lambda t:D(t['start_us']))
        if not candidates:raise ValueError('missing example: '+name)
        examples[name]=candidates[0]['id']
    for name,direction in [('input_h2d','host_to_device'),('token_d2h','device_to_host')]:
        t=tasks_by_id[examples[name]]
        exact.require(direction in queues_by_id[t['queue_id']]['enqueue']['name'],'copy direction not established')

    groups=defaultdict(list)
    for r in phases:
        if r['index']>=1:groups[r['kind']].append(r)
    phase_summary={kind:dict(calls=len(rs),wall_us=exact.stats(r['wall_us'] for r in rs),
        thread_cpu_us=exact.stats(r['thread_cpu_us'] for r in rs),self_wall_us=exact.stats(r['self_wall_us'] for r in rs))
        for kind,rs in groups.items()}
    controls=exact.read(root/'reference/responses.json')
    diag=exact.read(run/'responses.json')[0]
    recovery=exact.read(root/'recovery/responses.json')[0]
    overhead=dict(reference_wall_us=exact.stats(r['wall_us'] for r in controls),
        diagnostic_wall_us=diag['wall_us'],diagnostic_over_reference_median=diag['wall_us']/statistics.median(r['wall_us'] for r in controls),
        recovery_wall_us=recovery['wall_us'],limits='One diagnostic vs three controls; combines profiling/observer overhead and run variation.')
    req=next(r for r in phases if r['kind']=='request')
    exact.require(len({r['tid'] for r in phases})==1,'more than one Python phase thread')
    exact.require(abs(sum(r['self_wall_us'] for r in phases)-req['wall_us'])<0.001,'exclusive wall accounting')
    executor=[r for r in phases if r['kind']=='executor_submit']
    exact.require(len(executor)==64 and all(r['future_done_on_return'] for r in executor),'executor future not already done')
    # Retain focused CPU operators, including operators with no device kernel.
    frame=next(r for r in phases if r['kind']=='engine_step' and r['index']==32)
    focused_ops=[dict(index=i,event=e,phase_id=(p['id'] if (p:=enclosing(e)) else None)) for i,e in op_events
        if D(frame['trace_start_us'])<=D(e['ts']) and exact.end(e)<=D(frame['trace_end_us'])]
    data.update(phases=phases,phase_sources=sources,phase_summary=phase_summary,synchronizations=sync,
        focused_cpu_ops=focused_ops,top_gaps=gaps,examples=examples,overhead=overhead,
        request_cpu_thread_id=req['tid'],source_manifest=exact.read(root/'source_manifest.json'))
    out=root/'analysis';save(out/'evidence.json',data)
    summary=dict(exact=data['summary'],phases=len(phases),queues=len(data['queues']),
        device_tasks_with_queue=sum(t['queue_id'] is not None for t in data['tasks']),
        unattributed_runtime=[t for t in data['tasks'] if t['step'] is None],
        cpu_sync_calls=len(sync),sync_by_phase={kind:dict(count=sum(s['kind']==kind for s in sync),
            duration_us=str(sum(D(str(s['event']['dur'])) for s in sync if s['kind']==kind))) for kind in sorted({s['kind'] for s in sync})},
        request_phase={k:req[k] for k in ('wall_us','thread_cpu_us','wall_minus_thread_cpu_us')},
        decode_phase_summary=phase_summary,overhead=overhead,all_executor_futures_done_on_return=True,
        binding_restored=exact.read(run/'binding_recovery.json')['restored'],recovery=status['recovery'])
    save(out/'summary.json',summary)
    save(out/'validation.json',dict(exact_kernel_csv_coverage=True,complete_steps=64,
        single_python_phase_thread=True,nested_exclusive_accounting=True,
        source_and_collector_hashes_verified=True,examples=list(examples)))
    print(json.dumps(summary,ensure_ascii=False,default=str,indent=2))


if __name__=='__main__':main()
