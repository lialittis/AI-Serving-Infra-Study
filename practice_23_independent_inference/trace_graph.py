"""Exact profiler flow joins and stream/event HB; adapted from Practice 21."""
import csv,json,sys
from bisect import bisect_left,bisect_right
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'practice_17_vllm_multistream'))
from analyze_run import number,end,inside,only,require,resolve_colliding_host,intersection,union
sys.path.insert(0,str(ROOT/'practice_20_kv_offload_overlap'))
from hb import Graph
def read(p):return json.loads(p.read_text())

def extract(root):
    events=json.loads(only(root.rglob('trace_view.json'),'trace').read_text(),parse_float=Decimal)
    if isinstance(events,dict):events=events['traceEvents']
    with only(root.rglob('kernel_details.csv'),'kernel CSV').open() as f:kernels=list(csv.DictReader(f))
    points,starts,finishes=defaultdict(list),defaultdict(list),defaultdict(list)
    def point(e):return e['pid'],e['tid'],number(e['ts'])
    for i,e in enumerate(events):
        if e.get('ph')=='X':points[point(e)].append((i,e))
        elif e.get('ph')=='s':starts[e.get('cat'),str(e['id'])].append(e)
        elif e.get('ph')=='f':finishes[(e.get('cat'),)+point(e)].append(e)
    observations=read(root/'observations.json');records={r['label']:r for r in observations}
    scopes={e['name']:e for e in events if e.get('ph')=='X' and e.get('name') in records}
    require(set(scopes)==set(records),'missing diagnostic scope')
    queues=[]
    for e in events:
        if e.get('ph')=='X' and e.get('cat')=='dequeue':
            for finish in finishes[('async_task_queue',)+point(e)]:
                enqueue=only(starts['async_task_queue',str(finish['id'])],'queue source')
                queues.append((e,enqueue,str(finish['id'])))
    # Limit candidates by exact interval bounds, then retain the original
    # queue/flow containment proof. This is not nearest-timestamp matching.
    queue_index={}
    for tid in {q[0]['tid'] for q in queues}:
        qs=sorted([q for q in queues if q[0]['tid']==tid],key=lambda q:number(q[0]['ts']))
        queue_index[tid]=([number(q[0]['ts']) for q in qs],qs,max(number(q[0]['dur']) for q in qs))
    def containing_queues(cann):
        if cann['tid'] not in queue_index:return []
        times,qs,longest=queue_index[cann['tid']];start=number(cann['ts'])
        return [q for q in qs[bisect_left(times,start-longest):bisect_right(times,start)]
                if number(q[0]['ts'])<=start and end(cann)<=end(q[0])]
    wait_calls=[(i,e) for i,e in enumerate(events) if e.get('name')=='AscendCL@aclrtStreamWaitEvent']
    collisions=0
    def source(task,cat,cann=None):
        nonlocal collisions
        fs=finishes[(cat,)+point(task)]
        if not fs:return None
        finish=only(fs,'flow endpoint');ss=starts[cat,str(finish['id'])]
        if len(ss)>1 and cat=='async_npu' and cann:
            collisions+=1
            hosts=[only(points[point(s)],'host op') for s in ss]
            i,e,_=resolve_colliding_host(hosts,cann,containing_queues(cann));return i,e,str(finish['id'])
        i,e=only(points[point(only(ss,'flow source'))],'flow source range')
        return i,e,str(finish['id'])
    csv_index=defaultdict(list)
    for i,r in enumerate(kernels):csv_index[r['Name'],r['Stream ID'].strip(),r['Task ID'].strip(),number(r['Start Time(us)'])].append(i)
    graph=Graph();tasks=[];used=set();by_scope=defaultdict(list);by_stream=defaultdict(list);unknown=[];outside=[]
    mapping=defaultdict(set)
    for index,e in enumerate(events):
        a=e.get('args',{})
        if e.get('ph')!='X' or 'Task Type' not in a or e['name'] in ('PROFILING_ENABLE','PROFILING_DISABLE'):continue
        cann=source(e,'HostToDevice');host=source(e,'async_npu',cann[1] if cann else None)
        t=dict(id='k:'+str(index),kind='device',name=e['name'],task_type=a['Task Type'],
            stream=str(a['Physic Stream Id']),task_id=str(a['Task Id']),trace_index=index,
            start_us=str(e['ts']),end_us=str(end(e)),duration_us=str(e['dur']),
            is_compute=a['Task Type'] in ('AI_CORE','AI_VECTOR_CORE','MIX_AIC','MIX_AIV','AI_CPU'),
            scope=None,trial=None,stage=None,connection_id=str(a.get('connection_id')),
            host_trace_index=host[0] if host else None,cann_trace_index=cann[0] if cann else None,
            host_operator=host[1]['name'] if host else None,cann_api=cann[1]['name'] if cann else None,
            torch_flow=host[2] if host else None,cann_flow=cann[2] if cann else None)
        if cann:require(str(cann[1]['args']['connection_id'])==t['connection_id'],'CANN connection mismatch')
        if host:
            candidates=[(label,s) for label,s in scopes.items() if inside(s,host[1])]
            if candidates:
                label,_=min(candidates,key=lambda pair:number(pair[1]['dur']));r=records[label]
                t.update(scope=label,trial=r['trial'],stage=r['kind'],task=r.get('task'),role=r.get('role'));by_scope[label].append(t)
                mapping[r['stream_handle']].add(t['stream'])
            else:outside.append(t['id'])
        else:
            require(e['name']=='PLACE_HOLDER_SQE','unknown task without host flow '+e['name']);unknown.append(t['id'])
        matches=csv_index[e['name'],t['stream'],t['task_id'],number(e['ts'])]
        if matches:
            j=only(matches,'CSV task identity');require(j not in used,'duplicate kernel');used.add(j)
            require(abs(number(kernels[j]['Duration(us)'])-number(e['dur']))<=Decimal('.001'),'CSV duration')
            t['csv_row']=j;t['is_compute']=not any(s in t['name'] for s in ('EVENT_','MEMCPY','PLACE_HOLDER'))
        require(not t['is_compute'] or 'csv_row' in t,'compute missing CSV')
        graph.node(t['id'],**{k:v for k,v in t.items() if k!='id'});tasks.append(t);by_stream[t['stream']].append(t)
    require(len(used)==len(kernels),'CSV rows not fully accounted')
    require(all(len(v)==1 for v in mapping.values()),'ambiguous logical/physical mapping')
    for stream,ts in by_stream.items():
        ts.sort(key=lambda t:(number(t['start_us']),int(t['task_id'])))
        for a,b in zip(ts,ts[1:]):graph.edge(a['id'],b['id'],'stream_order',stream=stream)
    last={};event_edges=[];same_stream_waits=0;api_waits=[]
    for r in observations:
        if not r['kind'].startswith('event_'):continue
        ts=by_scope[r['label']];key=r['event_handle']
        if r['kind']=='event_record':
            last[key]=only((t for t in ts if t['name']=='EVENT_RECORD'),'native event record')
        elif r['kind']=='event_wait':
            record=last.get(key);require(record is not None,'wait without observed record')
            for wait in [t for t in ts if t['name']=='EVENT_WAIT']:
                edge=dict(source=record['id'],target=wait['id'],kind='event_wait',event_handle=key,
                          record_scope=record['scope'],wait_scope=r['label'])
                graph.edge(**edge);event_edges.append(edge)
            if not any(t['name']=='EVENT_WAIT' for t in ts):
                stream=only(mapping[r['stream_handle']],'wait stream mapping')
                scope=scopes[r['label']]
                # Native wait APIs can return without emitting an EVENT_WAIT SQE.
                # Require exact enqueue/dequeue -> CANN identity, then represent
                # the API's stream-order guarantee as a logical barrier, never as
                # an invented device task or a timestamp-derived dependency.
                calls=[]
                for ci,c in wait_calls:
                    for dequeue,enqueue,correlation in containing_queues(c):
                        if c['tid']==dequeue['tid'] and number(dequeue['ts'])<=number(c['ts']) and end(c)<=end(dequeue) and inside(scope,enqueue):
                            calls.append((ci,correlation))
                ci,correlation=only(calls,'native elided wait API association')
                if stream==record['stream']:
                    same_stream_waits+=1  # Same-stream FIFO already orders it.
                else:
                    barrier='api:'+r['label']
                    graph.node(barrier,kind='runtime_wait',trial=r['trial'],stream=stream,
                        scope=r['label'],cann_trace_index=ci,queue_correlation=correlation,
                        note='Successful native stream wait; no physical EVENT_WAIT task observed')
                    graph.edge(record['id'],barrier,'event_wait_api',event_handle=key,cann_trace_index=ci)
                    after=[t for t in by_stream[stream] if t['host_trace_index'] is not None and
                           events[t['host_trace_index']]['tid']==scope['tid'] and
                           number(events[t['host_trace_index']]['ts'])>=end(scope)]
                    require(bool(after),'logical wait without subsequent stream work')
                    target=min(after,key=lambda t:(number(events[t['host_trace_index']]['ts']),number(t['start_us'])))
                    graph.edge(barrier,target['id'],'stream_submission_order',stream=stream,
                        evidence='same host thread, native task queue and same logical stream')
                    api_waits.append(dict(node=barrier,record=record['id'],next_task=target['id'],event_handle=key))
        elif r['kind']=='event_synchronize':
            require(key in last,'sync without record')
            host_id='h:'+r['label'];graph.node(host_id,kind='host_completion',trial=r['trial'],scope=r['label'])
            graph.edge(last[key]['id'],host_id,'host_sync',event_handle=key)
    return dict(graph=graph,tasks=tasks,observations=observations,scopes=scopes,by_scope=by_scope,mapping={k:sorted(v) for k,v in mapping.items()},collisions=collisions,api_waits=api_waits,event_edges=event_edges,same_stream_waits=same_stream_waits,unknown=unknown,outside=outside)
