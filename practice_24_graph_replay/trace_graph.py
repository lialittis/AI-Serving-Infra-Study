"""Exact profiler flow joins and stream/event HB; including captured model tasks and replay completion."""
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
    captures=read(root/'captures.json');models={}
    for capture in captures:
        dump=read(root/'dumps'/capture['dump']);ids={int(d['args']['Model Id']) for d in dump}
        mid=only(ids,'one model per captured graph');require(mid not in models,'reused live model ID')
        require(len({d['args']['Stream Id'] for d in dump})==1,'multi-internal-stream graph needs a new contract')
        models[mid]=dict(capture=capture,dump=dump,replays=[])
    observations=read(root/'observations.json');records={r['label']:r for r in observations}
    scopes={e['name']:e for e in events if e.get('ph')=='X' and e.get('name') in records}
    require(set(scopes)==set(records),'missing diagnostic scope')
    scope_index={}
    for tid in {s['tid'] for s in scopes.values()}:
        ss=sorted(((label,s) for label,s in scopes.items() if s['tid']==tid),key=lambda x:number(x[1]['ts']))
        require(all(end(a[1])<=number(b[1]['ts']) for a,b in zip(ss,ss[1:])),'overlapping diagnostic scopes')
        scope_index[tid]=([number(s['ts']) for _,s in ss],ss)
    def containing_scope(e,native=False):
        if e['tid'] not in scope_index:return None
        times,ss=scope_index[e['tid']];i=bisect_right(times,number(e['ts']))-1
        if i<0:return None
        label,s=ss[i]
        return label if (native or s['pid']==e['pid']) and end(e)<=end(s) else None
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
    def native_scope(cann):
        # CANN ranges use a synthetic process lane; Thread Id is the native
        # calling thread. Full containment in exactly one RF scope is required.
        return containing_scope(cann,native=True)
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
    replay_calls=defaultdict(list)
    for i,e in enumerate(events):
        if e.get('ph')=='X' and e['name']=='AscendCL@aclmdlRIExecuteAsync':
            replay_calls[str(e['args']['connection_id'])].append((i,e))
    graph=Graph();tasks=[];used=set();by_scope=defaultdict(list);by_stream=defaultdict(list);unknown=[];outside=[]
    mapping=defaultdict(set)
    for index,e in enumerate(events):
        a=e.get('args',{})
        if e.get('ph')!='X' or 'Task Type' not in a or e['name'] in ('PROFILING_ENABLE','PROFILING_DISABLE'):continue
        # Captured tasks have no fresh per-kernel host submission. The profiler
        # may retain dangling flow IDs from capture outside this trace. Resolve
        # them through the live model/dump/replay contract below instead.
        internal=a.get('Model Id') in models
        cann=None if internal else source(e,'HostToDevice')
        if cann is None and e['name'] in ('MODEL_EXECUTE','NOTIFY_WAIT'):
            ci,c=only(replay_calls[str(a['connection_id'])],'replay exact native connection')
            cann=(ci,c,'connection:'+str(a['connection_id']))
        host=None if internal else source(e,'async_npu',cann[1] if cann else None)
        t=dict(id='k:'+str(index),kind='device',name=e['name'],task_type=a['Task Type'],
            stream=str(a['Physic Stream Id']),task_id=str(a['Task Id']),trace_index=index,
            start_us=str(e['ts']),end_us=str(end(e)),duration_us=str(e['dur']),
            is_compute=a['Task Type'] in ('AI_CORE','AI_VECTOR_CORE','MIX_AIC','MIX_AIV','AI_CPU'),
            model_id=a.get('Model Id'),scope=None,trial=None,stage=None,connection_id=str(a.get('connection_id')),
            host_trace_index=host[0] if host else None,cann_trace_index=cann[0] if cann else None,
            host_operator=host[1]['name'] if host else None,cann_api=cann[1]['name'] if cann else None,
            torch_flow=host[2] if host else None,cann_flow=cann[2] if cann else None)
        if cann:require(str(cann[1]['args']['connection_id'])==t['connection_id'],'CANN connection mismatch')
        label=None
        if host:
            label=containing_scope(host[1])
            if label:
                r=records[label]
                t.update(scope=label,trial=r['trial'],stage=r['kind'],task=r.get('task'),role=r.get('role'));by_scope[label].append(t)
                mapping[r['stream_handle']].add(t['stream'])
            else:outside.append(t['id'])
        elif cann and native_scope(cann[1]):
            label=native_scope(cann[1]);r=records[label]
            t.update(scope=label,trial=r['trial'],stage=r['kind'],task=r.get('task'),role=r.get('role'),
                native_scope_method='same native thread and full CANN interval containment')
            by_scope[label].append(t);mapping[r['stream_handle']].add(t['stream'])
        elif t['model_id'] in models:
            t['stage']='graph_internal'
        else:
            require(e['name']=='PLACE_HOLDER_SQE','unknown task without host flow '+repr((e,cann)));unknown.append(t['id'])
        matches=csv_index[e['name'],t['stream'],t['task_id'],number(e['ts'])]
        if matches:
            j=only(matches,'CSV task identity');require(j not in used,'duplicate kernel');used.add(j)
            require(abs(number(kernels[j]['Duration(us)'])-number(e['dur']))<=Decimal('.001'),'CSV duration')
            t['csv_row']=j;t['is_compute']=not any(s in t['name'] for s in ('EVENT_','MEMCPY','PLACE_HOLDER'))
        require(not t['is_compute'] or 'csv_row' in t,'compute missing CSV')
        tasks.append(t);by_stream[t['stream']].append(t)
    require(len(used)==len(kernels),'CSV rows not fully accounted')
    require(all(len(v)==1 for v in mapping.values()),'ambiguous logical/physical mapping')
    replays=[]
    for r in observations:
        if r['kind']!='replay':continue
        model=only((m for m in models.values() if m['capture']['case']==r['trial'].split('-r')[0] and m['capture']['name']==r['task']),'replay graph source contract')
        boundary=by_scope[r['label']]
        require(sorted(t['name'] for t in boundary)==['MODEL_EXECUTE','NOTIFY_WAIT'],'replay launch and wait coverage')
        launch=only((t for t in boundary if t['name']=='MODEL_EXECUTE'),'launch')
        wait=only((t for t in boundary if t['name']=='NOTIFY_WAIT'),'completion')
        require(launch['cann_trace_index']==wait['cann_trace_index'] and launch['cann_api']=='AscendCL@aclmdlRIExecuteAsync','replay native connection')
        replay=dict(id=r['label'],trial=r['trial'],task=r['task'],model_id=only({d['args']['Model Id'] for d in model['dump']},'model id'),
            launch=launch['id'],completion=wait['id'],caller_stream=launch['stream'],cann_trace_index=launch['cann_trace_index'])
        model['replays'].append(replay);replays.append(replay)
    require(len(replays)==len(replay_calls)==40,'replay native API coverage')
    require(len({r['cann_trace_index'] for r in replays})==40,'unique replay native API')
    byid={t['id']:t for t in tasks}
    for mid,model in models.items():
        group=sorted((t for t in tasks if t['model_id']==mid),key=lambda t:number(t['start_us']))
        dump=model['dump'];size=len(dump);rs=model['replays']
        require(len(group)==len(rs)*size,'complete graph task repetitions')
        for ordinal,replay in enumerate(rs):
            chunk=group[ordinal*size:(ordinal+1)*size]
            for t,d in zip(chunk,dump):
                a=d['args']
                require((t['stream'],t['task_id'])==(str(a['Stream Id']),str(a['Task Id'])),'exact dump stream/task sequence')
                types={'KERNEL_AIVEC':'AI_VECTOR_CORE','KERNEL_AICORE':'AI_CORE','KERNEL_MIX_AIV':'MIX_AIV','NOTIFY_RECORD':'NOTIFY_RECORD','MEMCPY_ASYNC':'SDMA_SQE'}
                require(types[a['Task Type']]==t['task_type'],'dump core type')
                t.update(trial=replay['trial'],task=replay['task'],scope=replay['id'],replay=replay['id'],dump_symbol=d['name'])
                by_scope[replay['id']].append(t)
            launch=byid[replay['launch']];wait=byid[replay['completion']]
            require(number(launch['start_us'])<=number(chunk[0]['start_us']) and number(chunk[-1]['end_us'])<=number(wait['end_us']),'replay task enclosure')
            require(chunk[-1]['name']=='NOTIFY_RECORD','graph terminal notify')
            replay.update(internal_stream=chunk[0]['stream'],internal_tasks=[t['id'] for t in chunk],
                evidence='source-pinned graph object callsite + unique live dump Model ID + complete stream/task sequence + exact CANN launch/wait connection and enclosure; no nearest-time join')
            graph.edge(launch['id'],chunk[0]['id'],'graph_launch',replay=replay['id'])
            graph.edge(chunk[-1]['id'],wait['id'],'graph_completion',replay=replay['id'],
                evidence='replay completion semantics; exact NOTIFY identifier pair not claimed')
    for t in tasks:graph.node(t['id'],**{k:v for k,v in t.items() if k!='id'})
    for stream,ts in by_stream.items():
        ts.sort(key=lambda t:(number(t['start_us']),int(t['task_id'])))
        for a,b in zip(ts,ts[1:]):
            require(number(a['end_us'])<=number(b['start_us']),'same-stream task overlap')
            graph.edge(a['id'],b['id'],'stream_order',stream=stream)
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
                if not calls:
                    calls=[(ci,None) for ci,c in wait_calls if native_scope(c)==r['label']]
                ci,correlation=only(calls,'native elided wait API association')
                if stream==record['stream']:
                    same_stream_waits+=1  # Same-stream FIFO already orders it.
                else:
                    barrier='api:'+r['label']
                    graph.node(barrier,kind='runtime_wait',trial=r['trial'],stream=stream,
                        scope=r['label'],cann_trace_index=ci,queue_correlation=correlation,
                        note='Successful native stream wait; no physical EVENT_WAIT task observed')
                    graph.edge(record['id'],barrier,'event_wait_api',event_handle=key,cann_trace_index=ci)
                    def submitted(t):
                        index=t['host_trace_index'] if t['host_trace_index'] is not None else t['cann_trace_index']
                        if index is None:return None
                        host=events[index]
                        if host['tid']==scope['tid'] and number(host['ts'])>=end(scope):return number(host['ts'])
                        return None
                    after=[t for t in by_stream[stream] if submitted(t) is not None]
                    require(bool(after),'logical wait without subsequent stream work')
                    target=min(after,key=lambda t:(submitted(t),number(t['start_us'])))
                    graph.edge(barrier,target['id'],'stream_submission_order',stream=stream,
                        evidence='same host thread, native task queue and same logical stream')
                    api_waits.append(dict(node=barrier,record=record['id'],next_task=target['id'],event_handle=key))
        elif r['kind']=='event_synchronize':
            require(key in last,'sync without record')
            host_id='h:'+r['label'];graph.node(host_id,kind='host_completion',trial=r['trial'],scope=r['label'])
            graph.edge(last[key]['id'],host_id,'host_sync',event_handle=key)
    return dict(replays=replays,graph=graph,tasks=tasks,observations=observations,scopes=scopes,by_scope=by_scope,mapping={k:sorted(v) for k,v in mapping.items()},collisions=collisions,api_waits=api_waits,event_edges=event_edges,same_stream_waits=same_stream_waits,unknown=unknown,outside=outside)
