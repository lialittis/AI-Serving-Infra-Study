"""Join stream lifetimes, graph dumps, exact task flows and event generations."""
import argparse
from collections import Counter,defaultdict
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import statistics


def require(ok,message):
    if not ok:raise ValueError(message)


def only(xs,message):
    xs=list(xs);require(len(xs)==1,message+' (found %d)'%len(xs));return xs[0]


def num(x):return Decimal(str(x))
def end(e):return num(e['ts'])+num(e.get('dur',0))
def inside(a,b):return a['pid']==b['pid'] and a['tid']==b['tid'] and num(a['ts'])<=num(b['ts']) and end(b)<=end(a)
def load(p):return json.loads(p.read_text())
def jsonlines(root):return [json.loads(s) for p in sorted(root.glob('*.jsonl')) for s in p.read_text().splitlines()]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def validate_dag(nodes,edges):
    ids={n['id'] for n in nodes};degree={k:0 for k in ids};out=defaultdict(list)
    for e in edges:
        require(e['source'] in ids and e['target'] in ids,'unknown DAG endpoint')
        out[e['source']].append(e['target']);degree[e['target']]+=1
    ready=[k for k,d in degree.items() if not d];visited=0
    while ready:
        k=ready.pop();visited+=1
        for target in out[k]:
            degree[target]-=1
            if not degree[target]:ready.append(target)
    require(visited==len(ids),'execution dependency cycle')


def live_graph(graphs,model_id,time_ns):
    candidates=[g for g in graphs if model_id in g['model_ids'] and num(g['capture_time_ns'])<=num(time_ns) and
                (g['destroy_time_ns'] is None or num(time_ns)<num(g['destroy_time_ns']))]
    require(len(candidates)<=1,'ambiguous graph model id')
    return candidates[0] if candidates else None


def check_handle(found,stream):
    require(len(found)==1 and int(stream) in found[0]['runtime_ids'],'submitted handle / physical stream mismatch')
    return found[0]['id']


def compute_overlap(tasks):
    points=[]
    for t in tasks:
        if t['csv'] is not None and num(t['end_us'])>num(t['start_us']):
            points.extend([(num(t['start_us']),1,t['stream']),(num(t['end_us']),-1,t['stream'])])
    counts=Counter();last=None;total=Decimal(0);peak=0
    for time,change,stream in sorted(points):
        if last is not None and sum(v>0 for v in counts.values())>1:total+=time-last
        counts[stream]+=change;peak=max(peak,sum(v>0 for v in counts.values()));last=time
    return dict(cross_stream_compute_overlap_us=str(total),peak_observed_compute_streams=peak)


def analyze(run):
    command=load(run/'command.json');complete=load(run/'complete.json')
    require(not complete['baseline'],'diagnostic input required')
    for name,sha in load(run/'source_hashes.json').items():require(digest(run/name)==sha,'source hash mismatch '+name)
    responses=load(run/'responses.json');require(len(responses)==2,'two formal requests required')
    for r in responses:require(r['response']['usage']['completion_tokens']==complete['batch']*4,'incomplete response')
    py=jsonlines(run/'stream_events');native=jsonlines(run/'native');base=jsonlines(run/'events')
    require(not [e for e in py if e['kind']=='observer_error'],'Python observer errors')
    require(not [e for e in base if e.get('event')=='trace_error'],'base observer errors')
    entries={e['id']:e for e in py if e['kind']=='call'}
    returns={e['id']:e for e in py if e['kind']=='return'}
    spans={k:dict(v,end_ns=returns[k]['monotonic_ns'],returned=returns[k]['returned'],self_after=returns[k]['self_after']) for k,v in entries.items() if k in returns}
    def encloses(p,n):return p['pid']==n['pid'] and p['tid']==n['tid'] and p['monotonic_ns']<=n['begin_ns'] and n['end_ns']<=p['end_ns']
    ready=load(run/'ready.json')['monotonic_ns']
    def stage(ns):
        for i,r in enumerate(responses):
            if r['start_monotonic_ns']<=ns<=r['end_monotonic_ns']:return 'request-'+str(i)
        return 'initialization' if ns<ready else 'warmup_or_control'
    for i,n in enumerate(native):n.update(index=i,stage=stage(n['begin_ns']))
    # Explicit creates and destroys define handle generations, not Python ids.
    lifetimes=[];active={};generation=Counter()
    for n in sorted(native,key=lambda n:n['begin_ns']):
        key=(n['pid'],n['stream'])
        if n['result']==0 and n['api'] in ('aclrtCreateStream','aclrtCreateStreamWithConfig','aclrtCreateStreamV2'):
            require(key not in active,'stream handle recreated while live')
            generation[key]+=1
            parents=[p for p in spans.values() if encloses(p,n)]
            parents.sort(key=lambda p:p['end_ns']-p['monotonic_ns'])
            item=dict(id='%d:%s:g%d'%(key[0],key[1],generation[key]),pid=n['pid'],handle=n['stream'],generation=generation[key],
                      create=n,destroy=None,creation_scope=parents[0]['id'] if parents else None,python_acquisitions=[],runtime_ids=[])
            active[key]=item;lifetimes.append(item)
        elif n['result']==0 and n['api'] in ('aclrtDestroyStream','aclrtDestroyStreamForce'):
            if key in active:active.pop(key)['destroy']=n
    def lifetime(pid,handle,ns):
        return [s for s in lifetimes if s['pid']==pid and s['handle']==handle and s['create']['end_ns']<=ns and (s['destroy'] is None or ns<=s['destroy']['end_ns'])]
    objects=[]
    for p in spans.values():
        if p['module']=='torch_npu.npu.streams' and p['function']=='Stream.__new__':
            obj=p['returned']
            if not isinstance(obj,dict) or 'handle' not in obj:continue
            found=lifetime(p['pid'],obj['handle'],p['end_ns'])
            require(len(found)<=1,'ambiguous stream lifetime')
            record=dict(call=p['id'],pid=p['pid'],stage=stage(p['monotonic_ns']),object=obj,stack=p['stack'],lifetime=found[0]['id'] if found else None,
                        acquisition='wrap_existing' if p['arguments'].get('kwargs',{}).get('stream_id') is not None else 'pool_or_new_resource')
            objects.append(record)
            if found:
                found[0]['python_acquisitions'].append(p['id'])
                if obj['stream_id_query_result']==0 and obj['runtime_stream_id'] not in found[0]['runtime_ids']:found[0]['runtime_ids'].append(obj['runtime_stream_id'])
    require(lifetimes and objects,'missing stream creation/acquisition evidence')
    # Model RI handle comes from the native capture end, not from Python id().
    graphs=[]
    for d in (e for e in py if e['kind']=='graph_dump'):
        capture=spans[d['call']]
        api=only((n for n in native if n['api']=='aclmdlRICaptureEnd' and n['result']==0 and encloses(capture,n)),'native capture end')
        dumped=load(run/d['path'])
        parents=[p for p in spans.values() if p.get('partition') and p['pid']==capture['pid'] and p['tid']==capture['tid'] and p['monotonic_ns']<=capture['monotonic_ns'] and capture['end_ns']<=p['end_ns']]
        parent=min(parents,key=lambda p:p['end_ns']-p['monotonic_ns']) if parents else None
        model_ids=sorted({int(e['args']['Model Id']) for e in dumped})
        streams=sorted({int(e['args']['Stream Id']) for e in dumped})
        destroyed=sorted((n for n in native if n['api']=='aclmdlRIDestroy' and n['result']==0 and n['pid']==api['pid'] and n['object']==api['object'] and api['end_ns']<n['begin_ns']),key=lambda n:n['begin_ns'])
        destroy=destroyed[0] if destroyed else None
        offset=capture['time_ns']-capture['monotonic_ns']
        graphs.append(dict(id=d['call'],pid=d['pid'],python_graph=d['graph_id'],native_model=api['object'],capture_stream=api['stream'],
                           capture_native_index=api['index'],partition=parent.get('partition') if parent else None,
                           capture_start_ns=capture['monotonic_ns'],capture_time_ns=capture['time_ns'],destroy_native=destroy,
                           destroy_time_ns=destroy['begin_ns']+offset if destroy else None,
                           model_ids=model_ids,streams=streams,dump=d['path'],dump_tasks=dumped))
    trace_path=only((run/'profiler').rglob('trace_view.json'),'profiler trace')
    events=json.loads(trace_path.read_text(),parse_float=Decimal)
    if isinstance(events,dict):events=events['traceEvents']
    csv_path=only((run/'profiler').rglob('kernel_details.csv'),'kernel CSV')
    with csv_path.open() as f:rows=list(csv.DictReader(f))
    csv_index=defaultdict(list)
    for j,r in enumerate(rows):csv_index[r['Name'],r['Stream ID'].strip(),r['Task ID'].strip(),num(r['Start Time(us)'])].append((j,r))
    xs=defaultdict(list);starts=defaultdict(list);finishes=defaultdict(list)
    def point(e):return e['pid'],e['tid'],num(e['ts'])
    for i,e in enumerate(events):
        if e.get('ph')=='X':xs[point(e)].append(i)
        elif e.get('ph')=='s':starts[e.get('cat'),str(e['id'])].append(i)
        elif e.get('ph')=='f':finishes[(e.get('cat'),)+point(e)].append(i)
    def source(task,cat):
        fs=finishes[(cat,)+point(task)]
        if not fs:return None,dict(status='missing_flow')
        fi=only(fs,'unique flow finish');f=events[fi]
        ss=starts[cat,str(f['id'])]
        if len(ss)!=1:return None,dict(status='ambiguous_flow',candidates=ss,finish=fi)
        si=ss[0];hi=only(xs[point(events[si])],'unique flow source')
        return hi,dict(status='exact',start=si,finish=fi,flow_id=str(f['id']))
    base_entries={e['label']:e for e in base if e.get('event')=='enter'}
    step_phases={}
    for i,response in enumerate(responses):
        prefix=response['response']['id']
        runners=[e for e in base_entries.values() if e['kind']=='runner' and any(rid.startswith(prefix+'-') for rid in (e.get('request_ids') or [e.get('request_id') or '']))]
        runners.sort(key=lambda e:e['monotonic_ns'])
        require(len(runners)>=4,'missing request scheduling steps')
        if not command['case'].startswith('sampling'):require(len(runners)==4,'unexpected single-request scheduling shape')
        for j,runner in enumerate(runners):
            if command['case'].startswith('sampling'):
                counts=list(runner['scheduled_tokens'].values())
                kind='decode' if max(counts)==1 else 'mixed' if min(counts)==1 else 'prefill'
                phase='step-%d-%s'%(j+1,kind)
            else:phase='prefill' if j==0 else 'decode-%d'%j
            step_phases[runner['step']]='request-%d/'%i+phase
    scopes=[(i,e) for i,e in enumerate(events) if e.get('ph')=='X' and e.get('name') in base_entries]
    tasks=[];used=set()
    for i,e in enumerate(events):
        a=e.get('args',{})
        if e.get('ph')!='X' or 'Task Type' not in a or e['name'] in ('PROFILING_ENABLE','PROFILING_DISABLE'):continue
        ci,cf=source(e,'HostToDevice');hi,hf=source(e,'async_npu')
        if ci is not None:require(events[ci]['args']['connection_id']==a['connection_id'],'runtime connection mismatch')
        containing=sorted([(j,s) for j,s in scopes if hi is not None and inside(s,events[hi])],key=lambda js:num(js[1]['dur']))
        observation=base_entries[containing[0][1]['name']] if containing else None
        match=csv_index[e['name'],str(a['Physic Stream Id']),str(a['Task Id']),num(e['ts'])]
        csv_evidence=None
        if match:
            j,row=only(match,'CSV identity');require(j not in used,'duplicate CSV identity');used.add(j)
            require(abs(num(row['Duration(us)'])-num(e['dur']))<=Decimal('.001'),'CSV duration mismatch')
            csv_evidence=dict(index=j,core_type=row['Accelerator Core'],block_num=row['Block Num'],mix_block_num=row['Mix Block Num'])
        graph=live_graph(graphs,a.get('Model Id'),num(e['ts'])*1000)
        dump_match=[]
        if graph:
            dump_match=[(j,d) for j,d in enumerate(graph['dump_tasks']) if int(d['args']['Stream Id'])==a['Physic Stream Id'] and int(d['args']['Task Id'])==a['Task Id']]
            require(len(dump_match)<=1,'ambiguous graph task')
            # Debug dump names are compiled symbols; profiler can expose an
            # aclnn operation alias. Identity is model generation + stream/task.
            if dump_match and a['Task Type']=='AI_VECTOR_CORE':
                require(dump_match[0][1]['args']['Task Type']=='KERNEL_AIVEC','graph dump core type mismatch')
        tasks.append(dict(id='k:'+str(i),trace_index=i,name=e['name'],stream=str(a['Physic Stream Id']),task_id=str(a['Task Id']),model_id=a.get('Model Id'),
                          start_us=str(e['ts']),end_us=str(end(e)),duration_us=str(e['dur']),task_type=a['Task Type'],csv=csv_evidence,
                          host_index=hi,cann_index=ci,host_flow=hf,cann_flow=cf,connection_id=a.get('connection_id'),
                          host_name=events[hi]['name'] if hi is not None else None,
                          scope=observation['label'] if observation else None,scope_kind=observation['kind'] if observation else None,
                          phase=step_phases.get(observation.get('step'),'unattributed') if observation else 'unattributed',
                          step=observation.get('step') if observation else None,graph=graph['id'] if graph else None,
                          partition=graph['partition'] if graph else None,dump_task_index=dump_match[0][0] if dump_match else None,
                          dump_kernel_symbol=dump_match[0][1]['name'] if dump_match else None))
    require(len(used)==len(rows),'unmatched kernel CSV rows')
    handle_checks=[]
    for t in tasks:
        if t['host_index'] is None:continue
        launches=[base_entries[s['name']] for _,s in scopes if base_entries[s['name']]['kind']=='launch' and inside(s,events[t['host_index']])]
        for launch in launches:
            found=lifetime(launch['pid'],str(launch['runtime_stream']),launch['monotonic_ns'])
            check_handle(found,t['stream'])
            handle_checks.append(dict(task=t['id'],scope=launch['label'],lifetime=found[0]['id'],stream=t['stream']))
    # Native-to-profiler: same API/thread and exact sequence within HTTP windows.
    # Wall-clock consistency is a check, never a nearest-time matching rule.
    worker_pids={int(s['pid']) for s in base_entries.values()}
    require(len(worker_pids)==1,'this analyzer requires one profiled worker process')
    offsets={pid:statistics.median(e['time_ns']-e['monotonic_ns'] for e in py if e['pid']==pid) for pid in worker_pids}
    def request_window_us(t):return any(num(r['start_ns'])/1000<=num(t)<=num(r['end_ns'])/1000 for r in responses)
    ng=defaultdict(list);cg=defaultdict(list)
    for n in native:
        if n['pid'] in worker_pids and n['stage'].startswith('request-'):ng[n['tid'],n['api']].append(n)
    for i,e in enumerate(events):
        if e.get('ph')=='X' and e.get('name','').startswith('AscendCL@') and request_window_us(e['ts']):
            cg[e.get('args',{}).get('Thread Id',e['tid']),e['name'].split('@',1)[1]].append((i,e))
    native_to_cann={};unmatched_api=[]
    for key,ns in ng.items():
        cs=cg.get(key,[])
        if len(ns)!=len(cs):unmatched_api.append(dict(thread=key[0],api=key[1],native=len(ns),profiler=len(cs)));continue
        ns.sort(key=lambda n:n['begin_ns']);cs.sort(key=lambda ie:num(ie[1]['ts']))
        for n,(i,e) in zip(ns,cs):
            delta=abs(num(n['begin_ns']+offsets[n['pid']])-num(e['ts'])*1000)
            require(delta<1000000,'native/profiler sequence clock mismatch')
            native_to_cann[n['index']]=dict(trace_index=i,method='same API/thread + complete ordered sequence + clock consistency',clock_delta_ns=str(delta))
    bycann=defaultdict(list)
    for t in tasks:
        if t['cann_index'] is not None:bycann[t['cann_index']].append(t)
    for n in native:
        proof=native_to_cann.get(n['index'])
        if not proof or n['stream']=='0':continue
        for task in bycann[proof['trace_index']]:
            found=lifetime(n['pid'],n['stream'],n['begin_ns'])
            ident=check_handle(found,task['stream'])
            handle_checks.append(dict(task=task['id'],native=n['index'],lifetime=ident,stream=task['stream']))
    replays=[]
    for n in native:
        if n['api']!='aclmdlRIExecuteAsync' or not n['stage'].startswith('request-') or n['result']!=0:continue
        graph=only((g for g in graphs if g['pid']==n['pid'] and g['native_model']==n['object'] and g['capture_start_ns']<n['begin_ns'] and
                    (g['destroy_native'] is None or n['end_ns']<g['destroy_native']['begin_ns'])),'replay captured graph')
        proof=native_to_cann.get(n['index'])
        connection=events[proof['trace_index']]['args']['connection_id'] if proof else None
        boundary=[t for t in tasks if t['connection_id']==connection and t['name'] in ('MODEL_EXECUTE','NOTIFY_WAIT')] if proof else []
        if proof:require(Counter(t['name'] for t in boundary)==Counter(['MODEL_EXECUTE','NOTIFY_WAIT']),'replay boundary connection mismatch')
        for task in boundary:
            ident=check_handle(lifetime(n['pid'],n['stream'],n['begin_ns']),task['stream'])
            handle_checks.append(dict(task=task['id'],native=n['index'],lifetime=ident,stream=task['stream']))
        replays.append(dict(id='replay:'+str(n['index']),native=n,graph=graph['id'],partition=graph['partition'],cann=proof,
                           cann_event=events[proof['trace_index']] if proof else None,boundary_tasks=[t['id'] for t in boundary]))
    byid={t['id']:t for t in tasks}
    replay_ordinals=Counter()
    for r in sorted(replays,key=lambda r:r['native']['begin_ns']):
        key=r['graph'],r['native']['stage'];replay_ordinals[key]+=1
        r['phase']=r['native']['stage']+'/decode-'+str(replay_ordinals[key])
    for graph in graphs:
        rs=sorted((r for r in replays if r['graph']==graph['id']),key=lambda r:r['native']['begin_ns'])
        if not rs:continue
        group=sorted((t for t in tasks if t['graph']==graph['id']),key=lambda t:num(t['start_us']))
        size=len(graph['dump_tasks'])+1
        require(len(group)==len(rs)*size,'graph execution task coverage mismatch')
        for ordinal,r in enumerate(rs):
            chunk=group[ordinal*size:(ordinal+1)*size]
            require([t['dump_task_index'] for t in chunk[:-1]]==list(range(size-1)) and chunk[-1]['name']=='NOTIFY_RECORD','graph execution task pattern mismatch')
            r['internal_tasks']=[t['id'] for t in chunk]
            r['membership_evidence']='live model identity + exact dump stream/task IDs + repeated per-graph sequence + matching replay boundaries'
            if r['boundary_tasks']:
                launch=only((byid[k] for k in r['boundary_tasks'] if byid[k]['name']=='MODEL_EXECUTE'),'model execute boundary')
                wait=only((byid[k] for k in r['boundary_tasks'] if byid[k]['name']=='NOTIFY_WAIT'),'graph wait boundary')
                require(num(launch['start_us'])<=num(chunk[0]['start_us']) and num(chunk[-1]['end_us'])<=num(wait['end_us']),'graph tasks outside replay completion boundary')
            for t in chunk+[byid[k] for k in r['boundary_tasks']]:t.update(replay=r['id'],phase=r['phase'])
    # Native event generations: record/reset/destruction can reuse one handle.
    last_record={};event_generation=Counter();dependencies=[]
    for n in sorted(native,key=lambda n:n['begin_ns']):
        if n['result']!=0:continue
        key=n['pid'],n['object'];api=n['api']
        if api=='aclrtRecordEvent':event_generation[key]+=1;last_record[key]=n
        elif api in ('aclrtDestroyEvent','aclrtResetEvent'):last_record.pop(key,None)
        elif api in ('aclrtStreamWaitEvent','aclrtStreamWaitEventWithTimeout','aclrtSynchronizeEvent'):
            producer=last_record.get(key)
            if producer is not None:
                require(producer['end_ns']<=n['begin_ns'],'overlapping event host operations need explicit order')
                dependencies.append(dict(kind='device_event_wait' if 'StreamWait' in api else 'host_event_completion',event=n['object'],generation=event_generation[key],
                                         pid=n['pid'],record_native=producer['index'],wait_native=n['index'],source_stream=producer['stream'],target_stream=n['stream'],
                                         cross_stream=('StreamWait' in api and producer['stream']!=n['stream']),
                                         record_cann=native_to_cann.get(producer['index']),wait_cann=native_to_cann.get(n['index'])))
    edges=[]
    for stream in {t['stream'] for t in tasks}:
        group=sorted((t for t in tasks if t['stream']==stream),key=lambda t:num(t['start_us']))
        for a,b in zip(group,group[1:]):
            require(num(a['end_us'])<=num(b['start_us']),'overlapping tasks in one physical stream')
            edges.append(dict(source=a['id'],target=b['id'],kind='stream_order'))
    completion_nodes=[]
    for dep in dependencies:
        if dep['kind']=='device_event_wait' and dep['record_cann'] and dep['wait_cann']:
            rec=[t for t in bycann[dep['record_cann']['trace_index']] if t['name']=='EVENT_RECORD']
            wait=[t for t in bycann[dep['wait_cann']['trace_index']] if t['name']=='EVENT_WAIT']
            if len(rec)==len(wait)==1:
                require(num(rec[0]['end_us'])<=num(wait[0]['end_us']),'event wait completes before record')
                edges.append(dict(source=rec[0]['id'],target=wait[0]['id'],kind='event_wait_completion',event=dep['event'],generation=dep['generation']))
        elif dep['kind']=='host_event_completion' and dep['record_cann'] and dep['wait_cann']:
            rec=[t for t in bycann[dep['record_cann']['trace_index']] if t['name']=='EVENT_RECORD']
            wait=events[dep['wait_cann']['trace_index']]
            if len(rec)!=1:continue
            require(num(rec[0]['end_us'])<=end(wait),'CPU event wait returned before device record')
            node=dict(id='host-wait:'+str(dep['wait_native']),kind='host_event_completion',name=wait['name'],
                      start_us=str(wait['ts']),end_us=str(end(wait)),trace_index=dep['wait_cann']['trace_index'])
            completion_nodes.append(node)
            edges.append(dict(source=rec[0]['id'],target=node['id'],kind='host_event_completion',event=dep['event'],generation=dep['generation']))
            later=[t for t in tasks if t['host_index'] is not None and events[t['host_index']]['tid']==wait['tid'] and num(events[t['host_index']]['ts'])>=end(wait)]
            if later:
                earliest=min(num(events[t['host_index']]['ts']) for t in later)
                following=[t for t in later if num(events[t['host_index']]['ts'])==earliest]
                dep['next_submitted_tasks']=[t['id'] for t in following]
                for task in following:edges.append(dict(source=node['id'],target=task['id'],kind='host_after_wait',host_trace_index=task['host_index']))
    validate_dag(tasks+completion_nodes,edges)
    inventory=[]
    for sid in sorted({t['stream'] for t in tasks},key=int):
        group=[t for t in tasks if t['stream']==sid]
        resources=[s for s in lifetimes if s['pid'] in worker_pids and int(sid) in s['runtime_ids']]
        active_graphs={t['graph'] for t in group if t['graph']}
        gs=[g for g in graphs if g['id'] in active_graphs]
        inventory.append(dict(stream=sid,tasks=len(group),compute_tasks=sum(t['csv'] is not None for t in group),
                              lifetimes=[s['id'] for s in resources],graphs=[g['id'] for g in gs],partitions=[g['partition'] for g in gs],
                              origin='python/native_stream' if resources else 'graph_internal_resource' if gs else 'unresolved',
                              creation_proven=bool(resources),graph_membership_proven=bool(gs)))
    return dict(schema_version=1,case=command['case'],streams=inventory,lifetimes=lifetimes,python_objects=objects,python_spans=list(spans.values()),graphs=graphs,replays=replays,
                tasks=tasks,completion_nodes=completion_nodes,edges=edges,event_dependencies=dependencies,native=native,native_to_cann=native_to_cann,handle_checks=handle_checks,
                summary=dict(device_tasks=len(tasks),compute_tasks=len(rows),physical_streams=len(inventory),native_creates=len(lifetimes),
                             graph_captures=len(graphs),formal_replays=len(replays),graph_tasks_matched=sum(t['dump_task_index'] is not None for t in tasks),
                             resource_limit_calls=sum(n['api']=='aclrtSetStreamResLimit' for n in native),unresolved_streams=sum(s['origin']=='unresolved' for s in inventory),**compute_overlap(tasks)),
                gaps=dict(unmatched_native_profiler_apis=unmatched_api,unreturned_python_calls=sorted(set(entries)-set(returns)),
                          graph_dump_errors=[e for e in py if e['kind']=='graph_dump_unavailable'],
                          internal_stream_creation='Graph membership is not a native creation call. CANN-internal allocation not necessarily visible at the audited API boundary.',
                          notify_pairing='Model membership and replay boundaries do not alone prove an exact NOTIFY identifier pair.'),
                provenance=dict(trace_sha256=digest(trace_path),csv_sha256=digest(csv_path)),responses=responses)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);a=p.parse_args()
    result=analyze(a.run);out=a.run/'analysis';out.mkdir(exist_ok=True)
    (out/'stream_evidence.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str)+'\n')
    print(json.dumps(result['summary']))


if __name__=='__main__':main()
