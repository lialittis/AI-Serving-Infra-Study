"""Offline reconstruction: observed stream/event HB versus required branch data edges."""
import argparse,csv,hashlib,json,statistics,sys
from collections import defaultdict,Counter
from decimal import Decimal
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'practice_17_vllm_multistream'))
from analyze_run import number,end,inside,only,require,resolve_colliding_host,intersection,union
sys.path.insert(0,str(ROOT/'practice_20_kv_offload_overlap'))
from hb import Graph


def read(p):return json.loads(p.read_text())
def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def length(xs):return sum((b-a for a,b in union(xs)),Decimal(0))
def intervals(ts):return [(number(t['start_us']),number(t['end_us'])) for t in ts]
def stats(xs):
    q=statistics.quantiles(xs,n=4,method='inclusive') if len(xs)>1 else [xs[0]]*3
    return dict(n=len(xs),median=statistics.median(xs),q1=q[0],q3=q[2],min=min(xs),max=max(xs))


def analyze(root):
    plan=read(root/'plan.json');complete=read(root/'completed.json');require(complete['status']=='passed','incomplete capture')
    for f,v in read(root/'sources.json').items():
        require(hashlib.sha256((root/f).read_bytes()).hexdigest()==v['sha256'],'source mismatch '+f)
    manifest=read(root/'sources.json')
    for f,expected in read(Path(__file__).with_name('contracts.json'))['sources'].items():
        require(manifest[f]['sha256']==expected,'unaudited implementation '+f)
    require(read(root/'weight_check.json')['unchanged'],'weights changed during experiment')
    checks=read(root/'correctness.json');require(len(checks)==len(plan['tokens'])*3,'numerical coverage')
    require(all(c['exact_modes'] and c['finite'] and c['cpu_reference_pass'] and
        all(c['routing'][k] for k in ('logits_valid','selected_scores_valid','weights_valid','distinct_ids'))
        for c in checks),'numerical failure')
    require(all(c['all_outputs_equal'] for c in read(root/'lifetime_checks.json')),'input reuse failure')
    rows=read(root/'measurements.json');require(len(rows)==len(plan['tokens'])*plan['rounds']*4,'sample count')
    require(len({(r['tokens'],r['round'],r['mode']) for r in rows})==len(rows),'duplicate performance sample')
    performance=[]
    for n in plan['tokens']:
        modes={m:stats([r['wall_us_per_call'] for r in rows if r['tokens']==n and r['mode']==m])
               for m in ('serial','parallel','shared_only','routed_only')}
        paired=[]
        for r in range(plan['rounds']):
            values={x['mode']:x['wall_us_per_call'] for x in rows if x['tokens']==n and x['round']==r}
            paired.append(100*(values['parallel']/values['serial']-1))
        performance.append(dict(tokens=n,modes=modes,paired_change_percent=stats(paired),
            parallel_faster_pairs=sum(p<0 for p in paired),
            pooled_iqr_disjoint=modes['parallel']['q3']<modes['serial']['q1'] or modes['serial']['q3']<modes['parallel']['q1']))
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
    collisions=0
    def source(task,cat,cann=None):
        nonlocal collisions
        fs=finishes[(cat,)+point(task)]
        if not fs:return None
        finish=only(fs,'flow endpoint');ss=starts[cat,str(finish['id'])]
        if len(ss)>1 and cat=='async_npu' and cann:
            collisions+=1
            hosts=[only(points[point(s)],'host op') for s in ss]
            i,e,_=resolve_colliding_host(hosts,cann,queues);return i,e,str(finish['id'])
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
                t.update(scope=label,trial=r['trial'],stage=r['kind']);by_scope[label].append(t)
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
                for ci,c in enumerate(events):
                    if c.get('name')!='AscendCL@aclrtStreamWaitEvent':continue
                    for dequeue,enqueue,correlation in queues:
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
    trials=read(root/'trials.json');require(len(trials)==len(plan['tokens'])*plan['profile_repeats']*2,'diagnostic coverage')
    requirements=[];summaries=[]
    pairs=[('input','router'),('input','routed'),('router','routed'),('input','shared_up'),
           ('shared_up','shared_down'),('input','shared_down'),('shared_down','merge'),('routed','merge')]
    for trial in trials:
        ts=[t for t in tasks if t['trial']==trial['id']];compute=[t for t in ts if t['is_compute']]
        stages={k:sorted([t for t in compute if t['stage']==k],key=lambda t:number(t['start_us']))
                for k in ('input','router','routed','shared_up','shared_down','merge')}
        require(all(stages.values()),'missing branch or merge '+trial['id'])
        branch=stages['shared_up']+stages['shared_down']
        streams={t['stream'] for t in compute}
        require(len(streams)==(1 if trial['mode']=='serial' else 2),'unexpected compute streams')
        overlap=length(intersection(intervals(stages['routed']),intervals(branch)))
        if trial['mode']=='serial':require(overlap==0,'serial compute overlap')
        for a,b in pairs:requirements.append(dict(source=stages[a][-1]['id'],target=stages[b][0]['id'],
            kind='data_dependency',trial=trial['id'],producer=a,consumer=b,evidence='observed original method tensor boundaries + audited Qwen2/Ascend implementation'))
        # Metadata proves the merged tensors are the exact branch outputs.
        rs={r['kind']:r for r in observations if r['trial']==trial['id'] and r['kind'] in stages}
        require(set(x['ptr'] for x in rs['merge']['inputs'])=={rs['routed']['output']['ptr'],rs['shared_down']['output']['ptr']},'merge input identity')
        input_ptr=rs['input']['output']['ptr']
        require(input_ptr==rs['router']['inputs'][0]['ptr'],'router input identity')
        require(input_ptr==rs['routed']['inputs'][0]['ptr'] and
                rs['router']['output']['ptr']==rs['routed']['inputs'][1]['ptr'],'routed input identities')
        require(input_ptr==rs['shared_up']['inputs'][0]['ptr'],'shared up input identity')
        require(input_ptr==rs['shared_down']['inputs'][0]['ptr'] and
                rs['shared_up']['output']['ptr']==rs['shared_down']['inputs'][1]['ptr'],'shared down input identities')
        ro,so=rs['routed']['output'],rs['shared_down']['output']
        require(int(ro['ptr'])+ro['bytes']<=int(so['ptr']) or int(so['ptr'])+so['bytes']<=int(ro['ptr']),
                'overlapping live branch outputs')
        summaries.append(dict(**trial,device_tasks=len(ts),compute_tasks=len(compute),streams=sorted(streams),
            branch_busy_us={k:str(length(intervals(v))) for k,v in stages.items()},
            branch_overlap_us=str(overlap),device_span_us=str(max(number(t['end_us']) for t in compute)-min(number(t['start_us']) for t in compute)),
            stages={k:[t['id'] for t in v] for k,v in stages.items()}))
    verified=graph.verify(requirements);violations=[r for r in verified if not r['satisfied']]
    summary=dict(status='passed' if not violations else 'failed',trials=summaries,performance=performance,
        diagnostic_trials=len(trials),device_tasks=len(tasks),compute_tasks=sum(t['is_compute'] for t in tasks),
        event_wait_edges=len(event_edges),cross_stream_wait_apis_without_device_task=len(api_waits),same_stream_waits_without_device_task=same_stream_waits,required_data_edges=len(verified),unsatisfied_data_edges=len(violations),
        unknown_runtime_placeholders=len(unknown),tasks_outside_diagnostic_scopes=len(outside),
        collision_resolutions=collisions,stream_mapping={k:sorted(v) for k,v in mapping.items()},
        all_cross_mode_exact=True,cpu_max_abs=max(c['cpu_max_abs'] for c in checks),
        topk_tie_choice_rows=sum(c['routing']['different_topk_rows'] for c in checks),
        complete_exact_kernel_memory_dag=False)
    output=root/'analysis';output.mkdir(exist_ok=True)
    save(output/'summary.json',summary)
    data=dict(summary=summary,nodes=list(graph.nodes.values()),edges=graph.edges,requirements=verified,records=observations)
    (output/'execution_graph.json').write_text(json.dumps(data,ensure_ascii=False,separators=(',',':'))+'\n')
    save(output/'unsatisfied.json',violations)
    require(not violations,'data dependencies lack happens-before')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('trials','performance')},indent=2))
    return data


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);analyze(p.parse_args().run)
