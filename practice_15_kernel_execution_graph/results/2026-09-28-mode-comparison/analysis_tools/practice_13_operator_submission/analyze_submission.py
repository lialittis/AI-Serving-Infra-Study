"""Join actual compile/load records, API parameters, queue flows and NPU tasks.

Compiler host timestamps and profiler timestamps are separate clocks; never
subtract them to infer device transfer timing. Runtime queue joins use real
correlation IDs and explicit OS thread identity across profiler PID namespaces.
"""
import argparse
from bisect import bisect_left,bisect_right
from collections import Counter,defaultdict
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'practice_09_operator_trace'))
from analyze_all_ops import audit as audit_all,cpu_parents
from summarize_profile import read_trace,require,only,inside,number


def load(p):return json.loads(p.read_text())
def end(e):return number(e['ts'])+number(e.get('dur',0))
def duration_ns(a,b):return str(Decimal(b['monotonic_ns']-a['monotonic_ns'])/Decimal(1000000))
def overlap(a,b):return max(Decimal(0),min(end(a),end(b))-max(number(a['ts']),number(b['ts'])))


def analyze(run,records=None,trace=None):
    records=records if records is not None else [json.loads(s) for p in (run/'events').glob('*.jsonl') for s in p.read_text().splitlines()]
    records=sorted(records,key=lambda e:(e['pid'],e['monotonic_ns'],e['tid'])) if all('pid' in e for e in records) else records
    require(not any(e['event']=='trace_error' for e in records),'instrumentation errors')
    require(load(run/'shutdown.json')['server_exit_code']==0,'unclean shutdown')
    require(not load(run/'cache_before.json')['triton_cache_exists'],'Triton cache was not initially empty')
    controls=load(run/'profile_control.json');require([(e['endpoint'],e['status']) for e in controls]==[('/start_profile',200),('/stop_profile',200)],'profiler controls')
    command=load(run/'command.json')['argv'];require('--enforce-eager' in command and '--num-gpu-blocks-override' not in command,'expected eager with native pool sizing')
    response=load(run/'response.json');prompt=load(run/'prompt_info.json')
    require(response['usage']==dict(prompt_tokens=prompt['prompt_tokens'],completion_tokens=4,total_tokens=prompt['prompt_tokens']+4) or
            response['usage']['prompt_tokens']==prompt['prompt_tokens'] and response['usage']['completion_tokens']==4,'wrong response token counts')
    for p,meta in load(run/'source_manifest.json').items():require(hashlib.sha256((run/p).read_bytes()).hexdigest()==meta['sha256'],'source mismatch '+p)
    artifacts=load(run/'compiler_artifacts.json')
    omitted=load(run/'unarchived_build_intermediates.json') if (run/'unarchived_build_intermediates.json').exists() else {}
    for p,meta in omitted.items():
        require(p.endswith('/precompiled.h.gch') and artifacts.get(p)==meta,'invalid omitted intermediate')
    for p,meta in artifacts.items():
        if p in omitted and not (run/p).exists():continue
        require(hashlib.sha256((run/p).read_bytes()).hexdigest()==meta['sha256'],'compiler artifact mismatch '+p)
    for p,digest in load(run/'instrumentation_hashes.json').items():
        require(hashlib.sha256((run/p).read_bytes()).hexdigest()==digest,'instrumentation snapshot mismatch '+p)
    entries={e['label']:e for e in records if e['event']=='enter'}
    exits={e['label']:e for e in records if e['event']=='exit'}
    require(set(entries)==set(exits),'unbalanced observations')
    require(Counter(e['label'] for e in records if e['event']=='exit')==Counter(exits.keys()),'duplicate return labels')
    require(Counter(e['label'] for e in records if e['event']=='enter')==Counter(entries.keys()),'duplicate observation labels')
    for e in entries.values():
        if e['kind'] in ('buffer_copy','to_list'):
            require(e.get('source_tensor',{}).get('kind')=='tensor','missing copy source tensor metadata')
    trace_path=only(list((run/'profiler').rglob('trace_view.json')),'trace')
    events=trace if trace is not None else read_trace(trace_path)
    with only(list((run/'profiler').rglob('kernel_details.csv')),'kernel CSV').open() as f:kernels=list(csv.DictReader(f))
    coverage,rows,hosts,kernel_names,_,queues=audit_all(events,kernels)
    scopes={e['name']:e for e in events if e.get('ph')=='X' and e.get('name','').startswith('P13/')}
    require(set(scopes)=={k for k,v in entries.items() if v['measured']},'measured scopes/profiler mismatch')
    rid=only(list({e['request_id'] for e in entries.values() if e['measured']}),'measured request')
    require(rid.startswith(response['id']+'-'),'request correlation mismatch')
    schedule=sorted([e for e in records if e['event']=='schedule' and rid in e['scheduled_tokens']],key=lambda e:e['step'])
    require([e['scheduled_tokens'][rid] for e in schedule]==[prompt['prompt_tokens'],1,1,1],'wrong prefill/decode sequence')
    phase={e['step']:('prefill' if i==0 else 'decode-'+str(i)) for i,e in enumerate(schedule)}
    compile_records=[];loads=[]
    ready=load(run/'ready.json')['time_ns'];window=load(run/'request_window.json')
    for label,entry in entries.items():
        out=exits[label];kind=entry['kind']
        if kind not in ('compile','load_binary','init_handles','launcher_init'):continue
        lifecycle='startup' if entry['time_ns']<ready else ('warmup' if entry['time_ns']<window['start_ns'] else 'measured')
        item=dict(id=label,kind=kind,host_stage=lifecycle,entry=entry,exit=out,duration_ms=duration_ns(entry,out),
                  clock='host monotonic; not a device timestamp')
        if kind=='compile':
            digest=out['binary']['sha256']
            item['binary_artifacts']=[p for p,v in artifacts.items() if v['sha256']==digest]
            require(item['binary_artifacts'],'compiled binary not archived')
            compile_records.append(item)
        else:loads.append(item)
    require(compile_records and all(c['exit']['ran_compiler_stages'] for c in compile_records),'no cold compiler pipeline evidence')
    native_loads=[e for e in loads if e['kind']=='load_binary']
    require(native_loads,'missing binary registration observations')
    for item in native_loads:
        item['compile_candidates']=[e['id'] for e in compile_records if e['exit']['binary']['sha256']==item['entry']['binary']['sha256'] and e['exit']['monotonic_ns']<=item['entry']['monotonic_ns']]
        require(item['compile_candidates'],'load without prior compiled binary')
        item['device_residency']='runtime binary/function registration observed; exact code DMA/residency timing not exposed'
    for initialized in [x for x in loads if x['kind']=='init_handles']:
        binary=only([x for x in native_loads if x['entry']['parent']==initialized['id']],'init/binary registration link')
        require(binary['entry']['binary']==initialized['entry']['binary'],'initialization binary mismatch')
        require(binary['exit']['returned_handles'][:2]==[initialized['exit']['module_handle'],initialized['exit']['function_handle']],
                'binary registration handle mismatch')
        initialized['native_load_record']=binary['id']
    launches=[e for e in entries.values() if e['kind']=='launch' and e['measured']]
    require(launches,'missing measured launcher observations')
    launch_records=[]
    for e in launches:
        candidates=[x for x in loads if x['kind']=='init_handles' and x['exit'].get('function_handle')==e['function_handle'] and
                    x['entry']['pid']==e['pid'] and x['exit']['monotonic_ns']<=e['monotonic_ns']]
        loaded=only(candidates,'launch/function initialization link')
        require(loaded['entry']['kernel_hash']==e['packed_metadata']['hash'],'launch metadata binary hash mismatch')
        launch_records.append(dict(entry=e,exit=exits[e['label']],load_record=loaded['id'],kernel_hash=loaded['entry']['kernel_hash']))
    all_complete=[e for e in events if e.get('ph')=='X']
    by_point=defaultdict(list)
    for e in all_complete:by_point[(e['pid'],e['tid'],number(e['ts']))].append(e)
    endpoints=defaultdict(list);starts=defaultdict(list)
    for e in events:
        if e.get('ph')=='f':endpoints[(e.get('cat'),e['pid'],e['tid'],number(e['ts']))].append(e)
        if e.get('ph')=='s':starts[(e.get('cat'),str(e['id']))].append(e)
    def source(task,cat):
        f=only(endpoints[(cat,task['pid'],task['tid'],number(task['ts']))],cat+' endpoint')
        s=only(starts[(cat,str(f['id']))],cat+' start')
        return only(by_point[(s['pid'],s['tid'],number(s['ts']))],cat+' source')
    cpu_ids=[i for i,e in enumerate(events) if e.get('ph')=='X' and e.get('cat')=='cpu_op']
    parents=cpu_parents(events,cpu_ids);has_child=set(p for p in parents.values() if p is not None)
    leaves=[events[i] for i in cpu_ids if i not in has_child]
    def index_by_lane(items,event_key):
        lanes=defaultdict(list)
        for item in items:
            e=event_key(item);lanes[(e['pid'],e['tid'])].append((number(e['ts']),item))
        return {k:(sorted(v,key=lambda x:x[0])) for k,v in lanes.items()}
    queue_lanes=index_by_lane(queues,lambda q:q['enqueue'])
    queue_starts={k:[x[0] for x in v] for k,v in queue_lanes.items()}
    runtime_by_tid=defaultdict(list)
    for e in all_complete:
        if e['name'].startswith(('AscendCL@','Runtime@')):runtime_by_tid[e['tid']].append(e)
    for tid in runtime_by_tid:runtime_by_tid[tid].sort(key=lambda e:number(e['ts']))
    runtime_starts={tid:[number(e['ts']) for e in es] for tid,es in runtime_by_tid.items()}
    full=[];examples={};queue_joined=0
    for row in rows:
        task=events[row['trace_index']]
        if row['route']=='profiler_control':continue
        host=source(task,'async_npu');cann=source(task,'HostToDevice')
        containing=[(label,s) for label,s in scopes.items() if inside(s,host)]
        containing.sort(key=lambda pair:number(pair[1]['dur']))
        labels=[label for label,_ in containing]
        context=entries[labels[0]] if labels else None
        # Start with the exact host operation. Join its queue pair by correlation
        # flow, then verify the CANN launch is on that worker OS thread/in range.
        lane=(host['pid'],host['tid']);qstarts=queue_starts.get(lane,[])
        qitems=queue_lanes.get(lane,[])[bisect_left(qstarts,number(host['ts'])):bisect_right(qstarts,end(host))]
        pairs=[q for _,q in qitems if inside(host,q['enqueue']) and
               q['dequeue']['tid']==cann['args'].get('Thread Id',cann['tid']) and
               number(q['dequeue']['ts'])<=number(cann['ts']) and end(cann)<=end(q['dequeue'])]
        require(len(pairs)<=1,'ambiguous queue/device relationship')
        pair=pairs[0] if pairs else None
        if pair:queue_joined+=1
        runtime_apis=[]
        if pair:
            deq=pair['dequeue']
            times=runtime_starts.get(deq['tid'],[])
            candidates=runtime_by_tid[deq['tid']][bisect_left(times,number(deq['ts'])):bisect_right(times,end(deq))]
            runtime_apis=[e for e in candidates if end(e)<=end(deq)]
        item=dict(task=row,device_event=task,host_operator=host,cann_launch=cann,queue=pair,runtime_apis=runtime_apis,
                  phase=phase.get(context['step'],'outside_model') if context else 'outside_model',
                  step=context['step'] if context else None,parameter_scope_labels=labels,
                  cpu_activity_limit='overlapping instrumented intervals, not OS CPU utilization or proof of instruction execution throughout',
                  queue_delay_us=str(number(pair['dequeue']['ts'])-end(pair['enqueue'])) if pair else None,
                  device_start_minus_launch_entry_us=str(number(task['ts'])-number(cann['ts'])))
        # Keep every task's ordering/correlation. Detailed resources and overlaps
        # are expanded once per representative name/phase to keep the report readable.
        full.append({k:v for k,v in item.items() if k not in ('parameter_scopes','cpu_main_during_device','cpu_worker_during_device')})
        variant=next((entries[label]['kind'] for label in labels if entries[label]['kind'] in ('buffer_copy','to_list')),'other') if row['kernel']=='MEMCPY_ASYNC' else ''
        item['example_variant']=variant
        key=(row['kernel'],item['phase'],variant)
        if key not in examples:
            parameters=[dict(label=label,entry=entries[label],exit=exits[label]) for label in labels
                        if entries[label]['kind'] in ('launch','jit_run','torch_api','fia','cache_write','linear','buffer_copy','to_list')]
            main_activity=[dict(event=e,overlap_us=str(overlap(e,task))) for e in leaves
                           if e['pid']==host['pid'] and e['tid']==host['tid'] and overlap(e,task)>0]
            worker_activity=[dict(event=q['dequeue'],overlap_us=str(overlap(q['dequeue'],task))) for q in queues
                             if pair and q['dequeue']['tid']==pair['dequeue']['tid'] and overlap(q['dequeue'],task)>0]
            scope_activity=[dict(label=label,kind=entries[label]['kind'],event=span,overlap_us=str(overlap(span,task)))
                            for label,span in scopes.items() if span['pid']==host['pid'] and span['tid']==host['tid'] and overlap(span,task)>0]
            scope_activity.sort(key=lambda x:number(x['event']['dur']))
            item.update(parameter_scopes=parameters,cpu_main_during_device=main_activity,cpu_worker_during_device=worker_activity,
                        cpu_python_scopes_during_device=scope_activity)
            examples[key]=item
    for launch in launch_records:
        tasks=[x['task']['trace_index'] for x in full if launch['entry']['label'] in x['parameter_scope_labels']]
        require(len(tasks)==1,'Triton launch/device task link is not unique')
        launch['device_trace_indices']=tasks
    streams=defaultdict(list)
    for task in full:streams[str(task['task']['stream_id'])].append(task)
    stream_order=[]
    for stream,tasks in sorted(streams.items()):
        tasks.sort(key=lambda x:number(x['device_event']['ts']))
        overlaps=[(a['task']['task_id'],b['task']['task_id']) for a,b in zip(tasks,tasks[1:]) if end(a['device_event'])>number(b['device_event']['ts'])]
        stream_order.append(dict(stream_id=stream,task_count=len(tasks),task_ids_in_execution_order=[x['task']['task_id'] for x in tasks],
                                 overlapping_neighbors=overlaps,note='observed device order, not a claim about unobserved runtime policy'))
    completions=[]
    for e in entries.values():
        if e['kind']!='to_list':continue
        label=e['label'];span=scopes[label]
        children=[x for x in entries.values() if x.get('parent')==label]
        record=only([x for x in children if x['kind']=='event_record'],'completion event record')
        wait=only([x for x in children if x['kind']=='synchronize'],'completion event wait')
        require(record['object_id']==wait['object_id']==e['event_object_id'],'completion event identity mismatch')
        require(exits[record['label']]['monotonic_ns']<=wait['monotonic_ns']<exits[label]['monotonic_ns'],
                'completion record/wait/return ordering mismatch')
        related=[x for x in full if label in x['parameter_scope_labels']]
        copies=[x for x in related if x['host_operator']['name']=='acl_memcpy_device_to_host']
        require(len(copies)==1,'completion device-to-host copy')
        native_wait=only([x for x in all_complete if x.get('name')=='Event::synchronize' and inside(span,x) and inside(x,scopes[wait['label']])],'native Event::synchronize')
        runtime_wait=only([x for x in all_complete if x.get('name')=='AscendCL@aclrtSynchronizeEvent'
                           and x['args'].get('Thread Id',x['tid'])==wait['tid']
                           and number(scopes[wait['label']]['ts'])<=number(x['ts'])
                           and end(x)<=end(scopes[wait['label']])],'runtime event wait')
        require(end(copies[0]['device_event'])<=end(runtime_wait),'copy completes after wait returned')
        completions.append(dict(phase=phase[e['step']],entry=e,exit=exits[label],record=record,wait=wait,
                                copy_task=copies[0]['task'],copy= copies[0]['device_event'],
                                wait_profiler=scopes[wait['label']],native_wait=native_wait,runtime_wait=runtime_wait,to_list_profiler=span))
    require(len(completions)==len(schedule),'missing step completion')
    summary=dict(run=run.name,prompt_tokens=prompt['prompt_tokens'],output_tokens=4,steps=len(schedule),
                 response=response['choices'][0]['text'],device_tasks=coverage['device_task_count'],
                 correlated_tasks=coverage['correlated_device_tasks'],queue_pairs=coverage['queue_pairs_verified'],
                 device_tasks_with_queue=queue_joined,compiler_calls=len(compile_records),
                 compile_stages=dict(Counter(e['host_stage'] for e in compile_records)),
                 binary_loads=len(native_loads),measured_triton_launches=len(launches),
                 measured_compiles=sum(e['host_stage']=='measured' for e in compile_records),
                 measured_binary_loads=sum(e['host_stage']=='measured' for e in native_loads),
                 measured_scopes=len(scopes),trace_sha256=hashlib.sha256(trace_path.read_bytes()).hexdigest(),
                 omitted_host_build_intermediates=len(omitted),code_dma_timing_observed=False,raw_native_argument_bytes_observed=False,
                 compiler_clock='host monotonic',execution_clock='profiler aligned host/device')
    return dict(summary=summary,compile_records=compile_records,load_records=loads,triton_launches=launch_records,
                observations=list(entries.values()),returns=list(exits.values()),device_tasks=full,
                examples=list(examples.values()),queue_pairs=queues,stream_order=stream_order,completions=completions,host_inventory=hosts,kernel_inventory=kernel_names,
                limits=['CANN/ATB binary build timestamps are not inferred from runtime trace',
                        'load_binary bounds native registration; exact kernel code transfer timing is not exposed',
                        'parameters are visible Python API/launcher arguments, not every native tiling/workspace byte',
                        'same-stream execution and real correlation flows, not CPU call order alone',
                        'host/device interval overlap is observed activity, not OS scheduling utilization',
                        'instrumented single-request eager trace, not a performance benchmark'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);a=p.parse_args()
    data=analyze(a.run);dest=a.run/'analysis';dest.mkdir(exist_ok=True)
    (dest/'submission_evidence.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    (dest/'summary.json').write_text(json.dumps(data['summary'],ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(data['summary'],ensure_ascii=False))

if __name__=='__main__':main()
