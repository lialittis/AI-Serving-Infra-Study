"""Build a per-event / per-replay evidence ledger without inferring missing flows.

CPU-only; accepts resource_schema=1 captures. Older runs remain historical baselines.
Every profiler CPU op and device task is inventoried, including unassociated tasks.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from analyze_reuse import analyze, end, inside, load, number, only, read_trace, require


def tensors(value, path=''):
    if isinstance(value,dict):
        if all(k in value for k in ('data_ptr','storage_ptr','shape','dtype','device')):
            yield path,value
        else:
            for k,v in value.items():
                for item in tensors(v,path+'.'+k if path else k):yield item
    elif isinstance(value,list):
        for i,v in enumerate(value):
            for item in tensors(v,path+'['+str(i)+']'):yield item


def signature(value):
    return {k:value[k] for k in ('device','dtype','shape','stride','data_ptr','storage_ptr','storage_offset','element_size')}


def scalars(value, path=''):
    if isinstance(value,dict):
        if 'data_ptr' in value or value.get('contents')=='not inspected':return
        for k,v in value.items():
            for item in scalars(v,path+'.'+k):yield item
    elif isinstance(value,list):
        for i,v in enumerate(value):
            for item in scalars(v,path+'['+str(i)+']'):yield item
    else:yield path,value


def check_replay(capture, before, after):
    require(before['graph_object_id']==capture['graph_object_id']==after['graph_object_id'], 'replay graph identity mismatch')
    require(before['wrapper_object_id']==capture['wrapper_object_id']==after['wrapper_object_id'], 'replay wrapper identity mismatch')
    require(before['graph_pool']==capture['graph_pool']==after['graph_pool'], 'replay graph pool mismatch')
    require(before['batch_descriptor']==capture['batch_descriptor'], 'replay batch descriptor mismatch')
    require(before['argument_count']==before['placeholder_count'], 'replay arguments missing placeholders')
    require(before['captured_input_addresses']==before['input_addresses']==capture['input_addresses'],
            'replay input addresses differ from capture')
    require(dict(scalars(before['arguments']))==dict(scalars(capture['arguments'])), 'replay scalar argument changed')
    require(before['keyword_arguments']==capture['keyword_arguments'], 'replay keyword arguments changed')
    expected={p:signature(t) for p,t in tensors(capture['arguments'])}
    actual={p:signature(t) for p,t in tensors(before['arguments'])}
    require(actual==expected, 'replay tensor layout/storage mismatch')
    require({p:signature(t) for p,t in tensors(before['persistent_output'])}==
            {p:signature(t) for p,t in tensors(capture['persistent_output'])}, 'replay pre-call output storage mismatch')
    require({p:signature(t) for p,t in tensors(after['persistent_output'])}==
            {p:signature(t) for p,t in tensors(capture['persistent_output'])}, 'replay output storage mismatch')
    # Tensor contents are allowed to differ; scalar arguments require separate visibility.
    return dict(input_addresses='matched_capture',input_layouts='matched_capture',
                output_storage='matched_capture',graph_pool='matched_capture',scalar_arguments='matched_capture',
                tensor_values='not_read',hidden_workspace_lifetime='not_observed',
                complete_resource_safety='not_proven')


def build_inventory(events, scopes):
    """Exact flow endpoints only; containment classifies host context, not device cause."""
    points=defaultdict(list);flow=defaultdict(list)
    for i,e in enumerate(events):
        if e.get('ph')=='X':points[(e['pid'],e['tid'],number(e['ts']))].append(i)
        elif e.get('ph') in ('s','f'):flow[(e.get('cat'),str(e.get('id')),e['ph'])].append(i)
    scope_intervals=defaultdict(list)
    for label,s in scopes.items():
        scope_intervals[(s['pid'],s['tid'])].append((label,number(s['ts']),end(s),number(s['dur'])))
    def context(e):
        begin,finish=number(e['ts']),end(e)
        containing=[x for x in scope_intervals[(e['pid'],e['tid'])] if x[1]<=begin and finish<=x[2]]
        return min(containing,key=lambda x:x[3])[0] if containing else None
    cpu=[];device=[];contexts={}
    for i,e in enumerate(events):
        if e.get('ph')=='X' and e.get('cat')=='cpu_op':
            contexts[i]=context(e)
            cpu.append(dict(id='trace:'+str(i),trace=e,scope=contexts[i],
                            resource_detail='see enclosing observed scope; per-ATen tensor addresses not collected'))
    ends=defaultdict(list)
    for i,e in enumerate(events):
        if e.get('ph')=='f':ends[(e.get('cat'),e['pid'],e['tid'],number(e['ts']))].append(i)
    for i,e in enumerate(events):
        if e.get('ph')!='X' or 'Task Type' not in e.get('args',{}):continue
        associations=[]
        for cat in ('async_npu','HostToDevice'):
            for ei in ends[(cat,e['pid'],e['tid'],number(e['ts']))]:
                f=events[ei]
                for si in flow[(cat,str(f['id']),'s')]:
                    s=events[si]
                    candidates=points[(s['pid'],s['tid'],number(s['ts']))]
                    for hi in candidates:
                        h=events[hi]
                        if cat=='async_npu' and h.get('cat')!='cpu_op':continue
                        if cat=='HostToDevice' and h.get('args',{}).get('connection_id')!=e['args'].get('connection_id'):continue
                        associations.append(dict(category=cat,flow_id=str(f['id']),start='trace:'+str(si),
                            end='trace:'+str(ei),host='trace:'+str(hi),host_event=h,
                            host_scope=contexts.get(hi) if cat=='async_npu' else None))
        device.append(dict(id='trace:'+str(i),trace=e,associations=associations,
                           attribution='exact_flow' if associations else 'unassociated',
                           resource_addresses='not provided by device profiler'))
    return cpu,device


def audit(run, records=None, trace=None):
    data,chains,_=analyze(run,records,trace)
    records=records if records is not None else [json.loads(line) for p in (run/'events').glob('*.jsonl') for line in p.read_text().splitlines()]
    require(any(e['event']=='trace_installed' and e.get('resource_schema')==1 for e in records), 'resource schema missing; recapture required')
    require(all(e.get('mode')==data['mode'] for e in records if e['event']=='trace_installed' and e.get('resource_schema')==1), 'resource mode mismatch')
    events=trace if trace is not None else read_trace(run/data['trace_file'])
    scopes={e['name']:e for e in events if e.get('ph')=='X' and e.get('name','').startswith('P12/')}
    entries={e['label']:e for e in records if e['event']=='scope_enter'}
    exits={e['label']:e for e in records if e['event']=='scope_exit'}
    captures={}
    for e in records:
        if e['event']=='graph_capture_resources':
            r=e['resources'];key=(e['pid'],r['wrapper_object_id'],r['graph_object_id'],e['monotonic_ns'])
            require(key not in captures,'duplicate capture observation')
            captures[key]=e
    cpu,device=build_inventory(events,scopes)
    for event in cpu:
        context=entries.get(event['scope'],{})
        event.update(role=context.get('role'),step=context.get('step'))
    cpu_by_id={e['id']:e for e in cpu}
    for task in device:
        origins=[cpu_by_id[a['host']] for a in task['associations'] if a['category']=='async_npu' and a['host'] in cpu_by_id]
        roles={e['role'] for e in origins if e['role'] is not None}
        steps={e['step'] for e in origins if e['step'] is not None}
        task.update(role=next(iter(roles)) if len(roles)==1 else None,
                    step=next(iter(steps)) if len(steps)==1 else None)
    by_scope=defaultdict(list)
    for task in device:
        for a in task['associations']:
            if a['category']=='async_npu' and a['host_scope']:by_scope[a['host_scope']].append(task['id'])
    ledger=[];replays=[];barriers=[]
    ordered=sorted(entries.values(),key=lambda e:(number(scopes[e['label']]['ts']),e['label']))
    for entry in ordered:
        label=entry['label'];s=scopes[label];out=exits[label]
        req=data['requests'][entry['role']]
        step=next((x for x in req['steps'] if x['step']==entry['step']),None)
        kind=entry['kind'];parent=entry.get('parent_label')
        step_assignment='runner/scheduler trace step'
        if kind in ('allocate','pool_allocate'):
            allocation=entry if kind=='allocate' else entries.get(parent,{})
            computed=allocation.get('computed')
            require(computed in (0,126),'allocator step cannot be resolved from computed tokens')
            step=req['steps'][0 if computed==0 else 1]
            step_assignment='request computed_tokens; allocation occurs before schedule return increments trace step'
        require(parent is None or (parent in scopes and inside(scopes[parent],s)), 'invalid parent scope')
        if kind in ('pool_allocate','pool_free','allocate','manager_free'):
            responsibility='Scheduler / KVCacheManager / BlockPool'
        elif kind in ('acl_dispatch','graph_replay'):
            responsibility='ModelRunner prepares buffers; ACLGraphWrapper dispatches; torch-npu/CANN submits'
        elif kind in ('event_record','event_wait','native_synchronize','to_list'):
            responsibility='native transfer/event implementation; runtime enforces submitted dependencies'
        else:responsibility='ModelRunner / operator implementation'
        resources=dict(tensors(entry))
        record=dict(id=label,kind=kind,role=entry['role'],step=step['step'] if step else entry['step'],
                    host_trace_step=entry['step'],step_assignment=step_assignment,phase=step['phase'] if step else None,
                    host_scope=s,parent=parent,responsibility=responsibility,
                    resources=resources,observed_entry=entry,observed_exit=out,
                    direct_device_tasks=sorted(set(by_scope[label])),
                    logical_owner=dict(request_id=entry['request_id'],physical_block=1,
                                       allocation_interval=req['allocation']['name'],pool_free=req['pool_free']['name']),
                    completion_boundary=step['native_transfer_scope']['name'] if step else None,
                    limitations=['scope return is not device completion','address equality is not allocation generation',
                                 'unobserved tensors / hidden workspace have unknown lifetime'])
        ledger.append(record)
        if kind=='to_list':
            children=[e for e in entries.values() if e.get('parent_label')==label]
            rec=only([e for e in children if e['kind']=='event_record'],'native event record observation')
            sync=only([e for e in children if e['kind']=='native_synchronize'],'native event synchronize observation')
            require(rec['object_id']==sync['object_id']==entry['transfer_event_object_id'], 'native event object identity mismatch')
            require(end(scopes[rec['label']])<=number(scopes[sync['label']]['ts']), 'native event record/wait order mismatch')
            require(rec['current_stream']==sync['current_stream'], 'native event observation stream mismatch')
            require(inside(step['native_wait'],scopes[sync['label']]), 'native synchronize is outside profiler wait wrapper')
            barriers.append(dict(id=label,role=entry['role'],step=entry['step'],event_object_id=rec['object_id'],
                record_occurrence=rec['label'],wait_occurrence=sync['label'],
                cpu_destination=entry['destination'],sampled_source=entry['sampled_tensor'],
                device_tasks=step['transfer_tasks'],native_wait=step['native_wait'],
                note='event object can be reused; this record/wait occurrence identifies the completion boundary'))
        elif kind=='graph_replay':
            wrapper=entries.get(parent,{})
            require(wrapper.get('kind')=='acl_dispatch','replay without ACL parent')
            before=wrapper['resources'];after=exits[parent]['resources_after']
            candidates=[v for (pid,wid,gid,tm),v in captures.items()
                        if pid==entry['pid'] and wid==before['wrapper_object_id']
                        and gid==entry['graph_object_id'] and tm<entry['monotonic_ns']]
            require(candidates,'replay capture baseline missing')
            # Python object IDs can be reused during startup recapture. Select the
            # latest observed capture of this wrapper+graph before this submission.
            capture=max(candidates,key=lambda e:e['monotonic_ns'])
            require(entry['graph_object_id']==before['graph_object_id'],'replay/wrapper graph mismatch')
            checks=check_replay(capture['resources'],before,after)
            require(step is not None and step['phase']=='decode','unexpected replay outside decode')
            require(end(s)<=number(step['native_wait']['ts']),'replay host submission after result wait')
            replays.append(dict(id=label,role=entry['role'],step=entry['step'],partition=wrapper['partition'],
                host_scope=s,graph_object_id=entry['graph_object_id'],capture=capture,
                capture_occurrence='capture:{}:{}:{}'.format(capture['pid'],capture['resources']['wrapper_object_id'],capture['monotonic_ns']),
                resources_before=before,resources_after=after,checks=checks,
                current_stream=entry['current_stream'],completion_boundary=step['native_transfer_scope']['name'],
                direct_device_tasks=record['direct_device_tasks'],
                device_execution_mapping='only exact flow links listed; MODEL_EXECUTE/internal kernels not assigned by time',
                reuse_condition='next update must follow last device use; this run observes serial native result waits',
                hidden_resources='workspace/internal temporaries not enumerated; not a complete lifetime proof'))
    normalized_scopes={e['id']:e for e in ledger}
    for event in cpu:
        if event['scope'] in normalized_scopes:
            event['step']=normalized_scopes[event['scope']]['step']
    # Validate the observed metadata producer -> consumer path. No device values
    # are read: tensor addresses and operator semantics identify the resources.
    task_by_id={t['id']:t for t in device}
    copies=[e for e in ledger if e['kind']=='buffer_copy']
    require(len(copies)==32,'missing buffer copy records')
    for copy in copies:
        count=copy['observed_entry']['copied_rows']
        tasks=[task_by_id[i]['trace'] for i in copy['direct_device_tasks']]
        if count==0:
            require(not tasks,'zero-length buffer copy unexpectedly submitted a device task')
            copy['copy_effect']='zero_length_no_device_task'
        else:
            require(len(tasks)==1 and tasks[0]['name']=='MEMCPY_ASYNC','nonempty buffer copy flow missing')
            copy['copy_effect']='native_nonblocking_H2D_with_exact_flow'
    preparation_edges=[]
    slot_events=[e for e in ledger if e['kind']=='slot_prepare']
    require(len(slot_events)==4,'missing slot preparation records')
    for slot in slot_events:
        entry=slot['observed_entry']
        tasks=[task_by_id[i]['trace'] for i in slot['direct_device_tasks']]
        kernel=only([t for t in tasks if t['name']=='_compute_slot_mapping_kernel'],'slot mapping kernel flow')
        table_copy=only([e for e in copies if e['role']==slot['role'] and e['step']==slot['step'] and
                        e['observed_entry']['tensors']['device_destination']['data_ptr']==entry['block_table']['data_ptr']],
                        'block table H2D producer')
        copy_task=task_by_id[only(table_copy['direct_device_tasks'],'block table copy task')]['trace']
        require(copy_task['args']['Physic Stream Id']==kernel['args']['Physic Stream Id'] and
                end(copy_task)<=number(kernel['ts']),'block table copy/slot kernel ordering missing')
        current_writes=[e for e in ledger if e['role']==slot['role'] and e['step']==slot['step'] and e['kind']=='cache_write']
        require(len(current_writes)==24,'missing KV consumers of slot mapping')
        for write in current_writes:
            require(write['observed_entry']['tensors']['slot_mapping']['data_ptr']==entry['slot_mapping']['data_ptr'],
                    'slot producer/consumer storage mismatch')
            kv=only([c for c in chains if c['python_scope']['name']==write['id']],'KV write chain')['kernel']
            require(kernel['args']['Physic Stream Id']==kv['args']['Physic Stream Id'] and
                    end(kernel)<=number(kv['ts']),'slot mapping/KV write device ordering missing')
        preparation_edges.append(dict(id='preparation:'+slot['id'],role=slot['role'],step=slot['step'],block_table_copy=table_copy['id'],
              slot_prepare=slot['id'],consumers=[w['id'] for w in current_writes],
              block_table_storage=entry['block_table'],slot_storage=entry['slot_mapping'],
              copy_task=copy_task,slot_kernel=kernel,
              evidence='exact flows + same storage arguments + same physical stream + device order',
              values='not read; this verifies mapping-resource preparation order, not every slot value'))
    require(len(barriers)==4,'missing per-step native completion boundaries')
    require(len(replays)==(50 if data['mode']=='graph' else 0),'missing per-replay resource records')
    require(sum(e['kind']=='prepare' for e in ledger)==4,'missing per-step preparation records')
    forwards=[e for e in ledger if e['kind']=='forward']
    require(len(forwards)==4,'missing model forward resource records')
    weights=[{p:signature(t) for p,t in tensors(e['observed_entry']['weights'])} for e in forwards]
    require(weights[0] and all(w==weights[0] for w in weights),'model weight storage changed')
    for record in ledger + replays:
        current=[e for e in ledger if e['role']==record['role'] and e['step']==record['step']]
        record['preparation_events']=[e['id'] for e in current if e['kind'] in ('prepare','buffer_copy','slot_prepare')]
        record['preparation_evidence']='actual host scopes and exact device flows where available; not proof that every input value is ready'
        if record['completion_boundary']:
            record['completion_evidence']=next(b for b in barriers if b['id']==record['completion_boundary'])['record_occurrence']
        record['overwrite_rule']='storage must remain valid through last device access; scope return alone does not permit overwrite'
    # Explicit record-return aliases; not a last-writer claim or a dependency edge.
    returns=[]
    for rec in ledger:
        source=rec['observed_exit']
        for field in ('returned_resources','output_resources'):
            for path,t in tensors(source.get(field)):
                returns.append((end(rec['host_scope']),rec['id'],field+'.'+path,t))
    for replay in replays:
        aliases=[]
        for path,t in tensors(replay['resources_before']['arguments']):
            candidates=[(tm,label,p,old) for tm,label,p,old in returns
                        if tm<=number(replay['host_scope']['ts']) and old['device']==t['device'] and old['data_ptr']==t['data_ptr']]
            if candidates:
                tm,label,p,old=max(candidates,key=lambda x:x[0])
                aliases.append(dict(input=path,earlier_event=label,earlier_path=p,address=t['data_ptr'],
                                    interpretation='latest observed return at same address; not proof of last writer or readiness'))
        replay['earlier_return_aliases']=aliases
    summary=dict(mode=data['mode'],observed_scopes=len(ledger),replay_records=len(replays),
                 capture_baselines=len(captures),native_completion_boundaries=len(barriers),
                 buffer_copy_records=len(copies),zero_length_copies=sum(e['copy_effect']=='zero_length_no_device_task' for e in copies),
                 verified_slot_preparation_chains=len(preparation_edges),
                 cpu_events=len(cpu),device_tasks=len(device),unassociated_device_tasks=sum(not t['associations'] for t in device),
                 kv_operator_chains=len(chains),weight_tensors=len(weights[0]),weight_storage_stable=True,
                 per_aten_tensor_addresses=False,hidden_workspace_lifetime_proven=False,
                 additional_device_waits=False,device_values_read=False)
    return dict(schema=1,run=run.name,trace_file=data['trace_file'],trace_sha256=data['trace_sha256'],
                event_id_namespace='trace:N is zero-based index in the original traceEvents array',summary=summary,events=ledger,replays=replays,
                completion_boundaries=barriers,preparation_chains=preparation_edges,cpu_events=cpu,device_tasks=device,
                limits=['every cpu_op and device task in the profiler window is inventoried, not every hardware instruction',
                        'tensor resources observed at Python boundaries, not every ATen input/output',
                        'capture/current storage metadata equality does not prove value readiness or allocator lifetime',
                        'kernel semantics and actual flows support KV access links; no per-address device memory tracing',
                        'PIECEWISE graph only; attention/KV remain outside capture; serial A/B only'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);a=p.parse_args()
    data=audit(a.run);dest=a.run/'analysis';dest.mkdir(exist_ok=True)
    (dest/'resource_ledger.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    (dest/'resource_summary.json').write_text(json.dumps(data['summary'],ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(data['summary'],ensure_ascii=False))

if __name__=='__main__':main()
