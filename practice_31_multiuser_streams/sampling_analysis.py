"""P31b: per-step random/model submission, synchronization and exact overlap."""
from collections import Counter, defaultdict
from decimal import Decimal
import statistics


def num(x):return Decimal(str(x))
def end(e):return num(e['ts'])+num(e.get('dur',0))
def stats(xs):
    values=[float(x) for x in xs]
    return dict(n=len(values),min=min(values),median=statistics.median(values),max=max(values),sum=sum(values)) if values else None


def analyze_sampling(result, records, events, enabled):
    from analyze import intersect_length
    tasks_by_step=defaultdict(list)
    for t in result['tasks']:tasks_by_step[t['step']].append(t)
    scopes_by_step=defaultdict(dict)
    for s in result['scopes']:scopes_by_step[s['step']][s['stage']]=s
    q_by_step=defaultdict(list);consume_by_step=defaultdict(list)
    for r in records:
        if r['kind']=='q_ready':q_by_step[r['key']].append(r)
        if r['kind']=='q_consume':consume_by_step[r['key']].append(r)
    native_sync=[(i,e) for i,e in enumerate(events) if e.get('ph')=='X' and e.get('name')=='AscendCL@aclrtSynchronizeEvent']
    results=[];issues=[]
    expected='async_exponential' if enabled else 'inline_exponential'
    for step in result['steps']:
        key=step['key'];ts=tasks_by_step[key];scopes=scopes_by_step[key]
        try:
            qscope=scopes[expected];forward_scope=scopes['forward']
            forward=[t for t in ts if t['stage']=='forward' and t['is_compute']]
            model_streams={t['stream'] for t in forward}
            if len(model_streams)!=1:raise ValueError('Ambiguous model compute stream')
            model_stream=next(iter(model_streams))
            q=[t for t in ts if t['stage']==expected and t['is_compute'] and t['stream']!=model_stream]
            if not q:raise ValueError('No independently observed random compute')
            random_streams={t['stream'] for t in q}
            if len(random_streams)!=1:raise ValueError('Ambiguous random stream')
            random_stream=next(iter(random_streams))
            consumer_stages=(('sampling_math',) if 'sampling_math' in scopes else ('sampler','sampling_math')) if enabled else ('inline_exponential',)
            consumers=[t for t in ts if t['is_compute'] and t['stage'] in consumer_stages and t['stream']==model_stream
                       and t['host_index'] is not None
                       and result['hosts'][str(t['host_index'])]['name'].startswith(('aten::div','aclnnInplaceDiv'))]
            if not consumers:raise ValueError('No traced probs/q consumer')
            produced=q_by_step[key]
            if len(produced)!=1:raise ValueError('Expected one q metadata record')
            produced=produced[0]
            if produced['generator_count']!=0:raise ValueError('Per-request generator changes experimental path')
            metadata=produced['q']
            if metadata['shape'][0]!=len(step['requests']) or metadata['dtype']!='torch.float32':
                raise ValueError('q shape/dtype does not match scheduled batch')
            consumed=consume_by_step[key]
            if enabled:
                if len(consumed)!=1 or consumed[0]['q']!=metadata or consumed[0]['event_handle']!=produced['event_handle']:
                    raise ValueError('Precomputed q/event identity mismatch')
            elif consumed:raise ValueError('Off path unexpectedly consumes precomputed q')
            q_intervals=[(num(t['start_us']),num(t['end_us'])) for t in q]
            f_intervals=[(num(t['start_us']),num(t['end_us'])) for t in forward]
            intersection=intersect_length(q_intervals,f_intervals)
            dsa_overlap=intersect_length([(num(t['start_us']),num(t['end_us'])) for t in q if t.get('core_type')=='DSA_SQE'],f_intervals)
            ai_overlap=intersect_length([(num(t['start_us']),num(t['end_us'])) for t in q if t.get('core_type')!='DSA_SQE'],f_intervals)
            qfirst=min(num(t['start_us']) for t in q);qlast=max(num(t['end_us']) for t in q)
            ffirst=min(num(t['start_us']) for t in forward);flast=max(num(t['end_us']) for t in forward)
            first_consumer=min(consumers,key=lambda t:num(t['start_us']))
            event_records=[t for t in ts if t['stage']==expected and t['name']=='EVENT_RECORD' and t['stream']==random_stream]
            if not event_records:raise ValueError('No random branch event record')
            # Current pinned implementation records q-ready last in the branch.
            ready=max(event_records,key=lambda t:num(t['start_us']))
            if qlast>num(ready['start_us']):raise ValueError('Random kernel extends beyond q-ready event')
            waits=[t for t in ts if t['name']=='EVENT_WAIT' and t['stage']==expected and t['stream']==model_stream]
            if enabled:
                wait_scope=scopes.get('q_wait')
                if wait_scope is None:
                    # Early qualification captured the Python API interval and
                    # exact event identity, before adding a dedicated scope.
                    apis=[r for r in records if r['kind']=='stream_api_return' and r['key']==key
                          and r['operation']=='Event.synchronize' and r['event_handle']==produced['event_handle']]
                    if len(apis)!=1:raise ValueError('Missing exact q Event.synchronize identity')
                    api=apis[0]
                    wait_scope=dict(tid=api['tid'],start_us=str(num(api['start_ns'])/1000),end_us=str(num(api['end_ns'])/1000))
                syncs=[(i,e) for i,e in native_sync if e.get('args',{}).get('Thread Id',e.get('tid'))==wait_scope['tid']
                       and num(wait_scope['start_us'])<=num(e['ts']) and end(e)<=num(wait_scope['end_us'])]
                if len(syncs)!=1:raise ValueError('Missing/ambiguous CANN host event synchronization')
                i,e=syncs[0]
                if num(ready['end_us'])>end(e):raise ValueError('Host wait returns before ready event completes')
                if waits:raise ValueError('Unexpected consumer EVENT_WAIT on enabled path')
                # Use one profiler clock domain. Python wall_ns is not a safe
                # boundary for sub-millisecond Torch/CANN events after clock
                # alignment. The pinned code has one probs/q div after this wait.
                consumers=[t for t in consumers if num(result['hosts'][str(t['host_index'])]['start_us'])>=end(e)]
                if len({t['host_index'] for t in consumers})!=1:raise ValueError('Missing/ambiguous post-wait probs/q division')
                first_consumer=min(consumers,key=lambda t:num(t['start_us']))
                ch=result['hosts'][str(first_consumer['host_index'])]
                if end(e)>num(ch['start_us']):raise ValueError('Consumer submitted before host q wait returns')
                sync=dict(kind='host_event_synchronize',native_trace_index=i,name=e['name'],start_us=str(e['ts']),
                          end_us=str(end(e)),duration_us=str(e['dur']),ready_event=ready['id'],consumer=first_consumer['id'])
                if num(qscope['end_us'])>num(forward_scope['start_us']):raise ValueError('q producer not before host forward')
            else:
                if len(waits)!=1:raise ValueError('Missing/ambiguous device stream wait')
                wait=waits[0]
                if num(ready['end_us'])>num(wait['end_us']) or num(wait['end_us'])>num(first_consumer['start_us']):
                    raise ValueError('Device record/wait/consumer order violated')
                if 'q_wait' in scopes:raise ValueError('Unexpected precomputed-q host wait on off path')
                sync=dict(kind='device_event_wait',task=wait['id'],start_us=wait['start_us'],end_us=wait['end_us'],
                          duration_us=wait['duration_us'],ready_event=ready['id'],consumer=first_consumer['id'],
                          note='device interval; not CPU blocked time')
                if num(forward_scope['end_us'])>num(qscope['start_us']):raise ValueError('Inline q producer not after host forward')
            def first_native(group):
                values=[result['hosts'][str(t['cann_index'])] for t in group if t['cann_index'] is not None]
                if len(values)!=len(group):raise ValueError('Missing native submission correlation')
                return min(num(h['start_us']) for h in values)
            qsubmit=first_native(q);fsubmit=first_native(forward)
            phases=sorted({r['phase'] for r in step['scheduler']['requests'].values()})
            results.append(dict(key=key,batch=len(step['requests']),phases=phases,model_stream=model_stream,random_stream=random_stream,
                                q_metadata=metadata,q_event_handle=produced['event_handle'],q_identity_checked=enabled,
                                q_scope=qscope,forward_scope=forward_scope,synchronization=sync,
                                random_compute_tasks=[t['id'] for t in q],forward_compute_tasks=len(forward),
                                q_start_us=str(qfirst),q_end_us=str(qlast),forward_start_us=str(ffirst),forward_end_us=str(flast),
                                q_native_submit_start_us=str(qsubmit),forward_native_submit_start_us=str(fsubmit),
                                native_submission_delta_us=str(qsubmit-fsubmit),
                                q_end_minus_forward_start_us=str(qlast-ffirst),
                                same_step_random_forward_overlap_us=str(intersection),dsa_forward_overlap_us=str(dsa_overlap),
                                other_random_forward_overlap_us=str(ai_overlap),consumer=first_consumer['id']))
        except (KeyError,ValueError,TypeError) as exc:
            issues.append(dict(step=key,error=str(exc)))
    groups=[]
    for batch,phase in sorted({(s['batch'],'+'.join(s['phases'])) for s in results}):
        subset=[s for s in results if s['batch']==batch and '+'.join(s['phases'])==phase]
        groups.append(dict(batch=batch,phase=phase,steps=len(subset),
                           overlapping_steps=sum(num(s['same_step_random_forward_overlap_us'])>0 for s in subset),
                           random_forward_overlap_us=str(sum((num(s['same_step_random_forward_overlap_us']) for s in subset),Decimal(0))),
                           dsa_forward_overlap_us=str(sum((num(s['dsa_forward_overlap_us']) for s in subset),Decimal(0))),
                           other_random_forward_overlap_us=str(sum((num(s['other_random_forward_overlap_us']) for s in subset),Decimal(0))),
                           native_submission_delta_us=stats(s['native_submission_delta_us'] for s in subset),
                           q_end_minus_forward_start_us=stats(s['q_end_minus_forward_start_us'] for s in subset),
                           synchronization_duration_us=stats(s['synchronization']['duration_us'] for s in subset)))
    summary=dict(precompute=enabled,expected_steps=len(result['steps']),validated_steps=len(results),issues=issues,
                 synchronization_kind='host_event_synchronize' if enabled else 'device_event_wait',
                 random_forward_overlap_us=str(sum((num(s['same_step_random_forward_overlap_us']) for s in results),Decimal(0))),
                 overlapping_steps=sum(num(s['same_step_random_forward_overlap_us'])>0 for s in results),
                 batch_phase_groups=groups,output_equality_required=False,
                 note='q_ready metadata is captured at Python producer return, not evidence of device completion; device event and native wait establish completion.')
    by_id={t['id']:t for t in result['tasks']}
    patterns=defaultdict(lambda:dict(intervals=0,duration=Decimal(0),example=None))
    for interval in result.get('overlap_evidence',[]):
        overlapping=[by_id[i] for i in interval['tasks']]
        signature=tuple(sorted((t['stage'],t['name'],t.get('core_type')) for t in overlapping))
        group=patterns[signature];group['intervals']+=1
        group['duration']+=num(interval['end_us'])-num(interval['start_us'])
        if group['example'] is None:
            group['example']=dict(interval=interval,tasks=overlapping)
    summary['overlap_patterns']=[dict(tasks=[dict(stage=s,name=n,core_type=c) for s,n,c in signature],
                                       intervals=p['intervals'],overlap_us=str(p['duration']),example=p['example'])
                                 for signature,p in sorted(patterns.items(),key=lambda item:-item[1]['duration'])]
    # Retain one full-batch decode example and one greatest-overlap example, or
    # the least separated example if every step has zero overlap.
    candidates=[s for s in results if s['phases']==['decode']] or results
    examples=[]
    if candidates:examples.append(max(candidates,key=lambda s:(s['batch'],num(s['same_step_random_forward_overlap_us']))))
    if results:
        best=max(results,key=lambda s:num(s['same_step_random_forward_overlap_us']))
        if best not in examples:examples.append(best)
    # Small reviewable evidence survives without publishing the full task table.
    examples=[dict(s) for s in examples]
    for example in examples:
        ids=example['random_compute_tasks']+[example['consumer'],example['synchronization']['ready_event']]
        if 'task' in example['synchronization']:ids.append(example['synchronization']['task'])
        forward=[t for t in tasks_by_step[example['key']] if t['stage']=='forward' and t['is_compute']]
        if forward:ids.append(min(forward,key=lambda t:num(t['start_us']))['id'])
        evidence=[]
        for ident in ids:
            t=by_id[ident]
            evidence.append(dict(task=t,host_operator=result['hosts'].get(str(t['host_index'])),
                                 native_submission=result['hosts'].get(str(t['cann_index']))))
        example['task_evidence']=evidence
        sync_index=example['synchronization'].get('native_trace_index')
        if sync_index is not None:example['native_synchronization']=events[sync_index]
    return dict(summary=summary,steps=results,examples=examples)
