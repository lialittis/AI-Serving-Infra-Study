"""Join native profiler flows with CPU block and transfer protocol observations."""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'practice_17_vllm_multistream'))
from analyze_run import number, end, only, require, resolve_colliding_host, intersection, union
sys.path.pop(0)
from hb import Graph


def load_records(root):
    return sorted((json.loads(line) for p in root.glob('events-*.jsonl')
                   for line in p.read_text().splitlines()), key=lambda r: r['monotonic_ns'])


def analyze(root):
    command = json.loads((root/'command.json').read_text())
    require(command['phase']=='diagnostic', 'diagnostic required')
    require(json.loads((root/'shutdown.json').read_text())['exit_code']==0, 'unclean shutdown')
    for key in ('profile_start', 'profile_stop'):
        require(json.loads((root/(key+'.json')).read_text())['status']==200, 'profiler API failure')
    manifest=json.loads((root/'source_manifest.json').read_text())
    for path, metadata in manifest.items():
        require(hashlib.sha256((root/path).read_bytes()).hexdigest()==metadata['sha256'], 'source changed')
    contracts=json.loads(Path(__file__).with_name('contracts.json').read_text())
    for path,expected in contracts['sources'].items():
        require(manifest[path]['sha256']==expected, 'unaudited source contract: '+path)
    records = load_records(root)
    require(not any(r['kind']=='unsupported_preemption' for r in records), 'preemption outside scope')
    trace_path = only(root.rglob('trace_view.json'), 'trace')
    print('reading trace', trace_path, flush=True)
    events = json.loads(trace_path.read_text(), parse_float=Decimal)
    if isinstance(events, dict):
        events = events['traceEvents']
    complete, starts, finishes = defaultdict(list), defaultdict(list), defaultdict(list)
    def point(e):
        return e['pid'], e['tid'], number(e['ts'])
    scopes, scopes_by_thread = {}, defaultdict(list)
    for i, e in enumerate(events):
        ph = e.get('ph')
        if ph=='X':
            complete[point(e)].append((i,e))
            if e['name'].startswith('P20/'):
                scopes[e['name']]=e
                scopes_by_thread[e['pid'],e['tid']].append(e)
        elif ph=='s':
            starts[e.get('cat'),str(e['id'])].append(e)
        elif ph=='f':
            finishes[(e.get('cat'),)+point(e)].append(e)
    entries = {r['label']:r for r in records if r['kind']=='scope_begin'}
    exits = {r['label']:r for r in records if r['kind']=='scope_end'}
    require(set(entries)==set(exits), 'unbalanced observer scopes')
    require(set(scopes)<=set(entries), 'scope without observation')
    # CPU RF spans do not propagate to the pre-existing copy thread. Its native
    # loop is sequential: one batched CANN call and one event record per job.
    # Pair by this checked order/count contract, NOT by aligning wall clocks.
    copy_records=[r for r in records if r['kind']=='scope_begin' and r['label'].endswith('/copy')]
    published={(r['direction'],r['event_index']):r for r in records if r['kind']=='dma_published'}
    background_tids={r['tid'] for r in copy_records}
    batch_scope={}
    for tid in background_tids:
        jobs=[r for r in copy_records if r['tid']==tid]
        calls=sorted(((i,e) for i,e in enumerate(events) if e.get('ph')=='X' and
                      e.get('tid')==tid and e.get('name')=='AscendCL@aclrtMemcpyBatchAsync'),
                     key=lambda pair:number(pair[1]['ts']))
        require(len(calls)==len(jobs),'native FIFO batched-copy count mismatch')
        for (i,e),r in zip(calls,jobs):
            batch_scope[i]=r['label']
    pid_by_tid=defaultdict(set)
    for r in records:
        pid_by_tid[r['tid']].add(r['pid'])
    dequeue_by_tid=defaultdict(list)
    for i,e in enumerate(events):
        if e.get('ph')=='X' and e.get('cat')=='dequeue':
            dequeue_by_tid[e['tid']].append((i,e))
    dequeue_times={}
    for tid,items in dequeue_by_tid.items():
        items.sort(key=lambda pair:number(pair[1]['ts']))
        dequeue_times[tid]=[number(e['ts']) for _,e in items]
    queues = None
    collision_count = 0
    def source(task, category, cann=None):
        nonlocal queues, collision_count
        ends = finishes[(category,)+point(task)]
        if not ends:
            return None
        finish = only(ends, category+' endpoint')
        candidates = starts[category,str(finish['id'])]
        if len(candidates)>1 and category=='async_npu' and cann is not None:
            if queues is None:
                queues=[]
                for e in events:
                    if e.get('ph')=='X' and e.get('cat')=='dequeue':
                        for f in finishes[('async_task_queue',)+point(e)]:
                            enqueue=only(starts['async_task_queue',str(f['id'])], 'queue start')
                            queues.append((e,enqueue,str(f['id'])))
            hosts=[only(complete[point(s)], 'host') for s in candidates]
            i,e,_=resolve_colliding_host(hosts,cann,queues)
            collision_count += 1
            return i,e
        return only(complete[point(only(candidates,category+' start'))],category+' source')

    graph=Graph()
    host_nodes={}
    thread_last={}
    for i,r in enumerate(records):
        identifier=graph.node('h:%d'%i, kind='host', operation=r['kind'], record_index=i,
                              thread='%s:%s'%(r['pid'],r['tid']), monotonic_ns=r['monotonic_ns'])
        host_nodes[id(r)]=identifier
        key=(r['pid'],r['tid'])
        if key in thread_last:
            graph.edge(thread_last[key],identifier,'host_program_order')
        thread_last[key]=identifier
    host_scope={label:host_nodes[id(r)] for label,r in entries.items()}
    tasks, unresolved = [], []
    corrected_thread_attributions=0
    host_sources={}
    for index,e in enumerate(events):
        args=e.get('args',{})
        if e.get('ph')!='X' or 'Task Type' not in args or e['name'] in ('PROFILING_ENABLE','PROFILING_DISABLE'):
            continue
        cann=source(e,'HostToDevice')
        host=source(e,'async_npu',cann[1] if cann else None)
        anchor_kind='torch_cpu_flow'
        if cann:
            c=cann[1]
            ds=dequeue_by_tid[c['tid']]
            position=bisect_right(dequeue_times.get(c['tid'],[]),number(c['ts']))-1
            if position>=0 and end(c)<=end(ds[position][1]):
                d=ds[position][1]
                enqueue=only(starts['async_task_queue',str(d['args']['correlation_id'])], 'real enqueue')
                true_host=only(complete[point(enqueue)], 'enqueue range')
                if host and (host[1]['pid'],host[1]['tid'])!=(true_host[1]['pid'],true_host[1]['tid']):
                    corrected_thread_attributions+=1
                host=true_host
                anchor_kind='CANN connection -> dequeue -> enqueue correlation'
            elif c['tid'] in pid_by_tid:
                host=(cann[0],dict(c,pid=only(pid_by_tid[c['tid']],'CANN caller process')))
                anchor_kind='direct CANN call inside same-thread observer bounds'
        # Some runtime calls only have a CANN flow. Preserve that evidence.
        if host is None and cann is not None:
            host=cann
        if cann:
            require(str(cann[1]['args']['connection_id'])==str(args['connection_id']), 'connection mismatch')
        item=dict(id='k:%d'%index,kind='device',name=e['name'],task_type=args['Task Type'],
                  stream=str(args['Physic Stream Id']),task_id=str(args['Task Id']),
                  start_us=str(e['ts']),end_us=str(end(e)),duration_us=str(e['dur']),
                  trace_index=index,host_trace_index=host[0] if host else None,
                  cann_trace_index=cann[0] if cann else None,
                  host_anchor_kind=anchor_kind,
                  connection_id=str(args['connection_id']))
        if host:
            host_sources[host[0]]=host[1]
        else:
            unresolved.append(item['id'])
        tasks.append(item)
    # Sweep nested scopes on each host thread; avoid tasks × scopes quadratic work.
    host_labels={}
    hosts_by_thread=defaultdict(list)
    for i,e in host_sources.items():
        hosts_by_thread[e['pid'],e['tid']].append((i,e))
    for key,hosts in hosts_by_thread.items():
        bounds=sorted(scopes_by_thread[key],key=lambda e:number(e['ts']))
        active=[]; cursor=0
        for i,e in sorted(hosts,key=lambda p:number(p[1]['ts'])):
            while cursor<len(bounds) and number(bounds[cursor]['ts'])<=number(e['ts']):
                active.append(bounds[cursor]); cursor+=1
            active=[s for s in active if end(s)>=number(e['ts'])]
            containing=[s for s in active if end(s)>=end(e)]
            host_labels[i]=min(containing,key=lambda s:number(s['dur']))['name'] if containing else None
    for tid in background_tids:
        api=[r for r in records if r['kind']=='event_api' and r['api']=='record' and r['tid']==tid]
        device_records=sorted((t for t in tasks if t['name']=='EVENT_RECORD' and
                               t['host_trace_index'] in host_sources and
                               host_sources[t['host_trace_index']]['tid']==tid),
                              key=lambda t:number(host_sources[t['host_trace_index']]['ts']))
        require(len(api)==len(device_records)==len([r for r in copy_records if r['tid']==tid]),
                'native FIFO event publication count mismatch')
        jobs=[r for r in copy_records if r['tid']==tid]
        for t,r,job in zip(device_records,api,jobs):
            require(published[job['direction'],job['event_index']]['event_handle']==r['event_handle'],
                    'FIFO publication handle mismatch')
            host_labels[t['host_trace_index']]=r['label']
    by_scope=defaultdict(list); by_stream=defaultdict(list)
    with only(root.rglob('kernel_details.csv'),'kernel CSV').open() as f:
        csv_rows=list(csv.DictReader(f))
    csv_index=defaultdict(list)
    for i,row in enumerate(csv_rows):
        csv_index[row['Name'],row['Stream ID'].strip(),row['Task ID'].strip(),number(row['Start Time(us)'])].append(i)
    matched=set()
    for t in tasks:
        label=batch_scope.get(t['cann_trace_index']) or host_labels.get(t['host_trace_index'])
        r=entries.get(label,{})
        t.update(scope=label,scope_kind=label.rsplit('/',1)[-1] if label else None,
                 step=r.get('step'),scheduled=r.get('scheduled',{}))
        matches=csv_index[t['name'],t['stream'],t['task_id'],number(t['start_us'])]
        if matches:
            idx=only(matches,'CSV identity')
            require(idx not in matched,'duplicate CSV')
            require(abs(number(csv_rows[idx]['Duration(us)'])-number(t['duration_us']))<=Decimal('.001'),'CSV duration')
            t['kernel_csv_row']=idx; matched.add(idx)
        t['is_compute']='kernel_csv_row' in t
        graph.node(t['id'], **{k:v for k,v in t.items() if k!='id'})
        if label:
            graph.edge(host_scope[label],t['id'],'host_submission',scope=label)
            by_scope[label].append(t)
        by_stream[t['stream']].append(t)
    require(len(matched)==len(csv_rows),'unmatched kernel CSV rows')
    for stream,items in by_stream.items():
        ordered=sorted(items,key=lambda t:number(t['start_us']))
        for a,b in zip(ordered,ordered[1:]):
            graph.edge(a['id'],b['id'],'stream_order',stream=stream)

    # Successful event queries and synchronizes are completion evidence.
    last_record={}; external_events=0
    for r in records:
        if r['kind']!='event_api':
            continue
        key=(r['pid'],r['event_handle'])
        if r['api']=='record':
            candidates=[t for t in by_scope[r['label']] if t['name']=='EVENT_RECORD']
            last_record[key]=only(candidates,'event record task')['id'] if candidates else None
        elif r['api']=='synchronize' or (r['api']=='query' and r['result']):
            record=last_record.get(key)
            if record:
                graph.edge(record,host_nodes[id(r)],'host_sync' if r['api']=='synchronize' else 'event_query_complete',
                           event_handle=r['event_handle'])
            else:
                external_events+=1
        elif r['api']=='wait':
            record=last_record.get(key)
            waits=[t for t in by_scope[r['label']] if t['name']=='EVENT_WAIT']
            if record:
                for wait in waits:
                    graph.edge(record,wait['id'],'event_wait',event_handle=r['event_handle'])

    submits={(r['direction'],r['event_index']):r for r in records if r['kind']=='dma_submit'}
    layouts=[r for r in records if r['kind']=='cache_layout']
    layout=only(layouts,'cache layout') if command['mode']!='recompute' else None
    transfers=[]
    for label,r in entries.items():
        if not label.endswith('/copy'):
            continue
        key=(r['direction'],r['event_index'])
        submit=submits[key]
        graph.edge(host_nodes[id(submit)],host_scope[label],'host_queue',event_index=r['event_index'])
        copies=[t for t in by_scope[label] if 'MEMCPY' in t['name'].upper() or 'MEMCPY' in t['task_type'].upper()]
        copies.sort(key=lambda t:number(t['start_us']))
        require(bool(copies),'DMA batch without device memcpy')
        require(len(copies)==len(r['src_blocks'])*len(layout['tensors']), 'batch/device memcpy coverage')
        transfers.append(dict(label=label,direction=r['direction'],event_index=r['event_index'],
                              src_blocks=r['src_blocks'],dst_blocks=r['dst_blocks'],num_bytes=r['num_bytes'],
                              device_nodes=[t['id'] for t in copies],
                              host_node=host_scope[label],record=r))

    # Block content generations follow fresh allocation, never a prefix-hit touch.
    generations=defaultdict(int); histories=defaultdict(list)
    for r in records:
        if r['kind']=='pool' and r['operation']=='get_new_blocks':
            for b in r['after']:
                key=(r['medium'],b['id']); generations[key]+=1
                histories[key].append((r['monotonic_ns'],generations[key],host_nodes[id(r)]))
    def generation(medium,block_id,at):
        values=[g for time,g,_ in histories[medium,block_id] if time<=at]
        return values[-1] if values else 0

    # Requirements are block-level, with native multi-kernel KV calls kept opaque.
    # Use the entire KV producer/consumer call interval across all 24 layers.
    accesses=[]
    for label,r in entries.items():
        if not label.endswith('/_model_forward') or label not in scopes:
            continue
        current=by_scope[label]
        writes=[t for t in current if t['name']=='ReshapeAndCacheNdKernel']
        reads=[t for t in current if t['name']=='FusedInferAttentionScore']
        if not writes or not reads:
            continue
        require(len(writes)==len(reads)==24, 'expected all 24 Qwen attention layers')
        tables=r.get('block_tables',[])
        require(len(tables)==1,'single KV group required')
        scheduled=r['scheduled']
        for rid,blocks in zip(r['request_ids'],tables[0]):
            if rid not in scheduled:
                continue
            # Last block is the active write block for decode. Prefill writes all
            # blocks conservatively; FIA read ranges include all mapped blocks.
            write_blocks=blocks if scheduled[rid]>1 else blocks[-1:]
            for b in blocks:
                gen=generation('NPU',b,r['monotonic_ns'])
                for mode,selected in [('R',reads),('W',writes if b in write_blocks else [])]:
                    if selected:
                        ordered=sorted(selected,key=lambda t:number(t['start_us']))
                        accesses.append(dict(medium='NPU',block=b,generation=gen,mode=mode,
                            first=ordered[0]['id'],last=ordered[-1]['id'],at=r['monotonic_ns'],
                            request=rid,source='attention call contract; conservative mapped-block range'))
    for t in transfers:
        for medium,blocks,mode in [('NPU' if t['direction']=='D2H' else 'CPU',t['src_blocks'],'R'),
                                   ('CPU' if t['direction']=='D2H' else 'NPU',t['dst_blocks'],'W')]:
            first,last=t['device_nodes'][0],t['device_nodes'][-1]
            for b in blocks:
                accesses.append(dict(medium=medium,block=b,generation=generation(medium,b,t['record']['monotonic_ns']),
                                     mode=mode,first=first,last=last,at=t['record']['monotonic_ns'],
                                     direction=t['direction'],event_index=t['event_index'],source='DMA pointer batch'))
    # Generate only cross-DMA hazards: read-read needs no order; intra-forward
    # native workspace/data dependencies are explicitly outside this KV study.
    history=defaultdict(list); required=[]
    for access in sorted(accesses,key=lambda x:x['at']):
        key=access['medium'],access['block']
        for old in history[key]:
            if old['mode']=='R' and access['mode']=='R':
                continue
            if 'direction' not in old and 'direction' not in access:
                continue
            if old['last']==access['first']:
                continue
            required.append(dict(source=old['last'],target=access['first'],
                kind='storage_reuse' if old['generation']!=access['generation'] else 'data_dependency',
                hazard=old['mode']+access['mode'],medium=key[0],block=key[1],
                from_generation=old['generation'],to_generation=access['generation']))
        # Keep accesses until the next write; reads remain outstanding together.
        if access['mode']=='W':
            history[key]=[access]
        else:
            history[key].append(access)
    # Publication/release is a separate obligation, not just a future DMA read.
    for transfer in transfers:
        index=str(transfer['event_index'])
        if transfer['direction']=='D2H':
            completed=[r for r in records if r['kind']=='scheduler_completed_begin' and index in r['stored']]
            require(len(completed)==1, 'missing/duplicate scheduler store completion')
        else:
            metadata_records=[r for r in records if r['kind']=='worker_metadata' and
                              r['metadata']['load_event']==transfer['event_index']]
            require(bool(metadata_records),'load without worker metadata')
            reqs=set(metadata_records[0]['metadata']['load_event_to_reqs'][index])
            completed=[r for r in records if r['kind']=='scheduler_completed_begin' and
                       reqs.intersection(r['received'])]
            require(len(completed)==1,'missing/duplicate scheduler load completion')
        begin=completed[0]
        finish=next(r for r in records if r['kind']=='scheduler_completed_end' and
                    (r['pid'],r['tid'])==(begin['pid'],begin['tid']) and
                    r['monotonic_ns']>begin['monotonic_ns'])
        required.append(dict(source=transfer['device_nodes'][-1],target=host_nodes[id(finish)],
            kind='cache_publish' if transfer['direction']=='D2H' else 'load_release',
            direction=transfer['direction'],event_index=transfer['event_index']))
    print('verify',len(tasks),'tasks',len(required),'KV requirements',flush=True)
    checked=graph.verify(required)
    violations=[r for r in checked if not r['satisfied']]
    overlap=[]
    computations=[t for t in tasks if t['is_compute']]
    for t in transfers:
        dma=[graph.nodes[i] for i in t['device_nodes']]
        bounds=[(number(d['start_us']),number(d['end_us'])) for d in dma]
        lanes={d['stream'] for d in dma}
        low=min(a for a,b in bounds); high=max(b for a,b in bounds)
        other=[c for c in computations if c['stream'] not in lanes
               and number(c['start_us'])<high and number(c['end_us'])>low]
        total=sum((b-a for a,b in intersection(bounds,[(number(c['start_us']),number(c['end_us'])) for c in other])),Decimal(0))
        b_tasks=[c for c in other if any('-B-' in rid for rid in c['scheduled'])]
        b_overlap=sum((b-a for a,b in intersection(bounds,[(number(c['start_us']),number(c['end_us'])) for c in b_tasks])),Decimal(0))
        overlap.append(dict(direction=t['direction'],event_index=t['event_index'],num_bytes=t['num_bytes'],
                            overlap_us=str(total),b_compute_overlap_us=str(b_overlap)))
    summary=dict(mode=command['mode'],device_tasks=len(tasks),compute_tasks=len(csv_rows),
                 streams=dict(Counter(t['stream'] for t in tasks)),transfers=len(transfers),
                 flow_collisions_resolved=collision_count,unassociated_device_tasks=len(unresolved),
                 unassociated_task_names=dict(Counter(graph.nodes[i]['name'] for i in unresolved)),
                 corrected_cross_thread_attributions=corrected_thread_attributions,
                 external_event_completions=external_events,acyclic=True,
                 required_edges=len(checked),unsatisfied_requirements=len(violations),
                 requirements_by_kind=dict(Counter(e['kind'] for e in checked)),
                 overlap=overlap,complete_exact_model_data_dag=False,
                 kv_range_contract='Conservative block-level call boundaries; no native workspace inspection')
    for t in transfers:
        t.pop('record')
    output=root/'analysis'; output.mkdir(exist_ok=True)
    data=dict(summary=summary,nodes=list(graph.nodes.values()),edges=graph.edges,requirements=checked,
              transfers=transfers,accesses=accesses,records=records,memory_layout=layout,
              byte_range_contract='For each layout tensor and block b: [base + b*page_bytes, base + (b+1)*page_bytes).')
    (output/'execution_graph.json').write_text(json.dumps(data,ensure_ascii=False,separators=(',',':'))+'\n')
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (output/'unsatisfied.json').write_text(json.dumps(violations,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    analyze(p.parse_args().run)
