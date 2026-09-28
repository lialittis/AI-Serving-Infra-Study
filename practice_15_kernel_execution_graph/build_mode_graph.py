"""Reconstruct eager/PIECEWISE execution, preserving missing replay correlations.

Offline, stdlib only. A device node is never dropped for lacking host flow.
Exact flow/connection IDs establish attribution; time windows only label phases.
"""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sys

from build_model_graph import validate_dag, digest, same_view, tensor_span
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'practice_09_operator_trace'))
from summarize_profile import read_trace, number, only, require, inside

CONTROL = {'PROFILING_ENABLE', 'PROFILING_DISABLE'}
INVALID_MODEL = 4294967295


def end(e):
    return number(e['ts']) + number(e.get('dur', 0))


def load(p):
    return json.loads(p.read_text())


def verify_files(run):
    for name in ('source_manifest.json', 'instrumentation_hashes.json', 'compiler_artifacts.json'):
        omitted = load(run / 'unarchived_build_intermediates.json') if (run / 'unarchived_build_intermediates.json').exists() else {}
        for path, meta in load(run / name).items():
            if name == 'compiler_artifacts.json' and path in omitted and not (run / path).exists():
                require(path.endswith('/precompiled.h.gch') and meta == omitted[path], 'invalid omitted artifact')
                continue
            require(digest(run / path) == (meta if isinstance(meta, str) else meta['sha256']), 'fingerprint mismatch: ' + path)


def analyze(run, records=None, events=None):
    verify_files(run)
    command = load(run / 'command.json')
    mode, argv = command['mode'], command['argv']
    require('--num-gpu-blocks-override' not in argv, 'expected native KV pool sizing')
    if mode == 'graph':
        require('--enforce-eager' not in argv, 'graph command forces eager')
        conf = json.loads(argv[argv.index('--compilation-config') + 1])
        require(conf == dict(mode=3, cudagraph_mode='PIECEWISE', cudagraph_capture_sizes=[1], custom_ops=['all']), 'unexpected graph config')
    else:
        require(mode == 'eager' and '--enforce-eager' in argv, 'unexpected mode')
    require(load(run / 'shutdown.json')['server_exit_code'] == 0, 'unclean shutdown')
    require([(x['endpoint'], x['status']) for x in load(run / 'profile_control.json')] == [('/start_profile', 200), ('/stop_profile', 200)], 'profile controls failed')
    response = load(run / 'response.json')
    prompt = load(run / 'prompt_info.json')
    require(response['usage']['prompt_tokens'] == prompt['prompt_tokens'] == 10 and response['usage']['completion_tokens'] == 4, 'unexpected workload')
    records = records if records is not None else [json.loads(s) for p in sorted((run / 'events').glob('*.jsonl')) for s in p.read_text().splitlines()]
    require(not any(e['event'] == 'trace_error' for e in records), 'instrumentation errors')
    entries = {e['label']: e for e in records if e['event'] == 'enter'}
    exits = {e['label']: e for e in records if e['event'] == 'exit'}
    require(set(entries) == set(exits), 'unbalanced observations')
    for action, mapping in [('enter', entries), ('exit', exits)]:
        require(Counter(e['label'] for e in records if e['event'] == action) == Counter(mapping.keys()), 'duplicate scope labels')
    artifacts = load(run / 'compiler_artifacts.json')
    ready = load(run / 'ready.json')['time_ns']
    window = load(run / 'request_window.json')
    compilation = []
    require(not load(run / 'cache_before.json')['triton_cache_exists'], 'expected isolated cold Triton cache')
    for e in entries.values():
        if e['kind'] not in ('compile', 'init_handles', 'load_binary'):
            continue
        out = exits[e['label']]
        stage = 'startup' if e['time_ns'] < ready else 'warmup' if e['time_ns'] < window['start_ns'] else 'measured'
        if e['kind'] == 'compile':
            require(out['ran_compiler_stages'], 'expected cold compile stages')
            require(any(m['sha256'] == out['binary']['sha256'] for m in artifacts.values()), 'compiled binary not archived')
        elif e['kind'] == 'load_binary':
            require(any(c['kind'] == 'compile' and c['pid'] == e['pid'] and
                        exits[c['label']]['monotonic_ns'] <= e['monotonic_ns'] and
                        exits[c['label']]['binary'] == e['binary'] for c in entries.values()), 'binary registration without compiled binary')
        compilation.append(dict(entry=e, exit=out, stage=stage))
    require(compilation, 'missing compilation/load observations')
    for e in entries.values():
        if e['kind'] == 'launch' and e['measured']:
            init = only([x for x in compilation if x['entry']['kind'] == 'init_handles' and
                         x['entry']['pid'] == e['pid'] and x['exit']['monotonic_ns'] <= e['monotonic_ns'] and
                         x['exit']['function_handle'] == e['function_handle']], 'launch binary registration')
            require(init['entry']['kernel_hash'] == e['packed_metadata']['hash'], 'launch binary hash mismatch')
    rid = only(list({e['request_id'] for e in entries.values() if e['measured']}), 'measured request')
    require(rid.startswith(response['id'] + '-'), 'request mismatch')
    schedule = sorted([e for e in records if e['event'] == 'schedule' and rid in e['scheduled_tokens']], key=lambda e: e['step'])
    require([e['scheduled_tokens'][rid] for e in schedule] == [10, 1, 1, 1], 'unexpected scheduling')
    phases = {e['step']: 'prefill' if i == 0 else 'decode-' + str(i) for i, e in enumerate(schedule)}
    trace = only(list((run / 'profiler').rglob('trace_view.json')), 'trace')
    events = events if events is not None else read_trace(trace)
    scopes = {e['name']: e for e in events if e.get('ph') == 'X' and e.get('name', '').startswith('P13/')}
    require(set(scopes) == {k for k, e in entries.items() if e['measured']}, 'scope/profiler mismatch')
    complete, flows, endpoints = defaultdict(list), defaultdict(list), defaultdict(list)
    def point(e):
        return e['pid'], e['tid'], number(e['ts'])
    for i, e in enumerate(events):
        if e.get('ph') == 'X':
            complete[point(e)].append(i)
        if e.get('ph') in ('s', 'f'):
            flows[e.get('cat'), str(e['id']), e['ph']].append(i)
            if e['ph'] == 'f':
                endpoints[(e.get('cat'),) + point(e)].append(i)
    def source(task, category):
        fs = endpoints[(category,) + point(task)]
        if not fs:
            return None, dict(status='missing_endpoint')
        fi = only(fs, 'unique flow endpoint')
        flow = str(events[fi]['id'])
        starts = flows[category, flow, 's']
        if not starts:
            return None, dict(status='missing_start', flow_id=flow, endpoint_index=fi)
        si = only(starts, 'unique flow start')
        require(len(flows[category, flow, 'f']) == 1, 'ambiguous flow target')
        ei = only(complete[point(events[si])], 'unique flow source event')
        src = events[ei]
        if category == 'HostToDevice':
            require(src.get('args', {}).get('connection_id') == task['args'].get('connection_id'), 'CANN connection mismatch')
        else:
            require(src.get('cat') == 'cpu_op', 'torch flow source is not CPU operator')
        return ei, dict(status='exact', flow_id=flow, source_index=ei, start_index=si, endpoint_index=fi)
    queue_pairs = []
    for (cat, ident, ph), indices in flows.items():
        if cat == 'async_task_queue' and ph == 's':
            start = events[only(indices, 'enqueue flow')]
            finish = events[only(flows[cat, ident, 'f'], 'dequeue flow')]
            a = events[only(complete[point(start)], 'enqueue')]
            b = events[only(complete[point(finish)], 'dequeue')]
            require(a.get('cat') == 'enqueue' and b.get('cat') == 'dequeue', 'queue endpoint type')
            require(str(a['args']['correlation_id']) == str(b['args']['correlation_id']) == ident, 'queue correlation mismatch')
            queue_pairs.append(dict(correlation_id=ident, enqueue=a, dequeue=b))
    require(len(queue_pairs) == sum(cat == 'async_task_queue' and ph == 'f' for cat, ident, ph in flows), 'unpaired queue flows')
    csv_path = only(list((run / 'profiler').rglob('kernel_details.csv')), 'kernel CSV')
    with csv_path.open() as f:
        kernels = list(csv.DictReader(f))
    csv_index = defaultdict(list)
    for j, row in enumerate(kernels):
        csv_index[row['Name'], str(row['Stream ID']), str(row['Task ID']), number(row['Start Time(us)'])].append((j, row))
    used_csv = set()
    nodes, edges, byid, tasks = [], [], {}, []
    def node(n):
        if n['id'] not in byid:
            nodes.append(n)
            byid[n['id']] = n
        return n['id']
    def edge(a, b, kind, evidence, semantics):
        edges.append(dict(source=a, target=b, kind=kind, evidence=evidence, semantics=semantics))
    def host_node(i, kind):
        e = events[i]
        return node(dict(id=kind + ':' + str(i), kind=kind, name=e['name'], pid=e['pid'], tid=e['tid'],
                         start_us=str(e['ts']), end_us=str(end(e)), event=e))
    streams = defaultdict(list)
    for i, task in enumerate(events):
        args = task.get('args', {})
        if task.get('ph') != 'X' or 'Task Type' not in args or task['name'] in CONTROL:
            continue
        hi, hp = source(task, 'async_npu')
        ci, cp = source(task, 'HostToDevice')
        host, cann = events[hi] if hi is not None else None, events[ci] if ci is not None else None
        labels = sorted([s for s, e in scopes.items() if host and inside(e, host)], key=lambda s: number(scopes[s]['dur']))
        phase = phases.get(entries[labels[0]]['step'], 'unattributed') if labels else 'unattributed'
        candidates = [q for q in queue_pairs if host and cann and inside(host, q['enqueue']) and
                      q['dequeue']['tid'] == cann.get('args', {}).get('Thread Id', cann['tid']) and
                      number(q['dequeue']['ts']) <= number(cann['ts']) and end(cann) <= end(q['dequeue'])]
        require(len(candidates) <= 1, 'ambiguous queue attribution')
        match = csv_index.get((task['name'], str(args['Physic Stream Id']), str(args['Task Id']), number(task['ts'])), [])
        if match:
            j, row = only(match, 'unique kernel CSV row')
            require(j not in used_csv and abs(number(row['Duration(us)']) - number(task['dur'])) <= number('0.001'), 'kernel CSV mismatch')
            used_csv.add(j)
        else:
            require(args['Task Type'] not in ('AI_CORE', 'AI_VECTOR_CORE', 'AI_CPU'), 'compute kernel missing CSV')
        n = dict(id='k:' + str(i), kind='kernel', name=task['name'], phase=phase,
                 phase_evidence='exact_host_scope' if labels else 'unknown', stream=str(args['Physic Stream Id']),
                 device_lane=str(task['pid']), task_id=args['Task Id'], model_id=args.get('Model Id'),
                 start_us=str(task['ts']), end_us=str(end(task)), trace_index=i,
                 host_operator=host['name'] if host else 'unknown: no exact torch flow',
                 queue=candidates[0] if candidates else None, scopes=labels,
                 flow_evidence=dict(torch=hp, cann=cp), kernel_csv_verified=bool(match),
                 raw_stream_handle=None, device_event=task)
        if match:
            n['core_usage'] = dict(csv_index=j, accelerator_core=row['Accelerator Core'],
                                   block_num=row['Block Num'], mix_block_num=row['Mix Block Num'],
                                   status='reported' if number(row['Block Num'] or '0') > 0 else 'unknown_or_not_reported',
                                   meaning='profiler task core counts; not physical core IDs or utilization; not KV storage blocks')
        if hi is None or ci is None:
            require(mode == 'graph' and (args.get('Model Id') not in (None, INVALID_MODEL) or
                    task['name'] in ('MODEL_EXECUTE', 'NOTIFY_WAIT')), 'unexpected missing direct task flow')
        node(n)
        tasks.append(n)
        streams[n['device_lane'], n['stream']].append(n)
        if hi is not None:
            h = host_node(hi, 'host')
        if ci is not None:
            c = host_node(ci, 'submission')
            edge(c, n['id'], 'launch', cp, 'exact runtime flow; launch return need not precede device start')
            if hi is not None:
                edge(h, c, 'dispatch', dict(torch=hp, cann=cp, queue=n['queue']), 'same device task joined by exact profiler flows')
        for label in labels:
            if entries[label]['kind'] == 'launch':
                n['raw_stream_handle'] = str(entries[label]['runtime_stream'])
    require(len(used_csv) == len(kernels), 'unmatched kernel CSV rows')
    for key, group in streams.items():
        group.sort(key=lambda n: (number(n['start_us']), n['trace_index']))
        for a, b in zip(group, group[1:]):
            require(number(a['end_us']) <= number(b['start_us']), 'overlapping same-stream tasks')
            edge(a['id'], b['id'], 'stream_order', dict(device_lane=key[0], stream=key[1]), 'observed consecutive tasks; not proof of tensor dependency')
    for kind in ('host', 'submission'):
        lanes = defaultdict(list)
        for n in nodes:
            if n['kind'] == kind:
                lanes[n['pid'], n['tid']].append(n)
        for group in lanes.values():
            group.sort(key=lambda n: (number(n['start_us']), n['id']))
            for a, b in zip(group, group[1:]):
                edge(a['id'], b['id'], kind + '_issue_order', 'same PID/TID', 'observed call entry order only')
    by_scope = {s: [n for n in tasks if s in n['scopes']] for s in scopes}
    completions = []
    for e in sorted([e for e in entries.values() if e['measured'] and e['kind'] == 'to_list'], key=lambda x: x['step']):
        label = e['label']
        records_in = [x for x in entries.values() if x.get('parent') == label and x['kind'] == 'event_record']
        rec = only(records_in, 'result event record')
        wait = only([x for x in entries.values() if x.get('parent') == label and x['kind'] == 'synchronize'], 'result event wait')
        require(rec['object_id'] == wait['object_id'] == e['event_object_id'], 'completion event mismatch')
        require(exits[rec['label']]['monotonic_ns'] <= wait['monotonic_ns'], 'wait precedes record return')
        task = only([n for n in by_scope[rec['label']] if n['name'] == 'EVENT_RECORD'], 'result record task')
        copy = only([n for n in by_scope[label] if n['name'] == 'MEMCPY_ASYNC' and n['host_operator'] == 'acl_memcpy_device_to_host'], 'result D2H')
        native = only([v for v in events if v.get('name') == 'AscendCL@aclrtSynchronizeEvent' and
                       v.get('args', {}).get('Thread Id', v.get('tid')) == wait['tid'] and
                       number(scopes[wait['label']]['ts']) <= number(v['ts']) and end(v) <= end(scopes[wait['label']])], 'native completion wait')
        require(number(task['end_us']) <= end(native) and number(copy['end_us']) <= number(task['start_us']), 'completion boundary violated')
        wid = node(dict(id='wait:' + label, kind='wait_return', name='CPU event synchronize returned', phase=phases[e['step']],
                        start_us=str(end(native)), end_us=str(end(native)), event_object=wait['object_id']))
        edge(task['id'], wid, 'event_sync', dict(record=rec['label'], wait=wait['label'], native=native), 'event completed before CPU wait return')
        following = [n for n in nodes if n['kind'] == 'host' and n['pid'] == wait['pid'] and n['tid'] == wait['tid'] and number(n['start_us']) >= end(native)]
        if following:
            h = min(following, key=lambda n: number(n['start_us']))
            edge(wid, h['id'], 'host_after_wait', wait['label'], 'same CPU thread issues operation after completion wait')
        completions.append(dict(phase=phases[e['step']], wait_end_us=str(end(native)), record_task=task['id'], copy_task=copy['id']))
    require(len(completions) == 4, 'missing result completion')
    # Time windows identify a request step, NOT a replay/FX/host-op attribution.
    previous = None
    for completion in completions:
        upper = number(completion['wait_end_us'])
        for n in tasks:
            if n['phase'] == 'unattributed' and (previous is None or previous <= number(n['start_us'])) and number(n['end_us']) <= upper:
                n['phase'] = completion['phase']
                n['phase_evidence'] = 'between_native_completion_boundaries_only'
        previous = upper
    captures = [e for e in records if e['event'] == 'graph_capture']
    replays = []
    for e in entries.values():
        if not e['measured'] or e['kind'] != 'graph_replay':
            continue
        parent = entries[e['parent']]
        require(parent['kind'] == 'acl_dispatch', 'replay missing ACL parent')
        s = parent['resources']
        require(s['graph_id'] == e['graph_id'] and s['runtime_mode'] == 'PIECEWISE', 'replay graph identity/mode mismatch')
        prior = [c for c in captures if c['pid'] == e['pid'] and c['monotonic_ns'] < e['monotonic_ns'] and
                 c['resources']['wrapper_id'] == s['wrapper_id'] and c['resources']['graph_id'] == s['graph_id']]
        baseline = max(prior, key=lambda c: c['monotonic_ns']) if prior else None
        require(baseline is not None, 'replay lacks capture baseline')
        old = baseline['resources']
        require(s['input_addresses'] == s['captured_input_addresses'] == old['input_addresses'], 'replay input address mismatch')
        require(s['inputs'] == old['inputs'] and s['output'] == old['output'] and s['graph_pool'] == old['graph_pool'], 'replay layout/output/pool changed')
        scope = scopes[e['label']]
        runtime = [(i, v) for i, v in enumerate(events) if v.get('ph') == 'X' and
                   v.get('name') == 'AscendCL@aclmdlRIExecuteAsync' and
                   v.get('args', {}).get('Thread Id', v.get('tid')) == e['tid'] and
                   number(scope['ts']) <= number(v['ts']) and end(v) <= end(scope)]
        ri, native = only(runtime, 'native replay execute')
        rnode = node(dict(id='replay:' + e['label'], kind='replay', name='NPUGraph.replay ' + s['partition'], phase=phases[e['step']],
                         start_us=str(scope['ts']), end_us=str(end(scope)), resources=s,
                         capture_occurrence=dict(pid=baseline['pid'], monotonic_ns=baseline['monotonic_ns']),
                         runtime_stream=e['runtime_stream']))
        runtime_id = host_node(ri, 'submission')
        edge(rnode, runtime_id, 'replay_runtime_call', dict(scope=e['label'], native_index=ri),
             'native async execute occurs inside replay on the same host thread; return is not device completion')
        connection = native['args']['connection_id']
        boundary_tasks = [n for n in tasks if n['device_event']['args'].get('connection_id') == connection and
                          n['name'] in ('MODEL_EXECUTE', 'NOTIFY_WAIT')]
        require(Counter(n['name'] for n in boundary_tasks) == Counter(['MODEL_EXECUTE', 'NOTIFY_WAIT']), 'replay runtime boundary tasks missing')
        for n in boundary_tasks:
            require(n['phase'] == phases[e['step']], 'replay outside completion window')
            n['phase_evidence'] = 'exact_replay_runtime_connection'
            n['replay_label'] = e['label']
            n['runtime_connection'] = dict(native_index=ri, connection_id=connection)
            edge(runtime_id, n['id'], 'runtime_connection', n['runtime_connection'],
                 'exact native connection_id; one async execute submits multiple boundary tasks; no per-kernel flow invented')
        associated = by_scope[e['label']]
        # Only tasks with an actual host flow can be attributed to this replay.
        for n in associated:
            edge(rnode, n['id'], 'replay_dispatch', dict(scope=e['label'], flow=n['flow_evidence']), 'host replay scope contains exact task attribution; not replay return/completion ordering')
        replays.append(dict(label=e['label'], partition=s['partition'], phase=phases[e['step']], graph_id=e['graph_id'],
                            capture=byid[rnode]['capture_occurrence'], resources_verified=True,
                            directly_correlated_tasks=[n['id'] for n in associated],
                            runtime_api=native, boundary_tasks=[n['id'] for n in boundary_tasks]))
    partitions = {'submod_' + str(i) for i in range(0, 49, 2)}
    for step, phase in phases.items():
        dispatches = [e for e in entries.values() if e['measured'] and e['kind'] == 'acl_dispatch' and e['step'] == step]
        bodies = [e for e in entries.values() if e['measured'] and e['kind'] == 'partition_body' and e['step'] == step]
        rp = [r for r in replays if r['phase'] == phase]
        if mode == 'graph':
            require(len(dispatches) == 25 and {e['resources']['partition'] for e in dispatches} == partitions, 'missing graph dispatch partitions')
            if phase == 'prefill':
                require(len(bodies) == 25 and not rp and all(e['resources']['runtime_mode'] == 'NONE' for e in dispatches), 'unexpected prefill path')
            else:
                require(not bodies and len(rp) == 25 and {r['partition'] for r in rp} == partitions, 'unexpected decode replay path')
        else:
            require(not dispatches and not bodies and not rp, 'eager unexpectedly captured/replayed')
    # Direct API parameters support all KV/FIA chains. Replay-internal RoPE lacks
    # an actual runtime launcher scope, so never synthesize its data edge.
    chains = []
    for att in [e for e in entries.values() if e['measured'] and e['kind'] == 'attention']:
        selected = by_scope[att['label']]
        cache = only([n for n in selected if n['name'] == 'ReshapeAndCacheNdKernel'], 'attention KV writer')
        fia = only([n for n in selected if n['name'] == 'FusedInferAttentionScore'], 'attention FIA')
        def api(n, name):
            return only([entries[s] for s in n['scopes'] if entries[s].get('operator') == name], 'operator API metadata')
        cw, fa = api(cache, 'atb::_npu_reshape_and_cache'), api(fia, 'npu::npu_fused_infer_attention_score')
        for n in (cache, fia):
            require(n['flow_evidence']['torch']['status'] == n['flow_evidence']['cann']['status'] == 'exact', 'attention flow missing')
        phase = phases[att['step']]
        chains.append(dict(layer=att['layer'], phase=phase, cache=cache['id'], fia=fia['id']))
        if fa['kwargs']['block_table'] is not None:
            for key in ('key', 'value'):
                a, b = cw['kwargs'][key + '_cache'], fa['kwargs'][key]
                require(a['device'] == b['device'] and a['storage_ptr'] == b['storage_ptr'] and tensor_span(a) == tensor_span(b), 'KV pool mismatch')
                edge(cache['id'], fia['id'], 'storage_candidate', dict(resource=key, tensor=b), 'shared KV pool; exact indexed byte overlap unobserved')
        ropes = [n for n in tasks if n['phase'] == phase and n['name'] == '_triton_rope' and number(n['start_us']) < number(cache['start_us']) and any(entries[s]['kind'] == 'launch' for s in n['scopes'])]
        if mode == 'graph' and phase != 'prefill':
            continue
        rope = max(ropes, key=lambda n: number(n['start_us'])) if ropes else None
        require(rope is not None, 'missing direct RoPE')
        launch = only([entries[s] for s in rope['scopes'] if entries[s]['kind'] == 'launch'], 'direct RoPE launcher')
        for target, src, arg, metadata in [(cache, 'k_ptr', 'key', cw), (fia, 'q_ptr', 'query', fa)]:
            require(same_view(launch['named_arguments'][src], metadata['kwargs'][arg]), 'RoPE/attention view mismatch')
            edge(rope['id'], target['id'], 'data_contract', dict(producer=launch['label'], consumer=metadata['label'], tensor=metadata['kwargs'][arg]), 'source-backed scoped RoPE write/read; not cross-stream synchronization')
    require(len(chains) == 96 and all(len({c['layer'] for c in chains if c['phase'] == p}) == 24 for p in phases.values()), 'incomplete attention layers')
    validate_dag(nodes, edges)
    valid_models = [n for n in tasks if n['model_id'] not in (None, INVALID_MODEL)]
    summary = dict(mode=mode, kernel_tasks=len(tasks), graph_nodes=len(nodes), graph_edges=len(edges),
                   physical_streams=len(streams), compute_csv_rows=len(kernels),
                   tasks_with_both_flows=sum(n['flow_evidence']['torch']['status'] == n['flow_evidence']['cann']['status'] == 'exact' for n in tasks),
                   tasks_with_queue=sum(n['queue'] is not None for n in tasks),
                   verified_queue_pairs=len(queue_pairs),
                   replay_boundary_tasks_with_runtime_connection=sum('runtime_connection' in n for n in tasks),
                   replay_calls=len(replays), model_execute_tasks=sum(n['name'] == 'MODEL_EXECUTE' for n in tasks),
                   tasks_with_model_id=len(valid_models),
                   model_tasks_without_both_flows=sum(n['flow_evidence']['torch']['status'] != 'exact' or n['flow_evidence']['cann']['status'] != 'exact' for n in valid_models),
                   edges_by_kind=dict(Counter(e['kind'] for e in edges)), native_completion_boundaries=len(completions),
                   attention_invocations=len(chains), complete_data_graph=False,
                   compiler_calls=sum(x['entry']['kind'] == 'compile' for x in compilation),
                   measured_compiles=sum(x['entry']['kind'] == 'compile' and x['stage'] == 'measured' for x in compilation),
                   measured_binary_loads=sum(x['entry']['kind'] == 'load_binary' and x['stage'] == 'measured' for x in compilation),
                   per_phase={p: dict(tasks=sum(n['phase'] == p for n in tasks), replays=sum(r['phase'] == p for r in replays)) for p in phases.values()},
                   response=response['choices'][0]['text'])
    return dict(schema_version=1, nodes=nodes, edges=edges, summary=summary, replays=replays, completions=completions,
                attention_chains=chains, compilation=compilation,
                streams=[dict(device_lane=k[0], stream=k[1], tasks=len(v)) for k, v in sorted(streams.items())],
                parameter_scopes={s: dict(entry=entries[s], exit=exits[s]) for s in scopes},
                provenance=dict(run=run.name, trace_sha256=digest(trace), csv_sha256=digest(csv_path), builder_sha256=digest(Path(__file__))),
                limits=['运行后重建的执行证据图，不是可执行 ACL Graph。',
                        'PIECEWISE capture [1]；prefill 走编译 callable，decode 普通分区 replay，attention 直接调用。' if mode == 'graph' else 'eager 单请求匹配基线。',
                        '所有设备任务保留；缺失 flow 不按时间最近补造 host/FX/replay 归属。',
                        '无 host flow 任务的 phase 只由原生完成边界窗口标注，不代表逐分区关联。',
                        'Model Id、NPUGraph 对象 ID、stream 句柄不同；未建立其映射时不连边。',
                        '未解析 NOTIFY 配对，不把不同 stream 上时间相邻的任务画成同步边。',
                        '同 stream 顺序、CPU 调用归属、数据契约是不同语义；无数据边不等于独立。',
                        '带插桩/profiler 的单请求观测，不是性能基准；不验证 FULL graph、并发或完整内存安全。'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    graph = analyze(args.run)
    dest = args.run / 'analysis'
    dest.mkdir(exist_ok=True)
    (dest / 'execution_graph.json').write_text(json.dumps(graph, ensure_ascii=False, indent=2) + '\n')
    (dest / 'summary.json').write_text(json.dumps(graph['summary'], ensure_ascii=False, indent=2) + '\n')
    template = Path(__file__).with_name('mode_viewer.html').read_text()
    payload = json.dumps(graph, ensure_ascii=False, separators=(',', ':')).replace('<', '\\u003c')
    (dest / 'index.html').write_text(template.replace('__GRAPH_DATA__', payload))
    from export_mode_focus import export_focus
    export_focus(graph, dest)
    print(json.dumps(graph['summary'], ensure_ascii=False))


if __name__ == '__main__':
    main()
