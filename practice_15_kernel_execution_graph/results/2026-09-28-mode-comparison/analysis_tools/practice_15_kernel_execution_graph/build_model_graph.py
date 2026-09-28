"""Build an evidence-typed execution graph from a revalidated Practice 13 run.

Only the Python standard library is needed. No time-nearest attribution and no
global address-based last-writer inference: allocation lifetimes are not logged.
"""
import argparse
from collections import Counter, defaultdict, deque
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'practice_13_operator_submission'))
from analyze_submission import analyze, end, number, only, require


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_span(t):
    """A strided view's bounding span, NOT proof all enclosed bytes are accessed."""
    require(t.get('kind') == 'tensor', 'expected tensor metadata')
    require(len(t['shape']) == len(t['stride']), 'invalid tensor rank')
    require(all(n >= 0 for n in t['shape']), 'negative shape')
    require(t['data_ptr'] == t['storage_ptr'] + t['storage_offset'] * t['element_size'],
            'tensor pointer/offset mismatch')
    if any(n == 0 for n in t['shape']):
        return None
    offsets = [(n - 1) * s for n, s in zip(t['shape'], t['stride'])]
    return (t['data_ptr'] + sum(min(0, x) for x in offsets) * t['element_size'],
            t['data_ptr'] + (sum(max(0, x) for x in offsets) + 1) * t['element_size'])


def same_view(a, b):
    tensor_span(a)
    tensor_span(b)
    return all(a[k] == b[k] for k in
               ('device', 'data_ptr', 'storage_ptr', 'shape', 'stride', 'element_size'))


def validate_dag(nodes, edges):
    ids = {n['id'] for n in nodes}
    require(len(ids) == len(nodes), 'duplicate node ID')
    incoming = dict.fromkeys(ids, 0)
    successors = defaultdict(list)
    for e in edges:
        require(e['source'] in ids and e['target'] in ids, 'dangling edge')
        require(e['source'] != e['target'], 'self edge')
        incoming[e['target']] += 1
        successors[e['source']].append(e['target'])
    ready = deque(k for k, count in incoming.items() if not count)
    visited = 0
    while ready:
        visited += 1
        for target in successors[ready.popleft()]:
            incoming[target] -= 1
            if not incoming[target]:
                ready.append(target)
    require(visited == len(nodes), 'execution graph contains a cycle')


def build(data):
    nodes, edges, existing = [], [], {}
    observations = {e['label']: e for e in data['observations']}
    tasks = data['device_tasks']
    def node(n):
        if n['id'] not in existing:
            existing[n['id']] = n
            nodes.append(n)
        return n['id']
    def edge(a, b, kind, evidence, **extra):
        edges.append(dict(source=a, target=b, kind=kind, evidence=evidence, **extra))
    def kid(t):
        return 'k:' + str(t['task']['trace_index'])
    streams, hosts, launches = defaultdict(list), defaultdict(dict), defaultdict(dict)
    scope_tasks = defaultdict(list)
    for t in tasks:
        row, dev, host, launch = (t[k] for k in ('task', 'device_event', 'host_operator', 'cann_launch'))
        ident = kid(t)
        n = dict(id=ident, kind='kernel', name=row['kernel'], phase=t['phase'],
                 step=t['step'], stream=str(row['stream_id']), device_lane=str(dev['pid']),
                 task_id=row['task_id'], trace_index=row['trace_index'],
                 start_us=str(dev['ts']), end_us=str(end(dev)),
                 host_operator=host['name'], queue=t['queue'],
                 scopes=t['parameter_scope_labels'], raw_stream_handle=None)
        launch_obs = [observations[s] for s in n['scopes'] if observations[s]['kind'] == 'launch']
        if launch_obs:
            ob = only(launch_obs, 'unique launcher')
            n['raw_stream_handle'] = str(ob['runtime_stream'])
            n['launcher_scope'] = ob['label']
        node(n)
        for label in n['scopes']:
            scope_tasks[label].append(t)
        streams[dev['pid'], str(row['stream_id'])].append(n)
        def event_node(ev, kind):
            key = '{}:{}:{}:{}'.format(kind, ev['pid'], ev['tid'], ev['ts'])
            return node(dict(id=key, kind=kind, name=ev['name'], phase=t['phase'],
                             pid=ev['pid'], tid=ev['tid'], start_us=str(ev['ts']), end_us=str(end(ev))))
        h, c = event_node(host, 'host'), event_node(launch, 'submission')
        hosts[host['pid'], host['tid']][h] = existing[h]
        launches[launch['pid'], launch['tid']][c] = existing[c]
        edge(h, c, 'dispatch', dict(queue=t['queue']['correlation_id'] if t['queue'] else None,
                                   torch_flow=row['torch_flow_id'], cann_flow=row['cann_flow_id']),
             semantics='attribution through verified queue; not host-return-before-device-start')
        edge(c, ident, 'launch', dict(flow=row['cann_flow_id'], connection=dev['args']['connection_id']),
             semantics='launch entry causes task; launch return may follow device start')
    for groups, kind in ((hosts, 'host_issue_order'), (launches, 'submission_order')):
        for group in groups.values():
            ordered = sorted(group.values(), key=lambda n: (number(n['start_us']), n['id']))
            for a, b in zip(ordered, ordered[1:]):
                edge(a['id'], b['id'], kind, 'same PID/TID, observed entry order',
                     semantics='entry order only; does not serialize separate NPU streams')
    stream_summary = []
    for (lane, stream), group in sorted(streams.items()):
        group.sort(key=lambda n: (number(n['start_us']), n['trace_index']))
        for a, b in zip(group, group[1:]):
            require(number(a['end_us']) <= number(b['start_us']), 'overlapping same-stream tasks')
            edge(a['id'], b['id'], 'stream_order', dict(device_lane=lane, stream=stream),
                 semantics='observed consecutive tasks on one physical stream; unobserved tasks may exist')
        stream_summary.append(dict(device_lane=lane, stream=stream, tasks=len(group)))

    # Native completion events are reused: identify each record occurrence by
    # its unique scope, not by Python object address alone.
    for completion in data['completions']:
        record = completion['record']
        candidates = [t for t in scope_tasks[record['label']] if t['task']['kernel'] == 'EVENT_RECORD']
        task = only(candidates, 'completion record task')
        require(record['object_id'] == completion['wait']['object_id'], 'event identity mismatch')
        wait = completion['runtime_wait']
        require(end(task['device_event']) <= end(wait), 'event completes after synchronize return')
        w = node(dict(id='wait:' + record['label'], kind='wait_return', name='CPU event synchronize returned',
                      phase=completion['phase'], start_us=str(end(wait)), end_us=str(end(wait)),
                      event_object=str(record['object_id']), generation=record['label']))
        edge(kid(task), w, 'event_sync', dict(record=record['label'], wait=completion['wait']['label']),
             semantics='event completion precedes successful CPU wait return')
        following = [n for n in hosts.get((completion['wait']['pid'], completion['wait']['tid']), {}).values()
                     if number(n['start_us']) >= end(wait)]
        if following:
            next_host = min(following, key=lambda n: number(n['start_us']))
            edge(w, next_host['id'], 'host_after_wait', dict(wait=completion['wait']['label']),
                 semantics='same CPU thread issues next observed operator after synchronous wait returns')

    # Deliberately narrow data contracts. The adjacent RoPE and this attention
    # invocation have source-backed roles and actual matching tensor views.
    # KV indices are not dumped: cache-pool relationships remain conservative.
    attention = [o for o in observations.values() if o.get('measured') and o['kind'] == 'attention']
    used_ropes = set()
    sorted_tasks = sorted(tasks, key=lambda t: number(t['device_event']['ts']))
    for att in attention:
        related = scope_tasks[att['label']]
        caches = [t for t in related if t['task']['kernel'] == 'ReshapeAndCacheNdKernel']
        fias = [t for t in related if t['task']['kernel'] == 'FusedInferAttentionScore']
        cache, fia = only(caches, 'attention cache task'), only(fias, 'attention FIA task')
        def api(t, name):
            return only([observations[s] for s in t['parameter_scope_labels']
                         if observations[s].get('operator') == name], 'operator parameters')
        cw = api(cache, 'atb::_npu_reshape_and_cache')
        fa = api(fia, 'npu::npu_fused_infer_attention_score')
        prior = [t for t in sorted_tasks if t['step'] == cache['step'] and
                 t['task']['kernel'] == '_triton_rope' and
                 number(t['device_event']['ts']) < number(cache['device_event']['ts'])]
        require(prior, 'missing preceding RoPE')
        rope = prior[-1]
        require(kid(rope) not in used_ropes, 'RoPE task reused across attention invocations')
        used_ropes.add(kid(rope))
        launch = only([observations[s] for s in rope['parameter_scope_labels']
                       if observations[s]['kind'] == 'launch'], 'RoPE launcher')
        args = launch['named_arguments']
        for dst, src_arg, dst_arg, parameters in (
                (cache, 'k_ptr', 'key', cw), (fia, 'q_ptr', 'query', fa)):
            require(same_view(args[src_arg], parameters['kwargs'][dst_arg]), 'RoPE/attention view mismatch')
            proof = dict(producer_scope=launch['label'], consumer_scope=parameters['label'],
                         producer_argument=src_arg, consumer_argument=dst_arg,
                         tensor=parameters['kwargs'][dst_arg], scope=att['label'])
            edge(kid(rope), kid(dst), 'data_contract', proof,
                 semantics='RAW from source-backed RoPE write and attention read; scoped tensor identity')
        if fa['kwargs']['block_table'] is not None:
            for key in ('key', 'value'):
                a, b = cw['kwargs'][key + '_cache'], fa['kwargs'][key]
                require(a['device'] == b['device'] and a['storage_ptr'] == b['storage_ptr'] and
                        tensor_span(a) == tensor_span(b), 'KV pool mismatch')
                edge(kid(cache), kid(fia), 'storage_candidate',
                     dict(writer_scope=cw['label'], reader_scope=fa['label'], resource=key,
                          storage_ptr=str(a['storage_ptr']), tensor=b),
                     semantics='partial indexed write/read of shared KV pool; exact byte overlap unobserved')
    require(used_ropes == {kid(t) for t in tasks if t['task']['kernel'] == '_triton_rope'},
            'unpaired RoPE task')
    validate_dag(nodes, edges)
    handles = defaultdict(set)
    for n in nodes:
        if n.get('raw_stream_handle'):
            handles[n['raw_stream_handle']].add((n['device_lane'], n['stream']))
    kinds = Counter(e['kind'] for e in edges)
    covered = {e[k] for e in edges if e['kind'] == 'data_contract' for k in ('source', 'target')}
    selected_kinds = {'launch', 'linear', 'torch_api', 'buffer_copy', 'to_list'}
    returns = {e['label']: e for e in data['returns']}
    parameters = {label: dict(entry=o, exit=returns[label],
                             tasks=[kid(t) for t in scope_tasks[label]],
                             attribution='scope-level parameters; multi-kernel scopes do not expose internal workspaces')
                  for label, o in observations.items() if o.get('measured') and o['kind'] in selected_kinds}
    return dict(schema_version=1, nodes=nodes, edges=edges, streams=stream_summary,
                parameter_scopes=parameters,
                stream_handle_mappings=[dict(raw_handle=h, observed_lanes=sorted(v)) for h, v in handles.items()],
                summary=dict(kernel_tasks=len(tasks), graph_nodes=len(nodes), graph_edges=len(edges),
                             edges_by_kind=dict(kinds), tasks_with_data_contract=len(covered),
                             raw_stream_observations=sum(n.get('raw_stream_handle') is not None for n in nodes),
                             parameter_scopes=len(parameters),
                             cross_stream_event_edges=0, complete_data_graph=False),
                limits=['Single-request eager model trace; not an ACL graph capture/replay graph.',
                        'Physical stream IDs and raw runtime handles are distinct namespaces.',
                        'Data contracts cover scoped RoPE-to-attention views only; other accesses are unknown.',
                        'KV storage candidates do not prove indexed byte overlap; no KV values were read.',
                        'No global last-writer inference: recycled addresses lack allocation lifetime IDs.',
                        'No cross-stream synchronization observed in this model run.',
                        'Host issue order and attribution edges are not device completion dependencies.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = analyze(args.run)  # Revalidate ORIGINAL trace/CSV/queue/parameters, not cached analysis.
    graph = build(data)
    paths = list((args.run / 'events').glob('*.jsonl'))
    paths += list((args.run / 'profiler').rglob('trace_view.json'))
    paths += list((args.run / 'profiler').rglob('kernel_details.csv'))
    paths += [args.run / 'source_manifest.json', args.run / 'command.json']
    supplement = args.run / 'contract_sources/manifest.json'
    if supplement.exists():
        for name, metadata in json.loads(supplement.read_text()).items():
            source = supplement.parent / name
            require(digest(source) == metadata['sha256'], 'contract source mismatch')
            paths.append(source)
        paths.append(supplement)
    graph['provenance'] = dict(source_run=str(args.run), source_summary=data['summary'],
                              inputs={str(p.relative_to(args.run)): digest(p) for p in sorted(paths)},
                              builder_sha256=digest(Path(__file__)))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'execution_graph.json').write_text(json.dumps(graph, ensure_ascii=False, indent=2) + '\n')
    (args.output / 'summary.json').write_text(json.dumps(graph['summary'], indent=2) + '\n')
    from render_graph import render
    render(graph, args.output / 'index.html')
    from export_graph import export_focus
    export_focus(graph, args.output)
    print(json.dumps(graph['summary'], ensure_ascii=False))


if __name__ == '__main__':
    main()
