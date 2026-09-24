"""Verify two-stream probe flows, event generations, CSV and owned tensor dataflow."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

from build_model_graph import digest, end, number, only, require, validate_dag
from analyze_submission import inside, read_trace
from render_graph import render


def build_probe(run, metadata=None, events=None):
    meta = metadata if metadata is not None else json.loads((run / 'probe.json').read_text())
    require(meta['output_equals_five'] is True, 'probe arithmetic failed')
    for name, expected in meta['source_sha256'].items():
        require(digest(run / name) == expected, 'probe source hash mismatch')
    trace = only(list((run / 'profiler').rglob('trace_view.json')), 'probe trace')
    events = events if events is not None else read_trace(trace)
    with only(list((run / 'profiler').rglob('kernel_details.csv')), 'kernel CSV').open() as stream:
        kernels = list(csv.DictReader(stream))
    records = meta['records']
    expected_names = ['produce_y', 'record_1', 'wait_1', 'consume_y', 'record_2',
                      'wait_2', 'consume_z', 'record_3', 'host_wait_3']
    require([r['name'] for r in records] == expected_names, 'probe operation order mismatch')
    require([r['sequence'] for r in records] == list(range(9)), 'probe sequence mismatch')
    for a, b in zip(records, records[1:]):
        require(a['host_end_ns'] <= b['host_start_ns'], 'probe host order mismatch')
    scopes = {}
    for r in records:
        scopes[r['label']] = only([e for e in events if e.get('ph') == 'X' and e['name'] == r['label']],
                                  'probe scope')
    complete, starts, finishes = defaultdict(list), defaultdict(list), defaultdict(list)
    def point(e):
        return e['pid'], e['tid'], number(e['ts'])
    for i, e in enumerate(events):
        if e.get('ph') == 'X':
            complete[point(e)].append((i, e))
        elif e.get('ph') == 's':
            starts[e.get('cat'), str(e['id'])].append(e)
        elif e.get('ph') == 'f':
            finishes[(e.get('cat'),) + point(e)].append(e)
    def source(task, cat):
        finish = only(finishes[(cat,) + point(task)], cat + ' endpoint')
        start = only(starts[cat, str(finish['id'])], cat + ' start')
        i, e = only(complete[point(start)], cat + ' source')
        return i, e, str(finish['id'])
    nodes, edges, by_scope, streams = [], [], defaultdict(list), defaultdict(list)
    existing = set()
    def add(n):
        if n['id'] not in existing:
            nodes.append(n)
            existing.add(n['id'])
        return n['id']
    def edge(a, b, kind, proof, semantics):
        edges.append(dict(source=a, target=b, kind=kind, evidence=proof, semantics=semantics))
    used_csv = set()
    for i, task in enumerate(events):
        if task.get('ph') != 'X' or 'Task Type' not in task.get('args', {}):
            continue
        if task['name'] in {'PROFILING_ENABLE', 'PROFILING_DISABLE'}:
            continue
        hi, host, torch_flow = source(task, 'async_npu')
        ci, cann, cann_flow = source(task, 'HostToDevice')
        require(cann['args']['connection_id'] == task['args']['connection_id'], 'CANN connection mismatch')
        record = only([r for r in records if inside(scopes[r['label']], host)], 'task observation scope')
        if task['name'] in {'EVENT_RECORD', 'EVENT_WAIT'}:
            expected_api = {'EVENT_RECORD': 'AscendCL@aclrtRecordEvent',
                            'EVENT_WAIT': 'AscendCL@aclrtStreamWaitEvent'}[task['name']]
            require(cann['name'] == expected_api, 'event runtime API mismatch')
        kid = 'k:' + str(i)
        n = dict(id=kid, kind='kernel', name=task['name'], phase='two-stream-probe',
                 stream=str(task['args']['Physic Stream Id']), device_lane=str(task['pid']),
                 task_id=task['args']['Task Id'], trace_index=i,
                 start_us=str(task['ts']), end_us=str(end(task)), host_operator=host['name'],
                 raw_stream_handle=record['raw_stream_handle'], host_stream_id=record['host_stream_id'],
                 scopes=[record['label']], observation=record)
        add(n)
        by_scope[record['name']].append(n)
        streams[n['device_lane'], n['stream']].append(n)
        for idx, ev, kind in ((hi, host, 'host'), (ci, cann, 'submission')):
            add(dict(id=kind+':'+str(idx), kind=kind, name=ev['name'], phase='two-stream-probe',
                     pid=ev['pid'], tid=ev['tid'], start_us=str(ev['ts']), end_us=str(end(ev)), event=ev))
        edge('host:'+str(hi), 'submission:'+str(ci), 'dispatch',
             dict(torch_flow=torch_flow, cann_flow=cann_flow), 'same device task joined by two exact profiler flows')
        edge('submission:'+str(ci), kid, 'launch', dict(flow=cann_flow), 'CANN launch entry, not launch completion')
        if task['name'] not in {'EVENT_RECORD', 'EVENT_WAIT'}:
            matches = [j for j, row in enumerate(kernels) if row['Name'] == task['name'] and
                       str(row['Stream ID']) == n['stream'] and str(row['Task ID']) == str(n['task_id']) and
                       number(row['Start Time(us)']) == number(task['ts']) and
                       abs(number(row['Duration(us)']) - number(task['dur'])) <= number('0.001')]
            match = only(matches, 'probe kernel CSV match')
            require(match not in used_csv, 'duplicate CSV use')
            used_csv.add(match)
    require(len(used_csv) == len(kernels) == 3, 'probe kernel coverage mismatch')
    require(sum(len(v) for v in by_scope.values()) == 8, 'expected three kernels and five event tasks')
    for r in records[:-1]:
        only(by_scope[r['name']], 'one task per probe operation')
    stream_maps = defaultdict(set)
    for n in nodes:
        if n['kind'] == 'kernel':
            stream_maps[n['raw_stream_handle']].add((n['device_lane'], n['stream']))
    require(len(stream_maps) == 2 and all(len(v) == 1 for v in stream_maps.values()), 'raw/physical stream mapping mismatch')
    require(len(streams) == 2, 'expected two physical streams')
    for group in streams.values():
        group.sort(key=lambda n: number(n['start_us']))
        for a, b in zip(group, group[1:]):
            require(number(a['end_us']) <= number(b['start_us']), 'probe stream overlap')
            edge(a['id'], b['id'], 'stream_order', dict(stream=a['stream']), 'same physical stream')
    latest, record_nodes = {}, {}
    for r in records:
        if r['name'].startswith('record_'):
            require(r['generation'] == len(record_nodes) + 1, 'record generation mismatch')
            n = only(by_scope[r['name']], 'event record task')
            require(n['name'] == 'EVENT_RECORD', 'wrong event record task')
            latest[r['event_handle']] = r
            record_nodes[r['generation']] = n
        elif r['generation']:
            prior = latest.get(r['event_handle'])
            require(prior and prior['generation'] == r['generation'], 'wait refers to stale or unknown event generation')
            src = record_nodes[r['generation']]
            if r['name'].startswith('wait_'):
                dst = only(by_scope[r['name']], 'event wait task')
                require(dst['name'] == 'EVENT_WAIT', 'wrong event wait task')
                require(src['stream'] != dst['stream'], 'expected cross-stream wait')
                require(number(src['end_us']) <= number(dst['end_us']), 'wait completes before record')
                edge(src['id'], dst['id'], 'event_wait', dict(event_handle=r['event_handle'], generation=r['generation'],
                     record_scope=prior['label'], wait_scope=r['label']),
                     'record completion precedes wait completion; wait submission itself is asynchronous')
            else:
                scope = scopes[r['label']]
                wait = only([e for e in events if e.get('name') == 'AscendCL@aclrtSynchronizeEvent' and
                             e.get('args', {}).get('Thread Id', e.get('tid')) == scope['tid'] and
                             number(scope['ts']) <= number(e['ts']) and end(e) <= end(scope)], 'native host wait')
                require(number(src['end_us']) <= end(wait), 'CPU wait returned before final event')
                ident = add(dict(id='wait:3', kind='wait_return', name='CPU synchronize returned',
                                 phase='two-stream-probe', start_us=str(end(wait)), end_us=str(end(wait))))
                edge(src['id'], ident, 'event_sync', dict(event_handle=r['event_handle'], generation=3),
                     'CPU wait returned after final event completion')
    # Named resources are live throughout, and each simple API has one verified
    # kernel. These conditions are intentionally stronger than the model trace.
    roles = {'produce_y': (['x'], ['y']), 'consume_y': (['y'], ['z']), 'consume_z': (['x', 'z'], ['out'])}
    ranges = sorted((int(t['data_ptr']), int(t['data_ptr']) + t['bytes']) for t in meta['resources'].values())
    require(all(a[1] <= b[0] for a, b in zip(ranges, ranges[1:])), 'owned tensors unexpectedly alias')
    writers = {}
    for r in records:
        if r['name'] not in roles:
            continue
        require((r['reads'], r['writes']) == roles[r['name']], 'probe resource role mismatch')
        n = only(by_scope[r['name']], 'data kernel')
        for resource in r['reads']:
            if resource in writers:
                edge(writers[resource], n['id'], 'data_contract', dict(resource=resource, tensor=meta['resources'][resource]),
                     'RAW; distinct live tensors, explicit out= API, one correlated kernel per operation')
        for resource in r['writes']:
            writers[resource] = n['id']
    validate_dag(nodes, edges)
    kinds = dict(Counter(e['kind'] for e in edges))
    return dict(schema_version=1, nodes=nodes, edges=edges,
                streams=[dict(device_lane=k[0], stream=k[1], tasks=len(v)) for k, v in streams.items()],
                stream_handle_mappings=[dict(raw_handle=h, observed_lanes=sorted(v)) for h, v in stream_maps.items()],
                summary=dict(kernel_tasks=8, graph_nodes=len(nodes), graph_edges=len(edges), edges_by_kind=kinds,
                             cross_stream_event_edges=kinds['event_wait'], kernel_csv_verified=3,
                             complete_data_graph=False, output_equals_five=True),
                provenance=dict(source_run=str(run), trace_sha256=digest(trace), metadata_sha256=digest(run/'probe.json'),
                                builder_sha256=digest(Path(__file__))),
                limits=['Controlled two-stream calibration workload; not the vLLM model.',
                        'Data edges describe the four explicit live API tensors; hidden runtime workspace is unobserved.',
                        'One event handle has three record generations. Host wait submission does not wait for device completion.',
                        'Host stream ID, runtime handle and physical stream ID are distinct namespaces.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    graph = build_probe(args.run)
    dest = args.run / 'analysis'
    dest.mkdir(exist_ok=True)
    (dest/'execution_graph.json').write_text(json.dumps(graph, indent=2)+'\n')
    (dest/'summary.json').write_text(json.dumps(graph['summary'], indent=2)+'\n')
    render(graph, dest/'index.html')
    # A compact export includes the final CPU completion boundary as well.
    selected = {n['id']: n for n in graph['nodes'] if n['kind'] in {'kernel', 'wait_return'}}
    lines = ['digraph execution {', 'rankdir=LR;', 'node [shape=box, fontname="sans-serif"];']
    for ident, n in selected.items():
        label = n['name']+'\n'+('stream '+n['stream'] if 'stream' in n else 'CPU')
        if n.get('observation', {}).get('generation'):
            label += '\ngeneration '+str(n['observation']['generation'])
        lines.append('{} [label={}];'.format(json.dumps(ident), json.dumps(label)))
    for e in graph['edges']:
        if e['source'] in selected and e['target'] in selected:
            label = e['kind']
            if e['kind'] == 'event_wait':
                label += ' g'+str(e['evidence']['generation'])
            elif e['kind'] == 'data_contract':
                label += ' '+e['evidence']['resource']
            lines.append('{} -> {} [label={}];'.format(json.dumps(e['source']), json.dumps(e['target']), json.dumps(label)))
    lines.append('}')
    (dest/'execution_graph.dot').write_text('\n'.join(lines)+'\n')
    print(json.dumps(graph['summary']))


if __name__ == '__main__':
    main()
