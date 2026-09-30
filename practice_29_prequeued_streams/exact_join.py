"""P28 analyze_trial reused verbatim except native Notify connection fallback.

Source: practice_28_native_decode_streams/analyze.py; see README.
"""
import argparse
from collections import Counter, defaultdict
import csv
from decimal import Decimal
import importlib.util
import json
from pathlib import Path
import sys

from common import save, overlap, union, sha256


def helpers():
    repo = next(p for p in Path(__file__).resolve().parents if (p / 'practice_26_decode_utilization/analyze.py').exists())
    spec = importlib.util.spec_from_file_location('p26_evidence_helpers', repo / 'practice_26_decode_utilization/analyze.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read(path):
    return json.loads(path.read_text(), parse_float=Decimal)


def analyze_trial(root, run):
    h = helpers()
    number, end, only, require = h.number, h.end, h.only, h.require
    trace_path = only(root.rglob('trace_view.json'), 'one trace')
    csv_path = only(root.rglob('kernel_details.csv'), 'one CSV')
    events = read(trace_path)
    if isinstance(events, dict):
        events = events['traceEvents']
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    records = {r['label']: r for r in read(root / 'observations.json')}
    scopes = {e['name']: e for e in events if e.get('ph') == 'X' and e.get('name') in records}
    require(set(scopes) == set(records), 'missing scope')
    points, starts, finishes = defaultdict(list), defaultdict(list), defaultdict(list)
    def point(e):
        return e['pid'], e['tid'], number(e['ts'])
    for index, event in enumerate(events):
        phase = event.get('ph')
        if phase == 'X':
            points[point(event)].append((index, event))
        elif phase == 's':
            starts[event.get('cat'), str(event['id'])].append(event)
        elif phase == 'f':
            finishes[(event.get('cat'),) + point(event)].append(event)

    queues = []
    for event in events:
        if event.get('ph') == 'X' and event.get('cat') == 'dequeue':
            for finish in finishes[('async_task_queue',) + point(event)]:
                queues.append((event, only(starts['async_task_queue', str(finish['id'])], 'enqueue'), str(finish['id'])))
    def source(event, category, native=None):
        fs = finishes[(category,) + point(event)]
        if not fs:
            return None
        finish = only(fs, 'flow finish')
        ss = starts[category, str(finish['id'])]
        if category == 'async_npu' and len(ss) > 1 and native:
            candidates = [q for q in queues if q[0]['tid'] == native['tid'] and number(q[0]['ts']) <= number(native['ts']) and end(native) <= end(q[0])]
            index, host, _ = h.resolve_colliding_host([only(points[point(s)], 'host') for s in ss], native, candidates)
            return index, host, str(finish['id'])
        index, host = only(points[point(only(ss, 'flow source'))], 'source range')
        return index, host, str(finish['id'])
    def containing(event, native=False):
        tid = event.get('args', {}).get('Thread Id', event['tid']) if native else event['tid']
        matches = [(label, s) for label, s in scopes.items() if s['tid'] == tid and
                   (native or s['pid'] == event['pid']) and number(s['ts']) <= number(event['ts']) and end(event) <= end(s)]
        return min(matches, key=lambda x: number(x[1]['dur']))[0] if matches else None

    def native_scope(event):
        label = containing(event, True)
        if label:
            return label
        qs = [q for q in queues if q[0]['tid'] == event['tid'] and number(q[0]['ts']) <= number(event['ts']) and end(event) <= end(q[0])]
        if not qs:
            return None
        q = only(qs, 'native call queue membership')
        _, enqueue = only(points[point(q[1])], 'enqueue range')
        return containing(enqueue)

    models = {}
    for g in read(run / 'graphs.json'):
        dump = read(run / g['path'])
        mid = only({int(d['args']['Model Id']) for d in dump}, 'one model ID')
        require(mid not in models, 'duplicate live model ID')
        models[mid] = dict(graph=g, dump=dump)
    native_replays = defaultdict(list)
    for index, event in enumerate(events):
        if event.get('ph') == 'X' and event.get('name') in ('AscendCL@aclmdlRIExecuteAsync', 'AscendCL@aclrtWaitAndResetNotify', 'AscendCL@aclrtRecordNotify'):
            native_replays[str(event['args']['connection_id'])].append((index, event))
    csv_index = defaultdict(list)
    for index, row in enumerate(rows):
        csv_index[row['Name'], row['Stream ID'].strip(), row['Task ID'].strip(), number(row['Start Time(us)'])].append(index)
    tasks, used, by_scope = [], set(), defaultdict(list)
    for index, event in enumerate(events):
        args = event.get('args', {})
        if event.get('ph') != 'X' or 'Task Type' not in args or event['name'] in ('PROFILING_ENABLE', 'PROFILING_DISABLE'):
            continue
        internal = args.get('Model Id') in models
        native = None if internal else source(event, 'HostToDevice')
        if native is None and event['name'] in ('MODEL_EXECUTE', 'NOTIFY_WAIT', 'NOTIFY_RECORD'):
            ni, ne = only(native_replays[str(args['connection_id'])], 'native replay connection')
            native = ni, ne, 'connection:' + str(args['connection_id'])
        host = None if internal else source(event, 'async_npu', native[1] if native else None)
        label = containing(host[1]) if host else None
        if label is None and native:
            label = containing(native[1], True)
        record = records.get(label, {})
        task = dict(id=index, name=event['name'], stream=str(args['Physic Stream Id']),
                    task_id=str(args['Task Id']), model_id=args.get('Model Id'),
                    start_us=str(event['ts']), end_us=str(end(event)), duration_us=str(event['dur']),
                    role=record.get('role'), scope=label, is_compute=False,
                    host_index=host[0] if host else None, cann_index=native[0] if native else None,
                    host_flow=host[2] if host else None, cann_flow=native[2] if native else None)
        matches = csv_index[event['name'], task['stream'], task['task_id'], number(event['ts'])]
        if matches:
            ci = only(matches, 'CSV identity')
            require(ci not in used, 'duplicate CSV task')
            used.add(ci)
            h.validate_csv(task, rows[ci])
            task.update(is_compute=True, csv_row=ci, counters=rows[ci], core_type=rows[ci]['Accelerator Core'],
                        block_num=h.core_count(rows[ci]['Block Num']), mix_block_num=h.core_count(rows[ci]['Mix Block Num']))
        tasks.append(task)
        if label:
            by_scope[label].append(task)
    require(len(used) == len(rows), 'CSV coverage')

    for mid, model in models.items():
        records_for_graph = [r for r in records.values() if r['kind'] == 'replay' and r['python_id'] == model['graph']['python_id']]
        records_for_graph.sort(key=lambda r: number(scopes[r['label']]['ts']))
        internal = sorted((t for t in tasks if t['model_id'] == mid), key=lambda t: number(t['start_us']))
        dump = model['dump']
        size = len(dump) + (dump[-1]['args']['Task Type'] != 'NOTIFY_RECORD')
        require(len(internal) == size * len(records_for_graph), 'graph task coverage')
        for index, record in enumerate(records_for_graph):
            boundary = by_scope[record['label']]
            launch = only((t for t in boundary if t['name'] == 'MODEL_EXECUTE'), 'graph launch')
            wait = only((t for t in boundary if t['name'] == 'NOTIFY_WAIT'), 'graph completion')
            require(launch['cann_index'] == wait['cann_index'], 'graph connection mismatch')
            chunk = internal[index * size:(index + 1) * size]
            h.validate_graph_chunk(chunk, dump, launch, wait)
            for task in chunk:
                task.update(role=record['role'], scope=record['label'], replay=record['label'],
                            submission_kind='graph_replay', submission_index=launch['cann_index'])
    compute = [t for t in tasks if t['is_compute']]
    for task in compute:
        if 'submission_kind' not in task:
            task.update(submission_kind='direct', submission_index=task['cann_index'])
    require(all(t['role'] in ('A', 'B') for t in compute), 'unattributed compute')
    intervals = {role: [(number(t['start_us']), number(t['end_us'])) for t in compute if t['role'] == role] for role in ('A', 'B')}
    require(all(intervals.values()), 'missing branch')
    start = min(a for v in intervals.values() for a, b in v)
    finish = max(b for v in intervals.values() for a, b in v)
    both = overlap(intervals['A'], intervals['B'])
    coverage = sum((b-a for a, b in union(intervals['A'] + intervals['B'])), Decimal(0))
    trial = json.loads((root / 'trial.json').read_text())
    require(len({(scopes['P28/forward/'+r]['pid'], scopes['P28/forward/'+r]['tid']) for r in ('A','B')}) == 1,
            'more than one CPU submission thread')
    first_role, second_role = trial['order']
    first_finish = max(b for a, b in intervals[first_role])
    second_first = min((t for t in compute if t['role'] == second_role), key=lambda t: number(t['start_us']))
    submission = second_first['submission_index']
    branch_timing = dict(order=trial['order'],
        second_host_start_minus_first_compute_end_us=str(number(scopes['P28/forward/'+second_role]['ts'])-first_finish),
        second_compute_start_minus_first_compute_end_us=str(number(second_first['start_us'])-first_finish),
        second_native_submit_minus_first_compute_end_us=str(number(events[submission]['ts'])-first_finish) if submission is not None else None)
    profile_info = read(only(root.rglob('profiler_info_0.json'), 'one profiler config'))
    experimental = profile_info['config']['experimental_config']
    metric = 'ACL_AICORE_PIPE_UTILIZATION' if trial['profile'] == 'pipe' else 'ACL_AICORE_NONE'
    require(experimental['_aic_metrics'] == metric and experimental['_profiler_level'] == 'Level1', 'profiler configuration drift')
    if trial['strategy'] == 'serial':
        require(both == 0, 'serial branch overlap violates control')
    # These scopes observe the actual record/wait calls. Event-object identity
    # is guaranteed by pair() and frozen code; timestamps alone do not prove it.
    synchronization = dict(evidence='Python event identity + exact task flows + observed ordering',
                           roles={}, forward_cpu_synchronizations=[])
    if 'P28/origin' in scopes:
        origin = only((t for t in by_scope['P28/origin'] if t['name'] == 'EVENT_RECORD'), 'origin event record')
        for role in ('A', 'B'):
            wait_label = 'P28/wait_origin/' + role
            waits = [t for t in by_scope[wait_label] if t['name'] == 'EVENT_WAIT']
            call_index, call = only(((i, e) for i, e in enumerate(events) if e.get('ph') == 'X' and
                e.get('name') == 'AscendCL@aclrtStreamWaitEvent' and native_scope(e) == wait_label), 'native origin wait call')
            terminal = only((t for t in by_scope['P28/terminal/' + role] if t['name'] == 'EVENT_RECORD'), 'terminal event record')
            join = scopes['P28/join/' + role]
            first, last = min(a for a, b in intervals[role]), max(b for a, b in intervals[role])
            if waits:
                wait = only(waits, 'origin event wait')
                require(number(origin['end_us']) <= number(wait['end_us']), 'origin wait completion before producer')
                require(number(wait['end_us']) <= first, 'compute before input ready')
                wait_evidence = dict(task=wait['id'], status='device_wait_observed')
            else:
                require(number(origin['end_us']) <= number(call['ts']), 'missing device wait while producer incomplete')
                require(end(call) <= first, 'compute before wait call returned')
                wait_evidence = dict(task=None, status='no_device_wait_in_trace; producer_already_complete_before_native_call')
            require(last <= number(terminal['start_us']), 'terminal before compute completes')
            require(number(terminal['end_us']) <= end(join), 'CPU join before terminal')
            require(all(end(scopes['P28/forward/'+r]) <= number(join['ts']) for r in ('A','B')), 'CPU wait between submissions')
            synchronization['roles'][role] = dict(origin=origin['id'], wait=wait_evidence, wait_cann_index=call_index,
                                                 terminal=terminal['id'], cpu_join='P28/join/'+role)
        for index, event in enumerate(events):
            if event.get('ph') == 'X' and 'AscendCL@' in event.get('name','') and 'Synchronize' in event['name']:
                label = containing(event, True)
                if label and records[label]['kind'] in ('forward', 'replay'):
                    synchronization['forward_cpu_synchronizations'].append(dict(index=index, scope=label, name=event['name'], duration_us=str(event['dur'])))
    else:
        synchronization['evidence'] = 'missing instrumented event scopes; runtime dependencies not verified'
    groups = defaultdict(list)
    for task in compute:
        groups[task['role'], task['name']].append(task)
    cores = []
    for (role, name), group in sorted(groups.items()):
        pipeline = defaultdict(list)
        for task in group:
            for key, value in task['counters'].items():
                if key.startswith(('aic_', 'aiv_', 'aicore_', 'cube_utilization')):
                    try:
                        pipeline[key].append(float(value))
                    except (TypeError, ValueError):
                        pass
        cores.append(dict(role=role, name=name, count=len(group),
            duration_us=h.stats(t['duration_us'] for t in group),
            core_types=sorted({t['core_type'] for t in group}),
            block_nums=sorted({t['block_num'] for t in group if t['block_num'] is not None}),
            mix_block_nums=sorted({t['mix_block_num'] for t in group if t['mix_block_num'] is not None}),
            unknown_block_tasks=sum(t['block_num'] is None for t in group),
            pipeline={k:h.stats(v) for k,v in pipeline.items()}))
    result = dict(trial=trial, compute_tasks=len(compute), stream_ids=sorted({t['stream'] for t in compute}),
                  overlap_us=str(both), compute_union_us=str(coverage), device_compute_span_us=str(finish-start),
                  coverage=float(coverage/(finish-start)), tasks=tasks, synchronization=synchronization, cores=cores, branch_timing=branch_timing,
                  provenance=dict(trace=str(trace_path.relative_to(root)), trace_sha256=sha256(trace_path),
                                  csv=str(csv_path.relative_to(root)), csv_sha256=sha256(csv_path)),
                  host_events=json.loads(json.dumps({str(i):events[i] for i in sorted({t[k] for t in tasks for k in ('host_index','cann_index') if t[k] is not None})}, default=str)),
                  cpu_scopes={label: {k: str(s[k]) for k in ('ts', 'dur', 'pid', 'tid')} for label, s in scopes.items()})
    save(root / 'analysis.json', result)
    return result
