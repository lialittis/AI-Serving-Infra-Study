"""Join observed accesses to physical tasks, audit completeness, analyze three DAGs.

The projected graph is explicitly conditional. Opaque native workspace is NOT
made observable by TorchDispatchMode. Only the observed stream graph is currently
eligible for a dependency-preserving plan without additional assumptions.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

from dag import MemoryFrontier, metrics, ns, schedule_scopes, topological
from kv_ranges import KVIndices, pool_ranges

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT.parent / 'practice_15_kernel_execution_graph'),
               str(ROOT.parent / 'practice_13_operator_submission')]
from analyze_submission import analyze, inside, number, read_trace
from build_model_graph import tensor_span


def tensors(value):
    if isinstance(value, dict):
        if value.get('kind') == 'tensor':
            yield value
        else:
            for child in value.values():
                yield from tensors(child)
    elif isinstance(value, list):
        for child in value:
            yield from tensors(child)


def view_ranges(t, limit=4096):
    """Exact union of a view's element bytes, with an explicit bounded fallback."""
    span = tensor_span(t)
    if span is None:
        return [], 'empty'
    size = t['element_size']
    intervals = [(span[0], span[0] + size)]
    for count, stride in sorted(zip(t['shape'], t['stride']), key=lambda x: abs(x[1])):
        step = abs(stride) * size
        if count <= 1 or step == 0:
            continue
        if len(intervals) == 1 and step <= intervals[0][1] - intervals[0][0]:
            intervals = [(intervals[0][0], intervals[0][1] + (count - 1) * step)]
            continue
        if count * len(intervals) > limit:
            return [span], 'bounding_fallback'
        expanded = sorted((a + i * step, b + i * step) for a, b in intervals for i in range(count))
        intervals = []
        for a, b in expanded:
            if intervals and a <= intervals[-1][1]:
                intervals[-1] = (intervals[-1][0], max(intervals[-1][1], b))
            else:
                intervals.append((a, b))
    return intervals, 'view_bytes'


def accesses(value, mode):
    result = []
    for t in tensors(value):
        if not t['device'].startswith('npu'):
            continue  # Host accesses are not inferred from NPU kernels.
        span = tensor_span(t)
        if span is None:
            continue
        if not ('storage_generation' in t and 'storage_bytes' in t):
            raise ValueError('missing allocation identity; old captures cannot be upgraded')
        lo, hi = span
        if lo < t['storage_ptr'] or hi > t['storage_ptr'] + t['storage_bytes']:
            raise ValueError('view exceeds storage')
        intervals, precision = view_ranges(t)
        for a, b in intervals:
            result.append(dict(device=t['device'], lo=a, hi=b,
                               generation=t['storage_generation'], mode=mode,
                               precision=precision))
    return result


TRITON_WRITES = {
    '_triton_rope': {'q_ptr', 'k_ptr'},
    '_compute_slot_mapping_kernel': {'slot_mapping_ptr'},
    'token_bin_counts_and_mask_kernel': {'bin_counts_ptr'},
    'apply_all_penalties_kernel': {'logits_ptr'},
}


def contract(entry, returned, kernel_names, kv):
    reads, writes = [], []
    kind = entry.get('kind', 'dispatch')
    status = 'operator_boundary'
    if entry['event'] == 'dispatch_enter':
        for arg in entry['arguments']:
            reads += accesses(arg['value'], 'R')
            if arg['write']:
                writes += accesses(arg['value'], 'W')
        writes += accesses(returned.get('returned'), 'W')
    elif kind == 'launch':
        if len(set(kernel_names)) != 1 or kernel_names[0] not in TRITON_WRITES:
            return [], 'unknown_triton_contract'
        args = entry['named_arguments']
        reads += accesses(args, 'R')
        for name in TRITON_WRITES[kernel_names[0]]:
            if name not in args:
                raise ValueError('Triton contract argument missing: ' + name)
            writes += accesses(args[name], 'W')
        status = 'triton_tensor_bounds'
    elif kind == 'torch_api':
        # Redispatch invokes OpOverload with positional tensors even when the
        # outer Python API used kwargs. Bind the audited native schema order.
        names = (['key', 'value', 'key_cache', 'value_cache', 'slot_indices']
                 if entry['operator'] == 'atb::_npu_reshape_and_cache' else ['query', 'key', 'value'])
        if len(entry['args']) > len(names):
            raise ValueError('unsupported native positional signature')
        args = dict(entry['kwargs'])
        for name, value in zip(names, entry['args']):
            if name in args:
                raise ValueError('duplicate native argument')
            args[name] = value
        if not all(name in args for name in names):
            raise ValueError('missing native tensor arguments')
        if entry['operator'] == 'npu::npu_fused_infer_attention_score':
            args.setdefault('block_table', None)
        entry = dict(entry, kwargs=args)
        if entry['operator'] == 'atb::_npu_reshape_and_cache':
            slots = kv.pool_slots(entry)
            reads += accesses({k: v for k, v in args.items() if k not in ('key_cache', 'value_cache')}, 'R')
            for name in ('key_cache', 'value_cache'):
                writes += pool_ranges(args[name], slots, 'W') if slots is not None else accesses(args[name], 'W')
            status = 'indexed_kv_cpu_contract' if slots is not None else 'indexed_kv_whole_pool'
        elif entry['operator'] == 'npu::npu_fused_infer_attention_score':
            slots = kv.pool_slots(entry)
            for name, value in args.items():
                if name in ('key', 'value') and slots is not None:
                    for tensor in tensors(value):
                        reads += pool_ranges(tensor, slots, 'R')
                else:
                    reads += accesses(value, 'R')
            writes += accesses(returned.get('returned'), 'W')
            status = 'native_attention_kv_cpu_contract' if slots is not None else 'native_attention_boundary'
        else:
            return [], 'unknown_native_contract'
    else:
        return [], 'unobserved_accesses'
    return reads + writes, status


def build(data, records, trace):
    if any(e['event'] in ('trace_error', 'dispatch_error') for e in records):
        raise ValueError('capture contains instrumentation errors')
    entries = {e['label']: e for e in records if e['event'] in ('enter', 'dispatch_enter')}
    returns = {e['label']: e for e in records if e['event'] in ('exit', 'dispatch_exit')}
    kv = KVIndices(records, entries, returns)
    dispatch = [e for e in records if e['event'] == 'dispatch_enter']
    exits = [e for e in records if e['event'] == 'dispatch_exit']
    if (Counter(e['label'] for e in dispatch) != Counter(e['label'] for e in exits)
            or len({e['label'] for e in dispatch}) != len(dispatch)):
        raise ValueError('unbalanced/duplicate dispatcher records')
    scopes = {e['name']: e for e in trace if e.get('ph') == 'X'
              and e.get('name', '').startswith(('P18/', 'P13/'))}
    if {e['label'] for e in dispatch} != {k for k in scopes if k.startswith('P18/')}:
        raise ValueError('dispatcher/profiler scope mismatch')
    tasks = sorted(data['device_tasks'], key=lambda t: number(t['device_event']['ts']))
    if len({t['task']['stream_id'] for t in tasks}) != 1:
        raise ValueError('adapter currently requires the audited single physical stream capture')
    nodes, groups, owners = [], {}, {}
    for i, t in enumerate(tasks):
        event, row = t['device_event'], t['task']
        # Prefer actual direct launcher/native arguments to an opaque parent op.
        labels = [label for label in t['parameter_scope_labels']
                  if entries[label]['kind'] in ('launch', 'torch_api')]
        if not labels:
            labels = [e['label'] for e in dispatch
                      if not e['operator'].startswith('profiler.')
                      and inside(scopes[e['label']], t['host_operator'])]
        owner = min(labels, key=lambda label: number(scopes[label]['dur'])) if labels else None
        in_forward = any(entries[k]['kind'] == 'forward' for k in t['parameter_scope_labels'])
        n = dict(id=f'n{i:04d}', name=row['kernel'], duration_ns=ns(event['dur']),
                 start_ns=ns(event['ts']), stream=str(row['stream_id']), phase=t['phase'],
                 step=t['step'], in_forward=in_forward, trace_index=row['trace_index'],
                 owner=owner, host_operator=t['host_operator']['name'])
        if i and nodes[-1]['start_ns'] + nodes[-1]['duration_ns'] > n['start_ns']:
            raise ValueError('observed same-stream overlap')
        nodes.append(n)
        if owner:
            groups.setdefault(owner, []).append(n)
        owners[n['id']] = owner
    per_node = defaultdict(list)
    projected, audit, opaque_nodes = [], [], set()
    for label, members in groups.items():
        access, status = contract(entries[label], returns[label], [n['name'] for n in members], kv)
        if not access or status.startswith('unknown'):
            opaque_nodes.update(n['id'] for n in members)
        reads = [a for a in access if a['mode'] == 'R']
        writes = [a for a in access if a['mode'] == 'W']
        per_node[members[0]['id']] += reads
        per_node[members[-1]['id']] += writes
        for a, b in zip(members, members[1:]):
            projected.append(dict(source=a['id'], target=b['id'], kind='opaque_internal_order'))
        audit.append(dict(scope=label, kind=entries[label].get('kind', 'dispatch'),
                          operator=entries[label].get('operator'), status=status,
                          kernels=[n['id'] for n in members], read_ranges=len(reads),
                          write_ranges=len(writes), exact_per_kernel=False))
    frontier = MemoryFrontier()
    for n in nodes:
        for access in per_node[n['id']]:
            frontier.access(n['id'], access)
    projected += frontier.edges
    # Unknown tasks are global barriers in the projection, not independent work.
    # Also retain autoregressive request-step / model-boundary order; inference
    # cannot prepare decode inputs before sampled output becomes available.
    for i, n in enumerate(nodes):
        if owners[n['id']] is None or n['id'] in opaque_nodes:
            projected += [dict(source=p['id'], target=n['id'], kind='unknown_access_barrier')
                          for p in nodes[:i]]
            projected += [dict(source=n['id'], target=p['id'], kind='unknown_access_barrier')
                          for p in nodes[i + 1:]]
    runs = []
    for n in nodes:
        key = (n['phase'], n['in_forward'])
        if not runs or runs[-1][0] != key:
            runs.append((key, []))
        runs[-1][1].append(n)
    for (_, left), (_, right) in zip(runs, runs[1:]):
        for a in left:
            for b in right:
                projected.append(dict(source=a['id'], target=b['id'], kind='host_phase_boundary'))
    # Deduplicate structural edges but keep byte witnesses for hazard types.
    projected = list({json.dumps(e, sort_keys=True): e for e in projected}.values())
    observed = [dict(source=a['id'], target=b['id'], kind='stream_order')
                for a, b in zip(nodes, nodes[1:])]
    topological(nodes, projected)
    by_id = {n['id']: n for n in nodes}
    if any(by_id[e['source']]['start_ns'] > by_id[e['target']]['start_ns'] for e in projected):
        raise ValueError('dependency contradicts observed order')
    return dict(schema_version=1, nodes=nodes,
                graphs={'observed': observed, 'projected': projected,
                        'certified': observed}, contracts=audit,
                external_read_ranges=frontier.external_reads,
                audit=dict(dispatch_calls=len(dispatch), device_tasks=len(nodes),
                           dispatcher_reentry_scopes=dict(Counter(e['kind'] for e in records if e['event'] == 'dispatch_mode_scope')),
                           forward_tasks=sum(n['in_forward'] for n in nodes),
                           attributed_tasks=sum(owner is not None for owner in owners.values()),
                           unattributed_task_names=dict(Counter(n['name'] for n in nodes if owners[n['id']] is None)),
                           unobserved_tasks=[n['id'] for n in nodes if owners[n['id']] is None],
                           multikernel_scopes=sum(len(g['kernels']) > 1 for g in audit),
                           unknown_contract_tasks=sorted(opaque_nodes),
                           kv_slot_launches_reconstructed=len(kv.slots),
                           contract_status=dict(Counter(g['status'] for g in audit)),
                           complete_task_inventory=True, complete_exact_data_dag=False,
                           native_workspace_observed=False, exact_kv_indices_observed=False,
                           kv_indices_from_cpu_source_contract=len(kv.slots) == 4,
                           live_stream_reassignment_eligible=False),
                assumptions=[
                    'Projected graph uses public operator boundary reads/writes, not native kernel memory traces.',
                    'Native private/global workspace and allocator stream ownership are not fully observed.',
                    'Tensor view byte sets describe possible public accesses; unsupported large strided sets use a bounding range.',
                    'KV indices reconstructed from CPU staging and synchronous source equivalence are contracts, not device memory traces.',
                    'Inputs read without a writer in this window are external; initialization is not proven.',
                    'Multi-kernel operator reads occur at first member and writes at last member, with internal order retained.',
                    'Unknown task accesses and host phase boundaries remain barriers.',
                    'Certified graph retains observed FIFO order until exact dependency audit passes.',
                    'Measured durations include instrumentation; fixed-duration schedules ignore resource contention and host release delays.',
                ])


def analyze_graph(graph, event_cost_ns=0):
    result = {}
    selections = {'request': graph['nodes']}
    for phase in sorted({n['phase'] for n in graph['nodes'] if n['in_forward']}):
        selections[phase + '/forward'] = [n for n in graph['nodes'] if n['phase'] == phase and n['in_forward']]
    for name, nodes in selections.items():
        ids = {n['id'] for n in nodes}
        result[name] = {}
        for kind, all_edges in graph['graphs'].items():
            edges = [e for e in all_edges if e['source'] in ids and e['target'] in ids]
            result[name][kind] = dict(metrics=metrics(nodes, edges),
                schedules=[schedule_scopes(nodes, edges, count, event_cost_ns) for count in (1, 2, 4, 8)],
                nodes=len(nodes), edges=len(edges),
                measured_window_ns=max(n['start_ns'] + n['duration_ns'] for n in nodes) - min(n['start_ns'] for n in nodes))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--event-cost-ns', type=int, default=0)
    args = parser.parse_args()
    command = json.loads((args.run / 'command.json').read_text())['argv']
    for flag in ('--no-async-scheduling', '--no-enable-chunked-prefill', '--no-enable-prefix-caching'):
        if flag not in command:
            raise ValueError('unsupported configuration: requires ' + flag)
    for flag in ('--tensor-parallel-size', '--max-num-seqs'):
        if flag not in command or command[command.index(flag) + 1] != '1':
            raise ValueError('unsupported configuration: requires ' + flag + '=1')
    expected = json.loads((ROOT / 'contracts.json').read_text())
    for name, digest in expected.items():
        source = args.run / 'contract_sources' / name
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise ValueError('unreviewed kernel contract source: ' + name)
    runner = args.run / 'sources/vllm_ascend/vllm_ascend/worker/model_runner_v1.py'
    if hashlib.sha256(runner.read_bytes()).hexdigest() != 'b912e997f3e9bfa1d7c50addc3c352c2034ca25ed836a8f1c1ce74e6260812bb':
        raise ValueError('unreviewed CPU positions equivalence contract')
    records = [json.loads(line) for p in sorted((args.run / 'events').glob('*.jsonl'))
               for line in p.read_text().splitlines()]
    data = analyze(args.run, records=records)
    traces = list((args.run / 'profiler').rglob('trace_view.json'))
    if len(traces) != 1:
        raise ValueError('expected one profiler trace')
    graph = build(data, records, read_trace(traces[0]))
    graph['provenance_contract_sources'] = expected
    graph['provenance'] = {str(p.relative_to(args.run)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in [traces[0], *(args.run / 'events').glob('*.jsonl')]}
    graph['analysis'] = analyze_graph(graph, args.event_cost_ns)
    snapshot = args.run / 'analysis_tools'
    snapshot.mkdir(exist_ok=True)
    graph['analysis_tool_hashes'] = {}
    for name in ('build_dag.py', 'dag.py', 'kv_ranges.py', 'contracts.json'):
        content = (ROOT / name).read_bytes()
        (snapshot / name).write_bytes(content)
        graph['analysis_tool_hashes'][name] = hashlib.sha256(content).hexdigest()
    output = args.run / 'analysis'
    output.mkdir(exist_ok=True)
    (output / 'data_dag.json').write_text(json.dumps(graph, indent=2) + '\n')
    print(json.dumps(graph['audit'], indent=2))
    for name, kinds in graph['analysis'].items():
        for kind in ('observed', 'projected'):
            m = kinds[kind]['metrics']
            print(name, kind, 'work/span(ns)', m['work_ns'], m['critical_path_ns'], 'W/CP', m['work_over_span'])


if __name__ == '__main__':
    main()
