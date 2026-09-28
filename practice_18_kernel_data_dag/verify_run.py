"""Check archived evidence, full-model coverage and every proposed event plan."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from dag import metrics, ns, topological, validate_schedule
from kv_ranges import KVIndices
from build_dag import read_trace, tensors


def verify(run, baseline=None):
    graph = json.loads((run / 'analysis/data_dag.json').read_text())
    for name, digest in graph['provenance'].items():
        if hashlib.sha256((run / name).read_bytes()).hexdigest() != digest:
            raise ValueError('evidence changed: ' + name)
    for name, digest in graph['analysis_tool_hashes'].items():
        if hashlib.sha256((run / 'analysis_tools' / name).read_bytes()).hexdigest() != digest:
            raise ValueError('analysis snapshot changed: ' + name)
    records = [json.loads(line) for p in (run / 'events').glob('*.jsonl') for line in p.read_text().splitlines()]
    if any('error' in e['event'] for e in records):
        raise ValueError('capture error')
    entries = {e['label']: e for e in records if e['event'] in ('enter', 'dispatch_enter')}
    returns = {e['label']: e for e in records if e['event'] in ('exit', 'dispatch_exit')}
    if set(entries) != set(returns):
        raise ValueError('unbalanced capture')
    trace = read_trace(next((run / 'profiler').rglob('trace_view.json')))
    for node in graph['nodes']:
        event = trace[node['trace_index']]
        if node['name'] != event['name'] or node['duration_ns'] != ns(event['dur']):
            raise ValueError('kernel inventory/timing mismatch')
    for phase in ('prefill', 'decode-1', 'decode-2', 'decode-3'):
        nodes = [n for n in graph['nodes'] if n['phase'] == phase and n['in_forward']]
        counts = Counter(n['name'] for n in nodes)
        for kernel in ('_triton_rope', 'ReshapeAndCacheNdKernel', 'FusedInferAttentionScore'):
            if counts[kernel] != 24:
                raise ValueError('not all 24 layers covered: ' + phase + '/' + kernel)
    if graph['audit']['unattributed_task_names'] != {'EVENT_RECORD': 8}:
        raise ValueError('unattributed memory/compute tasks remain')
    if graph['audit']['unknown_contract_tasks']:
        raise ValueError('unknown access contracts remain')
    # Regression guard for the concrete outer-custom-op blind spot: the final
    # attention output copy MUST read the FIA temporary, not just outer Q/K/V.
    raw_pairs = {(e['source'], e['target']) for e in graph['graphs']['projected'] if e['kind'] == 'RAW'}
    copies = []
    for c in graph['contracts']:
        entry = entries[c['scope']]
        if entry.get('operator') == 'aten.copy_.default':
            src = next(a['value'] for a in entry['arguments'] if a['name'] == 'src')
            copies.append((entry, c['kernels'][0], list(tensors(src))))
    verified_fia_copies = 0
    for c in graph['contracts']:
        if c['operator'] != 'npu::npu_fused_infer_attention_score':
            continue
        entry = entries[c['scope']]
        output = next(tensors(returns[c['scope']]['returned']))
        matches = [(copy, node) for copy, node, sources in copies
                   if copy['step'] == entry['step'] and copy['monotonic_ns'] > entry['monotonic_ns']
                   and any(t['storage_generation'] == output['storage_generation']
                           and t['device'] == output['device'] for t in sources)]
        if not matches or not any((c['kernels'][-1], node) in raw_pairs for _, node in matches):
            raise ValueError('missing actual FIA temporary -> output copy RAW dependency')
        verified_fia_copies += 1
    if verified_fia_copies != 96:
        raise ValueError('not all FIA output copies covered')
    kv = KVIndices(records, entries, returns)
    if len(kv.slots) != 4:
        raise ValueError('missing KV input reconstruction')
    slots = [r['values'] for r in kv.slots]
    first = slots[0][0]
    if slots != [list(range(first, first + 10)), [first + 10], [first + 11], [first + 12]]:
        raise ValueError('KV slots not consistent with prefill/decode positions')
    checked = 0
    for name, variants in graph['analysis'].items():
        nodes = graph['nodes'] if name == 'request' else [n for n in graph['nodes']
                if n['in_forward'] and name == n['phase'] + '/forward']
        ids = {n['id'] for n in nodes}
        for kind, analysis in variants.items():
            edges = [e for e in graph['graphs'][kind] if e['source'] in ids and e['target'] in ids]
            topological(nodes, edges)
            if metrics(nodes, edges) != analysis['metrics']:
                raise ValueError('metric mismatch')
            for plan in analysis['schedules']:
                validate_schedule(nodes, edges, plan)
                checked += 1
    audit = graph['audit']
    if audit['complete_exact_data_dag'] or audit['live_stream_reassignment_eligible']:
        raise ValueError('native workspace gap must not pass completeness gate')
    if baseline:
        previous = json.loads((baseline / 'analysis/execution_graph.json').read_text())
        expected = Counter((n['phase'], n['name']) for n in previous['nodes'] if n['kind'] == 'kernel')
        if Counter((n['phase'], n['name']) for n in graph['nodes']) != expected:
            raise ValueError('baseline device inventory changed')
        now, old = [json.loads((r / 'response.json').read_text()) for r in (run, baseline)]
        if now['choices'][0]['text'] != old['choices'][0]['text'] or now['usage'] != old['usage']:
            raise ValueError('baseline output changed')
    return dict(tasks=len(graph['nodes']), phases=4, layers_per_phase=24,
                verified_fia_output_copy_dependencies=verified_fia_copies,
                schedules_validated=checked, kv_slot_sequence=slots,
                baseline_matched=baseline is not None, complete_exact_data_dag=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--baseline', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.run, args.baseline), indent=2))
