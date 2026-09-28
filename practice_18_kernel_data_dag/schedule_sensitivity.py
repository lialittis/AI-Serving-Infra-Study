"""Sweep assumed cross-stream dependency latency; this does not run NPU work."""
import argparse
import json
from pathlib import Path

from dag import schedule_scopes


def sweep(graph, costs):
    rows = []
    for phase in ('prefill', 'decode-1', 'decode-2', 'decode-3'):
        nodes = [n for n in graph['nodes'] if n['in_forward'] and n['phase'] == phase]
        ids = {n['id'] for n in nodes}
        edges = [e for e in graph['graphs']['projected'] if e['source'] in ids and e['target'] in ids]
        work = sum(n['duration_ns'] for n in nodes)
        for cost in costs:
            plans = [schedule_scopes(nodes, edges, count, cost) for count in (1, 2, 4, 8)]
            best = min(plans, key=lambda p: (p['makespan_ns'], p['stream_count']))
            rows.append(dict(phase=phase, assumed_dependency_latency_ns=cost,
                selected_stream_count=best['stream_count'], selected_makespan_ns=best['makespan_ns'],
                work_ns=work, conditional_speedup=work / best['makespan_ns'],
                candidates=[{k: p[k] for k in ('stream_count', 'makespan_ns')}
                            | {'event_waits': len(p['events'])} for p in plans]))
    return dict(kind='offline assumption sensitivity; latency is not measured',
                resource_contention_modeled=False, live_reassignment_eligible=False, rows=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('graph', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--costs-ns', nargs='+', type=int, default=[0, 1000, 5000])
    args = parser.parse_args()
    result = sweep(json.loads(args.graph.read_text()), args.costs_ns)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    for row in result['rows']:
        print(row['phase'], row['assumed_dependency_latency_ns'],
              row['selected_stream_count'], round(row['conditional_speedup'], 5))
