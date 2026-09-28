"""Export an offline interactive report without raw pointers, hostnames or logs."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from dag import reachability


def reduce_edges(nodes, edges):
    """Unique-pair transitive reduction for display; analysis uses original edges."""
    _, _, pairs = reachability(nodes, edges)
    kinds = defaultdict(set)
    for e in edges:
        pair = (e['source'], e['target'])
        if pair in pairs:
            kinds[pair].add(e['kind'])
    return [dict(source=a, target=b, kinds=sorted(kinds[(a, b)])) for a, b in sorted(pairs)]


def export(graph):
    owners = {c['scope']: f'g{i:04d}' for i, c in enumerate(graph['contracts'])}
    nodes = [{k: n[k] for k in ('id', 'name', 'duration_ns', 'phase', 'in_forward', 'stream')}
             | {'group': owners.get(n['owner'])} for n in graph['nodes']]
    groups = [{k: c[k] for k in ('kind', 'operator', 'status', 'kernels', 'read_ranges', 'write_ranges', 'exact_per_kernel')}
              | {'id': owners[c['scope']]} for c in graph['contracts']]
    return dict(nodes=nodes, groups=groups, audit=graph['audit'], assumptions=graph['assumptions'],
                analysis=graph['analysis'],
                graphs={kind: reduce_edges(nodes, es) for kind, es in graph['graphs'].items()},
                edge_counts={kind: dict(Counter(e['kind'] for e in es)) for kind, es in graph['graphs'].items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('graph', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    graph = json.loads(args.graph.read_text())
    data = export(graph)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = dict(audit=data['audit'], edge_counts=data['edge_counts'],
                   assumptions=data['assumptions'], analysis={})
    for name, variants in data['analysis'].items():
        summary['analysis'][name] = {}
        for kind, value in variants.items():
            m = value['metrics']
            summary['analysis'][name][kind] = dict(nodes=value['nodes'], edges=value['edges'],
                measured_window_ns=value['measured_window_ns'],
                **{k: m[k] for k in ('work_ns', 'critical_path_ns', 'work_over_span', 'asap_peak_active')},
                schedules=[{k: s[k] for k in ('stream_count', 'event_cost_ns', 'makespan_ns', 'lower_bound_ns')}
                           | {'event_waits': len(s['events'])} for s in value['schedules']])
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    payload = json.dumps(data, ensure_ascii=False, separators=(',', ':')).replace('<', '\\u003c')
    template = Path(__file__).with_name('report.html').read_text()
    (args.output / 'index.html').write_text(template.replace('__DATA__', payload))
    print(args.output / 'index.html')


if __name__ == '__main__':
    main()
