"""Readable first-attention graph exports, selected from actual graph edges."""
import json
import shutil
import subprocess


def export_focus(graph, output):
    nodes = {n['id']: n for n in graph['nodes']}
    for phase in ('prefill', 'decode-1'):
        links = [e for e in graph['edges'] if e['kind'] == 'data_contract' and
                 nodes[e['target']]['phase'] == phase]
        if not links:
            continue
        first = min(links, key=lambda e: nodes[e['target']]['trace_index'])
        scope = first['evidence']['scope']
        selected = {n['id'] for n in graph['nodes'] if scope in n.get('scopes', [])}
        selected.add(first['source'])
        for e in graph['edges']:
            if e['kind'] == 'launch' and e['target'] in selected:
                selected.add(e['source'])
        for e in graph['edges']:
            if e['kind'] == 'dispatch' and e['target'] in selected:
                selected.add(e['source'])
        lines = ['digraph execution {', 'rankdir=TB;',
                 'graph [label="'+phase+' · first attention · observed execution", labelloc=t];',
                 'node [shape=box, style=rounded, fontname="sans-serif", fontsize=10];',
                 'edge [fontname="sans-serif", fontsize=8];']
        for kind in ('host', 'submission', 'kernel'):
            lines.append('subgraph cluster_'+kind+' { label='+json.dumps(kind)+';')
            for ident in sorted(selected):
                n = nodes[ident]
                if n['kind'] == kind:
                    label = n['name']
                    if kind == 'kernel':
                        label += '\nstream {} / task {}'.format(n['stream'], n['task_id'])
                    lines.append('{} [label={}];'.format(json.dumps(ident), json.dumps(label)))
            lines.append('}')
        colors = dict(stream_order='#607e9e', data_contract='#098366', storage_candidate='#ba862e')
        for e in graph['edges']:
            if e['source'] in selected and e['target'] in selected:
                lines.append('{} -> {} [label={}, color={}, style={}];'.format(
                    json.dumps(e['source']), json.dumps(e['target']), json.dumps(e['kind']),
                    json.dumps(colors.get(e['kind'], '#7d8792')),
                    'dashed' if e['kind'] == 'storage_candidate' else 'solid'))
        lines.append('}')
        dot = output / ('first_attention_'+phase+'.dot')
        dot.write_text('\n'.join(lines)+'\n')
        if shutil.which('dot'):
            subprocess.run(['dot', '-Tsvg', str(dot), '-o', str(dot.with_suffix('.svg'))], check=True)
