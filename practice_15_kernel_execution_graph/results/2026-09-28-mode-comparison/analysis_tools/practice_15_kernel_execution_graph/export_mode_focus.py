"""Export evidence-backed first-attention SVGs for every captured model step.

Execution edges are an induced subset of the validated execution graph. Optional
replay/attention tensor bindings prove equal scoped views, NOT device completion,
per-kernel producer identity, or a hardware synchronization dependency.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import textwrap
from xml.etree import ElementTree as ET

from build_model_graph import validate_dag, same_view
from build_mode_graph import analyze, require, only, number

PHASES = ('prefill', 'decode-1', 'decode-2', 'decode-3')
LAYER = 'model.layers.0.self_attn.attn'
EXECUTION_KINDS = {'launch', 'dispatch', 'runtime_connection', 'replay_runtime_call', 'replay_dispatch'}


def equal_view(a, b):
    require(a.get('kind') == b.get('kind') == 'tensor', 'binding needs actual tensor metadata')
    require(same_view(a, b) and a['dtype'] == b['dtype'] and a['storage_offset'] == b['storage_offset'], 'focus tensor view mismatch')


def focus(graph, phase):
    require(phase in PHASES, 'unknown phase')
    nodes = {n['id']: n for n in graph['nodes']}
    scopes = graph['parameter_scopes']
    chain = only([c for c in graph['attention_chains'] if c['phase'] == phase and c['layer'] == LAYER], 'first attention chain')
    attention_label = only([s for s in nodes[chain['cache']]['scopes'] if scopes[s]['entry']['kind'] == 'attention'], 'attention scope')
    attention = scopes[attention_label]['entry']
    selected = {n['id'] for n in graph['nodes'] if attention_label in n.get('scopes', [])}
    # Direct RoPE inclusion requires the previously verified scoped RAW edge.
    selected.update(e['source'] for e in graph['edges'] if e['kind'] == 'data_contract' and e['target'] in selected)
    bindings, replays = [], []
    if graph['summary']['mode'] == 'graph' and phase != 'prefill':
        for partition in ('submod_0', 'submod_2'):
            r = only([r for r in graph['replays'] if r['phase'] == phase and r['partition'] == partition], 'focus replay partition')
            replay_node = nodes['replay:' + r['label']]
            entry = scopes[r['label']]['entry']
            dispatch = scopes[entry['parent']]
            require(dispatch['entry']['parent'] == attention['parent'] and
                    dispatch['entry']['pid'] == attention['pid'] and dispatch['entry']['tid'] == attention['tid'] and
                    dispatch['entry']['step'] == attention['step'], 'replay/attention scope identity mismatch')
            require(r['resources_verified'] and entry['graph_id'] == replay_node['resources']['graph_id'], 'unverified replay resources')
            if partition == 'submod_0':
                require(dispatch['exit']['monotonic_ns'] <= attention['monotonic_ns'], 'producer dispatch must precede attention call')
                outputs = replay_node['resources']['output']
                require(outputs == dispatch['exit']['returned'] and len(outputs) == 5, 'returned partition output mismatch')
                for index, name in enumerate(('query', 'key', 'value', 'output')):
                    equal_view(outputs[index], attention['arguments'][name])
                    bindings.append(dict(name=name, left_scope=entry['parent'], left_field='returned[{}]'.format(index),
                                         right_scope=attention_label, right_field='arguments.' + name,
                                         tensor=outputs[index],
                                         semantics='same scoped tensor view; output is a supplied buffer, not proof this replay writes it' if name == 'output' else
                                         'same scoped tensor view; producer kernel inside replay is not identified'))
            else:
                require(scopes[attention_label]['exit']['monotonic_ns'] <= dispatch['entry']['monotonic_ns'], 'consumer dispatch must follow attention return')
                equal_view(attention['arguments']['output'], replay_node['resources']['inputs'][0])
                bindings.append(dict(name='attention_output', left_scope=attention_label, left_field='arguments.output',
                                     right_scope=entry['parent'], right_field='resources.inputs[0]',
                                     tensor=attention['arguments']['output'],
                                     semantics='same scoped output/input view; host return is not NPU completion'))
            replays.append(dict(replay=r, node=replay_node, dispatch_scope=entry['parent']))
            selected.add(replay_node['id'])
            selected.update(r['boundary_tasks'])
    # Close only over explicit incoming attribution edges. Do not include tasks
    # by nearest timestamp or by matching an unproven graph/Model ID namespace.
    changed = True
    while changed:
        before = set(selected)
        selected.update(e['source'] for e in graph['edges'] if e['kind'] in EXECUTION_KINDS and e['target'] in selected)
        changed = before != selected
    selected_nodes = [n for n in graph['nodes'] if n['id'] in selected]
    selected_edges = [dict(e, original_edge_index=i) for i, e in enumerate(graph['edges']) if e['source'] in selected and e['target'] in selected]
    validate_dag(selected_nodes, selected_edges)
    for n in selected_nodes:
        if n['kind'] == 'kernel':
            require(n['phase'] == phase, 'cross-phase focus task')
            require(n['flow_evidence']['torch']['status'] == n['flow_evidence']['cann']['status'] == 'exact' or
                    'runtime_connection' in n, 'unattributed device task cannot enter exact focus')
    unknown = [n['id'] for n in graph['nodes'] if n['kind'] == 'kernel' and n['phase'] == phase and
               n['flow_evidence']['torch']['status'] != 'exact' and 'runtime_connection' not in n]
    return dict(mode=graph['summary']['mode'], phase=phase, layer=LAYER,
                nodes=selected_nodes, edges=selected_edges, attention_scope=attention_label,
                attention_observation=scopes[attention_label], replay_context=replays,
                tensor_bindings=bindings, unknown_phase_tasks_not_assigned_to_layer=unknown,
                evidence_boundary='Execution edges retain original evidence and semantics. Tensor bindings are separate metadata associations, not execution edges.',
                source_provenance=graph['provenance'],
                exporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def wrapped(value, width=46):
    return '\n'.join(textwrap.wrap(value, width=width, break_long_words=True, break_on_hyphens=False))


def dot_text(data):
    q = lambda x: json.dumps(str(x), ensure_ascii=False)
    lines = ['digraph execution {', 'rankdir=TB;',
             'graph [fontname="sans-serif", fontsize=16, labelloc=t, nodesep=0.25, ranksep=0.6, pad=0.25,',
             'label=' + q('{} / {} · layer 0 attention\nExact observed attribution; node distance is not elapsed time'.format(data['mode'], data['phase'])) + '];',
             'node [shape=box, style="rounded,filled", fillcolor="#f1f6fc", color="#6687a5", fontname="sans-serif", fontsize=10];',
             'edge [fontname="sans-serif", fontsize=8, arrowsize=0.7];']
    byid = {n['id']: n for n in data['nodes']}
    clusters = [('cpu', 'CPU calls / replay (return does not mean NPU completion)', {'host', 'replay'}),
                ('runtime', 'CANN / native runtime submission', {'submission'}),
                ('device', 'NPU tasks / physical streams', {'kernel'})]
    for ident, title, kinds in clusters:
        lines.append('subgraph cluster_{} {{ label={}; color="#b9c8d7";'.format(ident, q(title)))
        for n in sorted(data['nodes'], key=lambda n: (number(n['start_us']), n['id'])):
            if n['kind'] not in kinds:
                continue
            label = wrapped(n['name'])
            if n['kind'] == 'kernel':
                label += '\nstream {} / task {}\n{} us'.format(n['stream'], n['task_id'], number(n['end_us']) - number(n['start_us']))
                if 'runtime_connection' in n:
                    label += '\nconnection=' + str(n['runtime_connection']['connection_id'])
            elif n['kind'] == 'replay':
                label += '\ngraph object=' + str(n['resources']['graph_id'])
            else:
                label += '\nthread=' + str(n['tid'])
            color = '#eaf3ff' if n['kind'] != 'replay' else '#eee8fa'
            if n['kind'] == 'kernel' and n['name'] in ('MODEL_EXECUTE', 'NOTIFY_WAIT'):
                color = '#eee8fa'
            lines.append('{} [id={}, label={}, fillcolor={}, tooltip={}];'.format(q(n['id']), q(n['id']), q(label), q(color), q(json.dumps(n, ensure_ascii=False))))
        lines.append('}')
    colors = dict(stream_order='#607e9e', data_contract='#087d58', storage_candidate='#b58024',
                  runtime_connection='#7851a9', replay_runtime_call='#7851a9', dispatch='#66717e', launch='#66717e')
    for i, e in enumerate(data['edges']):
        label = e['kind']
        proof = e['evidence']
        if e['kind'] == 'launch':
            label += '\nflow=' + str(proof['flow_id'])
        elif e['kind'] == 'dispatch':
            label += '\nflow=' + str(proof['torch']['flow_id'])
            if proof.get('queue'):
                label += '\nqueue=' + str(proof['queue']['correlation_id'])
        elif e['kind'] == 'runtime_connection':
            label += '\nid=' + str(proof['connection_id'])
        elif e['kind'] == 'storage_candidate':
            label += '\n' + str(proof['resource']) + ' pool (indexed overlap unknown)'
        style = 'dashed' if e['kind'] in ('storage_candidate', 'host_issue_order', 'submission_issue_order') else 'solid'
        lines.append('{} -> {} [id={}, label={}, color={}, style={}, tooltip={}];'.format(
            q(e['source']), q(e['target']), q('evidence-edge-' + str(i)), q(label), q(colors.get(e['kind'], '#a0aab3')), q(style), q(e['semantics'])))
    lines.append('}')
    return '\n'.join(lines) + '\n'


def add_panels(path, data):
    """Stack non-execution annotations below the graph without invisible edges."""
    uri = 'http://www.w3.org/2000/svg'
    ET.register_namespace('', uri)
    ET.register_namespace('xlink', 'http://www.w3.org/1999/xlink')
    tree = ET.parse(str(path))
    root = tree.getroot()
    _, _, width, height = map(float, root.attrib['viewBox'].split())
    width = max(width, 1100)
    panel = ET.SubElement(root, '{' + uri + '}g', {'id':'evidence-notes'})
    y = height + 12
    def box(x, y, w, h, color):
        ET.SubElement(panel, '{' + uri + '}rect', dict(x=str(x), y=str(y), width=str(w), height=str(h), fill=color, rx='5'))
    def text(x, y, value, size=12):
        element = ET.SubElement(panel, '{' + uri + '}text', {'x':str(x), 'y':str(y), 'font-family':'sans-serif', 'font-size':str(size), 'fill':'#193247'})
        element.text = value
    notes = ['Gray: exact call attribution / stream order. Green: scoped RAW. Amber dashed: shared KV pool, indexed overlap unknown.',
             'Purple: exact replay/runtime connection. CPU entry order and host return do NOT establish device completion.',
             'Every execution edge is copied from execution_graph.json; adjacent JSON retains full evidence and original edge indices.']
    if data['unknown_phase_tasks_not_assigned_to_layer']:
        notes += ['{} graph-internal tasks in this decode window have NO proven first-layer/replay attribution and are not assigned here.'.format(len(data['unknown_phase_tasks_not_assigned_to_layer'])),
                  'No nearest-time joins; no unproven cross-stream NOTIFY matching. Missing edges do not imply independence.']
    box(10, y, width - 20, len(notes) * 20 + 20, '#fff4de')
    for line in notes:
        y += 20
        text(20, y, line)
    y += 35
    if data['tensor_bindings']:
        text(20, y, 'Scoped tensor view equality: METADATA ONLY, not synchronization or per-kernel producer attribution', 14)
        y += 15
        columns = [20, width * .3, width * .58]
        box(10, y, width - 20, 30, '#dcecf1')
        for x, label in zip(columns, ['Observed returned/input view', 'Corresponding attention/partition view', 'Exact data_ptr / shape (BF16)']):
            text(x, y + 20, label)
        y += 30
        for i, binding in enumerate(data['tensor_bindings']):
            box(10, y, width - 20, 30, '#f0f7fa' if i % 2 == 0 else '#ffffff')
            left = 'submod_0 returned ' + binding['name'] if i < 4 else 'attention output buffer'
            right = 'attention ' + binding['name'] if i < 4 else 'submod_2 input[0]'
            tensor = binding['tensor']
            for x, label in zip(columns, [left, right, '{} / {}'.format(tensor['data_ptr'], tensor['shape'])]):
                text(x, y + 20, label)
            y += 30
        text(20, y + 20, 'The output buffer is preallocated: returning its view does not mean submod_0 writes the attention result.')
        y += 35
    root.set('viewBox', '0 0 {} {}'.format(width, y + 15))
    root.set('width', str(width) + 'pt')
    root.set('height', str(y + 15) + 'pt')
    tree.write(str(path), encoding='utf-8', xml_declaration=True)


def export_focus(graph, output):
    output.mkdir(parents=True, exist_ok=True)
    executable = shutil.which('dot')
    require(executable is not None, 'Graphviz dot is required for exact SVG exports')
    summary = []
    for phase in PHASES:
        data = focus(graph, phase)
        stem = output / ('first_attention_' + phase)
        stem.with_suffix('.json').write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
        stem.with_suffix('.dot').write_text(dot_text(data))
        subprocess.run([executable, '-Tsvg', str(stem.with_suffix('.dot')), '-o', str(stem.with_suffix('.svg'))], check=True)
        add_panels(stem.with_suffix('.svg'), data)
        summary.append(dict(phase=phase, nodes=len(data['nodes']), execution_edges=len(data['edges']),
                            tensor_bindings=len(data['tensor_bindings']),
                            unassigned_phase_tasks=len(data['unknown_phase_tasks_not_assigned_to_layer']),
                            svg=stem.with_suffix('.svg').name))
    (output / 'focus_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    # Revalidate the original trace, flows, runtime connections and tensor views.
    graph = analyze(args.run)
    print(json.dumps(export_focus(graph, args.run / 'analysis'), ensure_ascii=False))


if __name__ == '__main__':
    main()
