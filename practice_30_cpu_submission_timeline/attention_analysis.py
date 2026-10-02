"""One graph-external attention: nested Host accounting and exact device joins."""
import argparse
from collections import Counter
from decimal import Decimal as D
from html import escape
import json
from pathlib import Path

from forward_analysis import forest, end, sha
from analyze import validate_queue


def read(path):
    return json.loads(path.read_text())


def selected_parent(ops, phase):
    lo, hi = D(phase['trace_start_us']), D(phase['trace_end_us'])
    parents = [o for o in ops if o['event']['name'] == 'vllm::unified_attention_with_output'
               and o['event']['tid'] == phase['tid']
               and D(str(o['event']['ts'])) <= lo and hi <= end(o['event'])]
    if len(parents) != 1:
        raise ValueError('one exact enclosing attention required')
    return parents[0]


def analyze(root):
    data = read(root / 'analysis/evidence.json')
    meta = read(root / 'diagnostic/attention_detail.json')
    assert data['config']['mode'] == 'graph'
    assert meta['selected_calls'] == meta['context_calls'] == 1 and meta['restored']
    assert not meta['call']['capturing'] and meta['context_layer'] == meta['call']['layer_name']
    phases = [p for p in data['phases'] if p['kind'].startswith('attn_')]
    expected = {'attn_context', 'attn_backend', 'attn_kv_prepare', 'attn_kv_submit',
                'attn_dispatch', 'attn_fia', 'attn_fia_params', 'attn_fia_submit'}
    assert len(phases) == 8 and {p['kind'] for p in phases} == expected
    assert all(p['index'] == 32 for p in phases)
    backend = next(p for p in phases if p['kind'] == 'attn_backend')
    selected = selected_parent(data['focused_cpu_ops'], backend)
    parent = selected['event']
    lo, hi = D(str(parent['ts'])), end(parent)
    events = [dict(o['event'], id='cpu:' + str(o['index'])) for o in data['focused_cpu_ops']
              if o['event']['tid'] == backend['tid'] and lo <= D(str(o['event']['ts'])) and end(o['event']) <= hi]
    events += [dict(id=p['label'], name=p['kind'], kind='phase', tid=p['tid'],
                    ts=p['trace_start_us'], dur=str(D(p['trace_end_us']) - D(p['trace_start_us']))) for p in phases]
    tree = forest(events, lo, hi, backend['tid'])
    assert len(tree['roots']) == 1 and D(tree['unattributed_us']) == 0
    method_rows = []
    for n in tree['nodes']:
        if n['kind'] != 'phase' and n['name'] != parent['name']:
            continue
        children = [tree['nodes'][i] for i in n['children']]
        phase_us = sum((D(c['duration_us']) for c in children if c['kind'] == 'phase'), D(0))
        op_us = sum((D(c['duration_us']) for c in children if c['kind'] != 'phase'), D(0))
        assert phase_us + op_us + D(n['self_us']) == D(n['duration_us'])
        method_rows.append(dict(name=n['name'], inclusive_us=n['duration_us'],
            child_phase_us=str(phase_us), direct_cpu_op_us=str(op_us), residual_us=n['self_us'],
            direct_cpu_ops=[dict(name=c['name'],duration_us=c['duration_us']) for c in children if c['kind'] != 'phase']))
    labels = {p['label'] for p in phases}
    tasks = [t for t in data['tasks'] if t['scope'] in labels]
    assert tasks and all(t['step'] == backend['step'] and not t.get('replay') for t in tasks)
    queues = {q['id']: q for q in data['queues']}
    links = []
    for task in tasks:
        assert task['host_index'] is not None and task['cann_index'] is not None
        queue = queues[task['queue_id']]
        validate_queue(queue)
        assert queue['scope'] in labels
        links.append(dict(task=task, queue=queue,
            host=data['host_events'][str(task['host_index'])],
            cann=data['host_events'][str(task['cann_index'])]))
    compute = [t for t in tasks if t['is_compute']]
    assert any('ReshapeAndCache' in t['name'] for t in compute)
    assert any('FusedInferAttention' in t['name'] for t in compute)
    # Include all selected-scope queues, even if they produce no device task.
    selected_queues = [q for q in data['queues'] if q['scope'] in labels]
    others = sorted(D(str(o['event']['dur'])) for o in data['focused_cpu_ops']
                    if o['event']['name'] == parent['name'] and o['index'] != selected['index'])
    assert len(others) == 23
    replays = [p for p in data['phases'] if p['kind'] == 'replay' and p['index'] == 32]
    previous = max((p for p in replays if D(p['trace_end_us']) <= lo), key=lambda p:D(p['trace_end_us']))
    following = min((p for p in replays if D(p['trace_start_us']) >= hi), key=lambda p:D(p['trace_start_us']))
    sync = [s for s in data['synchronizations'] if s['event']['tid'] == backend['tid']
            and lo <= D(str(s['event']['ts'])) and end(s['event']) <= hi]
    return dict(run=root.name, metadata=meta, phases=phases, parent=parent,
        tree=tree, method_rows=method_rows, links=links, queues=selected_queues,
        synchronizations=sync, surrounding_replays=dict(previous=previous, following=following),
        observed_cpu_op_counts=dict(Counter(e['name'] for e in events if e.get('kind') != 'phase')),
        overhead=dict(selected_outer_us=str(hi-lo), other_23_median_us=str(others[11]),
            other_23_min_us=str(others[0]), other_23_max_us=str(others[-1]), request=data['overhead']),
        source_manifest=data['source_manifest'], evidence_sha256=sha(root/'analysis/evidence.json'),
        validation=dict(single_call=True, nested_partition=True, exact_queue_and_device_identity=True,
            graph_external_tasks=True, capture_during_measurement=data['graph']['capture_during_measurement'],
            decode32_replays=data['graph']['replay_by_step']['32']),
        limits=['Residual includes Python/framework glue, metadata inspection, wrapper and profiler cost; not pure Python time.',
                'Inner dual clocks and outer profiler boundaries differ; do not mix denominators.',
                'No observed explicit synchronization is not proof that native code never waits.',
                'No changed stream, device waits, operator arguments, or tensor data reads are introduced.',
                'Same-step other layers are an observation-cost indicator, not a matched causal control.'])


def timeline(data):
    bars = []
    for i, p in enumerate(sorted(data['phases'], key=lambda p:p['id'])):
        bars.append((p['kind'], D(p['trace_start_us']), D(p['trace_end_us']), p['label'], '#376cb1'))
    for q in data['queues']:
        for key, label, color in [('enqueue', 'CPU Enqueue', '#8855ad'), ('dequeue', '下发线程 Dequeue', '#8855ad')]:
            e = q[key]
            bars.append((label, D(str(e['ts'])), end(e), f"{e['name']} / {q['id']}", color))
    for link in data['links']:
        e, t = link['cann'], link['task']
        bars.append(('CANN 提交', D(str(e['ts'])), end(e), f"{e['name']} / connection {t['connection_id']}", '#8855ad'))
        bars.append((f"NPU stream {t['stream']}", D(t['start_us']), D(t['end_us']),
                     f"{t['name']} / task {t['task_id']} / {t['duration_us']} us", '#208977' if t['is_compute'] else '#d88b22'))
    labels = list(dict.fromkeys(row for row, *_ in bars))
    lo, hi = min(a for _,a,_,_,_ in bars), max(b for _,_,b,_,_ in bars)
    x = lambda v: 240 + float((v-lo)/(hi-lo))*1015
    height = 155 + len(labels)*39
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1290 {height}">',
        '<style>text{font:13px sans-serif;fill:#203545}</style>',
        f'<rect width="1290" height="{height}" fill="#f5f8fb"/>',
        '<text x="18" y="28" style="font-size:21px">一次 graph 图外 attention：Host 方法 → 队列 → CANN → NPU</text>',
        '<text x="18" y="53">蓝：Host 方法；紫：队列/提交；绿：设备计算；橙：设备拷贝/控制。宽度为实际时间，悬停查看身份。</text>',
        '<text x="18" y="75">只展示选中调用；CPU 和下发线程可重叠，设备任务完成可能晚于 Host 返回。新增 scope 会扰动时间。</text>']
    for i,label in enumerate(labels):
        out.append(f'<text x="12" y="{111+i*39}">{escape(label)}</text>')
    for row,a,b,name,color in bars:
        out.append(f'<rect x="{x(a):.3f}" y="{94+labels.index(row)*39}" width="{max(.5,x(b)-x(a)):.3f}" height="22" fill="{color}"><title>{escape(name)} · {b-a} µs</title></rect>')
    for i in range(6):
        out.append(f'<text x="{240+i*203}" y="{height-18}">{float(hi-lo)*i/5:.1f} µs</text>')
    return '\n'.join(out+['</svg>'])


def render(data, output):
    svg = timeline(data)
    call, fia = data['metadata']['call'], data['metadata']['fia_arguments']
    copy_calls = data['observed_cpu_op_counts'].get('aten::copy_', 0)
    copy_tasks = sum('MEMCPY' in l['task']['name'] for l in data['links'])
    output.mkdir(parents=True, exist_ok=True)
    (output/'attention.svg').write_text(svg)
    out = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>P30 · 一次 graph 图外 attention</title><style>body{font:16px/1.7 system-ui;max-width:1280px;margin:24px auto;padding:18px;color:#203545}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccd8e6;padding:8px;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f5fa;padding:14px}svg{width:100%}details{margin:16px 0}</style>',
        '<h1>一次 graph 图外 attention 的 CPU 调用</h1>',
        '<p>decode 32，第一层，预热后只增加一次细分。调用仍走原生 PIECEWISE 路径，输出与恢复检查见原始证据。Host 时间不等于 NPU 计算或纯 Python 时间。</p>',
        '<p><a href="attention.svg">独立 SVG</a> · <a href="attention.json">完整关联数据</a></p>',
        '<p>实际调用：get_attention_context → backend.forward → reshape_and_cache → forward_impl → forward_fused_infer_attention。FIA 内部先用 _get_fia_params 选择缓存和长度，再调用 torch-npu FIA，随后写回 output。</p>',
        f'<p>本轮 {escape(call["layer_name"])} / {escape(call["attn_state"])}：Q={call["query"]["shape"]}，K={call["key"]["shape"]}，KV 长度={call["seq_lens_list"]}，block size={fia["block_size"]}。长度参数是已有主机列表；_get_fia_params 将 NPU KV pool 变成 view，view 不代表设备数据复制。</p>',
        f'<p>{copy_calls} 个 copy_ Host 调用，关联到 {copy_tasks} 个设备 MEMCPY 任务。源码中内层返回 output，外层再次对 output 赋值；本轮无设备任务的第二次提交与别名自拷贝跳过相容，但未追入 native no-op 判断。选中范围显式 CANN Synchronize 记录数为 {len(data["synchronizations"])}；不能据此断言 native 内部绝无等待。</p>', svg,
        '<h2>函数内部双时钟</h2><table><tr><th>范围</th><th>墙钟 µs</th><th>线程 CPU µs</th><th>扣除子 scope 后墙钟 µs</th></tr>']
    for p in sorted(data['phases'], key=lambda p:p['id']):
        out.append(f'<tr><td>{p["kind"]}</td><td>{p["wall_us"]:.3f}</td><td>{p["thread_cpu_us"]:.3f}</td><td>{p["self_wall_us"]:.3f}</td></tr>')
    out += ['</table><p>自身墙钟仍含子包装边界成本；下面改用 profiler 边界，与本表分开阅读。</p>',
        '<h2>Profiler 嵌套范围分账</h2><table><tr><th>范围</th><th>包含子项 µs</th><th>直接子 scope µs</th><th>直接子 CPU op µs</th><th>尚未解释的自身余量 µs</th></tr>']
    for r in data['method_rows']:
        out.append('<tr>'+''.join(f'<td>{escape(str(r[k]))}</td>' for k in ('name','inclusive_us','child_phase_us','direct_cpu_op_us','residual_us'))+'</tr>')
    out += ['</table><p>每行后三项之和等于包含时间；不同层级不能再次相加。余量包括参数准备、框架与插桩开销，不强行归因。</p>',
        '<h2>精确关联的设备任务</h2><table><tr><th>任务</th><th>stream / task</th><th>connection / queue</th><th>设备 µs</th></tr>']
    for l in data['links']:
        t = l['task']
        out.append(f'<tr><td>{escape(t["name"])}</td><td>{t["stream"]} / {t["task_id"]}</td><td>{t["connection_id"]} / {escape(str(t["queue_id"]))}</td><td>{t["duration_us"]}</td></tr>')
    out.append('</table>')
    for title,key in [('实际分支与参数（仅形状/主机 metadata）','metadata'),('观察扰动','overhead'),
                      ('原生同步记录','synchronizations'),('前后 replay','surrounding_replays'),
                      ('源码位置与详细方法计时','phases'),('完整调用树','tree'),('校验与限制','validation'),('证据边界','limits')]:
        out.append(f'<details><summary>{title}</summary><pre>{escape(json.dumps(data[key],ensure_ascii=False,indent=2,default=str))}</pre></details>')
    (output/'index.html').write_text('\n'.join(out+['</html>']))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();data=analyze(a.run)
    render(data,a.output)
    (a.output/'attention.json').write_text(json.dumps(data,ensure_ascii=False,indent=2,default=str)+'\n')
    print(json.dumps({k:data[k] for k in ('method_rows','overhead','validation')},ensure_ascii=False,indent=2,default=str))


if __name__=='__main__':main()
