"""Non-overlapping forward scopes, followed by one precisely joined RoPE probe."""
import argparse
from collections import defaultdict
from decimal import Decimal as D
import hashlib
from html import escape
import json
from pathlib import Path


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def end(e):return D(str(e['ts']))+D(str(e['dur']))


def forest(events, begin, finish, tid):
    """Require true same-thread nesting; never sum a parent and its children."""
    begin,finish=D(str(begin)),D(str(finish))
    nodes=[];stack=[];roots=[]
    for raw in sorted(events,key=lambda e:(D(str(e['ts'])),-D(str(e['dur'])),str(e['id']))):
        if raw['tid']!=tid:raise ValueError('cross-thread event in CPU partition')
        a,b=D(str(raw['ts'])),end(raw)
        if not begin<=a<=b<=finish:raise ValueError('event outside forward scope')
        while stack and a>=D(nodes[stack[-1]]['end_us']):stack.pop()
        if stack and b>D(nodes[stack[-1]]['end_us']):raise ValueError('partially overlapping CPU events')
        node=dict(id=raw['id'],name=raw['name'],start_us=str(a),end_us=str(b),
            duration_us=str(b-a),parent=stack[-1] if stack else None,children=[],
            kind=raw.get('kind','cpu_op'))
        idx=len(nodes)
        if stack:nodes[stack[-1]]['children'].append(idx)
        else:roots.append(idx)
        nodes.append(node);stack.append(idx)
    for n in nodes:
        n['self_us']=str(D(n['duration_us'])-sum((D(nodes[i]['duration_us']) for i in n['children']),D(0)))
        if D(n['self_us'])<0:raise ValueError('negative self duration')
    covered=sum((D(nodes[i]['duration_us']) for i in roots),D(0))
    residual=finish-begin-covered
    if residual<0:raise ValueError('partition exceeds forward')
    assert sum((D(n['self_us']) for n in nodes),D(0))==covered
    return dict(nodes=nodes,roots=roots,denominator_us=str(finish-begin),
        covered_us=str(covered),unattributed_us=str(residual))


def partition(root):
    data=read(root/'analysis/evidence.json')
    f=next(p for p in data['phases'] if p['index']==32 and p['kind']=='forward')
    lo,hi=D(f['trace_start_us']),D(f['trace_end_us'])
    events=[]
    for op in data['focused_cpu_ops']:
        e=op['event']
        if e['tid']==f['tid'] and lo<=D(str(e['ts'])) and end(e)<=hi:
            events.append(dict(e,id='cpu:'+str(op['index'])))
    for p in data['phases']:
        if p['kind']=='replay' and p['index']==32:
            events.append(dict(id=p['label'],name='NPUGraph.replay scope',kind='replay',tid=p['tid'],
                ts=p['trace_start_us'],dur=str(D(p['trace_end_us'])-D(p['trace_start_us']))))
    tree=forest(events,lo,hi,f['tid'])
    groups=defaultdict(lambda:dict(calls=0,duration_us=D(0),roots=[]))
    for idx in tree['roots']:
        node=tree['nodes'][idx];g=groups[node['name']]
        g['calls']+=1;g['duration_us']+=D(node['duration_us']);g['roots'].append(idx)
    rows=[dict(name=name,**g,percent=float(g['duration_us']/(hi-lo)*100)) for name,g in groups.items()]
    rows.sort(key=lambda x:-x['duration_us'])
    rows.append(dict(name='未被这些范围覆盖的时间',calls=None,duration_us=D(tree['unattributed_us']),
        percent=float(D(tree['unattributed_us'])/(hi-lo)*100),roots=[]))
    attention=defaultdict(lambda:dict(calls=0,duration_us=D(0)))
    for i in tree['roots']:
        node=tree['nodes'][i]
        if node['name']!='vllm::unified_attention_with_output':continue
        attention['attention self（未被子范围解释）']['calls']+=1
        attention['attention self（未被子范围解释）']['duration_us']+=D(node['self_us'])
        for j in node['children']:
            child=tree['nodes'][j];g=attention[child['name']]
            g['calls']+=1;g['duration_us']+=D(child['duration_us'])
    return dict(run=root.name,mode=data['config']['mode'],phase=f,tree=tree,groups=rows,
        attention_children=[dict(name=n,**g) for n,g in sorted(attention.items(),key=lambda x:-x[1]['duration_us'])],
        evidence_sha256=sha(root/'analysis/evidence.json'),
        limits='Percentages use the forward profiler scope, including marker overhead. This differs from the inner dual-clock measurement. These are Host ranges, not kernel compute time.')


def rope_detail(root):
    data=read(root/'analysis/evidence.json');meta=read(root/'diagnostic/forward_detail.json')
    phases=[p for p in data['phases'] if p['kind'].startswith('rope_')]
    assert {p['kind'] for p in phases}=={'rope_python','rope_jit','rope_bind','rope_native'}
    assert len(phases)==4 and all(p['index']==32 for p in phases)
    native=next(p for p in phases if p['kind']=='rope_native')
    tasks=[t for t in data['tasks'] if t['scope']==native['label'] and t['is_compute']]
    assert len(tasks)==1 and 'rope' in tasks[0]['name'].lower()
    task=tasks[0];queue=next(q for q in data['queues'] if q['id']==task['queue_id'])
    assert task['host_index'] is not None and task['cann_index'] is not None
    host=data['host_events'][str(task['host_index'])];cann=data['host_events'][str(task['cann_index'])]
    lo,hi=D(native['trace_start_us']),D(native['trace_end_us'])
    assert lo<=D(str(queue['enqueue']['ts']))<=end(queue['enqueue'])<=hi
    ops=[o for o in data['focused_cpu_ops'] if o['event']['name']=='vllm::npu_rotary_embedding']
    selected=[o for o in ops if D(str(o['event']['ts']))<=lo and hi<=end(o['event'])]
    assert len(selected)==1
    parent=selected[0]['event']
    others=sorted(D(str(o['event']['dur'])) for o in ops if o!=selected[0])
    assert len(others)==23
    interval=lambda e:dict(name=e['name'],start_us=str(e['ts']),end_us=str(end(e)),tid=e['tid'],duration_us=str(e['dur']))
    timings=dict(native_scope_before_enqueue_us=str(D(str(queue['enqueue']['ts']))-lo),
        native_profiler_scope_us=str(hi-lo),
        enqueue_duration_us=str(queue['enqueue']['dur']),
        native_scope_after_enqueue_us=str(hi-end(queue['enqueue'])),
        selected_outer_rope_us=str(parent['dur']),other_23_rope_median_us=str(others[11]),
        other_23_rope_min_us=str(others[0]),other_23_rope_max_us=str(others[-1]),
        kernel_duration_us=task['duration_us'])
    return dict(run=root.name,metadata=meta,phases=phases,task=task,queue=queue,
        parent=interval(parent),host=interval(host),cann=interval(cann),timings=timings,
        source_manifest=data['source_manifest'],evidence_sha256=sha(root/'analysis/evidence.json'))


def rope_svg(detail):
    phases=detail['phases'];q=detail['queue'];t=detail['task'];bars=[]
    for row,kind in enumerate(('rope_python','rope_jit','rope_bind','rope_native')):
        p=next(p for p in phases if p['kind']==kind)
        bars.append((row,D(p['trace_start_us']),D(p['trace_end_us']),kind))
    for row,e in ((4,q['enqueue']),(5,q['dequeue'])):bars.append((row,D(str(e['ts'])),end(e),e['name']))
    bars.extend([(6,D(detail['cann']['start_us']),D(detail['cann']['end_us']),detail['cann']['name']),
                 (7,D(t['start_us']),D(t['end_us']),t['name'])])
    lo=min(a for _,a,_,_ in bars);hi=max(b for _,_,b,_ in bars)
    x=lambda v:230+float((v-lo)/(hi-lo))*1010
    out=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1280 510">',
        '<style>text{font:14px sans-serif;fill:#203545}</style><rect width="1280" height="510" fill="#f4f7fb"/>',
        '<text x="20" y="30" style="font-size:22px">一次 RoPE：Python → JIT → native launcher → 队列 → NPU</text>',
        '<text x="20" y="57">新增观测会扰动这一次调用；连线表示已验证关联，不表示逐指令因果或纯排队等待。</text>']
    labels=['rope_forward_triton','JITFunction.run','generated binder','native launcher','CPU Enqueue','下发线程 Dequeue','CANN launch','NPU kernel']
    for row,label in enumerate(labels):out.append(f'<text x="15" y="{105+row*46}">{label}</text>')
    for row,a,b,name in bars:
        out.append(f'<rect x="{x(a):.3f}" y="{88+row*46}" width="{max(.5,x(b)-x(a)):.3f}" height="23" fill="{ "#248c82" if row==7 else "#386eba"}"><title>{escape(name)} · {b-a} µs</title></rect>')
    for i in range(4,7):
        a,b=bars[i],bars[i+1]
        out.append(f'<path d="M{x(a[1]):.3f} {111+a[0]*46} L{x(b[1]):.3f} {88+b[0]*46}" stroke="#bc4260" stroke-width="1.5"/>')
    for i in range(6):out.append(f'<text x="{230+1010*i/5}" y="495">{float(hi-lo)*i/5:.1f} µs</text>')
    return '\n'.join(out+['</svg>'])


def render(data,out):
    page=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>Practice 30 · forward 内部细分</title><style>body{font:16px/1.7 system-ui;max-width:1250px;margin:24px auto;padding:18px;color:#203545}table{border-collapse:collapse;width:100%}td,th{border:1px solid #d2dce8;padding:9px;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f5fa;padding:15px}svg{width:100%}details{margin:14px 0}a{color:#2457ac}</style>',
        '<h1>forward 内部：已有 trace 的占比 + 一次 RoPE 细读</h1>',
        '<p>第一部分复用 eager-02 / graph-01，不增加测量；第二部分仅放大一个新请求 decode 32 的第一次 RoPE。Host 范围不是设备计算时间；父子时长不能重复累加。</p>']
    for mode,result in data['partitions'].items():
        tree=result['tree'];f=result['phase']
        page.append(f'<h2>{mode} · {result["run"]} · decode 32</h2><p>占比分母：forward profiler 范围 {tree["denominator_us"]} µs；范围内部双时钟：墙钟 {f["wall_us"]:.3f} / 线程 CPU {f["thread_cpu_us"]:.3f} µs。两种边界差异来自标记等观测开销，不混算。</p>')
        page.append('<table><tr><th>互不重叠的顶层范围</th><th>次数</th><th>Host 墙钟 µs</th><th>占 forward</th></tr>')
        for g in result['groups']:page.append(f'<tr><td>{escape(g["name"])}</td><td>{g["calls"] if g["calls"] is not None else "—"}</td><td>{float(g["duration_us"]):.3f}</td><td>{g["percent"]:.2f}%</td></tr>')
        page.append('</table><p>未覆盖部分可能包含框架衔接、参数处理和观测开销；不能全部称为 Python 或 CPU 等待。</p>')
        page.append('<details><summary>24 次图外 attention 的直接子范围合计（仍是 Host 时间）</summary><table><tr><th>直接子范围或剩余</th><th>次数</th><th>Host µs</th></tr>')
        for g in result['attention_children']:page.append(f'<tr><td>{escape(g["name"])}</td><td>{g["calls"]}</td><td>{float(g["duration_us"]):.3f}</td></tr>')
        page.append('</table></details>')
        page.append('<details><summary>展开第一处 attention 的包含关系（self 已扣除直接子范围）</summary><table><tr><th>调用</th><th>inclusive µs</th><th>self µs</th></tr>')
        attention=next(i for i in tree['roots'] if tree['nodes'][i]['name']=='vllm::unified_attention_with_output')
        def visit(idx,depth=0):
            n=tree['nodes'][idx]
            page.append(f'<tr><td>{"　"*depth}{escape(n["name"])}</td><td>{n["duration_us"]}</td><td>{n["self_us"]}</td></tr>')
            for c in n['children']:visit(c,depth+1)
        visit(attention);page.append('</table></details>')
    if data.get('rope'):
        d=data['rope'];page.append('<h2>只细读一次 RoPE</h2><p>这次新增包装只在 decode 32 的第一次 RoPE 内计时。预热缓存保持不变，无选中调用内编译；结果和恢复验证见原始记录。native 的内部细节以 profiler 事件为限。</p>')
        page.append('<a href="rope.svg">打开精确时间线 SVG</a>'+rope_svg(d))
        page.append('<table><tr><th>包装范围</th><th>墙钟 µs</th><th>线程 CPU µs</th><th>自身墙钟 µs</th></tr>')
        for p in d['phases']:page.append(f'<tr><td>{p["kind"]}</td><td>{p["wall_us"]:.3f}</td><td>{p["thread_cpu_us"]:.3f}</td><td>{p["self_wall_us"]:.3f}</td></tr>')
        page.append('</table><p>自身时间仍包含子包装边界成本；单次、微秒级诊断值不可作为稳定性能占比。下面的入队前/入队/入队后使用 native profiler 范围，不能与上表内部双时钟范围混算。新旧两轮不相减归因。</p><pre>'+escape(json.dumps(d['timings'],indent=2))+'</pre>')
        page.append('<details><summary>精确设备/队列身份、源码入口和观测边界</summary><pre>'+escape(json.dumps(dict(task=d['task'],queue=d['queue'],sources=[p.get('source') for p in d['phases']],metadata=d['metadata']),ensure_ascii=False,indent=2))+'</pre></details>')
        (out/'rope.svg').write_text(rope_svg(d))
    page.append('<p><a href="forward.json">完整机器可读证据</a> · <a href="../comparison/index.html">上一轮 eager / graph 对照</a></p></html>')
    (out/'index.html').write_text('\n'.join(page))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--eager',type=Path,required=True);p.add_argument('--graph',type=Path,required=True)
    p.add_argument('--detail',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    data=dict(partitions={mode:partition(root) for mode,root in [('eager',a.eager),('graph',a.graph)]})
    if a.detail:data['rope']=rope_detail(a.detail)
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'forward.json').write_text(json.dumps(data,ensure_ascii=False,default=str,indent=2)+'\n')
    render(data,a.output)
    print(json.dumps({mode:r['groups'] for mode,r in data['partitions'].items()},ensure_ascii=False,default=str,indent=2))


if __name__=='__main__':main()
