"""Offline CPU-first explorer and two standalone measured SVG timelines."""
import argparse
import base64
from decimal import Decimal as D
import gzip
from html import escape
import json
from pathlib import Path

NAMES={'schedule':'调度', 'state_update':'请求状态', 'prepare_inputs':'输入准备',
 'preprocess':'输入预处理', 'forward':'模型 forward', 'logits':'logits', 'sampling':'采样',
 'token_to_list':'token 回传与等待', 'logprobs_to_cpu':'logprob 回传',
 'scheduler_update':'更新调度器', 'output_processing':'生成输出', 'engine_step':'引擎单步',
 'core_step':'EngineCore 单步','execute':'runner.execute_model','executor_submit':'executor 提交',
 'sample':'sample_tokens','bookkeeping':'结果整理','request':'完整请求',
 'replay':'graph replay','graph_task_update_begin':'graph task update begin',
 'graph_task_update_end':'graph task update end','rope_python':'RoPE Python 包装',
 'rope_jit':'RoPE JIT.run','rope_bind':'RoPE 参数绑定','rope_native':'RoPE native launcher',
 'rope_compile':'RoPE 编译（若发生）'}
LEAVES=('schedule','state_update','prepare_inputs','preprocess','forward','logits','sampling',
        'token_to_list','logprobs_to_cpu','scheduler_update','output_processing')
COLORS={'forward':'#2864c7','logits':'#2864c7','sampling':'#cc7915','token_to_list':'#bd4250',
 'logprobs_to_cpu':'#bd4250','schedule':'#8055a0','scheduler_update':'#8055a0'}


def ns_end(e):return D(e['ts'])+D(str(e['dur']))


def bars(data,index):
    phases=[r for r in data['phases'] if r['index']==index]
    frame=next(r for r in phases if r['kind']=='engine_step')
    out=[]
    def add(row,begin,end,color,label,kind,key):
        out.append(dict(row=row,start=str(begin),end=str(end),color=color,label=label,kind=kind,key=key))
    for r in phases:
        if r['kind'] in LEAVES:
            add(0,r['trace_start_us'],r['trace_end_us'],COLORS.get(r['kind'],'#19887f'),NAMES[r['kind']],'phase',r['id'])
    step=frame['step'];tasks=[t for t in data['tasks'] if t['step']==step]
    graph=data['config']['mode']=='graph'
    streams=sorted({t['stream'] for t in tasks},key=int)
    labels=['CPU 阶段','PyTorch host','CPU Enqueue','下发线程 Dequeue','CANN launch']
    if graph:labels.append('CPU graph replay')
    device_rows={s:len(labels)+i for i,s in enumerate(streams)} if graph else {s:5 for s in streams}
    labels.extend(['NPU stream '+s for s in streams] if graph else ['NPU tasks'])
    sync_row=len(labels);labels.append('CPU 同步调用')
    if graph:
        for r in phases:
            if r['kind']=='replay':add(5,r['trace_start_us'],r['trace_end_us'],'#2864c7',r['uid'],'phase',r['id'])
    for i in sorted({t['host_index'] for t in tasks if t['host_index'] is not None}):
        e=data['host_events'][str(i)];add(1,e['ts'],ns_end(e),'#668da6',e['name'],'host',i)
    for q in data['queues']:
        if q['step']==step:
            for row,field,color in ((2,'enqueue','#657a99'),(3,'dequeue','#826496')):
                e=q[field];add(row,e['ts'],ns_end(e),color,e['name'],'queue',q['id'])
    for i in sorted({t['cann_index'] for t in tasks if t['cann_index'] is not None}):
        e=data['host_events'][str(i)];add(4,e['ts'],ns_end(e),'#976bae',e['name'],'host',i)
    for t in tasks:
        add(device_rows[t['stream']],t['start_us'],t['end_us'],'#1e8c86' if t['is_compute'] else '#c8912d',t['name'],'task',t['id'])
    for s in data['synchronizations']:
        if s['step']==index:
            e=s['event'];add(sync_row,e['ts'],ns_end(e),'#bd4250',e['name'],'sync',s['index'])
    lo=min(D(frame['trace_start_us']),*(D(t['start_us']) for t in tasks))
    hi=max(D(frame['trace_end_us']),*(D(t['end_us']) for t in tasks))
    return dict(index=index,start=str(lo),end=str(hi),bars=out,labels=labels,device_rows=device_rows)


def svg(view,title):
    lo,hi=D(view['start']),D(view['end']);width=1320;height=130+len(view['labels'])*46
    def x(t):return 215+float(D(t)-lo)/float(hi-lo)*1065
    out=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">',
      '<style>text{font:13px sans-serif;fill:#203545}.title{font-size:21px;font-weight:bold}</style>',
      '<rect width="100%" height="100%" fill="#f6f8fb"/>',
      f'<text x="20" y="30" class="title">{escape(title)}</text>',
      '<text x="20" y="54">实测墙钟区间；CPU 调用范围不等于持续运行。蓝：forward；红：结果回收/同步；绿：设备计算。</text>']
    for row,label in enumerate(view['labels']):
        y=80+row*46;out.append(f'<text x="20" y="{y+16}">{escape(label)}</text><path d="M215 {y+26}H1280" stroke="#d5dce5"/>')
    for b in view['bars']:
        xx=x(b['start']);length=max(.2,x(b['end'])-xx);y=80+b['row']*46
        out.append(f'<rect x="{xx:.3f}" y="{y}" width="{length:.3f}" height="22" fill="{b["color"]}" data-kind="{b["kind"]}" data-key="{escape(str(b["key"]))}"><title>{escape(b["label"])} · {float(D(b["end"])-D(b["start"])):.3f} µs</title></rect>')
    for i in range(6):
        out.append(f'<text x="{215+1065*i/5-10}" y="{height-29}">{float(hi-lo)*i/5000:.2f} ms</text>')
    return '\n'.join(out+['</svg>'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();data=json.loads((a.run/'analysis/evidence.json').read_text());summary=json.loads((a.run/'analysis/summary.json').read_text())
    a.output.mkdir(parents=True,exist_ok=True)
    request=next(r for r in data['phases'] if r['kind']=='request')
    overview=dict(start=request['trace_start_us'],end=request['trace_end_us'],
        labels=['调度 / 状态','输入准备','模型 forward / logits','采样','结果回传 / 等待','更新 / 输出'],bars=[])
    row={'schedule':0,'state_update':0,'prepare_inputs':1,'preprocess':1,'forward':2,'logits':2,
         'sampling':3,'token_to_list':4,'logprobs_to_cpu':4,'scheduler_update':5,'output_processing':5}
    for r in data['phases']:
        if r['kind'] in row:overview['bars'].append(dict(row=row[r['kind']],start=r['trace_start_us'],end=r['trace_end_us'],
            color=COLORS.get(r['kind'],'#19887f'),label=f'第 {r["index"]} 步 · {NAMES[r["kind"]]}',kind='phase',key=r['id']))
    mode=data['config']['mode']
    (a.output/'request.svg').write_text(svg(overview,f'Practice 30 · {mode} · 单请求 CPU 阶段总览'))
    (a.output/'decode32.svg').write_text(svg(bars(data,32),f'Practice 30 · {mode} · decode 32：CPU → 队列 → CANN → NPU'))
    # Source excerpts come from this run's actual installed-file snapshots.
    excerpts={}
    for kind,source in data['phase_sources'].items():
        entry=data['source_manifest'].get(source['path'])
        if entry:
            lines=(a.run/entry['path']).read_text().splitlines();line=source['line']
            excerpts[kind]=dict(source,collection=entry.get('collection','before/after audited'),
                text='\n'.join(f'{i+1}: {lines[i]}' for i in range(line-1,min(line+23,len(lines)))))
    # Keep complete evidence in the archive; the viewer needs the queue, phase
    # and task identity but not repeated CSV hardware counters for every kernel.
    payload=dict(summary=summary,phases=data['phases'],tasks=data['tasks'],queues=data['queues'],
        host_events=data['host_events'],synchronizations=data['synchronizations'],examples=data['examples'],
        top_gaps=data['top_gaps'],focused_cpu_ops=data['focused_cpu_ops'],sources=excerpts,names=NAMES,
        views={str(i):bars(data,i) for i in range(64)},overview=overview)
    packed=base64.b64encode(gzip.compress(json.dumps(payload,ensure_ascii=False,separators=(',',':')).encode(),mtime=0)).decode()
    template=Path(__file__).with_name('viewer.html').read_text()
    (a.output/'index.html').write_text(template.replace('__DATA__',packed).replace('__MODE__',mode))


if __name__=='__main__':main()
