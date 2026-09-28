"""Standalone offline timeline and verified block-dependency graph browser."""
import argparse
import json
from decimal import Decimal
from pathlib import Path


HTML='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>P1 · KV 迁移与计算依赖</title><style>
body{margin:0;background:#f4f6f5;color:#18383e;font:15px/1.7 system-ui,sans-serif}main{max-width:1300px;margin:auto;padding:32px 24px}h1{font-size:30px}h2{font-size:21px}section{background:white;border:1px solid #d4deda;border-radius:8px;padding:20px;margin:20px 0}select,button{font:inherit;padding:7px;margin:4px;border:1px solid #aabdb7;border-radius:4px;background:white}label{display:inline-block;margin-right:15px;max-width:100%}select{max-width:100%;box-sizing:border-box}pre{white-space:pre-wrap;font-size:12px;max-height:360px;overflow:auto;background:#f2f5f4;padding:15px}svg{width:100%;display:block}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #d4deda;text-align:left}small{color:#526b70}.scroll{overflow:auto}#dag{min-width:650px}.note{border-left:4px solid #bf893e;padding:12px;background:#fff4dc}a{color:#087b73}.controls{display:flex;flex-wrap:wrap;align-items:center}.stat{font-size:18px;color:#087b73}.edge{stroke:#8a6c3e;fill:none;stroke-width:1.4}.node{cursor:pointer}.node:hover rect{stroke:#ce872c;stroke-width:3}button:focus-visible,select:focus-visible{outline:3px solid #bc7936}</style>
<main><small>PRACTICE 20 · ASCEND · EAGER</small><h1>KV offload / reload 与计算重叠</h1>
<p>选择实际传输批次查看设备时间线；选择缓存块查看跨算子、传输及存储换代的依赖。依赖边单独生成，并在实际同步图中验证可达性。</p>
<div class="note">这是一份 KV 边界依赖图。完整窗口的设备任务已保留；原生 kernel 内部 workspace 的精确访存没有恢复。性能来自独立的无 profiler 测量。</div>
<section><h2>运行与传输</h2><div class="controls"><label>模式 <select id="run"></select></label><label>传输批次 <select id="transfer"></select></label></div><p id="stats" class="stat"></p><label>时间线缩放 <select id="zoom"><option value="1">1×</option><option value="10">10×</option><option value="100">100×</option></select></label><div class="scroll"><div id="timeline"></div></div><small>矩形宽度按真实设备时长绘制。横轴使用相对微秒；点击 kernel 或 DMA 查看关联证据。</small></section>
<section><h2>缓存块的数据与复用 DAG</h2><div class="controls"><label>存储 <select id="medium"><option>NPU</option><option>CPU</option></select></label><label>block <select id="block"></select></label><label>边类型 <select id="kind"><option value="">全部</option><option>data_dependency</option><option>storage_reuse</option></select></label></div><div class="scroll"><div id="dag"></div></div><p id="edgeinfo"></p><div id="ranges"></div></section>
<section><h2>选择项的证据</h2><pre id="detail">点击图中的节点。</pre></section>
<section><h2>独立性能对照</h2><div id="performance" class="scroll"></div><small>每模式、每种长度 10 个正式周期。串行组包含主机屏障；两种 offload 模式搬运字节相同。闭环完成时间不等于生产服务容量。</small></section>
<section><h2>图与证据边界</h2><p>HB 图区分 stream FIFO、主机程序顺序、后台队列交接、event 完成确认和提交边。下方 DAG 展示所需的数据／复用边，连线不按时间缩放。</p><p>后台线程的 CPU profiler scope 不可用时，使用真实 CANN connection、enqueue/dequeue 和已核验的 FIFO 批次顺序；不按最近时间猜测归属。</p><p>完整设备图、原始 trace、源码和哈希清单见 <a href="../RESULTS.md">实验报告</a>。</p></section></main>
<script>const DATA=__DATA__;
const $=id=>document.getElementById(id),esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));let current;
function options(id,values){$(id).innerHTML=values.map(v=>`<option value="${esc(v[0])}">${esc(v[1])}</option>`).join('')}
function show(o){$('detail').textContent=JSON.stringify(o,null,2)}
function initRun(){current=DATA.runs[Number($('run').value)];options('transfer',current.windows.map((w,i)=>[i,`${w.direction} #${w.event_index} · ${(w.num_bytes/1048576).toFixed(1)} MiB`]));let s=current.summary;$('stats').textContent=`${s.compute_tasks.toLocaleString()} 个计算任务 · ${s.device_tasks.toLocaleString()} 个设备任务 · ${s.required_edges} 条 KV 约束 · ${s.unsatisfied_requirements} 条未满足`;initBlocks();timeline()}
function timeline(){let w=current.windows[Number($('transfer').value)];if(!w){$('timeline').textContent='重算组没有 KV DMA。';return}let width=1200*Number($('zoom').value),left=100,lanes=[...new Set(w.tasks.map(t=>t.stream))].sort(),height=55+lanes.length*55;let x=v=>left+v/w.span_us*(width-left-20);let out=`<svg viewBox="0 0 ${width} ${height}" style="min-width:${width}px" role="img" aria-label="实际设备传输与计算时间线">`;
lanes.forEach((s,i)=>{let y=30+i*55;out+=`<text x="8" y="${y+18}" font-size="12">stream ${esc(s)}</text><line x1="${left}" x2="${width-20}" y1="${y+30}" y2="${y+30}" stroke="#ccd7d2"/>`});
w.tasks.forEach((t,i)=>{let y=30+lanes.indexOf(t.stream)*55,fill=t.dma?'#d58a3b':'#16867e';out+=`<rect class="node" data-task="${i}" x="${x(t.start_us)}" y="${y}" width="${t.duration_us/w.span_us*(width-left-20)}" height="26" fill="${fill}"><title>${esc(t.name)} · ${t.duration_us} µs</title></rect>`});out+=`<text x="${left}" y="${height-5}" font-size="12">0 µs</text><text x="${width-110}" y="${height-5}" font-size="12">${w.span_us.toFixed(2)} µs</text></svg>`;$('timeline').innerHTML=out;document.querySelectorAll('[data-task]').forEach(e=>e.onclick=()=>show(w.tasks[Number(e.dataset.task)]));show(w.evidence)}
function initBlocks(){let m=$('medium').value,ids=[...new Set(current.requirements.filter(e=>e.medium===m).map(e=>e.block))].sort((a,b)=>a-b);options('block',ids.map(i=>[i,i]));dag()}
function dag(){let medium=$('medium').value,block=Number($('block').value),kind=$('kind').value,edges=current.requirements.filter(e=>e.medium===medium&&e.block===block&&(!kind||e.kind===kind));let ids=[...new Set(edges.flatMap(e=>[e.source,e.target]))],level={},remaining=new Set(ids);while(remaining.size){let ready=[...remaining].filter(n=>!edges.some(e=>e.target===n&&remaining.has(e.source)));if(!ready.length)break;ready.forEach(n=>{level[n]=Math.max(0,...edges.filter(e=>e.target===n).map(e=>level[e.source]+1));remaining.delete(n)})}let groups={};ids.forEach(n=>(groups[level[n]]??=[]).push(n));let cols=Math.max(1,...Object.keys(groups).map(Number))+1,rows=Math.max(1,...Object.values(groups).map(g=>g.length)),width=Math.max(700,cols*245),height=rows*80+30,pos={};Object.entries(groups).forEach(([l,ns])=>ns.forEach((n,i)=>pos[n]=[10+Number(l)*245,15+i*80]));let out=`<svg viewBox="0 0 ${width} ${height}" style="min-width:${width}px"><defs><marker id="arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0L7 3.5L0 7" fill="#8a6c3e"/></marker></defs>`;edges.forEach(e=>{let a=pos[e.source],b=pos[e.target];out+=`<path class="edge" marker-end="url(#arrow)" d="M${a[0]+200},${a[1]+25} L${b[0]},${b[1]+25}"><title>${esc(e.kind)} ${esc(e.hazard)} · generation ${e.from_generation} → ${e.to_generation} · HB ${e.satisfied}</title></path>`});ids.forEach((n,i)=>{let p=pos[n],node=current.nodes[n];out+=`<g class="node" data-node="${i}"><rect x="${p[0]}" y="${p[1]}" width="200" height="50" rx="5" fill="#e8f2ed" stroke="#81a59a"/><text x="${p[0]+7}" y="${p[1]+19}" font-size="10">${esc((node.name||node.operation).slice(0,29))}</text><text x="${p[0]+7}" y="${p[1]+37}" font-size="10">${esc(n)} · stream ${esc(node.stream??'host')}</text></g>`});out+='</svg>';$('dag').innerHTML=edges.length?out:'该选择没有跨 DMA 的数据／复用边。';document.querySelectorAll('[data-node]').forEach(e=>e.onclick=()=>show({node:current.nodes[ids[Number(e.dataset.node)]],dependencies:edges.filter(x=>x.source===ids[Number(e.dataset.node)]||x.target===ids[Number(e.dataset.node)])}));$('edgeinfo').textContent=`${edges.length} 条边；generation 表示同一物理 block 的内容换代。`;let layout=current.layout?.tensors||[];$('ranges').textContent=layout.length?`覆盖 ${layout.length} 个 K/V 子张量。每个张量的字节范围为 base + ${block} × page_bytes 到 base + ${block+1} × page_bytes；具体布局见节点详情。`:''}
options('run',DATA.runs.map((r,i)=>[i,r.summary.mode]));$('run').onchange=initRun;$('transfer').onchange=timeline;$('zoom').onchange=timeline;$('medium').onchange=initBlocks;$('block').onchange=dag;$('kind').onchange=dag;
let table='<table><tr><th>前缀</th><th>模式</th><th>首 token / ms</th><th>A′ 完成 / ms</th><th>B 完成 / ms</th></tr>';DATA.performance.cases.forEach(c=>['native','serialized','recompute'].forEach(m=>{let s=c.modes[m];table+=`<tr><td>${c.prefix_tokens}</td><td>${m}</td><td>${s.reload_ttft_ms.median.toFixed(2)}</td><td>${s.reload_latency_ms.median.toFixed(2)}</td><td>${s.b_latency_ms.median.toFixed(2)}</td></tr>`}));$('performance').innerHTML=table+'</table>';initRun();
</script></html>'''


def compact(path):
    data=json.loads(path.read_text())
    device=[n for n in data['nodes'] if n['kind']=='device']
    by_id={n['id']:n for n in data['nodes']}
    selected={e[k] for e in data['requirements'] for k in ('source','target')}
    windows=[]
    for transfer in data['transfers']:
        dma=[by_id[n] for n in transfer['device_nodes']]
        low=min(Decimal(n['start_us']) for n in dma)-50
        high=max(Decimal(n['end_us']) for n in dma)+50
        nodes=[]
        for n in device:
            if Decimal(n['start_us'])<high and Decimal(n['end_us'])>low:
                nodes.append(dict(id=n['id'],name=n['name'],stream=n['stream'],
                                  start_us=float(Decimal(n['start_us'])-low),duration_us=float(n['duration_us']),
                                  dma=n['id'] in transfer['device_nodes'],step=n['step'],
                                  requests=list(n['scheduled']),host_anchor_kind=n['host_anchor_kind']))
        evidence=next(x for x in data['summary']['overlap'] if
                      x['direction']==transfer['direction'] and x['event_index']==transfer['event_index'])
        windows.append(dict(direction=transfer['direction'],event_index=transfer['event_index'],
                            num_bytes=transfer['num_bytes'],span_us=float(high-low),tasks=nodes,evidence=evidence))
    layout=next((r for r in data['records'] if r['kind']=='cache_layout'),None)
    # Public report omits raw pointers while preserving byte-stride descriptions.
    if layout:
        layout=dict(tensors=[{k:v for k,v in r.items() if 'ptr' not in k} for r in layout['tensors']])
    return dict(summary=data['summary'],windows=windows,requirements=data['requirements'],
                nodes={i:by_id[i] for i in selected},layout=layout)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graphs',type=Path,nargs='+',required=True)
    p.add_argument('--performance',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    data=dict(runs=[compact(path) for path in a.graphs],performance=json.loads(a.performance.read_text()))
    data['performance'].pop('samples',None)
    a.output.write_text(HTML.replace('__DATA__',json.dumps(data,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')))
    print(a.output,a.output.stat().st_size)


if __name__=='__main__':
    main()
