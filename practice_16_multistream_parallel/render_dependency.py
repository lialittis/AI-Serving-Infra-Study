"""Draw the measured A/B/C intervals and the single, optional event dependency."""
from decimal import Decimal
from html import escape
import json


def timeline(trials,span,interactive=False):
    width=1140;left=175;right=1100;height=60+len(trials)*187
    scale=(right-left)/float(span)
    out=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" role="img" aria-label="三 stream 依赖时间线">'%(width,height),
         '<rect width="100%" height="100%" fill="white"/>',
         '<style>text{font:13px system-ui,sans-serif;fill:#20314b}.task{cursor:pointer}.task:hover rect,.task:focus rect{stroke:#132d50;stroke-width:2}</style>',
         '<defs><marker id="arrow" markerWidth="7" markerHeight="7" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="none" stroke="#b5453d"/></marker></defs>',
         '<text x="175" y="22">设备执行时间（各轮独立归零，同一比例尺）；细竖线为 event record 标记</text>']
    for tick in range(0,int(span)+1,1000):
        x=left+tick*scale
        out.append('<path d="M%.3f 37 V%d" stroke="#e1e7f0"/><text x="%.3f" y="43">%.1f ms</text>'%(x,height-8,x,tick/1000))
    for i,trial in enumerate(trials):
        y=65+i*187;nodes=list(trial['nodes'].values());origin=min(Decimal(t['start_us']) for t in nodes)
        label='保留 A→B 等待' if trial['mode']=='event_wait' else '删除 A→B 等待'
        out.append('<text x="10" y="%d" font-weight="700">%s</text>'%(y,escape(label)))
        for bi,b in enumerate(['A','B','C']):
            ly=y+13+bi*37
            out.append('<text x="10" y="%d">%s · stream %s</text>'%(ly+18,b,escape(trial['physical_streams'][b])))
            out.append('<path d="M%d %d H%d" stroke="#dbe3ed"/>'%(left,ly+26,right))
        for task in nodes:
            x=left+float(Decimal(task['start_us'])-origin)*scale
            w=float(Decimal(task['duration_us']))*scale
            ly=y+13+['A','B','C'].index(task['branch'])*37
            op=task['operation']
            title='%s · %s us · %s'%(op,task['duration_us'],task['name'])
            attrs=' class="task" tabindex="0" data-label="%s"'%escape(task['label'],quote=True) if interactive else ''
            out.append('<g%s><title>%s</title>'%(attrs,escape(title)))
            if task['kind']=='record':
                out.append('<path d="M%.3f %d V%d" stroke="#23334b" stroke-width="1.5"/>'%(x,ly-2,ly+28))
            else:
                color=('#dbe3ed' if task['kind']=='device_wait' else '#235dba' if op=='produce-Y' else
                       '#749cda' if task['branch']=='A' else '#a34b97' if task['branch']=='B' else '#278c75')
                out.append('<rect x="%.3f" y="%d" width="%.3f" height="26" rx="3" fill="%s"/>'%(x,ly,w,color))
                text='等待 A' if task['kind']=='device_wait' else 'Y=X@X' if op=='produce-Y' else 'Z=Y×2' if op=='consume-Y' else op.split('-')[-1]
                if w>30:
                    out.append('<text x="%.3f" y="%d" style="fill:%s;font-size:11px">%s</text>'%(x+3,ly+17,'#20314b' if task['kind']=='device_wait' else 'white',text))
            out.append('</g>')
        if trial['dependency_wait']:
            a=left+float(Decimal(trial['dependency_wait']['record_end_us'])-origin)*scale
            b=left+float(Decimal(trial['dependency_wait']['wait_end_us'])-origin)*scale
            out.append('<path d="M%.3f %d L%.3f %d" stroke="#b5453d" stroke-width="1.5" marker-end="url(#arrow)"/>'%(a,y+40,b,y+47))
        relation={'B_finished_before_A_producer_started':'B 已经结束，A 才开始写入新 Y',
                  'B_started_after_A_producer_finished':'A 写入完成后，B 才开始读取 Y',
                  'producer_consumer_intervals_overlap':'A 写入与 B 读取的执行区间相交'}[trial['producer_consumer_relation']]
        out.append('<text x="175" y="%d">%s · A/C 计算重叠 %.3f ms</text>'%(y+143,relation,float(trial['A_C_compute_overlap_us'])/1000))
        out.append('<text x="175" y="%d">%s · 完成后 Z 前四个值：%s</text>'%(y+166,escape(trial['id']),escape(str(trial['Z_first_eight'][:4]))))
    out.append('</svg>')
    return '\n'.join(out)


def render(data,out):
    trials=data['trials']
    span=max(max(Decimal(k['end_us']) for k in t['nodes'].values())-min(Decimal(k['start_us']) for k in t['nodes'].values()) for t in trials)*Decimal('1.035')
    (out/'timeline.svg').write_text(timeline(trials,span))
    charts=''.join('<section class="trial" data-trial="%s">%s</section>'%(escape(t['id']),timeline([t],span,True)) for t in trials)
    options='<option value="all">全部轮次</option>'+''.join('<option>%s</option>'%escape(t['id']) for t in trials)
    names={'event_wait':'保留 B.wait_event(A_done)','no_wait':'删除跨流等待'}
    rows=''.join('<tr><td>%s</td><td>%.3f</td><td>%.3f–%.3f</td><td>%d / %d</td><td>%d / %d</td></tr>'%
                 (names[m],v['elapsed_median_ms'],v['elapsed_min_ms'],v['elapsed_max_ms'],v['correct_trials'],v['repeats'],v['stale_trials'],v['repeats']) for m,v in data['performance'].items())
    payload=json.dumps(data['tasks'],ensure_ascii=False).replace('<','\\u003c')
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Practice 16 · A→B 依赖与独立 C</title><style>
body{margin:0;background:#f2f5fa;color:#20314b;font:16px/1.65 system-ui,sans-serif}main{max-width:1160px;margin:32px auto;padding:0 20px}h1{font-size:30px;line-height:1.4}h2{font-size:21px}.panel{background:white;border:1px solid #dce4ef;border-radius:12px;padding:20px;margin:20px 0}.scroll{overflow:auto}.trial svg{width:100%;min-width:800px;display:block}table{width:100%;border-collapse:collapse;white-space:nowrap;font-variant-numeric:tabular-nums}td,th{text-align:left;padding:9px;border-bottom:1px solid #e2e8f0}.muted{color:#52647d;font-size:14px}select{font:inherit;max-width:100%;padding:6px 10px}.legend{display:flex;gap:18px;flex-wrap:wrap}.swatch{display:inline-block;width:12px;height:12px;margin-right:5px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f5fa;padding:15px;font-size:13px}a{color:#235db9}[hidden]{display:none!important}@media(max-width:600px){main{padding:0 12px}.panel{padding:12px}h1{font-size:25px}}
</style><main><p class="muted">PRACTICE 16 / CROSS-STREAM DEPENDENCY</p><h1>最后都等完了，为什么结果仍然错？</h1>
<p>A 生成 <code>Y = X @ X = 1</code>，B 消费 <code>Z = Y × 2 = 2</code>；C 独立计算 <code>W = V × 1.5 = 0.375</code>。每次开始前 Y 初始化为 −7。唯一删除的依赖是 B 的 <code>wait_event(A_done)</code>，CPU 始终在 A/B/C 全部完成后读取结果。</p>
<div class="panel"><h2>真实设备时间线</h2><div class="legend"><span><i class="swatch" style="background:#749cda"></i>A 的排队计算</span><span><i class="swatch" style="background:#235dba"></i>A 生成 Y</span><span><i class="swatch" style="background:#a34b97"></i>B 消费 Y</span><span><i class="swatch" style="background:#278c75"></i>C 独立计算</span><span><i class="swatch" style="background:#dbe3ed"></i>B 等待 A</span></div>
<p class="muted">A 在生成 Y 前先做真实矩阵计算，以扩大缺依赖时的观察窗口。两种模式工作量相同；这是受控演示，不代表一般负载必然出现同样的错误频率。点击色块查看原始 flow 与任务编号。</p>
<label for="trial">显示轮次：</label><select id="trial">OPTIONS</select><div class="scroll">CHARTS</div></div>
<div class="panel"><h2>完成后再检查，错误仍然存在</h2><div class="scroll"><table><thead><tr><th>策略</th><th>完整耗时中位数 ms</th><th>范围 ms</th><th>Z 全部正确</th><th>Z 全部为 −14</th></tr></thead><tbody>ROWS</tbody></table></div>
<p>耗时来自单独的无 profiler 测量。删除依赖后的耗时不能作为有效优化成绩：程序没有完成正确的计算。</p>
<p>缺少等待时，B 可能先用旧 Y=−7 算出 Z=−14。随后 A 把 Y 改成1，已经写出的 Z 不会自动改变。最终汇合只保证任务完成；重新执行一次 B，才能用新 Y 得到 Z=2。</p></div>
<div class="panel"><h2>独立 C 是否继续工作</h2><p>COVERAGE<code>wait_event</code>建立 A→B 的顺序，不把整个 NPU 或 Host 全局阻塞；最后的 Host join 则负责安全读取与退出。</p>
<p class="muted">矩形是 profiler 的设备任务区间，并非每个时钟周期的资源利用率。初始化、校验及完成后的修复重算不混入被测任务。</p></div>
<div class="panel"><h2>选中任务的证据</h2><pre id="detail" aria-live="polite">点击上方任务色块。</pre></div>
<p><a href="timeline.svg">六轮 SVG</a> · <a href="dependency_evidence.json">完整关联 JSON</a> · <a href="../../../CROSS_STREAM_DEPENDENCY.md">结论与复现</a> · <a href="../../../README.md">Practice 16</a></p>
<script type="application/json" id="data">PAYLOAD</script><script>
const tasks=JSON.parse(document.getElementById('data').textContent),byLabel=new Map(tasks.map(t=>[t.label,t]));
document.getElementById('trial').addEventListener('change',e=>document.querySelectorAll('.trial').forEach(s=>s.hidden=e.target.value!=='all'&&s.dataset.trial!==e.target.value));
function show(n){document.getElementById('detail').textContent=JSON.stringify(byLabel.get(n.dataset.label),null,2)}
document.querySelectorAll('.task').forEach(n=>{n.addEventListener('click',()=>show(n));n.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();show(n)}})});
</script></main></html>'''
    safe=[t for t in trials if t['mode']=='event_wait']
    covered=sum(Decimal(t['C_while_B_waiting_us'])>0 for t in safe)
    coverage='在 %d / %d 个保留依赖的轮次中，观测到 C 在 B 的设备等待期间执行。'%(covered,len(safe))
    (out/'index.html').write_text(html.replace('OPTIONS',options).replace('CHARTS',charts).replace('ROWS',rows).replace('PAYLOAD',payload).replace('COVERAGE',coverage))
