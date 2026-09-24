"""Small offline timeline with true device intervals and clickable kernel evidence."""
from decimal import Decimal
from html import escape
import json


def timeline(trials, tasks, span, interactive=False):
    width=1120
    left,right=170,1080
    height=65+len(trials)*142
    scale=(right-left)/float(span)
    out=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" role="img" aria-label="设备任务执行时间线">'%(width,height),
         '<rect width="100%" height="100%" fill="#fff"/>',
         '<style>text{font-family:system-ui,sans-serif;fill:#22324a;font-size:13px}.kernel{cursor:pointer}.kernel:hover rect,.kernel:focus rect{stroke:#10253f;stroke-width:2}</style>']
    out.append('<text x="170" y="23">时间：相对每轮第一个计算 kernel 的开始；所有轮次使用同一比例尺</text>')
    for tick in range(0,int(span)+1,1000):
        x=left+tick*scale
        out.append('<path d="M %.3f 40 V %d" stroke="#e2e8f0"/><text x="%.3f" y="43">%.1f ms</text>'%(x,height-10,x,tick/1000))
    for row,trial in enumerate(trials):
        y=65+row*142
        origin=Decimal(trial['start_us'])
        lanes=trial['physical_streams']
        out.append('<text x="12" y="%d" font-weight="700">%s</text>'%(y+10,escape(trial['id'])))
        for lane_idx,lane in enumerate(lanes):
            ly=y+22+lane_idx*33
            out.append('<text x="12" y="%d">stream %s</text>'%(ly+18,escape(lane)))
            out.append('<path d="M %d %d H %d" stroke="#dbe3ed"/>'%(left,ly+25,right))
        for task in tasks:
            if task['trial']!=trial['id'] or task['kind']!='compute':
                continue
            x=left+float(Decimal(task['start_us'])-origin)*scale
            w=float(Decimal(task['duration_us']))*scale
            ly=y+22+lanes.index(task['stream'])*33
            color='#3268c7' if task['branch']=='mm' else '#d57915'
            label=task['label'].rsplit('/',1)[-1]
            title='%s | stream %s | %s us | %s'%(label,task['stream'],task['duration_us'],task['name'])
            attrs=' class="kernel" tabindex="0" data-label="%s"'%escape(task['label'],quote=True) if interactive else ''
            out.append('<g%s><title>%s</title><rect x="%.3f" y="%d" width="%.3f" height="25" rx="3" fill="%s"/>'%(attrs,escape(title),x,ly,w,color))
            if w>38:
                out.append('<text x="%.3f" y="%d" style="fill:white;font-size:11px">%s</text>'%(x+3,ly+17,label))
            out.append('</g>')
        for overlap in trial['overlaps']:
            x=left+float(Decimal(overlap['start_us'])-origin)*scale
            w=float(Decimal(overlap['duration_us']))*scale
            out.append('<rect x="%.3f" y="%d" width="%.3f" height="7" fill="#25836f"/>'%(x,y+95,w))
        out.append('<text x="170" y="%d">kernel 跨度 %.3f ms · 两分支重叠 %.3f ms</text>'%
                   (y+120,float(trial['device_span_us'])/1000,float(trial['overlap_us'])/1000))
    out.append('</svg>')
    return '\n'.join(out)


def render(data,output):
    trials=data['trials']
    span=max(Decimal(t['device_span_us']) for t in trials)*Decimal('1.04')
    svg=timeline(trials,data['tasks'],span)
    (output/'timeline.svg').write_text(svg)
    charts=''.join('<section class="trial" data-trial="%s">%s</section>'%
                   (escape(t['id']),timeline([t],data['tasks'],span,True)) for t in trials)
    options='<option value="all">全部轮次</option>'+''.join('<option>%s</option>'%escape(t['id']) for t in trials)
    rows=''.join('<tr><td>%s</td><td>%s</td><td>%.3f</td><td>%.3f</td><td>%.3f%%</td></tr>'%
                 (escape(t['id']),', '.join(t['physical_streams']),float(t['device_span_us'])/1000,
                  float(t['overlap_us'])/1000,float(t['overlap_fraction_shorter_branch'])*100) for t in trials)
    payload=json.dumps(data['tasks'],ensure_ascii=False).replace('<','\\u003c')
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Practice 16 · 多 stream 并行</title><style>
body{margin:0;background:#f2f5fa;color:#20314b;font:16px/1.65 system-ui,sans-serif}main{max-width:1160px;margin:32px auto;padding:0 20px}h1{font-size:30px;line-height:1.3}h2{font-size:20px}p{max-width:960px}.panel{background:white;border:1px solid #dce4ef;border-radius:12px;padding:20px;margin:18px 0}.tag{font-size:13px;letter-spacing:.1em;color:#3268c7}.legend{display:flex;gap:24px;flex-wrap:wrap}.dot{display:inline-block;width:13px;height:13px;border-radius:3px;margin-right:6px}select{font:inherit;padding:6px 12px;border:1px solid #acbbce;border-radius:6px;margin-left:8px}.chart{overflow-x:auto}.trial svg{min-width:760px;width:100%;display:block}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}td,th{text-align:left;padding:9px;border-bottom:1px solid #e2e8f0}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f5fa;padding:16px;font-size:13px}a{color:#235db9}.muted{color:#52647d;font-size:14px}[hidden]{display:none!important}@media(max-width:600px){main{padding:0 12px}h1{font-size:25px}.panel{padding:12px}table{font-size:13px}}
</style><main><div class="tag">PRACTICE 16 / ASCEND 910B2C</div><h1>两个 stream，计算真的重叠了吗？</h1>
<p>同样的 4096×4096 FP16 矩阵乘法和 64M 元素 FP32 向量乘法，单 stream 顺序执行与双 stream 独立执行对照。数据彼此独立，输出一直保留到两条流完成。每轮每类算子的次数为该轮计算任务总数的一半，默认各六次。</p>
<div class="panel"><div class="legend"><span><i class="dot" style="background:#3268c7"></i>矩阵乘法 mm</span><span><i class="dot" style="background:#d57915"></i>向量乘法 mul</span><span><i class="dot" style="background:#25836f"></i>设备计算区间交集</span></div>
<p class="muted">矩形来自设备 profiler 的起止时间，非 CPU 调用耗时。所有轮次使用相同时间比例尺；可点击 kernel 查看真实 flow、stream 和任务编号。</p>
<label for="trial">显示轮次</label><select id="trial">OPTIONS</select><div class="chart">CHARTS</div></div>
<div class="panel"><h2>可核验的重叠</h2><div class="chart"><table><thead><tr><th>轮次</th><th>物理 stream</th><th>跨度 ms</th><th>重叠 ms</th><th>较短分支覆盖率</th></tr></thead><tbody>ROWS</tbody></table></div>
<p class="muted">跨度 = 最后一个计算 kernel 结束 − 第一个开始；重叠 = 两分支计算区间交集的并集长度。覆盖率 = 重叠时间 / 较短分支的累计计算时间。未将 event、输出校验、CPU 下发范围计入计算重叠。</p></div>
<div class="panel"><h2>选中 kernel 的证据</h2><pre id="detail" aria-live="polite">点击上方蓝色或橙色矩形。</pre></div>
<p>采集有 profiler 开销；时间线重叠不能等同于指令级资源占用，也不保证端到端加速。本练习没有执行 vLLM 请求或 KV block 复用。</p>
<p><a href="timeline.svg">导出 SVG</a> · <a href="evidence.json">完整证据 JSON</a> · <a href="summary.json">统计 JSON</a> · <a href="../../../README.md">复现说明</a></p>
<script type="application/json" id="tasks">PAYLOAD</script><script>
const tasks=JSON.parse(document.getElementById('tasks').textContent);const byLabel=new Map(tasks.map(t=>[t.label,t]));
document.getElementById('trial').addEventListener('change',e=>{document.querySelectorAll('.trial').forEach(s=>s.hidden=e.target.value!=='all'&&s.dataset.trial!==e.target.value)});
function show(node){document.getElementById('detail').textContent=JSON.stringify(byLabel.get(node.dataset.label),null,2)}
document.querySelectorAll('.kernel').forEach(n=>{n.addEventListener('click',()=>show(n));n.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();show(n)}})});
</script></main></html>'''
    html=html.replace('OPTIONS',options).replace('CHARTS',charts).replace('ROWS',rows).replace('PAYLOAD',payload)
    (output/'index.html').write_text(html)
