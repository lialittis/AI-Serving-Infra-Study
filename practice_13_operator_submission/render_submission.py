"""Render five questions about compilation, submission and CPU/NPU overlap."""
import argparse
import json
from pathlib import Path


def sequence_svg():
    """Mechanism sketch only; measured timing is rendered separately."""
    xs = [140, 410, 700, 1010]
    names = ['CPU主线程', '主机任务队列', 'CPU下发线程 / CANN', 'NPU stream']
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1150 630" role="img" aria-label="算子提交机制示意，非比例时间线">',
             '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="#087d75"/></marker></defs>',
             '<rect width="1150" height="630" fill="white"/>']
    for x, name in zip(xs, names):
        parts += ['<rect x="{}" y="10" width="230" height="42" rx="5" fill="#e7f2ef"/>'.format(x-115),
                  '<text x="{}" y="37" text-anchor="middle" font-family="sans-serif" font-size="16" fill="#17404a">{}</text>'.format(x, name),
                  '<line x1="{0}" x2="{0}" y1="55" y2="605" stroke="#a4bdb6" stroke-dasharray="5 5"/>'.format(x)]
    parts += ['<rect x="45" y="70" width="810" height="48" rx="4" fill="#fff1d9"/>',
              '<text x="70" y="91" font-size="15" fill="#17404a">启动 / 预热：Triton编译 → 二进制与函数注册</text>',
              '<text x="70" y="110" font-size="13" fill="#526c70">内部代码传输的精确时刻未直接观测</text>']
    arrows = [(0,1,155,'输入数据拷贝入队'),(1,2,200,'出队'),(2,3,245,'提交H2D'),
              (0,1,295,'句柄、地址、标量、grid、stream'),(1,2,340,'出队，准备原生参数包'),
              (2,3,385,'runtime launch'),(0,1,485,'采样后：D2H + event'),(1,2,530,'出队'),(2,3,575,'D2H + event')]
    for start, stop, y, label in arrows:
        a,b=xs[start],xs[stop]
        parts += ['<line x1="{}" x2="{}" y1="{}" y2="{}" stroke="#087d75" stroke-width="2" marker-end="url(#arrow)"/>'.format(a,b,y,y),
                  '<text x="{}" y="{}" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#17404a">{}</text>'.format((a+b)//2,y-10,label)]
    parts += ['<rect x="38" y="410" width="255" height="34" rx="4" fill="#fff1d9"/><text x="50" y="432" font-size="14" fill="#17404a">CPU继续后续操作（实际有重叠）</text>',
              '<rect x="900" y="410" width="210" height="34" rx="4" fill="#e7f2ef"/><text x="920" y="432" font-size="14" fill="#17404a">NPU执行已提交任务</text>',
              '<text x="45" y="615" font-size="14" fill="#17404a">CPU在原生event等待返回后读取结果；本图表示机制，间隔不代表测量值。</text></svg>']
    return ''.join(parts)


def render(run):
    evidence=json.loads((run/'analysis/submission_evidence.json').read_text())
    wanted={'_triton_rope','_compute_slot_mapping_kernel','ReshapeAndCacheNdKernel',
            'FusedInferAttentionScore','aclnnAddmm_MatMulCommon_MatMulV2',
            'aclnnMatmul_MatMulCommon_MatMulV2','MEMCPY_ASYNC','EVENT_RECORD'}
    examples=[e for e in evidence['examples'] if e['task']['kernel'] in wanted]
    # Convert absolute timestamps to relative decimals in Python so JS never loses
    # submicrosecond precision by subtracting ~1e15 floating-point timestamps.
    from decimal import Decimal
    for example in examples:
        selected=[('CPU调用',example['host_operator']),('CANN下发',example['cann_launch']),('NPU执行',example['device_event'])]
        if example['queue']:
            selected += [('CPU入队',example['queue']['enqueue']),('工作线程出队',example['queue']['dequeue'])]
        selected += [('CPU同时活动',x['event']) for x in example['cpu_main_during_device'][:5]]
        base=min(Decimal(str(e['ts'])) for _,e in selected)
        timeline=[]
        for lane,e in selected:
            timeline.append(dict(lane=lane,name=e['name'],start_us=float(Decimal(str(e['ts']))-base),
                                 duration_us=float(e['dur']),tid=e['tid']))
        example['timeline']=timeline
    tasks=[]
    origin=min(Decimal(str(x['device_event']['ts'])) for x in evidence['device_tasks'])
    for task in evidence['device_tasks']:
        row=task['task']
        tasks.append(dict(kernel=row['kernel'],phase=task['phase'],stream=row['stream_id'],task=row['task_id'],
                          start_us=float(Decimal(str(task['device_event']['ts']))-origin),
                          duration_us=float(task['device_event']['dur']),host=task['host_operator']['name'],
                          queue=task['queue']['correlation_id'] if task['queue'] else None,
                          parameters=task['parameter_scope_labels']))
    payload=dict(summary=evidence['summary'],examples=examples,compile_records=evidence['compile_records'],
                 load_records=evidence['load_records'],limits=evidence['limits'],tasks=tasks,
                 stream_order=evidence['stream_order'],completions=evidence['completions'])
    text='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Practice 13 · CPU怎样提交NPU任务</title><style>
body{overflow-wrap:anywhere;margin:0;background:#f3f7f6;color:#153c46;font:15px/1.75 system-ui,sans-serif}main{max-width:1180px;margin:auto;padding:38px 24px}h1{font-size:34px;line-height:1.3}h2{font-size:22px}small{color:#526c70}a{color:#087d75}.card{min-width:0;background:white;border:1px solid #d2dfda;border-radius:8px;padding:22px;margin:18px 0}.grid{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:18px}.grid .card{margin:0}.note{background:#fff1d9;border-left:4px solid #b7802c;padding:17px;margin:20px 0}button,select{font:inherit;padding:9px;border-radius:5px;border:1px solid #b9ccc5;background:white;color:#17404a}select{max-width:100%}.controls{display:flex;gap:12px;flex-wrap:wrap;align-items:center}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:9px;text-align:left;border-bottom:1px solid #dce7e1;vertical-align:top}pre{font:12px/1.6 monospace;white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f6f3;padding:13px;max-height:480px;overflow:auto}code{overflow-wrap:anywhere}.overflow{overflow:auto}svg{min-width:850px;width:100%}summary{cursor:pointer;color:#087d75}li{margin:7px 0}@media(max-width:760px){main{padding:22px 14px}h1{font-size:27px}.grid{grid-template-columns:1fr}}@media print{.controls{display:none}.card{break-inside:avoid}}</style>
<main><small>PRACTICE 13 · ASCEND 910B2C · QWEN2.5-0.5B · EAGER</small><h1>一次算子调用，<br>怎样变成NPU上的一次执行？</h1><p>请求：“请简要解释为什么天空是蓝色的。” 正式推理前预热一次，生成4个token。使用原生KV池；观察工具不额外等待或复制NPU数据。</p>
<div class="card"><h2>先看这五个答案</h2><ol><li><b>何时编译？</b>隔离空Triton缓存，记录实际编译阶段；CANN/ATB已安装二进制的历史构建时间不从trace推测。</li><li><b>何时加载？</b>记录Triton load_binary和函数句柄注册。注册返回不是代码DMA完成时间，底层精确传输时刻尚未暴露。</li><li><b>提交什么？</b>kernel句柄、grid、stream，以及当前tensor地址和标量。constexpr在编译时固化；原生隐藏workspace/tiling字节并未全量读取。</li><li><b>何时交给运行时？</b>主机调用与入队、工作线程出队、CANN下发分别计时；通过真实flow和队列correlation关联。</li><li><b>NPU工作时CPU在做什么？</b>下方列出同一时间内主线程和下发线程的已观测范围；范围重叠不等于CPU利用率。</li></ol><p id="summary"></p></div>
<div class="card"><h2>从准备到取回结果</h2><p>机制示意，不按时间比例绘制。下面的算子详情才是本次运行的测量时间线。</p><div class="overflow">__SEQUENCE__</div><a href="submission_sequence.svg">单独打开流程图</a></div>
<div class="controls"><label for="example">选择算子与阶段 </label><select id="example"></select></div>
<div class="card"><h2 id="title"></h2><p id="route"></p><div class="overflow" id="timeline"></div><p><small>同一profiler时间轴，单位µs，相对所选片段起点。窄条有最小显示宽度，精确时长见表；不同泳道不代表不同物理CPU核。</small></p><div class="overflow" id="times"></div></div>
<div class="grid"><div class="card"><h2>CPU提交了哪些参数？</h2><div id="parameters"></div></div><div class="card"><h2>NPU执行期间CPU的可见活动</h2><div class="overflow" id="overlap"></div><div class="note">没有可见范围不等于CPU空闲；范围可能包含等待、Python记录或其他开销。这是带插桩运行，不是正常服务的性能基准。</div></div></div>
<div class="card"><h2>冷启动：编译与注册</h2><p>以下采用主机monotonic时钟；不与上方设备时间戳相减。内存cache命中后可以直接使用已有kernel。</p><div class="overflow" id="compiles"></div><details><summary>编译、二进制hash及注册记录</summary><pre id="cold"></pre></details></div>
<div class="card"><h2>整个请求的设备执行顺序</h2><p id="order-note"></p><div class="controls"><label for="task-search">筛选名称或阶段 </label><input id="task-search" placeholder="如 prefill、rope、MEMCPY"><button id="previous">上一页</button><button id="next">下一页</button><span id="task-count"></span></div><div class="overflow" id="all-tasks"></div><p><small>按设备实际开始时间排序。包括计算、拷贝和event；不把一个Python算子当成一个kernel。参数观测范围ID可在完整JSON的observations中查找。</small></p></div>
<div class="card"><h2>CPU什么时候必须等结果？</h2><p>每步采样后，框架将token ID复制到CPU，记录event，再调用原生Event.synchronize；返回后才能转成Python列表。下面是本次运行的4个完成边界。</p><div class="overflow" id="completion"></div><details><summary>拷贝源与目标、event身份及等待范围</summary><pre id="completion-raw"></pre></details></div>
<div class="card"><h2>如何复现与追证据</h2><p><a href="../../../README.md">复现步骤与代码入口</a> · <a href="../../../RESULTS.md">结果与证据边界</a> · <a href="submission_evidence.json">全部提交、参数和任务记录</a> · <a href="summary.json">覆盖统计</a></p><details><summary>当前选中算子的原始证据</summary><pre id="raw"></pre></details></div></main>
<script id="data" type="application/json">__DATA__</script><script>
const d=JSON.parse(document.getElementById('data').textContent),$=id=>document.getElementById(id),esc=x=>String(x??'未观测').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
$('summary').textContent=`正式请求：${d.summary.steps}个执行步骤，${d.summary.device_tasks}个设备任务；${d.summary.compiler_calls}次Triton编译，正式请求期间${d.summary.measured_compiles}次；记录${d.summary.measured_triton_launches}次Triton launcher调用。`;
$('example').innerHTML=d.examples.map((e,i)=>'<option value="'+i+'">'+esc(e.phase+' · '+e.task.kernel+(e.example_variant?' / '+e.example_variant:''))+'</option>').join('');
$('example').value=String(d.examples.findIndex(e=>e.phase==='prefill'&&e.task.kernel==='_triton_rope'));
function table(rows){return '<table>'+rows.map(r=>'<tr>'+r.map(c=>'<td>'+esc(c)+'</td>').join('')+'</tr>').join('')+'</table>'}
function show(){const e=d.examples[Number($('example').value)];$('title').textContent=e.phase+' · '+e.task.kernel+(e.example_variant?' / '+e.example_variant:'');$('route').textContent=e.host_operator.name+' → '+(e.queue?'队列 '+e.queue.correlation_id+' → ':'队列关联未建立 → ')+e.cann_launch.name+' → NPU stream '+e.task.stream_id+' / task '+e.task.task_id;
 const t=e.timeline,max=Math.max(...t.map(x=>x.start_us+x.duration_us)),width=750,colors={'NPU执行':'#087d75','CPU同时活动':'#bc792d'};
 let svg='<svg viewBox="0 0 1100 '+(70+t.length*40)+'" role="img" aria-label="CPU与NPU执行时间线">';
 t.forEach((x,i)=>{const y=35+i*40;svg+='<text x="10" y="'+(y+17)+'" font-size="13" fill="#17404a">'+esc(x.lane)+'</text><rect x="'+(165+x.start_us/max*width)+'" y="'+y+'" width="'+Math.max(2,x.duration_us/max*width)+'" height="24" rx="3" fill="'+(colors[x.lane]||'#668990')+'"><title>'+esc(x.name)+' · '+x.duration_us+' µs</title></rect><text x="940" y="'+(y+17)+'" font-size="12" fill="#17404a">'+x.duration_us.toFixed(3)+' µs</text>'});svg+='</svg>';$('timeline').innerHTML=svg;
 $('times').innerHTML=table([['范围','线程/stream','开始µs','持续µs'],...t.map(x=>[x.lane+' / '+x.name,x.tid,x.start_us.toFixed(3),x.duration_us.toFixed(3)])]);
 const ps=e.parameter_scopes;const preferred=ps.find(x=>x.entry.kind==='launch')||ps.find(x=>x.entry.kind==='torch_api')||ps[0];
 if(preferred){const a=preferred.entry,args=a.named_arguments||a.kwargs||a.arguments||(a.source_tensor?{source:a.source_tensor,destination:a.destination,rows:a.rows}:a.args)||{};
 const rows=Object.entries(args).map(([k,v])=>[k,v&&v.kind==='tensor'?v.shape.join(' × '):JSON.stringify(v),v&&v.kind==='tensor'?v.dtype+' / '+v.device:'scalar / list',v&&v.kind==='tensor'?v.data_ptr:(a.constants&&k in a.constants?'编译常量':'')]);
 $('parameters').innerHTML='<p>真实 <code>'+esc(a.kind)+'</code> 调用边界；tensor地址属于本次进程。</p>'+(a.grid?'<p>grid = '+esc(a.grid.join(' × '))+'；stream handle = '+esc(a.runtime_stream)+'；function handle = '+esc(a.function_handle)+'</p>':'')+'<div class="overflow">'+table([['参数','shape或值','类型/位置','地址或绑定阶段'],...rows])+'</div><details><summary>完整参数、stride、offset和源码位置</summary><pre>'+esc(JSON.stringify(a,null,2))+'</pre></details>';
 }else $('parameters').innerHTML='<p>此任务没有独立参数观测范围；保留完整profiler关联，不伪造参数。</p>';
 let overlaps=[...e.cpu_main_during_device.map(x=>['主线程',x.event.name,x.overlap_us]),...e.cpu_worker_during_device.map(x=>['下发线程',x.event.name,x.overlap_us])];
 $('overlap').innerHTML=overlaps.length?table([['线程','观测范围','重叠µs'],...overlaps]):'<p>此设备区间没有匹配的CPU细粒度活动记录。</p>';
 const scopes=e.cpu_python_scopes_during_device||[];if(scopes.length)$('overlap').innerHTML+='<p>同时处于以下Python观测范围内（含记录开销）：</p>'+table([['范围','重叠µs'],...scopes.slice(0,4).map(x=>[x.kind,x.overlap_us])]);
 $('raw').textContent=JSON.stringify(e,null,2);
}
$('compiles').innerHTML=table([['kernel','阶段','编译耗时ms','实际编译pipeline'],...d.compile_records.map(x=>[x.exit.kernel_name,x.host_stage,Number(x.duration_ms).toFixed(3),x.exit.ran_compiler_stages])]);$('cold').textContent=JSON.stringify({compile:d.compile_records,load:d.load_records},null,2);$('example').onchange=show;show();
$('order-note').textContent=d.stream_order.map(x=>'stream '+x.stream_id+'：'+x.task_count+'个任务，观测到相邻任务重叠'+x.overlapping_neighbors.length+'对。').join(' ')+'这描述本次trace，不代表多stream也有相同全局顺序。';
let offset=0;function showTasks(){const query=$('task-search').value.toLowerCase(),rows=d.tasks.filter(x=>(x.kernel+' '+x.phase).toLowerCase().includes(query));const slice=rows.slice(offset,offset+40);$('task-count').textContent=rows.length+'条 / 当前'+(rows.length?offset+1:0)+'–'+Math.min(offset+40,rows.length);$('previous').disabled=offset===0;$('next').disabled=offset+40>=rows.length;$('all-tasks').innerHTML=table([['阶段','设备任务','stream / task','开始µs','持续µs','CPU来源','queue ID'],...slice.map(x=>[x.phase,x.kernel,x.stream+' / '+x.task,x.start_us.toFixed(3),x.duration_us.toFixed(3),x.host,x.queue])]);}
$('task-search').oninput=()=>{offset=0;showTasks()};$('previous').onclick=()=>{offset-=40;showTasks()};$('next').onclick=()=>{offset+=40;showTasks()};showTasks();
$('completion').innerHTML=table([['阶段','NPU→CPU复制task','event对象ID','Python token IDs'],...d.completions.map(x=>[x.phase,x.copy_task.task_id,x.entry.event_object_id,JSON.stringify(x.exit.token_ids)])]);$('completion-raw').textContent=JSON.stringify(d.completions,null,2);
</script></html>'''
    text=text.replace('__SEQUENCE__',sequence_svg()).replace('__DATA__',json.dumps(payload,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c'))
    (run/'analysis/submission_sequence.svg').write_text(sequence_svg())
    (run/'analysis/index.html').write_text(text)
    print('Rendered',run/'analysis/index.html')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);a=p.parse_args();render(a.run)
if __name__=='__main__':main()
