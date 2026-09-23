"""Render a standalone, offline timeline from validated Practice 12 evidence."""
import argparse
from decimal import Decimal
import html
import json
from pathlib import Path
from analyze_reuse import analyze, end, number


def render(run):
    data,chains,_=analyze(run)
    mode=data['mode']
    mode_label='eager' if mode=='eager' else 'PIECEWISE graph（capture [1]）'
    a,b=data['requests']['A'],data['requests']['B']
    base=number(a['allocation']['ts'])
    events=[]
    def add(title,t,owner,detail):
        events.append(dict(title=title,time_ms=str((t-base)/1000),owner=owner,detail=detail))
    add('A 分配 B1 完成',end(a['allocation']),'A','ref_cnt 0→1；空闲队列 [1]→[]。真实 allocator 分配。')
    add('A prefill 最后一次已观测 KV 访问结束',end(a['steps'][0]['last_kv_access']),'A','24 层 prefill 的 KV 写入 / FIA 已关联；本步主要读取当前 K/V。')
    add('A prefill 原生结果等待返回',end(a['steps'][0]['native_wait']),'A','采样 ID 已通过原生路径拷到 CPU。第一个输出 token 将进入 decode。')
    s=a['steps'][1]
    add('A decode 最后一次已观测 KV 访问结束',end(s['last_kv_access']),'A','第 23 层 FIA 完成；它通过 cache view 读取 B1 中历史及本轮 K/V。')
    for task in s['transfer_tasks']:
        add('A '+task['name']+' 完成',end(task),'A','原生 _to_list 范围中的设备任务，通过实际 flow 关联。与 KV kernel 位于同一 stream。')
    ready_before_wait=all(end(t)<=number(s['native_wait']['ts']) for t in s['transfer_tasks'])
    wait_detail=('本次 event 在 CPU 开始等待前已完成。' if ready_before_wait else '本次设备传输或 event 完成与主机等待范围重叠。')+'不能将等待范围耗时直接当作 KV kernel 耗时。'
    add('A decode 原生 Event 等待返回',end(s['native_wait']),'A',wait_detail)
    add('A 释放 B1 完成',end(a['pool_free']),'FREE','ref_cnt 1→0；空闲队列 []→[1]。这不是物理内存清零。')
    add('B 分配同一 B1 完成',end(b['allocation']),'B','同一个 CPU block 对象与 24 层相同的 K/V 设备存储，现在属于 B 的新一次分配。')
    first=min((c['kernel'] for c in chains if c['role']=='B'),key=lambda k:number(k['ts']))
    add('B 首次 KV 写入 kernel 开始',number(first['ts']),'B','本轮输入来自 B 的不同 prompt；写入目标仍为 B1。不读取设备值来比较 A/B 内容。')
    add('B decode 最后一次已观测 KV 访问结束',end(b['steps'][1]['last_kv_access']),'B','B 的四十八个 decode KV/FIA kernel 中最后一个结束。')
    add('B decode 原生结果等待返回',end(b['steps'][1]['native_wait']),'B','原生 D2H、event record、event synchronize 完成。')
    add('B 释放 B1 完成',end(b['pool_free']),'FREE','空闲队列恢复 [1]，KV pool 仍存在，服务关闭发生在采集之后。')
    events.sort(key=lambda e:Decimal(e['time_ms']))
    views=[]
    for i in range(24):
        layer='model.layers.{}.self_attn.attn'.format(i)
        views.append(dict(layer=layer,cache=a['steps'][0]['storage'][layer]))
    payload=dict(events=events,layers=views,counts={k:data[k] for k in ['scopes','device_tasks','kernel_csv_rows','verified_kv_operator_chains']})
    svg=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1080 235" role="img"><title>真实 B1 分配生命周期</title><desc>流程示意，间距不表示耗时。A 原生结果等待结束后才释放，B 随后复用。</desc><rect width="1080" height="235" fill="#f4f7f5"/><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10" fill="#4c7775"/></marker></defs>']
    boxes=[('A 分配 B1','ref_cnt = 1'),('A 访问 / 原生等待','24 层 · NPU→CPU'),('A 释放 B1','ref_cnt = 0'),('B 分配同一 B1','ref_cnt = 1'),('B 访问 / 原生等待','相同存储，新请求'),('B 释放 B1','ref_cnt = 0')]
    for i,(title,subtitle) in enumerate(boxes):
        x=20+i*175
        svg.append('<rect x="{}" y="58" width="160" height="94" rx="8" fill="{}" stroke="#b7cfca"/><text x="{}" y="94" text-anchor="middle" font-family="sans-serif" font-size="15" fill="#173e45">{}</text><text x="{}" y="124" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#476569">{}</text>'.format(x,'#e1efea' if i not in (2,5) else '#fff',x+80,title,x+80,subtitle))
        if i<5:svg.append('<path d="M{},106h15" stroke="#4c7775" marker-end="url(#arrow)"/>'.format(x+160))
    svg.append('<text x="20" y="198" font-family="sans-serif" font-size="13" fill="#426166">真实 allocator + 真实 NPU 任务；串行 __MODE__ 基线。没有人工 generation 字段，也没有额外插入设备等待。</text></svg>')
    dest=run/'analysis';dest.mkdir(exist_ok=True)
    text='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Practice 12 · B1 的真实生命周期</title>
<style>body{margin:0;background:#f4f7f5;color:#173e45;font:15px/1.8 system-ui,sans-serif}main{max-width:1120px;margin:auto;padding:38px 24px}h1{font-size:34px;line-height:1.4}h2{font-size:22px}a{color:#087d78}code{font-family:monospace;overflow-wrap:anywhere}small{color:#546e73}.card{padding:23px;background:white;border:1px solid #cddcd8;border-radius:9px;margin:20px 0}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}.controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap}button,select{font:inherit;padding:7px 12px;background:white;border:1px solid #b8cec8;border-radius:5px;color:#17434a;cursor:pointer}input{flex:1;accent-color:#087d78}button:focus-visible,input:focus-visible,select:focus-visible,a:focus-visible{outline:3px solid #b97024}table{width:100%;border-collapse:collapse;font-size:13px}td,th{padding:10px;border-bottom:1px solid #dce6e1;text-align:left}tr.active{background:#e1f0e9}tr[data-index]{cursor:pointer}.overflow{overflow:auto}.state{font-size:32px;color:#087d78;font-weight:700}.badge{background:#e6f0ea;padding:5px 9px;font-size:12px;border-radius:4px}.note{border-left:3px solid #ac722a;background:#fff1dc;padding:16px}.diagram svg{width:100%;min-width:740px;display:block}#detail{min-height:170px}li{margin:6px 0}@media(max-width:720px){.grid{grid-template-columns:1fr}main{padding:22px 16px}h1{font-size:28px}}@media print{.controls{display:none}.grid{display:block}.card{break-inside:avoid}}</style>
<main><small>ASCEND 910B2C · QWEN2.5-0.5B · PRACTICE 12 / __RUN__</small><h1>同一个 B1，先属于 A，再属于 B。</h1><p>原生配置把池限制为两个物理块：B0 保留作 null，B1 可分配。A、B 各有 126 输入 / 2 输出，顺序请求、__MODE__、无 prefix caching / async scheduling。</p>
<p>__MODE_NOTE__</p><div class="overflow diagram">__SVG__</div><div class="card"><div class="controls"><button id="prev">← 上一步</button><button id="play">播放</button><button id="next">下一步 →</button><label for="step">步骤</label><input id="step" type="range" min="0" value="0"><output id="count"></output></div><div class="grid"><div id="detail" aria-live="polite"></div><div><p>B1 当前占用（按选定里程碑）</p><div class="state" id="owner"></div><p id="ref"></p><small>状态来自真实 allocator 前后快照。这里只在已观测的边界更新，不声称精确定位到了引用计数修改的某条 CPU 指令。</small></div></div></div>
<div class="note"><b>验证到的是这一轮的执行顺序，不是通用 race 检测。</b> A 的最后一次已观测 KV 访问完成 → 原生结果等待返回 → 释放 → B 再分配 → B 写入。B 在 A 的 HTTP 响应后才发送；没有研究两请求重叠或异步调度下的复用。</div>
<div class="card"><h2>同一 profiler 时间轴上的里程碑</h2><p><small>相对 A 的 pool allocation 范围起点，单位 ms。图与播放间隔不按真实时间缩放；这里的数值来自原始 trace。</small></p><div class="overflow"><table><thead><tr><th>时间 ms</th><th>事件</th><th>B1 占用</th></tr></thead><tbody id="rows"></tbody></table></div></div>
<div class="card"><h2>24 层的存储身份都已核对</h2><label for="layer">选择层 </label><select id="layer"></select><div id="storage"></div><p><small>A/B、prefill/decode 的 cache 地址、shape、stride 与 offset 一致。不复制 NPU 数值，不把相同地址解释为相同逻辑生命周期。每层有自己的 K 和 V。</small></p></div>
<div class="grid"><div class="card"><h2>关联依据</h2><p id="counts"></p><p>每条 KV/FIA 链均核对 async_npu、HostToDevice、CANN connection ID 和计算 kernel CSV；结果传输另核对 MEMCPY_ASYNC 与 EVENT_RECORD。</p><p>CPU 等待来自原生 _to_list 中的 transfer_event.synchronize()。观测脚本没有增加设备同步或数据读回。</p></div><div class="card"><h2>明确保留的边界</h2><ul><li>KV 访问是按算子语义、参数和 layer 关联，并非设备内存指令追踪。</li><li>原生等待与设备完成分别核对；不能将等待耗时当作 KV 计算耗时。</li><li>block free 不等于物理清零；没有读取旧数据来检查清零或逐元素覆盖。</li><li>prefix sharing、并发请求、FULL graph、async scheduling 尚未在这里验证。</li></ul></div></div>
<p><a href="../../../RESULTS.md">实验结论</a> · <a href="reuse_evidence.json">结构化证据</a> · <a href="operator_links.json">192 条算子链</a> · <a href="lifetime_trace.json">时间线摘录</a> · <a href="lifecycle.svg">独立 SVG</a></p></main>
<script id="data" type="application/json">__DATA__</script><script>
const d=JSON.parse(document.getElementById('data').textContent),$=id=>document.getElementById(id);let index=0,timer=null;
function stop(){clearInterval(timer);timer=null;$('play').textContent='播放'}
function show(){let e=d.events[index];$('step').value=index;$('count').textContent=(index+1)+' / '+d.events.length;$('owner').textContent=e.owner==='FREE'?'FREE':e.owner+' 持有 B1';$('ref').textContent=e.owner==='FREE'?'ref_cnt=0 · free_queue=[1]':'ref_cnt=1 · free_queue=[]';$('detail').innerHTML='<h2>'+e.title+'</h2><span class="badge">t = '+Number(e.time_ms).toFixed(4)+' ms</span><p>'+e.detail+'</p>';$('prev').disabled=index===0;$('next').disabled=index===d.events.length-1;document.querySelectorAll('[data-index]').forEach(r=>r.classList.toggle('active',Number(r.dataset.index)===index))}
$('step').max=d.events.length-1;$('rows').innerHTML=d.events.map((e,i)=>'<tr data-index="'+i+'"><td>'+Number(e.time_ms).toFixed(4)+'</td><td>'+e.title+'</td><td>'+e.owner+'</td></tr>').join('');document.querySelectorAll('[data-index]').forEach(r=>r.onclick=()=>{stop();index=Number(r.dataset.index);show()});$('step').oninput=e=>{stop();index=Number(e.target.value);show()};$('prev').onclick=()=>{stop();index--;show()};$('next').onclick=()=>{stop();index++;show()};$('play').onclick=()=>{if(timer){stop();return}if(index===d.events.length-1)index=0;show();$('play').textContent='暂停';timer=setInterval(()=>{index++;show();if(index===d.events.length-1)stop()},2800)};document.addEventListener('visibilitychange',()=>{if(document.hidden)stop()});
$('layer').innerHTML=d.layers.map((e,i)=>'<option value="'+i+'">Layer '+i+'</option>').join('');function layer(){let e=d.layers[Number($('layer').value)];$('storage').innerHTML='<p><code>'+e.layer+'</code></p><table><tr><th>Tensor</th><th>池 data_ptr</th><th>B1 第 0 槽的地址</th></tr>'+e.cache.map((c,i)=>'<tr><td>'+(i===0?'K':'V')+'</td><td><code>'+c.data_ptr+'</code></td><td><code>'+(c.data_ptr+c.stride[0]*c.element_size)+'</code></td></tr>').join('')+'</table><p><code>shape=[2,128,2,64] · BF16 · npu:0</code></p>'}$('layer').onchange=layer;$('counts').textContent=d.counts.scopes+' 个主机范围；'+d.counts.verified_kv_operator_chains+' 条 KV/FIA 链；全窗口 '+d.counts.device_tasks+' 个设备任务 / '+d.counts.kernel_csv_rows+' 行计算 kernel CSV。';show();layer();
</script></html>'''
    text=text.replace('__MODE__',mode_label).replace('__RUN__',html.escape(run.name))
    mode_note=('所有步骤使用 eager，未观测到设备图重放。' if mode=='eager' else
               'Prefill：25个编译callable；decode：25个普通分区重放，attention/KV仍直接调用。全窗口50个MODEL_EXECUTE；'+str(data['replayed_tasks'])+'个重放任务的逐FX节点关联尚未建立。这不影响本次直接执行的KV/FIA链。')
    text=text.replace('__MODE_NOTE__',mode_note)
    svg=[part.replace('__MODE__',mode_label) for part in svg]
    (dest/'lifecycle.svg').write_text('\n'.join(svg)+'\n')
    text=text.replace('__SVG__','\n'.join(svg)).replace('__DATA__',json.dumps(payload,ensure_ascii=False).replace('<','\\u003c'))
    (dest/'reuse_viewer.html').write_text(text)
    (dest/'milestones.json').write_text(json.dumps(events,ensure_ascii=False,indent=2)+'\n')
    print('Rendered',dest/'reuse_viewer.html')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);args=p.parse_args();render(args.run)
if __name__=='__main__':main()
