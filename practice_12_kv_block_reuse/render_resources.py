"""Offline, searchable per-event / per-replay resource evidence viewer."""
import argparse
import html
import json
from pathlib import Path
from audit_resources import audit


def render(run, data=None):
    data=audit(run) if data is None else data
    summary=data['summary']
    # Full records live in the JSON ledger. Embed all events for file:// offline use.
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Practice 12 · 资源与执行记录</title><style>
:root{color-scheme:light}body{margin:0;background:#f3f6f5;color:#173c46;font:15px/1.7 system-ui,sans-serif}main{max-width:1280px;margin:auto;padding:32px 24px}h1{font-size:32px;line-height:1.4}h2{font-size:20px}a{color:#05766f}small{color:#546c71}button,input,select{font:inherit;padding:8px 12px;border:1px solid #bacdc8;border-radius:5px;background:white;color:#173c46}button{cursor:pointer}button[aria-pressed=true]{background:#087c74;color:white}.tabs,.filters{display:flex;gap:10px;flex-wrap:wrap;margin:18px 0}input{flex:1;min-width:160px}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.card{border:1px solid #d3dfdb;border-radius:7px;background:white;padding:18px}.stat{font-size:26px;color:#087c74}.note{border-left:4px solid #b37c2b;background:#fff2db;padding:17px;margin:22px 0}.grid{display:grid;grid-template-columns:340px minmax(0,1fr);gap:18px}.list{max-height:850px;overflow:auto}.item{display:block;width:100%;text-align:left;margin-bottom:7px;overflow-wrap:anywhere}.item.active{border-color:#087c74;background:#e4f1ec}pre{font:12px/1.55 monospace;white-space:pre-wrap;overflow-wrap:anywhere;max-height:470px;overflow:auto;padding:12px;background:#f3f7f5}code{overflow-wrap:anywhere}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border-bottom:1px solid #dce4df;padding:9px;text-align:left;vertical-align:top}.overflow{overflow:auto}details{margin-top:16px}summary{cursor:pointer;color:#05766f}.status{color:#77541c}#detail{min-width:0}button:focus-visible,input:focus-visible,select:focus-visible{outline:3px solid #bb7f23}@media(max-width:760px){main{padding:22px 14px}.grid{grid-template-columns:1fr}.list{max-height:300px}.stats{grid-template-columns:repeat(2,1fr)}h1{font-size:26px}}@media print{.list,.tabs,.filters{display:none}.grid{display:block}}</style>
<main><small>PRACTICE 12 / __RUN__ / __MODE__</small><h1>这次执行使用了什么资源，<br>哪些条件有证据，哪些还不知道？</h1><p>eager按观测事件浏览；graph可逐次选择实际NPUGraph.replay。两种模式均保留全窗口CPU算子和设备任务清单。</p><div class="stats">__STATS__</div><div class="note"><b>“地址相同”不等于“资源安全”。</b>本页区分捕获地址/布局核对、主机提交、原生完成等待和逻辑block归属。未知项保留为未知；没有额外设备同步或NPU数值读回。每个ATen算子的全部tensor地址、隐藏workspace生命周期尚未观测。</div>
<div class="tabs"><button data-tab="events">观测事件</button><button data-tab="replays">每次 Replay</button><button data-tab="cpu_events">全部 CPU 事件</button><button data-tab="device_tasks">全部设备任务</button><button data-tab="completion_boundaries">完成边界</button><button data-tab="preparation_chains">准备依赖链</button></div><div class="filters"><label>请求 <select id="role"><option value="">全部 / 未归属</option><option>A</option><option>B</option></select></label><input id="search" aria-label="搜索事件" placeholder="搜索名称、partition、地址或事件 ID"><span id="count"></span></div>
<div class="grid"><div><div id="list" class="list"></div><button id="more">再显示 80 条</button></div><section id="detail" class="card" aria-live="polite"></section></div><p><a href="resource_ledger.json">完整资源台账 JSON</a> · <a href="resource_summary.json">覆盖范围</a> · <a href="reuse_viewer.html">KV生命周期</a> · <a href="../../../README.md">复现方法</a></p></main>
<script id="payload" type="application/json">__DATA__</script><script>
const d=JSON.parse(document.getElementById('payload').textContent),$=id=>document.getElementById(id),esc=x=>String(x??'未观测').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let tab=d.summary.mode==='graph'?'replays':'events',selected=null,limit=80;
function title(x){return x.partition||x.kind||x.trace?.name||x.id}function raw(x){return '<pre>'+esc(JSON.stringify(x,null,2))+'</pre>'}
function detail(x){selected=x.id;let text='<small>'+esc(x.id)+'</small><h2>'+esc(title(x))+'</h2>';
if(tab==='replays'){text+='<p>请求 '+esc(x.role)+' · step '+esc(x.step)+' · graph object '+esc(x.graph_object_id)+'</p><p>本轮完成边界：<code>'+esc(x.completion_boundary)+'</code></p><h3>与捕获时核对</h3>'+raw(x.checks)+'<p class="status">主机replay返回仅表示提交返回。未按时间为MODEL_EXECUTE或内部kernel分配归属；隐藏workspace生命周期没有完整证明。</p><h3>输入资源</h3>'+raw(x.resources_before.arguments)+'<details><summary>捕获基线、输出、别名与完整记录</summary>'+raw(x)+'</details>'}
else if(tab==='events'){text+='<p>请求 '+esc(x.role)+' · '+esc(x.phase)+' · step '+esc(x.step)+'</p><p>责任：'+esc(x.responsibility)+'</p><p>父事件：<code>'+esc(x.parent)+'</code></p><p>完成边界：<code>'+esc(x.completion_boundary)+'</code></p><p>精确flow关联设备任务：'+x.direct_device_tasks.length+'</p><h3>逻辑归属</h3>'+raw(x.logical_owner)+'<h3>边界处可见的资源</h3>'+raw(x.resources)+'<details><summary>进入 / 返回状态与完整记录</summary>'+raw(x)+'</details>'}
else{text+='<p class="status">'+esc(x.attribution||x.resource_detail||x.evidence||x.note)+'</p>'+raw(x)}
$('detail').innerHTML=text;document.querySelectorAll('.item').forEach(b=>b.classList.toggle('active',b.dataset.id===selected));}
function show(){const q=$('search').value.toLowerCase(),role=$('role').value;let all=d[tab].filter(x=>(!role||x.role===role)&&(!q||JSON.stringify(x).toLowerCase().includes(q)));$('count').textContent=all.length+' 条';$('list').innerHTML=all.slice(0,limit).map(x=>'<button class="item" data-id="'+esc(x.id)+'">'+esc(title(x))+'<br><small>'+esc(x.role||'未归属')+' · '+esc(x.id)+'</small></button>').join('');$('more').hidden=all.length<=limit;document.querySelectorAll('.item').forEach(b=>b.onclick=()=>detail(all.find(x=>x.id===b.dataset.id)));document.querySelectorAll('[data-tab]').forEach(b=>b.setAttribute('aria-pressed',b.dataset.tab===tab));if(all.length)detail(all.find(x=>x.id===selected)||all[0]);else $('detail').innerHTML='<p>此筛选下没有记录。eager不存在设备图replay。</p>';}
document.querySelectorAll('[data-tab]').forEach(b=>b.onclick=()=>{tab=b.dataset.tab;selected=null;limit=80;$('role').value='';$('search').value='';show()});$('role').onchange=()=>{limit=80;show()};$('search').oninput=()=>{limit=80;show()};$('more').onclick=()=>{limit+=80;show()};show();
</script></html>'''
    stats=[('观测事件',summary['observed_scopes']),('Replay',summary['replay_records']),('CPU事件',summary['cpu_events']),('设备任务',summary['device_tasks'])]
    page=page.replace('__RUN__',html.escape(run.name)).replace('__MODE__',summary['mode'])
    page=page.replace('__STATS__',''.join('<div class="card"><small>'+name+'</small><div class="stat">'+str(count)+'</div></div>' for name,count in stats))
    page=page.replace('__DATA__',json.dumps(data,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c'))
    (run/'analysis/resource_viewer.html').write_text(page)
    print('Rendered',run/'analysis/resource_viewer.html')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);a=p.parse_args()
    # Use the independently validated audit from this run; regenerate it explicitly after changes.
    ledger=a.run/'analysis/resource_ledger.json'
    render(a.run,json.loads(ledger.read_text()) if ledger.exists() else None)

if __name__=='__main__':main()
