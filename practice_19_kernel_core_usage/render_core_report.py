"""Build a dependency-free offline report from validated evidence."""
import json


def render(data, output):
    cases = data['cases']
    capacity = data['hardware']['vector_core_num']
    points = [(70+i*78, 280-c['reported_vector_cores']*4) for i,c in enumerate(cases)]
    svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 840 340" role="img" aria-label="Token count versus reported Vector cores"><rect width="840" height="340" fill="#101c30"/><g fill="#e3edf9" font-family="sans-serif" font-size="14"><text x="35" y="28">KV write: reported Vector cores (AI_VECTOR_CORE)</text><path d="M55 65V280H800" stroke="#889ab5" fill="none"/><path d="M55 {0}H800" stroke="#e6b567" stroke-dasharray="5 5"/><text x="500" y="{1}">device capacity: {2} Vector cores</text>'.format(280-capacity*4,55,capacity)
    svg += '<polyline points="{}" fill="none" stroke="#62d4c7" stroke-width="3"/>'.format(' '.join('{},{}'.format(x,y) for x,y in points))
    for c,(x,y) in zip(cases,points):
        svg += '<circle cx="{}" cy="{}" r="5" fill="#62d4c7"/><text x="{}" y="{}">{}</text><text x="{}" y="305">{}</text>'.format(x,y,x-8,y-12,c['reported_vector_cores'],x-10,c['tokens'])
    svg += '<text x="340" y="330">Input tokens (categorical spacing)</text></g></svg>'
    (output/'core_counts.svg').write_text(svg)
    rows = ''.join('<tr><td>{}</td><td>{}</td><td>{:.3f}</td><td>{:.3f}</td><td>{:.3f}</td></tr>'.format(c['tokens'],c['reported_vector_cores'],c['host_completed_per_call_us']['median'],c['plain_kernel_us']['median'],c['pipe_kernel_us']['median']) for c in cases)
    payload = json.dumps(data,ensure_ascii=False,default=str).replace('<','\\u003c')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Practice 19 · 一个 KV kernel 使用多少核</title>
<style>body{background:#0b1424;color:#e3edf9;font:16px/1.7 system-ui,sans-serif;max-width:1100px;margin:auto;padding:24px}h1{font-size:28px}h2{font-size:21px}a{color:#70ddd0}section{background:#101c30;padding:20px;margin:20px 0;border:1px solid #31425a;border-radius:12px}img{width:100%}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;white-space:nowrap}th,td{text-align:left;padding:8px;border-bottom:1px solid #31425a}select,button{background:#233852;color:white;padding:8px;border:1px solid #889ab5;border-radius:5px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px;max-height:480px;overflow:auto}small,.note{color:#b8c8dc}.bar{height:14px;background:#62d4c7;margin:4px 0 12px}.track{background:#233852}.stats{display:flex;gap:30px;flex-wrap:wrap}.stats strong{font-size:28px;color:#70ddd0}label{display:inline-block;margin-right:15px}@media(max-width:600px){body{padding:12px}section{padding:12px}}</style>
<h1>Practice 19 · 一个 KV kernel 使用多少核？</h1>
<p>真实 Ascend910B2C · ATB ReshapeAndCacheNdKernel · BF16 · 单 stream · eager</p>
<section class="stats"><div><strong>24 / 48</strong><br>设备 Cube / Vector 核</div><div><strong>100 / 100</strong><br>完整 KV 池校验通过</div><div><strong>60</strong><br>精确关联的 KV 设备任务</div></section>
<p>本次固定布局下，报告的 Vector 核数随 token 数增长，到 48 封顶。它是任务的核数记录，不能告诉我们每个物理核的身份或忙碌时间。</p>
<section><h2>输入规模与任务核数</h2><img src="core_counts.svg" alt="1 至 256 token 的核数从 1 增长至 48 后保持 48"><p>横轴等距展示不同测试规模；虚线为设备查询所得容量。实测符合 min(tokens, 48)，仅适用于这组 dtype、shape 和连续 slot；没有改写 kernel 的 tiling 或手动指定核数。</p></section>
<section><h2>从一次调用追到设备任务</h2><label>Token 数 <select id="tokens"></select></label><label>采集模式 <select id="mode"><option value="plain">plain · 无流水线计数器</option><option value="pipe">pipe · PipeUtilization</option></select></label><label>重复 <select id="rep"><option>0</option><option>1</option><option>2</option></select></label><p id="chain"></p><p id="task"></p><p>CPU 调用 → PyTorch host 事件 → async_npu flow → NPU kernel；CANN HostToDevice flow 与 connection_id 再次核对，最后按名称、物理 stream、task ID、开始时间精确匹配 kernel_details.csv。</p><details><summary>查看所选任务的原始关联证据与 CSV 字段</summary><pre id="evidence"></pre></details></section>
<section><h2>耗时：保持两种测量边界</h2><div class="scroll"><table><thead><tr><th>tokens</th><th>Vector 核数</th><th>无 profiler 完成 μs/次</th><th>plain kernel μs</th><th>pipe kernel μs</th></tr></thead><tbody>ROWS</tbody></table></div><p>无 profiler：每轮连续提交 100 次写入，等待末尾 event 完成后停止计时，除以 100；取 7 轮中位数。含 CPU 提交开销和队列间隙，不是孤立 kernel 延迟。plain / pipe：独立采集的 3 个设备任务中位数，计数器可能扰动执行，不能与左列相除作为加速比。</p></section>
<section><h2>所选规模的流水线计数器</h2><div id="pipes"></div><p class="note">来自 pipe 的 3 次采集，展示原始 ratio 的中位数。指标可能重叠，不能相加为饼图，也不是整张芯片的利用率。MTE2 / MTE3 对应数据搬入 / 搬出路径；这组数据不足以断言 HBM 带宽瓶颈。</p></section>
<section><h2>数据与安全边界</h2><p>K、V 输入各为 [N, 2, 64]；两个 KV 池各为 [8, 128, 2, 64]。NPU slot tensor 保存 128…127+N。kernel 根据 slot 将 K/V 写入池中；这里的 KV block（存储分块）与 profiler 的 Block Num（任务核数）是不同概念。</p><p>初始化完成 → 单 stream 重置并等待 → KV 写入 → stream/event 完成等待 → CPU 回读完整池与输入校验。定时段不含重置和校验。反复使用预热后的相同 buffer，不能作为冷 HBM 带宽测量。</p><p>没有模型、vLLM 调度或 graph replay；实验隔离的是此前模型路径上真实使用的 ATB 算子。多于 48 个 token 并不需要超过 48 个核，核可处理更多工作；内部 token 到物理核的逐项分配没有被观测。</p></section>
<p><a href="core_evidence.json">完整派生证据 JSON</a> · <a href="../run.json">采集元数据与正确性</a> · <a href="../../../README.md">复现说明</a> · <a href="core_counts.svg">独立 SVG</a></p>
<script id="data" type="application/json">PAYLOAD</script><script>
const data=JSON.parse(document.getElementById('data').textContent), tok=document.getElementById('tokens'), mode=document.getElementById('mode'), rep=document.getElementById('rep');
for(const c of data.cases){const o=document.createElement('option');o.value=c.tokens;o.textContent=c.tokens;tok.append(o)}
function update(){const n=Number(tok.value),c=data.cases.find(c=>c.tokens===n),t=data.tasks.find(t=>t.tokens===n&&t.profile===mode.value&&t.repeat===Number(rep.value));document.getElementById('chain').textContent=`${t.label} → ${t.host.name} → ${t.name}`;document.getElementById('task').textContent=`物理 stream ${t.physical_stream} · task ${t.task_id} · ${t.core_type} · Block Num ${t.block_num} · Mix Block Num ${t.mix_block_num} · ${t.duration_us} μs`;document.getElementById('evidence').textContent=JSON.stringify(t,null,2);const el=document.getElementById('pipes');el.replaceChildren();for(const k of ['aiv_vec_ratio','aiv_scalar_ratio','aiv_mte2_ratio','aiv_mte3_ratio']){const value=c.pipeline_median[k],label=document.createElement('div'),track=document.createElement('div'),bar=document.createElement('div');label.textContent=`${k}: ${value.toFixed(3)} (${(value*100).toFixed(1)}%)`;track.className='track';bar.className='bar';bar.style.width=Math.max(0,Math.min(100,value*100))+'%';track.append(bar);el.append(label,track)}}
tok.value=String(data.cases.some(c=>c.tokens===48)?48:data.cases[0].tokens);for(const el of [tok,mode,rep])el.addEventListener('change',update);update();</script></html>'''
    page = page.replace('24 / 48', '{} / {}'.format(data['hardware']['cube_core_num'], capacity))
    page = page.replace('100 / 100', '{0} / {0}'.format(data['correctness_checks'])).replace('<strong>60</strong>', '<strong>{}</strong>'.format(len(data['tasks'])))
    (output/'index.html').write_text(page.replace('ROWS',rows).replace('PAYLOAD',payload))
