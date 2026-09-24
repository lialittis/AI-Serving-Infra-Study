"""Offline synchronized-completion comparison; early return is explicitly separate."""
from html import escape
import json


def render(data,out):
    names={'per_round':'每轮等待','final_only':'最后统一等待'}
    perf=data['performance'];diag=data['diagnostic']
    rows=''.join('<tr><td>%s</td><td>%d</td><td>%.3f</td><td>%.3f</td><td>%.3f–%.3f</td><td>全部正确</td></tr>'%
                 (names[k],v['waits_per_batch'],v['issue_median_ms'],v['completed_median_ms'],
                  v['completed_min_ms'],v['completed_max_ms']) for k,v in perf.items())
    options=''.join('<option value="%s">%s · %s</option>'%(s['id'],names[k],s['id'])
                    for k,v in perf.items() for s in v['samples'])
    samples={s['id']:dict(mode=k,**s) for k,v in perf.items() for s in v['samples']}
    snapshot_rows=''.join('<tr><td>%s / %d / %s</td><td>%s</td><td>%s</td><td>%s</td></tr>'%
                          (r['batch'],r['round'],r['branch'],escape(str(r['observed'])),r['expected'],
                           '已完成' if r['event_ready_after_read'] else '未完成') for r in diag['rows'][:8])
    payload=json.dumps(samples).replace('<','\\u003c')
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Practice 16 · 同步的代价与正确性</title><style>
body{background:#f2f5fa;color:#20314b;font:16px/1.65 system-ui,sans-serif;margin:0}main{max-width:1100px;margin:32px auto;padding:0 20px}h1{font-size:30px;line-height:1.4}h2{font-size:21px}.panel{background:white;border:1px solid #dce4ef;border-radius:12px;padding:20px;margin:20px 0}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;white-space:nowrap;font-variant-numeric:tabular-nums}td,th{text-align:left;padding:9px;border-bottom:1px solid #e2e8f0}.muted{font-size:14px;color:#52647d}select{padding:7px;font:inherit;max-width:100%}.bar{height:30px;display:flex;background:#e8edf5;margin:14px 0;border-radius:5px;overflow:hidden}.issue{background:#3268c7}.wait{background:#d57915}.swatch{display:inline-block;width:12px;height:12px;margin-right:5px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f5fa;padding:15px;font-size:13px}a{color:#235db9}@media(max-width:600px){main{padding:0 12px}.panel{padding:12px}h1{font-size:25px}}
</style><main><p class="muted">PRACTICE 16 / SYNCHRONIZATION COMPARISON</p><h1>CPU 提前返回，计算就更快了吗？</h1>
<p>两个独立 NPU stream：每轮等待完成，或连续提交多轮后统一等待。初始化屏障、流内顺序和最终清理始终保留。性能组没有 profiler，也不在计时中校验输出。</p>
<div class="panel"><h2>同样工作量，比较完成后的耗时</h2><div class="scroll"><table><thead><tr><th>策略</th><th>Host 等待次数</th><th>提交阶段中位数 ms</th><th>完整耗时中位数 ms</th><th>完整耗时范围 ms</th><th>最终结果</th></tr></thead><tbody>ROWS</tbody></table></div>
<p>本次完整耗时的中位数下降 <strong>REDUCTION%</strong>。这是一组固定负载的重复测量，不代表模型吞吐收益。</p>
<p class="muted">“提交阶段”对每轮等待策略包含轮内等待；对最后统一等待策略不包含末尾等待。完整耗时均从第一轮提交开始，到两个 stream 都完成为止；两种模式包含相同的采样拷贝和 event record。</p>
<label for="sample">查看单次测量：</label><select id="sample">OPTIONS</select>
<p class="muted"><span class="swatch issue"></span>提交阶段（可能含每轮等待）　<span class="swatch wait"></span>末尾等待</p><div class="bar"><div id="issue" class="issue"></div><div id="wait" class="wait"></div></div><p id="timing"></p></div>
<div class="panel"><h2>缺少 CPU 消费前的等待，会读到什么？</h2>
<p>独立诊断组：异步 D2H 拷贝提交后，立即读取预填为 <code>-7</code> 的 pinned CPU 缓冲区。BRANCH_READS 次分支采样中，BAD_READS 次值不正确，PENDING 次读取后的 event 查询仍未完成。</p>
<div class="scroll"><table><thead><tr><th>第一次诊断的各分支</th><th>提前读到的值</th><th>应得值</th><th>读取后 event 状态</th></tr></thead><tbody>SNAPSHOTS</tbody></table></div>
<p>最终补上等待后，全部样本和完整输出 tensor 正确。这说明错误发生在<strong>过早使用结果</strong>，不是 NPU 把算术算错。诊断组多了提前读取和 query，不纳入性能对比。</p></div>
<div class="panel"><h2>设备证据</h2><p>另采一轮 profiler：核对 144 个计算 kernel、24 次 D2H 采样拷贝、24 个终点 event，以及 12 次 Host event 等待。每条流上都满足“计算 → 拷贝 → event”；每轮等待有 8 次，最终等待有 2 次，提前读取组也在清理阶段补上 2 次。</p>
<p class="muted">本次默认四轮、每轮每分支六次算子；性能策略各重复七次。精确配置及每次结果以 JSON 为准。没有移除初始化依赖，没有共享输出存储，没有测试 KV block 抢占。</p></div>
<p><a href="sync_evidence.json">原始关联与全部测量 JSON</a> · <a href="../../../SYNC_COMPARISON.md">结论与复现</a> · <a href="../../../README.md">Practice 16</a></p>
<script type="application/json" id="data">PAYLOAD</script><script>
const samples=JSON.parse(document.getElementById('data').textContent);const max=Math.max(...Object.values(samples).map(s=>s.completed_ms))*1.02;
function show(){const s=samples[document.getElementById('sample').value];document.getElementById('issue').style.width=(s.issue_ms/max*100)+'%';document.getElementById('wait').style.width=(s.terminal_wait_ms/max*100)+'%';document.getElementById('timing').textContent='提交阶段 '+s.issue_ms.toFixed(3)+' ms；末尾等待 '+s.terminal_wait_ms.toFixed(3)+' ms；完整耗时 '+s.completed_ms.toFixed(3)+' ms。'}document.getElementById('sample').addEventListener('change',show);show();
</script></main></html>'''
    html=html.replace('ROWS',rows).replace('OPTIONS',options).replace('SNAPSHOTS',snapshot_rows).replace('PAYLOAD',payload)
    html=html.replace('REDUCTION','%.2f'%data['median_completed_reduction_percent'])
    html=html.replace('BRANCH_READS',str(diag['branch_reads'])).replace('BAD_READS',str(diag['incorrect_reads'])).replace('PENDING',str(diag['not_ready_after_read']))
    # The protocol counts are derived, including when CLI round counts change.
    summary=data['summary']
    html=html.replace('144 个计算',str(summary['profiled_compute_tasks'])+' 个计算')
    html=html.replace('24 次 D2H',str(summary['profiled_sample_copies'])+' 次 D2H')
    html=html.replace('24 个终点',str(summary['profiled_event_records'])+' 个终点')
    html=html.replace('12 次 Host',str(summary['profiled_host_waits'])+' 次 Host')
    html=html.replace('每轮等待有 8 次','每轮等待有 %d 次'%perf['per_round']['waits_per_batch'])
    (out/'index.html').write_text(html)
