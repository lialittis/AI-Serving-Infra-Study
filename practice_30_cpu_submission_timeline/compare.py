"""Compare matched eager/PIECEWISE requests; never compare profiler as baseline."""
import argparse
from html import escape
import json
from pathlib import Path
import statistics


def read(path):return json.loads(path.read_text())


def compare_outputs(a,b):
    shared=[];same_keys=True;ranks_equal=True
    for left,right in zip(a['logprobs'],b['logprobs']):
        same_keys &= left.keys()==right.keys()
        for token in left.keys() & right.keys():
            shared.append(abs(left[token]['logprob']-right[token]['logprob']))
            ranks_equal &= left[token]['rank']==right[token]['rank']
    return dict(tokens_equal=a['tokens']==b['tokens'],finish_reason_equal=a['finish_reason']==b['finish_reason'],
        logprob_steps_equal=len(a['logprobs'])==len(b['logprobs']),logprob_keys_equal=same_keys,
        shared_logprob_values=len(shared),shared_logprob_max_abs=max(shared,default=None),
        shared_ranks_equal=ranks_equal,exact_output_equal=a==b)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('eager',type=Path);p.add_argument('graph',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--eager-report',default='../eager-02/index.html')
    p.add_argument('--graph-report',default='../graph-01/index.html');a=p.parse_args()
    runs={'eager':a.eager,'graph':a.graph};details={}
    rev=[read(r/'revisions.json') for r in runs.values()]
    assert rev[0]==rev[1], 'version mismatch'
    sources=[read(r/'sources_before.json') for r in runs.values()]
    assert sources[0]==sources[1], 'audited source mismatch'
    settings=[read(r/'diagnostic/settings.json') for r in runs.values()]
    base=[{k:v for k,v in s.items() if k not in ('enforce_eager','compilation_config')} for s in settings]
    assert base[0]==base[1], 'base settings mismatch'
    prompts=[read(r/'diagnostic/request.json') for r in runs.values()]
    assert prompts[0]==prompts[1], 'workload mismatch'
    for mode,root in runs.items():
        evidence=read(root/'analysis/evidence.json');s=read(root/'analysis/summary.json')
        assert s['exact']['mode']==mode and read(root/'status.json')['status']=='passed'
        ref=read(root/'reference/responses.json')
        phases=[r for r in evidence['phases'] if r['index']==32 and r['kind']!='request']
        kinds=sorted({r['kind'] for r in phases})
        details[mode]=dict(reference_wall_us=[r['wall_us'] for r in ref],
            reference_thread_cpu_us=[r['thread_cpu_us'] for r in ref],
            reference_median_us=statistics.median(r['wall_us'] for r in ref),
            diagnostic=s['request_phase'],overhead=s['overhead'],sync=s['sync_by_phase'],
            focused_phases={k:dict(calls=sum(r['kind']==k for r in phases),
                wall_us=sum(r['wall_us'] for r in phases if r['kind']==k),
                thread_cpu_us=sum(r['thread_cpu_us'] for r in phases if r['kind']==k),
                self_wall_us=sum(r['self_wall_us'] for r in phases if r['kind']==k)) for k in kinds},
            exact=s['exact'],graph=s['graph'],top_gaps=evidence['top_gaps'],
            initialization=s['graph']['initialization'])
    outputs=compare_outputs(read(a.eager/'reference/responses.json')[0]['output'],
                            read(a.graph/'reference/responses.json')[0]['output'])
    result=dict(runs={k:str(v) for k,v in runs.items()},revisions=rev[0],matched_settings=True,
        matched_source_files=len(sources[0]),outputs=outputs,modes=details,
        reference_eager_over_graph=details['eager']['reference_median_us']/details['graph']['reference_median_us'],
        limits=['Three unprofiled requests per mode in isolated processes; not a throughput benchmark.',
                'Diagnostic phase comparisons include mode-dependent instrumentation overhead.',
                'Nested phase durations are not additive; replay is inside forward.',
                'No-compute coverage is not whole-chip idle or core utilization.'])
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    rows=[]
    def row(label,left,right):rows.append(f'<tr><th>{escape(label)}</th><td>{escape(str(left))}</td><td>{escape(str(right))}</td></tr>')
    for label,key in [('无 profiler 中位数 ms','reference_median_us')]:
        row(label,*(f'{d[key]/1000:.3f}' for d in details.values()))
    for key,label in [('wall_us','诊断请求墙钟 ms'),('thread_cpu_us','诊断请求线程 CPU ms')]:
        row(label,*(f'{d["diagnostic"][key]/1000:.3f}' for d in details.values()))
    for key in ('forward','prepare_inputs','sampling','token_to_list','logprobs_to_cpu','replay'):
        row('decode 32 '+key+' 墙钟 / CPU µs',*(f'{d["focused_phases"][key]["wall_us"]:.3f} / {d["focused_phases"][key]["thread_cpu_us"]:.3f}'
            if key in d['focused_phases'] else '未发生' for d in details.values()))
    for key,label in [('graph_replays','请求内 replay 次数'),('physical_streams','记录中的物理 stream 数'),('overlap_us','计算任务跨 stream 重叠 µs')]:
        row(label,*(d['exact'][key] for d in details.values()))
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Practice 30 · eager / graph 对照</title><style>body{font:16px/1.7 system-ui;max-width:1100px;margin:32px auto;padding:18px;color:#203545}table{border-collapse:collapse;width:100%}td,th{padding:10px;border:1px solid #ccd7df;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere}a{color:#2457ac}</style>
<h1>Practice 30 · 单请求 eager / graph</h1><p>相同模型、输入、输出长度和基础配置。graph 使用 PIECEWISE、capture size=1；捕获与编译在稳态请求之前。</p>
<p><a href="../eager-02/index.html">eager 精确时间线</a> · <a href="../graph-01/index.html">graph 精确时间线</a></p>
<p>无 profiler 请求用于对比延迟；诊断请求用于解释时序，不能把 profiler 下的时间差直接视为性能收益。阶段有嵌套：replay 属于 forward，不可重复累加。</p>
<table><tr><th>观测项</th><th>eager</th><th>graph PIECEWISE</th></tr>'''+''.join(rows)+'</table>'
    page=page.replace('../eager-02/index.html',escape(a.eager_report,quote=True)).replace('../graph-01/index.html',escape(a.graph_report,quote=True))
    page+='<h2>输出核验</h2><pre>'+escape(json.dumps(outputs,ensure_ascii=False,indent=2))+'</pre>'
    page+='<p>跨模式 logprob 不预设逐位一致，上面保留实际差值；模式内部参考、诊断与恢复输出另作精确一致性验证。多个 graph 内部 stream 不代表并行执行。</p>'
    page+='<h2>观测影响</h2><pre>'+escape(json.dumps({k:d['overhead'] for k,d in details.items()},ensure_ascii=False,indent=2))+'</pre>'
    page+='<p>完整数值与边界见 <a href="comparison.json">comparison.json</a>。同步调用不代表全部潜在等待；未覆盖的设备区间不证明整颗 NPU 空闲。</p></html>'
    (a.output/'index.html').write_text(page)
    print(json.dumps(dict(outputs=outputs,reference_eager_over_graph=result['reference_eager_over_graph']),indent=2))


if __name__=='__main__':main()
