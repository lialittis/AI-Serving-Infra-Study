"""Portable HTML report; measured tasks retain exact evidence identifiers."""
import argparse
from decimal import Decimal
import html
import json
from pathlib import Path

from common import save
from summarize import summarize


def escape(value):
    return html.escape(str(value), quote=True)


def table(headers, rows):
    return '<table><thead><tr>' + ''.join('<th>' + escape(x) + '</th>' for x in headers) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join('<td>' + escape(x) + '</td>' for x in row) + '</tr>' for row in rows) + '</tbody></table>'


def timeline(data, title):
    """SVG is an interactive evidence timeline, not an inferred utilization plot."""
    tasks = [t for t in data['tasks'] if t['is_compute']]
    cpu = {label:s for label,s in data['cpu_scopes'].items() if not label.startswith('P28/replay/')}
    origin = min([Decimal(t['start_us']) for t in tasks] + [Decimal(s['ts']) for s in cpu.values()])
    end = max([Decimal(t['end_us']) for t in tasks] + [Decimal(s['ts'])+Decimal(s['dur']) for s in cpu.values()])
    rows = sorted({(t['role'], t['stream']) for t in tasks}, key=lambda v: (v[0], int(v[1])))
    row_index = {key: i+1 for i, key in enumerate(rows)}
    width, left, scale = 1320, 165, 1100 / float(end-origin)
    height = 112 + 22 * len(rows)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="12" y="22" font-size="16">{escape(title)}</text>']
    for i in range(6):
        x = left + i * 220
        parts += [f'<line x1="{x}" x2="{x}" y1="42" y2="{height-25}" stroke="#ddd"/>',
                  f'<text x="{x}" y="38" font-size="11">{(end-origin)*i/5:.1f} μs</text>']
    for (role, stream), i in row_index.items():
        parts.append(f'<text x="8" y="{61+i*22}" font-size="12">{role} · stream {escape(stream)}</text>')
    parts.append('<text x="8" y="61" font-size="12">CPU submission</text>')
    for label,s in cpu.items():
        x = left + float(Decimal(s['ts'])-origin)*scale
        w = max(.15,float(s['dur'])*scale)
        color = '#b91c1c' if '/join/' in label else '#64748b'
        if '/forward/' in label:
            color = '#2563eb' if label.endswith('/A') else '#d97706'
        tip = escape(label+' '+json.dumps(s))
        parts.append(f'<rect x="{x:.4f}" y="48" width="{w:.4f}" height="15" fill="{color}"><title>{tip}</title></rect>')
    for t in tasks:
        x = left + float(Decimal(t['start_us'])-origin)*scale
        y = 48 + row_index[t['role'], t['stream']]*22
        w = max(.15, float(t['duration_us'])*scale)
        tooltip = escape(json.dumps({k:t.get(k) for k in ('name','role','stream','task_id','model_id','start_us','duration_us','scope','csv_row','host_index','cann_index')}, ensure_ascii=False))
        parts.append(f'<rect x="{x:.4f}" y="{y}" width="{w:.4f}" height="15" fill="{"#2563eb" if t["role"]=="A" else "#d97706"}"><title>{tooltip}</title></rect>')
    parts.append(f'<text x="12" y="{height-6}" font-size="12">CPU scopes + device compute intervals. CPU joins are red. A/B compute overlap: {escape(data["overlap_us"])} μs.</text></svg>')
    return '\n'.join(parts)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    summary = summarize(a.run)
    save(a.output / 'summary.json', summary)
    content = ['<h1>Practice 28 · 原生 decode 单流／双流</h1>',
        '<p>同进程、同提交线程、两个独立的真实 vLLM decode 状态。计时覆盖 forward、logits、sample；状态准备与数值验证在计时外。</p>',
        '<h2>资格与恢复</h2>',
        table(['检查', '结果'], [('套件',summary['status']['status']),
              *[(k,v) for k,v in summary['recovery'].items()]]),
        table(['后端','状态','完成数值检查数'], [(k,v['status']['status'],v['checks']) for k,v in summary['qualification'].items()])]
    diagnostics = [r for g in summary['diagnostics'] for r in g['trials']]
    if diagnostics and all(Decimal(r['overlap_us']) == 0 for r in diagnostics):
        timing_note = ('第二任务的首 kernel/replay 下发晚于第一任务计算结束。' if all(
            r['branch_timing']['second_native_submit_minus_first_compute_end_us'] is not None and
            Decimal(r['branch_timing']['second_native_submit_minus_first_compute_end_us']) > 0 for r in diagnostics) else '')
        content.insert(2, '<p class="notice"><strong>这批诊断没有观察到 A/B 计算重叠。</strong>'+timing_note+
            '是否缩短完成时间，需看下表的无 profiler 数据。此结论限定于本次提交方式及工作量，不能推出硬件不支持多流并行。</p>')
    content.insert(3, '<p><a href="../RESULTS.md">完整结论、只读 RoPE 表审计及限制</a> · <a href="../README.md">复现与代码阅读说明</a></p>')
    if summary['status']['status'] != 'passed':
        content.append('<p class="notice">资格未通过：不提供性能结论。失败不等于已证明多 stream 无效。</p>')
        for name, value in summary['qualification'].items():
            if value['status']['status'] == 'failed':
                content.append('<details open><summary>'+escape(name)+'</summary><pre>'+escape(value['status'].get('traceback',value['status']))+'</pre></details>')
    content += ['<h2>无 profiler 的完成时间</h2>',
        '<p>每后端三个独立进程。加速比为同进程 serial 中位数 ÷ parallel 中位数；大于 1 才表示缩短。区间为三个进程的最小／最大值，不是置信区间。</p>']
    rows = []
    for r in summary['performance']:
        for process in r['processes']:
            s, q = process['strategies']['serial'], process['strategies']['parallel']
            rows.append([r['mode'],r['case'],process['process'],f'{s["wall_us"]["median"]:.2f}',
                f'{q["wall_us"]["median"]:.2f}',f'{process["wall_speedup"]:.3f}×',f'{process["device_speedup"]:.3f}×'])
    content.append(table(['模式','场景','进程','serial μs','parallel μs','wall 加速比','设备 ready 加速比'], rows))
    content += ['<h2>实际设备计算区间</h2>', '<p>plain 和 PipeUtilization 的时间不用于性能结论。每张图来自单独的 pair；颜色区分任务，行区分实际物理 stream。矩形仅表示有 CSV 精确匹配的计算任务，悬停可查看证据标识。</p>']
    diagnostic_rows = []
    for group in summary['diagnostics']:
        for data in group['trials']:
            trial = data['trial']
            diagnostic_rows.append([group['process'], trial['case'], trial['strategy'], trial['order'],
                data['compute_tasks'], len(data['stream_ids']), data['overlap_us'],
                f'{100*data["coverage"]:.1f}%',
                data['branch_timing']['second_native_submit_minus_first_compute_end_us']])
    content.append(table(['采集','场景','策略','提交顺序','计算任务','计算 stream 数','重叠 μs','计算覆盖率',
                          '第二任务首 kernel 下发减第一任务计算结束 μs'], diagnostic_rows))
    content.append('<p>最后一列大于零，表示第二任务的首 kernel（或所在 graph replay）下发晚于第一任务计算结束。它记录先后关系，不独自证明下发滞后的内部原因。</p>')
    for path in sorted(a.run.glob('*-*/trials/*/analysis.json')):
        data = json.loads(path.read_text())
        trial = data['trial']
        name = path.parents[2].name + '-' + path.parent.name
        svg = timeline(data, name)
        (a.output / (name + '.svg')).write_text(svg)
        content.append('<details><summary>'+escape(name)+' · overlap '+escape(data['overlap_us'])+' μs</summary>'+svg+'</details>')
        if trial['profile'] == 'pipe':
            core_rows = []
            for core in sorted(data['cores'], key=lambda c:c['duration_us']['median']*c['count'], reverse=True):
                metrics = '; '.join(f'{k}: {v["median"]:.4g}' for k,v in core['pipeline'].items() if 'ratio' in k or 'utilization' in k)
                core_rows.append([core['role'],core['name'],core['count'],core['core_types'],core['block_nums'],core['mix_block_nums'],
                                  core['unknown_block_tasks'],metrics or '未报告'])
            content.append('<details><summary>'+escape(name)+' · 核字段及流水线计数</summary><p>零／缺失 Block Num 记为 unknown；ratio 是报告的 kernel 指标，不是整卡占用率。</p>'+table(
                ['任务','kernel','次数','核类型','Block Num','Mix Block Num','未知次数','流水线字段中位数'],core_rows)+'</details>')
    content += ['<h2>解释边界</h2><p>计算区间重叠、计算覆盖率、Block Num 与流水线 ratio 是不同指标；都不能独自证明整卡利用率。两个 runner 的状态切换包含实验适配开销，不能直接推导生产服务 QPS。同一 runner 的任意跨流执行也不由本实验担保。</p>',
        '<h2>版本</h2>',table(['远端路径','Git revision'],summary['revisions'].items()),
        '<p>相邻 summary.json 保存统计值；原始结果保存数值检查、输入哈希、存储范围、graph dump、trace 与 kernel_details.csv。代码及完整复现说明见练习 README。</p>']
    document = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Practice 28</title><style>body{max-width:1400px;margin:32px auto;padding:0 20px;font:16px/1.65 system-ui;color:#172033;background:#fafbfc}table{border-collapse:collapse;width:100%;font-size:14px;margin:16px 0}th,td{border:1px solid #ccd4df;padding:8px;text-align:left}th{background:#eaf0f8}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}details{margin:16px 0;padding:12px;border:1px solid #ccd4df;background:white}summary{cursor:pointer}svg{width:100%;height:auto}.notice{background:#fff3cd;padding:16px}</style><body>' + '\n'.join(content) + '</body></html>'
    (a.output / 'index.html').write_text(document)


if __name__ == '__main__':
    main()
