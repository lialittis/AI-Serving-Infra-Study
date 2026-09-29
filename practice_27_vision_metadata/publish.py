"""Package replayable raw evidence and offline P27 reports."""
import argparse
import base64
import gzip
import hashlib
import html
import importlib.util
import json
import shutil
from pathlib import Path

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('p25_report',HERE.parent/'practice_25_multimodal_overlap/render_report.py')
report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)


def read(p):return json.loads(p.read_text())
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('raw',type=Path);parser.add_argument('archive',type=Path);args=parser.parse_args()
    run=args.raw/'formal-r01';out=HERE/'results/published';out.mkdir(exist_ok=True);web=HERE/'report';web.mkdir(exist_ok=True)
    summary=read(run/'analysis/summary.json');assert summary['status']=='passed'
    for file in run.glob('*.json'):shutil.copyfile(file,out/file.name)
    shutil.copyfile(run/'analysis/summary.json',out/'summary.json')
    audits={p.stem:read(p) for p in sorted((run/'analysis').glob('*-sync.json'))};save(out/'sync_audit.json',audits)
    for variant in ('native','lengths','cached'):
        path=run/'analysis'/f'{variant}-graph.json.gz';shutil.copyfile(path,out/path.name)
        with gzip.open(path,'rt') as f:data=json.load(f)
        ids={n['id'] for n in data['nodes'] if n.get('trial')}
        view=dict(summary=data['summary'],nodes=[n for n in data['nodes'] if n['id'] in ids],
            edges=[e for e in data['edges'] if e['source'] in ids and e['target'] in ids],requirements=data['requirements'])
        packed=base64.b64encode(gzip.compress(json.dumps(view,ensure_ascii=False,separators=(',',':')).encode(),mtime=0)).decode()
        page=report.HTML.replace('PRACTICE 25','PRACTICE 27 · '+variant.upper()).replace('P3b · 视觉与语言并发','P27 · '+variant)
        page=page.replace('<h1>请求 A 的语言计算与请求 B 的视觉编码</h1>',f'<h1>{variant}：视觉元数据对照</h1><p><a href="index.html">返回三组比较</a> · 元数据预计算在计时外，准备成本见总报告。</p>')
        page=page.replace('../results/published/execution_graph.json.gz',f'../results/published/{variant}-graph.json.gz')
        page=page.replace('仅有 API 证据的等待保存在完整图中，不画成设备任务。','仅有 API 证据的等待保存在完整图中，不画成设备任务。默认 stream 可能仅承载起始 event；双流指 L/V 两条计算 stream。')
        (web/f'{variant}.html').write_text(page.replace('__DATA__',packed));del data,view,packed
    parts=[]
    with args.archive.open('rb') as f:
        while block:=f.read(48*1024*1024):
            p=out/f'evidence.tgz.part-{len(parts):02d}';p.write_bytes(block);parts.append(dict(file=p.name,bytes=len(block),sha256=digest(p)))
    save(out/'archives.json',dict(parts=parts,reassembled_sha256=digest(args.archive),bytes=args.archive.stat().st_size,
        remote='/data/tianchi/practice_27_vision_metadata/results',omitted='CANN buffers, databases and framework binary ranges remain remote; formal trace_view and kernel CSV retained in full'))
    save(out/'evidence_manifest.json',{str(p.relative_to(args.raw)):dict(bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(args.raw.rglob('*')) if p.is_file() and 'analysis' not in p.relative_to(args.raw).parts})
    save(out/'excluded_runs.json',{'qualification-r01':'Missing phase argument in harness call; failed before case construction; no performance samples.',
        'qualification-r02':'One-case numerical smoke passed for all six variant/mode combinations; excluded from performance.',
        'formal-r01':'Complete balanced formal run; sole source of performance statistics.'})
    table=[]
    for variant,s in summary['variants'].items():
        for p in s['performance']:
            for order in ('LV','VL'):
                modes=p['by_order'][order];t=next(t for t in s['trials'] if t['case']==p['case'] and t['mode']=='parallel' and t['order']==order)
                vals=[p['case'],variant,order,f"{modes['serial']['wall_us']['median']/1000:.3f}",f"{modes['parallel']['wall_us']['median']/1000:.3f}",
                    f"{p['paired'][order]['change_percent']['median']:+.2f}%",f"{t['overlap_us']/1000:.3f}",str(t['vision_native_syncs'])]
                table.append('<tr data-order="'+order+'" data-variant="'+variant+'">'+''.join('<td>'+html.escape(v)+'</td>' for v in vals)+'</tr>')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>P27 视觉元数据与同步</title><style>body{font:16px/1.6 system-ui;color:#193740;background:#f0f4f5;margin:0}main{max-width:1250px;padding:24px;margin:auto}section{padding:18px;background:white;margin:20px 0;border-radius:8px}.scroll{overflow:auto}table{border-collapse:collapse;white-space:nowrap}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}a{color:#187e83}select{padding:8px;margin:8px}h1{font-size:28px}</style><main><small>PRACTICE 27 · QWEN2.5-VL-3B · BF16 EAGER · ASCEND 910B2C</small><h1>视觉元数据预计算与同步消减</h1><p>比较原实现 native、仅预计算 attention 分段长度 lengths，以及完整固定形状元数据 cached。真实像素始终重新经过全部视觉网络。</p><section><h2>离线 kernel 执行图</h2><p><a href="native.html">native 原实现</a> · <a href="lengths.html">lengths 分段长度</a> · <a href="cached.html">cached 完整元数据</a></p><p>每个变体含 8 个场景 × 2 种流模式 × 2 种提交顺序；三个变体合计 96 个诊断 trial。数据依赖按事件和 FIFO 独立核验；不是全部底层访存 DAG。</p></section><section><h2>无 profiler 性能与独立诊断</h2><p>576 个正式计时样本；816 次资格/正式/诊断输出检查全部逐元素一致。表中单/双流时间为 ms/pair 中位数，变化为逐轮配对百分比的中位数。重叠和同步来自另跑的 profiler，不与无 profiler 计时混为同一轮。</p><label>提交顺序<select id="order"><option value="">全部</option><option>LV</option><option>VL</option></select></label><label>变体<select id="variant"><option value="">全部</option><option>native</option><option>lengths</option><option>cached</option></select></label><div class="scroll"><table id="comparison"><thead><tr><th>场景</th><th>变体</th><th>顺序</th><th>单流 ms</th><th>双流 ms</th><th>双流变化</th><th>计算重叠 ms</th><th>视觉同步 API</th></tr></thead><tbody>__ROWS__</tbody></table></div></section><section><h2>计时边界与代价</h2><p>计时覆盖 A 语言、B 视觉和后续 B 语言；A 视觉/前缀、输入传输与预处理在计时外。元数据准备耗时和常驻字节另见报告；复用场景收益不能直接解释为首次请求端到端收益。同步 API 可能嵌套，调用数不等于独立等待次数。一次进程、同日交替测量不代表跨日统计显著性。</p><p><a href="../RESULTS.md">完整结果与准备成本</a> · <a href="../README.md">复现方法</a> · <a href="../results/published/summary.json">机器可读汇总</a></p></section></main><script>function filter(){for(const r of document.querySelectorAll('#comparison tbody tr'))r.hidden=(document.querySelector('#order').value&&r.dataset.order!==document.querySelector('#order').value)||(document.querySelector('#variant').value&&r.dataset.variant!==document.querySelector('#variant').value)}document.querySelector('#order').onchange=filter;document.querySelector('#variant').onchange=filter;</script></html>'''
    (web/'index.html').write_text(page.replace('__ROWS__',''.join(table)))
    print('PUBLISHED',len(parts),'archive parts',flush=True)


if __name__=='__main__':main()
