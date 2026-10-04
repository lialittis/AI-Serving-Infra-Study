"""Self-contained offline report with gzip payload and linked canvas lanes."""
import base64
import gzip
import html
import json
from pathlib import Path


def render(result, path):
    def exact_clocks(value):
        if isinstance(value, dict):
            return {k: str(v) if str(k).endswith("_ns") and isinstance(v, int) else exact_clocks(v)
                    for k, v in value.items()}
        if isinstance(value, list):
            return [exact_clocks(v) for v in value]
        return value
    raw = json.dumps(exact_clocks(result), ensure_ascii=False, separators=(",", ":"), default=str).encode()
    payload = base64.b64encode(gzip.compress(raw, mtime=0)).decode()
    template = Path(__file__).with_name("viewer.html").read_text()
    path.write_text(template.replace("__P31_PAYLOAD__", payload).replace("__P31_TITLE__", html.escape(result["case"])))


def render_index(run, cases, summaries):
    if any("sampling" in s for s in summaries):
        return render_sampling_index(run, cases, summaries)
    rows = []
    by_c = {}
    for case, s in zip(cases, summaries):
        m = s["metrics"]
        d = s.get("device_metrics", {})
        values = [case.name, s.get("analysis_status", "benchmark"), m["requests"],
                  max(t["peak_http_inflight"] for t in m["trials"]),
                  round(sum(t["output_tokens_per_s"] for t in m["trials"]) / len(m["trials"]), 2),
                  round(m["ttft_ms"]["median"], 2), d.get("compute_streams", "—"), d.get("compute_overlap_us", "—")]
        rows.append('<tr><td><a href="%s/analysis/report.html">%s</a></td>%s</tr>' %
                    (case.name, html.escape(str(values[0])), "".join("<td>%s</td>" % html.escape(str(v)) for v in values[1:])))
        by_c.setdefault(s["concurrency"], {})[s["phase"]] = json.loads((case / "requests.json").read_text())
    checks = []
    for c, phases in by_c.items():
        baseline = {}
        changes = []
        for phase in ("benchmark", "diagnostic"):
            for r in phases.get(phase, []):
                key = (r["user"], r["round"])
                if key in baseline and baseline[key] != r["token_ids"]:
                    changes.append(dict(phase=phase, repetition=r["repetition"], user=key[0], round=key[1]))
                baseline.setdefault(key, r["token_ids"])
        checks.append(dict(concurrency=c, phases_present=sorted(phases), output_differences=changes,
                           comparison="same user/round input; compare all repetitions and diagnostic with first observed output"))
    (run / "output_comparison.json").write_text(json.dumps(checks, indent=2) + "\n")
    (run / "index.html").write_text('''<!doctype html><html lang="zh"><meta charset="utf-8"><title>P31 多用户实验</title>
<style>body{font:16px system-ui;margin:35px;line-height:1.6;background:#f5f7fc;color:#172335}table{border-collapse:collapse;background:white}td,th{padding:10px;border:1px solid #ddd}a{color:#1264bb}pre{white-space:pre-wrap}</style>
<h1>P31：单服务多用户与 NPU 执行并发</h1><p>点击案例进入离线交互时间线。benchmark 才用于性能比较；diagnostic 用于解释实际调度与 stream 行为。多个在途请求、合批、多 stream、计算重叠分别计量。</p>
<table><tr><th>案例</th><th>关联验证</th><th>请求数</th><th>峰值 HTTP 在途</th><th>输出 tokens/s（轮均值）</th><th>TTFT 中位数 ms</th><th>计算 streams</th><th>计算重叠 µs</th></tr>''' + "".join(rows) +
    '</table><h2>相同输入的输出核对</h2><p>差异会保留，不能把诊断插桩认定为性能或调度无扰动。不同合批形态也可能影响浮点数值。</p><pre>' + html.escape(json.dumps(checks, ensure_ascii=False, indent=2)) + '</pre></html>')


def render_sampling_index(run, cases, summaries):
    from collections import defaultdict
    import statistics
    grouped=defaultdict(list)
    rows=[]
    for case,s in zip(cases,summaries):
        m=s['metrics'];d=s.get('device_metrics',{});a=s.get('sampling_analysis',{})
        rate=statistics.mean(t['output_tokens_per_s'] for t in m['trials'])
        if s['phase']=='benchmark':grouped[s['concurrency'],s['precompute']].append(dict(case=case.name,tokens_per_s=rate,ttft_ms=m['ttft_ms']['median']))
        values=[s['phase'],'on' if s['precompute'] else 'off',m['requests'],round(rate,2),
                d.get('compute_streams','—'),d.get('compute_overlap_us','—'),a.get('random_forward_overlap_us','—'),
                a.get('overlapping_steps','—'),a.get('synchronization_kind','—'),s.get('analysis_status','benchmark')]
        rows.append('<tr><td><a href="'+case.name+'/analysis/report.html">'+case.name+'</a></td>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in values)+'</tr>')
    comparisons=[]
    for c in sorted({s['concurrency'] for s in summaries}):
        off=grouped[c,False];on=grouped[c,True]
        if off and on:
            om=statistics.mean(b['tokens_per_s'] for b in off);nm=statistics.mean(b['tokens_per_s'] for b in on)
            comparisons.append(dict(concurrency=c,off_blocks=off,on_blocks=on,off_mean_tokens_per_s=om,on_mean_tokens_per_s=nm,
                                    on_vs_off_percent=100*(nm/om-1),order='off,on,on,off',
                                    interpretation='Two service blocks per setting; descriptive result, not statistical proof of a stable speedup.'))
    (run/'sampling_comparison.json').write_text(json.dumps(comparisons,indent=2)+'\n')
    (run/'index.html').write_text('''<!doctype html><html lang="zh"><meta charset="utf-8"><title>P31b 随机采样提前生成对照</title>
<style>body{font:15px system-ui;margin:30px;line-height:1.6;background:#f5f7fc}table{border-collapse:collapse;background:white}td,th{border:1px solid #ccc;padding:7px}pre{white-space:pre-wrap}</style>
<h1>P31b：eager + 随机采样，提前生成开/关</h1><p>temperature=0.8、top_p=0.9；不设置 per-request seed。两组只切换 enable_async_exponential。关闭也可能使用第二条计算 stream。点击案例查看逐 step 的 q/forward 提交、同步和设备任务证据。</p>
<p>全窗口重叠与同一步 q/forward 重叠分别列出；CPU 同步 API 耗时与设备 EVENT_WAIT 时长不是同一个指标。所有性能数值取自无插桩 benchmark；随机输出不要求逐 token 相等。</p>
<table><tr><th>案例</th><th>运行</th><th>提前生成</th><th>请求</th><th>tokens/s</th><th>计算 streams</th><th>全窗口重叠 µs</th><th>同 step q/forward 重叠 µs</th><th>重叠 steps</th><th>q 同步</th><th>验证</th></tr>'''+''.join(rows)+
    '</table><h2>ABBA 无插桩性能</h2><pre>'+html.escape(json.dumps(comparisons,ensure_ascii=False,indent=2))+'</pre></html>')
