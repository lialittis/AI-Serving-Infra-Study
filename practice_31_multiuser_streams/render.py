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
