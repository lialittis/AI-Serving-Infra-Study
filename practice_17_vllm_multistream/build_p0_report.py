"""Publish compact P0 measurements and diagnostic timelines from local evidence."""
import argparse
from collections import defaultdict, deque
from decimal import Decimal
import hashlib
from html import escape
import json
from pathlib import Path
import shutil

from analyze_sampling_benchmark import analyze


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def validate_dag(graph):
    ids = {node["id"] for node in graph["nodes"]}
    if len(ids) != len(graph["nodes"]):
        raise ValueError("duplicate graph nodes")
    degrees = dict.fromkeys(ids, 0)
    targets = defaultdict(list)
    for edge in graph["edges"]:
        if edge["source"] not in ids or edge["target"] not in ids:
            raise ValueError("edge endpoint missing")
        degrees[edge["target"]] += 1
        targets[edge["source"]].append(edge["target"])
    ready = deque(node for node, degree in degrees.items() if degree == 0)
    visited = 0
    while ready:
        node = ready.popleft()
        visited += 1
        for target in targets[node]:
            degrees[target] -= 1
            if degrees[target] == 0:
                ready.append(target)
    if visited != len(ids):
        raise ValueError("execution graph contains a cycle")
    return {"acyclic": True, "nodes": len(ids), "edges": len(graph["edges"])}


def timeline(graph, destination):
    steps = graph["steps"]
    maximum = max(s["request_count"] for s in steps)
    candidates = [s for s in steps if s["step"] > 2 and s["request_count"] == maximum]
    step = candidates[0] if candidates else steps[-1]
    tasks = [t for t in graph["tasks"] if t["step"] == step["step"] and t["is_compute"]]
    start = min(Decimal(t["start_us"]) for t in tasks)
    finish = max(Decimal(t["end_us"]) for t in tasks)
    span = finish - start
    streams = sorted({t["stream"] for t in tasks})
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="230" viewBox="0 0 1200 230">',
             '<rect width="1200" height="230" fill="white"/>',
             '<g font-family="sans-serif" font-size="14" fill="#182330">',
             '<text x="20" y="25">%s; step %s; requests %s; overlap %s us</text>' %
             (escape(graph["summary"]["model"] + " / " + graph["summary"]["mode"]),
              step["step"], step["request_count"], step["overlap_us"])]
    for i, stream in enumerate(streams):
        y = 65 + i * 55
        parts.append('<text x="20" y="%d">stream %s</text>' % (y + 19, escape(stream)))
        parts.append('<line x1="125" x2="1170" y1="%d" y2="%d" stroke="#d8e0e8"/>' % (y+30, y+30))
        for task in tasks:
            if task["stream"] != stream:
                continue
            x = 125 + 1045 * (Decimal(task["start_us"]) - start) / span
            width = 1045 * Decimal(task["duration_us"]) / span
            kind = task["scope_kind"]
            color = "#e88924" if kind in ("async_exponential", "inline_exponential") else (
                "#287ab8" if kind == "forward" else "#78838f")
            title = escape("%s | %s | %s us" % (task["id"], task["name"], task["duration_us"]))
            parts.append('<rect x="%.4f" y="%d" width="%.4f" height="28" fill="%s"><title>%s</title></rect>' %
                         (x, y, width, color, title))
    parts += ['<text x="125" y="185">0</text>',
              '<text x="1020" y="185">%.3f us</text>' % span,
              '<text x="20" y="215">Blue: forward; orange: random branch; gray: sampler. Actual compute intervals; gaps retained.</text>',
              '</g></svg>']
    destination.write_text("\n".join(parts) + "\n")
    return step["step"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--llama", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--numerics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for name, source in [("qwen", args.qwen), ("llama", args.llama)]:
        summaries[name] = analyze(source)
        destination = output / "benchmarks" / name
        shutil.copytree(source, destination)
        save(destination / "summary.json", summaries[name])
    numerics = json.loads((args.numerics / "result.json").read_text())
    if numerics["status"] != "passed":
        raise ValueError("numerical validation failed")
    save(output / "numerics.json", numerics)
    traces = []
    for run in sorted(args.diagnostics.iterdir()):
        if not (run / "analysis" / "execution_graph.json").exists():
            continue
        graph = json.loads((run / "analysis" / "execution_graph.json").read_text())
        dag_check = validate_dag(graph)
        destination = output / "diagnostics" / run.name
        destination.mkdir(parents=True, exist_ok=False)
        for name in ["request.json", "command.json", "source_manifest.json", "artifact_manifest.json"]:
            if (run / name).exists():
                shutil.copy2(run / name, destination / name)
        save(destination / "summary.json", {"summary": graph["summary"], "steps": graph["steps"]})
        step = timeline(graph, destination / "timeline.svg")
        traces.append({"run": run.name, "representative_step": step, "dag_check": dag_check,
                       **graph["summary"]})
    if len(traces) != 8:
        raise ValueError("expected eight diagnostic traces")
    save(output / "summary.json", {"benchmarks": summaries, "diagnostics": traces,
                                  "numerical_status": numerics["status"]})
    lines = ["# P0：无 profiler 性能与独立 trace", "",
             "正的耗时降幅表示提前生成更快；负值表示更慢。所有耗时均为完整 HTTP 批量请求的接收时间。", "",
             "| 模型 | 提交 batch | 输出 token | 开 / ms（中位数） | 关 / ms（中位数） | 耗时降幅 | AB / BA 降幅 | 判断 |",
             "|---|---:|---:|---:|---:|---:|---|---|"]
    labels = {"uncertain": "不确定", "consistent_improvement_in_this_run": "本轮改善方向一致",
              "consistent_regression_in_this_run": "本轮退化方向一致"}
    for name, summary in summaries.items():
        for case in summary["cases"]:
            pairs = " / ".join("%+.2f%%" % p for p in case["pair_latency_reduction_percent"])
            lines.append("| %s | %d | %d | %.3f | %.3f | %+.2f%% | %s | %s |" %
                         (name, case["batch_size"], case["output_tokens"], case["enabled"]["median_ms"],
                          case["disabled"]["median_ms"], case["median_latency_reduction_percent"],
                          pairs, labels[case["classification"]]))
    lines += ["", "开／关各 10 个正式样本；预热排除。判断依据为两组方向一致且合并样本的四分位区间不重叠，",
              "仅作描述性判断，不是统计显著性检验。原始响应与测量位于 `benchmarks/`。", "",
              "## 单独采集的诊断", "", "这些 trace 有 profiler 和 Python 观测开销，不用于解释上述性能差值的大小。", "",
              "| 运行 | 实际 step | 重叠 step | compute 交集 / us | 逐 kernel 对齐数 | 代表性 step 时间线 |",
              "|---|---:|---:|---:|---:|---|"]
    for trace in traces:
        lines.append("| %s | %s | %s | %s | %s | [step %s](diagnostics/%s/timeline.svg) |" %
                     (trace["run"], trace["steps"], trace["overlap_steps"],
                      trace["total_model_random_overlap_us"], trace["kernel_csv_rows_verified"],
                      trace["representative_step"], trace["run"]))
    lines += ["", "代表时间线优先选择第一个满批 decode step，显示真实 kernel 区间与空隙，悬停可看任务名。",
              "完整 execution graph、原始 trace 与 CSV 在证据包中；这里保留逐 step 摘要及证据清单。", "",
              "独立数值验证见 [numerics.json](numerics.json)。当前设备仍报告历史硬件告警 `80C98001`；",
              "结果仅描述该主机本轮条件，不代表健康设备上的普遍性能。"]
    (output / "README.md").write_text("\n".join(lines) + "\n")
    manifest = {str(p.relative_to(output)): {"bytes": p.stat().st_size,
                 "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                for p in sorted(output.rglob("*")) if p.is_file() and p.name != "manifest.json"}
    save(output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
