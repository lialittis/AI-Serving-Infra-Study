"""Summarize the fixed 128-input/64-output, concurrency 1 vs 4 experiment."""

import argparse
import json
from pathlib import Path


def load_result(directory, concurrency):
    path = directory / f"concurrency-{concurrency}.json"
    result = json.loads(path.read_text())
    expected = {
        "max_concurrency": concurrency, "num_prompts": 20, "completed": 20,
        "failed": 0, "total_input_tokens": 2560, "total_output_tokens": 1280,
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError(f"{path}: expected {key}={value}, got {result.get(key)!r}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?",
                        default=Path(__file__).parent / "results" / "2026-09-18")
    args = parser.parse_args()
    one = load_result(args.directory, 1)
    four = load_result(args.directory, 4)
    print("| 指标 | 并发 1 | 并发 4 |")
    print("| --- | ---: | ---: |")
    for label, key in (
        ("成功请求", "completed"), ("失败请求", "failed"),
        ("总耗时 (s)", "duration"), ("请求吞吐 (req/s)", "request_throughput"),
        ("输出吞吐 (token/s)", "output_throughput"),
        ("平均 TTFT (ms)", "mean_ttft_ms"), ("平均 TPOT (ms)", "mean_tpot_ms"),
        ("P99 TTFT (ms)", "p99_ttft_ms"), ("P99 TPOT (ms)", "p99_tpot_ms"),
    ):
        print(f"| {label} | {one[key]:.2f} | {four[key]:.2f} |")
    ratio = four["output_throughput"] / one["output_throughput"]
    e2e = [r["mean_ttft_ms"] + 63 * r["mean_tpot_ms"] for r in (one, four)]
    print(f"\n输出吞吐倍数：{ratio:.3f}×。")
    print(f"固定输出 64 token 时，估算平均请求耗时：{e2e[0]:.2f} → {e2e[1]:.2f} ms。")
    print("估算公式为 TTFT + 63 × TPOT；它不包含客户端并发限流之前的等待时间。")
    print("每组仅 20 个请求；P99 和小幅差异不适合用于稳定性能结论。")


if __name__ == "__main__":
    main()
