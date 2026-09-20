"""Validate and summarize an archived real request trace, without vLLM/NPU."""

import argparse
from collections import Counter
import json
from pathlib import Path


def summarize(directory):
    events = sorted((json.loads(line)
                     for path in (directory / "events").glob("*.jsonl")
                     for line in path.read_text().splitlines()), key=lambda row: row["monotonic_ns"])
    by_kind = {}
    for event in events:
        by_kind.setdefault(event["event"], []).append(event)
    if by_kind.get("trace_error"):
        raise ValueError(f"instrumentation errors: {by_kind['trace_error']}")
    required = ("api_receive", "api_response", "scheduler_config", "engine_config",
                "scheduler_enqueue", "schedule", "worker_call", "runner_call",
                "model_forward", "attention_first_layer", "engine_output",
                "request_finish", "request_cleanup_return")
    missing = [kind for kind in required if not by_kind.get(kind)]
    if missing:
        raise ValueError(f"missing evidence: {missing}")
    response = json.loads((directory / "response.json").read_text())
    request = json.loads((directory / "request.json").read_text())
    if len(by_kind["api_receive"]) != 1 or len(by_kind["scheduler_enqueue"]) != 1:
        raise ValueError("expected exactly one inference request")
    if response["usage"]["completion_tokens"] != request["max_tokens"]:
        raise ValueError("unexpected output token count")
    rid = by_kind["scheduler_enqueue"][0]["request_id"]
    outputs = [out for event in by_kind["engine_output"] for out in event["outputs"]]
    if any(out["request_id"] != rid for out in outputs):
        raise ValueError("unrelated request in engine output")
    if sum(len(out["new_token_ids"]) for out in outputs) != request["max_tokens"]:
        raise ValueError("engine output token count does not match HTTP response")
    config = by_kind["scheduler_config"][0]
    if config["prefix_caching"] or config["async_scheduling"] or config["chunked_prefill"]:
        raise ValueError("baseline configuration differs from this exercise")
    schedule = by_kind["schedule"]
    for kind in ("worker_call", "runner_call", "model_forward", "attention_first_layer"):
        if len(by_kind[kind]) != len(schedule):
            raise ValueError(f"{kind} does not cover every scheduled step")
    lines = ["# Practice 07：实际请求追踪摘要", "", f"- Engine request ID：`{rid}`",
             f"- 输入 / 输出 token：{response['usage']['prompt_tokens']} / {response['usage']['completion_tokens']}",
             f"- HTTP finish_reason：`{response['choices'][0]['finish_reason']}`",
             f"- Executor：`{by_kind['engine_config'][0]['executor']}`",
             f"- Worker：`{by_kind['worker_call'][0]['implementation']}`",
             f"- Runner：`{by_kind['runner_call'][0]['implementation']}`",
             f"- 模型：`{by_kind['model_forward'][0]['model']}`",
             f"- Attention：`{by_kind['attention_first_layer'][0]['implementation']}`",
             f"- block_size：{config['block_size']}（本轮不解析映射）", "",
             "## 实际进程与线程", "", "| 观察点 | PID | TID |", "|---|---:|---:|"]
    for kind in ("api_receive", "schedule", "worker_call", "runner_call", "api_response"):
        event = by_kind[kind][0]
        lines.append(f"| {kind} | {event['pid']} | {event['tid']} |")
    lines += ["", "## 调度与模型输入", "",
              "| step | 本步前已计算 token | 本步调度 token | input_ids shape | positions shape |",
              "|---:|---:|---:|---|---|"]
    for event, forward in zip(schedule, by_kind["model_forward"]):
        if event["step"] != forward["step"]:
            raise ValueError("step correlation unavailable; extend tracer for this executor")
        lines.append(f"| {event['step']} | {event['before'][rid]['computed_tokens']} | "
                     f"{event['scheduled_tokens'][rid]} | {forward['input_ids']['shape']} | "
                     f"{forward['positions']['shape']} |")
    lines += ["", "## 观察到的源码入口", "", "| 事件 | 实际源码位置 |", "|---|---|"]
    for kind in required:
        event = by_kind[kind][0]
        source = event["source"]
        lines.append(f"| {kind} | `{source['file']}:{source['line']}` `{source['function']}` |")
    lines += ["", "## 证据边界", "",
              "- 使用 sys.setprofile/threading.setprofile 观察 Python call/return，不代表 NPU 完成时间。",
              "- 启动预热不计入用户请求的 model_forward/attention 事件，只保留真实 scheduler 执行期间的调用。",
              "- attention 只记录每个用户请求执行步的第一层，避免每层重复日志。",
              "- cleanup 只证明 scheduler 清理函数返回，不证明底层 NPU 内存被释放或清零。",
              "- 当前摘要的逐步关联针对单请求、同步调度、同进程 worker；其他配置需重新验证。",
              "- 这是带插桩的架构实验，不是性能基准。", "",
              "事件计数：`" + json.dumps(Counter(event['event'] for event in events), ensure_ascii=False) + "`", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = summarize(args.directory)
    (args.directory / "summary.md").write_text(result)
    print(result)


if __name__ == "__main__":
    main()
