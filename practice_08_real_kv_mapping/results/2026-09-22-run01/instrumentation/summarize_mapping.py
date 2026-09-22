"""Validate real manager -> CPU table -> NPU table -> slot -> KV writes."""

import argparse
import csv
import io
import json
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def analyze(root):
    events = sorted((json.loads(line) for path in (root / "events").glob("*.jsonl")
                     for line in path.read_text().splitlines()), key=lambda e: e["monotonic_ns"])
    kinds = {}
    for event in events:
        kinds.setdefault(event["event"], []).append(event)
    require(not kinds.get("trace_error"), f"trace errors: {kinds.get('trace_error')}")
    required = ("api_receive", "api_response", "scheduler_enqueue", "scheduler_config",
                "kv_allocate", "schedule", "kv_manager_blocks", "runner_block_table",
                "token_positions", "attention_kv_metadata", "kv_write_call", "kv_write_check",
                "engine_output", "request_finish", "request_cleanup_return")
    require(all(kinds.get(name) for name in required),
            "missing evidence: " + str([name for name in required if not kinds.get(name)]))
    request = json.loads((root / "request.json").read_text())
    response = json.loads((root / "response.json").read_text())
    require(len(kinds["scheduler_enqueue"]) == len(kinds["api_receive"]) == 1, "expected one request")
    require(kinds["api_response"][0]["response"] == response, "HTTP response mismatch")
    prompt_tokens = len(request["prompt"])
    output_tokens = request["max_tokens"]
    require(response["usage"]["prompt_tokens"] == prompt_tokens, "prompt token count mismatch")
    require(response["usage"]["completion_tokens"] == output_tokens, "output token count mismatch")
    require(response["choices"][0]["finish_reason"] == "length", "unexpected finish reason")
    config = kinds["scheduler_config"][0]
    require(config["block_size"] == 128 and not config["prefix_caching"]
            and not config["async_scheduling"] and not config["chunked_prefill"], "unsupported configuration")
    rid = kinds["scheduler_enqueue"][0]["request_id"]
    by_step = {}
    for name in ("schedule", "kv_manager_blocks", "runner_block_table", "token_positions",
                 "attention_kv_metadata", "kv_write_call", "kv_write_check", "engine_output"):
        rows = kinds[name]
        require(len(rows) == output_tokens, f"incomplete {name}")
        require([row["step"] for row in rows] == list(range(1, output_tokens + 1)), f"bad steps: {name}")
        by_step[name] = rows
    require(len(kinds["kv_allocate"]) == output_tokens, "allocation evidence incomplete")
    csv_rows, summary_rows = [], []
    computed = 0
    previous_blocks = []
    cache_addresses = None
    for index in range(output_tokens):
        step = index + 1
        count = prompt_tokens if index == 0 else 1
        expected_positions = list(range(computed, computed + count))
        schedule = by_step["schedule"][index]
        require(schedule["scheduled_tokens"] == {rid: count}, f"scheduled tokens mismatch step {step}")
        require(schedule["before"][rid]["computed_tokens"] == computed, f"computed tokens mismatch step {step}")
        allocation = kinds["kv_allocate"][index]
        require(allocation["success"] and allocation["request_id"] == rid, "allocation failed/unrelated")
        manager_groups = by_step["kv_manager_blocks"][index]["blocks"][rid]
        require(len(manager_groups) == 1, "only one full-attention KV group is supported")
        blocks = manager_groups[0]
        require(len(blocks) == (computed + count + 127) // 128, f"capacity mismatch step {step}")
        require(blocks[:len(previous_blocks)] == previous_blocks, "existing block mapping moved")
        require(allocation["request_block_ids"] == manager_groups, "allocation/manager disagree")
        require(allocation["new_block_ids"] == [blocks[len(previous_blocks):]], "new block IDs disagree")
        table = by_step["runner_block_table"][index]
        require(table["request_ids"] == [rid] and len(table["groups"]) == 1, "runner request/group mismatch")
        group = table["groups"][0]
        require(group["rows"] == [blocks], "CPU block table mismatch")
        require(group["block_size"] == group["physical_block_size"] == 128
                and group["blocks_per_phys_block"] == 1, "unsupported split blocks")
        positions = by_step["token_positions"][index]["positions"]
        metadata = by_step["attention_kv_metadata"][index]
        require(positions == metadata["positions"] == expected_positions, "position mismatch")
        require(metadata["block_tables"] == [blocks], "NPU block table mismatch")
        expected_slots = [blocks[p // 128] * 128 + p % 128 for p in positions]
        require(metadata["slot_mapping"] == expected_slots, f"attention slot mismatch step {step}")
        write = by_step["kv_write_call"][index]
        require(write["slot_mapping"] == expected_slots, f"write slot mismatch step {step}")
        require(write["key"]["shape"] == write["value"]["shape"] == [count, 2, 64], "unexpected new K/V shape")
        for name in ("key_cache", "value_cache"):
            require(write[name] == metadata[name], "attention/operator cache views differ")
            require(write[name]["shape"][1:] == [128, 2, 64], "unsupported KV layout")
            require(write[name]["shape"][0] > max(blocks), "block outside cache")
        addresses = (write["key_cache"]["data_ptr"], write["value_cache"]["data_ptr"])
        if cache_addresses is None:
            cache_addresses = addresses
        require(addresses == cache_addresses, "KV backing storage changed during request")
        check = by_step["kv_write_check"][index]
        require(check["tokens_checked"] == count and check["key_equal"] and check["value_equal"],
                f"actual KV write verification failed step {step}")
        out = by_step["engine_output"][index]["outputs"]
        require(len(out) == 1 and out[0]["request_id"] == rid and len(out[0]["new_token_ids"]) == 1,
                "unexpected engine output")
        for position, slot in zip(positions, expected_slots):
            csv_rows.append([step, position, position // 128, blocks[position // 128], position % 128, slot])
        summary_rows.append([step, positions[0], positions[-1], blocks, allocation["new_block_ids"][0],
                             expected_slots[0], expected_slots[-1]])
        computed += count
        previous_blocks = blocks
    require(computed > 128, "the experiment did not cross a block boundary")
    finish = kinds["request_finish"][0]
    require(finish["computed_tokens"] == computed and finish["output_tokens"] == output_tokens,
            "final request counters mismatch")
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["step", "token_position", "logical_block", "physical_block", "offset", "slot"])
    writer.writerows(csv_rows)
    lines = ["# Practice 08：真实 KV 映射校验", "", f"- Request：`{rid}`",
             f"- 输入 / 输出：{prompt_tokens} / {output_tokens} token",
             f"- 已计算并核对第一层 K/V 写入：{computed} 个 token 位置",
             f"- 第一层：`{by_step['attention_kv_metadata'][0]['layer']}`",
             f"- 最终 request block IDs（单 KV group）：`{previous_blocks}`", "",
             "| step | token positions | request blocks | 新分配 blocks | slots |",
             "|---:|---|---|---|---|"]
    for step, start, end, blocks, new, s0, s1 in summary_rows:
        lines.append(f"| {step} | {start}..{end} | {blocks} | {new} | {s0}..{s1} |")
    lines += ["", "## 第一层 KV 布局", ""]
    first = by_step["kv_write_call"][0]
    for name in ("key_cache", "value_cache"):
        info = first[name]
        lines += [f"- `{name}`：shape=`{info['shape']}`，stride=`{info['stride']}`，",
                  f"  dtype=`{info['dtype']}`，device=`{info['device']}`，element_size={info['element_size']} bytes。"]
    lines += ["", "所有步均通过 manager block IDs → CPU block table → NPU block table → slot_mapping 校验。",
              "每步第一层写入后，按观测到的 slot 读取实际 K/V cache，与算子输入逐元素完全相等。", "",
              "这是有同步和额外读操作的诊断运行，不用于性能或 race 结论；不覆盖其他层或其他请求。",
              "slot 是缓存中的 token 槽编号，不是 token ID，也不是 NPU 物理内存地址。", ""]
    return "\n".join(lines), output.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    summary, table = analyze(args.directory)
    (args.directory / "summary.md").write_text(summary)
    (args.directory / "token_mapping.csv").write_text(table)
    print(summary)


if __name__ == "__main__":
    main()
