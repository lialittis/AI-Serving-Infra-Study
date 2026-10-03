"""Interval sweep of the fill_exponential cross-stream reuse pattern.

Replicates the structure of vllm-ascend sample/sampler.py fill_exponential
(default non-greedy sampling path): q is allocated and written on the producer
stream, consumed by the main stream after wait_stream, released at "function
return", and the next iteration allocates on the producer stream again with no
leading wait. exponential_ is substituted by iteration-indexed fill_ so element
classification can identify whose data each consumer copy actually read.
"""
import argparse
from contextlib import nullcontext
import datetime
import hashlib
import inspect
import json
import os
from pathlib import Path
import threading
import time


def save(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def stats(npu):
    data = npu.memory_stats()
    return {key: data[key] for key in (
        "num_alloc_retries", "num_ooms", "allocated_bytes.all.current",
        "reserved_bytes.all.current", "active_bytes.all.current")}


def classify(observed_value, expected):
    if observed_value == expected:
        return "intact"
    return "overwritten"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("asis", "record", "leading-wait"), required=True)
    parser.add_argument("--gaps-us", type=int, nargs="+",
                        default=[0, 200, 1000, 5000, 25000, 50000])
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--elements", type=int, default=1 << 20)
    parser.add_argument("--consumer-backlog", type=int, default=0,
                        help="matrix multiplies enqueued on main before the consuming copy; "
                             "stands in for sampling kernels and main-stream congestion")
    parser.add_argument("--matrix", type=int, choices=(2048, 4096), default=4096)
    args = parser.parse_args()
    if not 2 <= args.iterations <= 100 or not 0 <= min(args.gaps_us) or max(args.gaps_us) > 1_000_000:
        parser.error("iterations 2..100, gaps 0..1000000 us required")
    if not 0 <= args.consumer_backlog <= 128:
        parser.error("consumer-backlog 0..128 required")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)

    import torch
    import torch_npu
    npu = torch_npu.npu
    npu.set_device(0)
    if npu.get_allocator_backend() != "native":
        raise RuntimeError("This experiment requires the native allocator")
    if torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("Deterministic uninitialized-memory filling would write candidates")
    npu.config.allow_internal_format = False
    producer = npu.Stream()   # analog of global_stream (e.g. stream 44)
    main = npu.current_stream()  # analog of the model/sampling stream (e.g. stream 46)
    elements = args.elements

    # Warm-up: exercise the exact allocation, fill, copy and backlog paths so
    # the allocator pools hold same-size blocks and the kernels are compiled.
    with npu.stream(producer):
        warm = [torch.empty((elements,), device="npu", dtype=torch.float32) for _ in range(3)]
        for tensor in warm:
            tensor.fill_(0.5)
    observed = [torch.empty((elements,), device="npu", dtype=torch.float32)
                for _ in range(args.iterations)]
    warm_probe = torch.empty((elements,), device="npu", dtype=torch.float32)
    with npu.stream(main):
        left = torch.full((args.matrix, args.matrix), 1 / 64, device="npu", dtype=torch.float16)
        right = torch.full_like(left, 1 / 64)
        scratch = torch.empty_like(left)
        for _ in range(3):
            torch.mm(left, right, out=scratch)
    npu.synchronize()
    with npu.stream(main):
        warm_probe.copy_(warm[0])
    npu.synchronize()
    if not bool((warm_probe == 0.5).all().item()):
        raise RuntimeError("Warm-up copy validation failed")
    if not bool(torch.all(scratch == args.matrix / 4096).item()):
        raise RuntimeError("Warm-up matrix validation failed")
    del warm, warm_probe

    # Cache stream handles before measurement (P32 lesson: npu_stream and
    # Stream.__hash__ drain host queues). No hashes or handle reads in the loop.
    stream_handles = {id(producer): str(producer.npu_stream), id(main): str(main.npu_stream)}
    stream_ids = {id(producer): str(producer.stream_id), id(main): str(main.stream_id)}
    npu.memory._record_memory_history(enabled="all", context=None, stacks="python", max_entries=10000)
    before_stats = stats(npu)
    traces_before = len(npu.memory._snapshot()["device_traces"][0])
    records = []
    intervals = []

    def run_interval(gap_us):
        # Each gap starts from an idle, fully synchronized device so intervals
        # do not inherit pending work from the previous one.
        npu.synchronize()
        rows = []
        for index in range(args.iterations):
            expected = index + 1
            with npu.stream(producer):
                if args.mode == "leading-wait":
                    # Fix analog of do_async_exponential's leading
                    # global_stream().wait_stream(current_stream()).
                    producer.wait_stream(main)
                q = torch.empty((elements,), device="npu", dtype=torch.float32)
                address = q.data_ptr()
                q.fill_(expected)
            # fill_exponential line 41 analog: consumer waits for the producer.
            main.wait_stream(producer)
            with npu.stream(main):
                for _ in range(args.consumer_backlog):
                    torch.mm(left, right, out=scratch)
                observed[index].copy_(q)   # probs.div_(q) analog: reads q
            if args.mode == "record":
                q.record_stream(main)      # register the consumer stream use
            del q                          # "function return" frees the block
            rows.append(dict(index=index, expected=expected, address=str(address)))
            if gap_us:
                time.sleep(gap_us / 1e6)
        # All device work for this interval has been submitted; wait, classify.
        npu.synchronize()
        for index, row in enumerate(rows):
            values = observed[index]
            is_expected = bool((values == row["expected"]).all().item())
            row["all_elements_expected"] = is_expected
            if not is_expected:
                row["mismatch_count"] = int((values != row["expected"]).sum().item())
                foreign = sorted({float(v) for v in torch.unique(values).tolist()
                                  if float(v) != float(row["expected"])})
                row["foreign_values"] = foreign
            row["outcome"] = "intact" if is_expected else "overwritten"
        for earlier in range(len(rows) - 1):
            rows[earlier]["next_address_reused"] = \
                rows[earlier]["address"] == rows[earlier + 1]["address"]
        rows[-1]["next_address_reused"] = None
        return rows

    start_mono = time.monotonic_ns()
    for gap_us in args.gaps_us:
        rows = run_interval(gap_us)
        intervals.append(dict(gap_us=gap_us, rows=rows))
        corrupt = sum(row["outcome"] == "overwritten" for row in rows)
        reuse = sum(bool(row["next_address_reused"]) for row in rows[:-1])
        print(f"gap={gap_us}us corrupt={corrupt}/{args.iterations} adjacent_reuse={reuse}/{args.iterations - 1}",
              flush=True)
    after_stats = stats(npu)
    end_mono = time.monotonic_ns()
    traces = npu.memory._snapshot()["device_traces"][0]
    first_address = int(intervals[0]["rows"][0]["address"])
    relevant = [dict(trace_index=i, **event) for i, event in enumerate(traces)
                if i >= traces_before and event.get("addr") == first_address]

    keys = ("ASCEND_HOME_PATH", "ASCEND_RT_VISIBLE_DEVICES", "TASK_QUEUE_ENABLE", "PER_STREAM_QUEUE",
            "PYTORCH_NPU_ALLOC_CONF", "PYTORCH_NO_NPU_MEMORY_CACHING", "ASCEND_LAUNCH_BLOCKING",
            "MULTI_STREAM_MEMORY_REUSE")
    installed = {}
    for name, path in (("probe.py", Path(__file__)),
                       ("streams.py", Path(inspect.getfile(npu.Stream))),
                       ("memory.py", Path(inspect.getfile(npu.memory)))):
        dest = out / "sources" / name
        dest.parent.mkdir(exist_ok=True)
        dest.write_bytes(path.read_bytes())
        installed[str(dest.relative_to(out))] = hashlib.sha256(dest.read_bytes()).hexdigest()
    save(out / "run.json", dict(schema=1, captured_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
         arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
         pid=os.getpid(), torch_version=torch.__version__, torch_npu_version=torch_npu.__version__,
         torch_git_version=torch.version.git_version, torch_npu_git_version=torch_npu.version.git_version,
         device=npu.get_device_name(0), allocator_backend=npu.get_allocator_backend(),
         environment={key: os.environ.get(key) for key in keys}, allow_internal_format=False,
         deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
         streams={"producer": dict(handle=stream_handles[id(producer)], id=stream_ids[id(producer)]),
                  "main": dict(handle=stream_handles[id(main)], id=stream_ids[id(main)])},
         tensor_bytes=elements * 4, elements=elements, source_sha256=installed,
         duration_mono_ns=end_mono - start_mono, thread_id=threading.get_native_id(),
         substitution="exponential_ is replaced by fill_(iteration+1); the device-write task, "
                      "its stream and the release timing keep the audited structure",
         memory_bound_description="iterations x 4 MiB observed buffers plus one 4 MiB q at a time; "
                                  "warm-up blocks cached; runtime/workspace additional",
         candidate_access="q is written on the producer stream and read once by the main-stream copy",
         records=records, trace_entries_for_first_address=relevant, intervals=intervals,
         statistics_before=before_stats, statistics_after=after_stats))
    print(f"Completed {len(args.gaps_us)} gaps x {args.iterations} iterations, mode={args.mode}", flush=True)


if __name__ == "__main__":
    main()
