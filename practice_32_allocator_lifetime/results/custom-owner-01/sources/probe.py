"""Bounded NPU allocator experiment; candidates are never read or written."""
import argparse
from contextlib import contextmanager, nullcontext
import datetime
import gzip
import hashlib
import inspect
import json
import os
from pathlib import Path
import queue
import threading
import time
import weakref


def save(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def locate(snapshot, address):
    """Include merged inactive blocks containing the original address."""
    found = []
    for segment in snapshot["segments"]:
        start = segment["address"]
        for block in segment["blocks"]:
            if start <= address < start + block["size"]:
                found.append(dict(address=start, size=block["size"], state=block["state"],
                                  requested_size=block["requested_size"],
                                  stream=segment["stream"], is_expandable=segment["is_expandable"],
                                  device=segment["device"]))
            start += block["size"]
    return found


def stats(npu):
    data = npu.memory_stats()
    return {key: data[key] for key in (
        "num_alloc_retries", "num_ooms", "allocated_bytes.all.current",
        "reserved_bytes.all.current", "active_bytes.all.current")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("omit", "record", "join"), required=True)
    parser.add_argument("--threads", type=int, choices=(1, 2), default=1)
    parser.add_argument("--owner", choices=("default", "custom"), default="default")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--backlog", type=int, default=64)
    parser.add_argument("--candidates", type=int, default=32)
    parser.add_argument("--matrix", type=int, choices=(2048, 4096), default=4096)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.repeats <= 10 or not 1 <= args.backlog <= 128 or not 1 <= args.candidates <= 32:
        parser.error("repeats 1..10, backlog 1..128, candidates 1..32 required")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)

    import torch
    import torch_npu
    npu = torch_npu.npu
    npu.set_device(0)
    if npu.get_allocator_backend() != "native":
        raise RuntimeError("This experiment requires the native allocator")
    if torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("Disable deterministic uninitialized-memory filling for address-only candidates")
    # Do not change global/service settings or installed code.
    npu.config.allow_internal_format = False
    owner = npu.current_stream() if args.owner == "default" else npu.Stream()
    consumer = npu.Stream()
    elements = 1 << 20
    size_bytes = elements * 4
    with npu.stream(consumer):
        left = torch.full((args.matrix, args.matrix), 1 / 64, device="npu", dtype=torch.float16)
        right = torch.full_like(left, 1 / 64)
        scratch = torch.empty_like(left)
        observed = torch.empty((elements,), device="npu", dtype=torch.float32)
        for _ in range(3):
            torch.mm(left, right, out=scratch)
    # Warm the exact copy path while retaining both tensors through completion.
    with npu.stream(owner):
        warm_source = torch.ones((elements,), device="npu", dtype=torch.float32)
        warm_candidates = [torch.empty_like(warm_source) for _ in range(args.candidates)]
    npu.synchronize()
    with npu.stream(consumer):
        observed.copy_(warm_source)
    npu.synchronize()
    if not bool(torch.all(observed == 1).item()):
        raise RuntimeError("Warm-up copy validation failed")
    if not bool(torch.all(scratch == args.matrix / 4096).item()):
        raise RuntimeError("Warm-up matrix validation failed")
    del warm_source, warm_candidates

    # Initialize event handles before measurement. Each trial has fresh events.
    trial_events = []
    for _ in range(args.repeats):
        markers = {name: npu.Event() for name in ("ready", "read_start", "read_end", "done")}
        for event in markers.values():
            event.record(consumer)
        trial_events.append(markers)
    npu.synchronize()
    npu.memory._record_memory_history(enabled="all", context=None, stacks="python", max_entries=10000)
    records, trials = [], []
    # Python npu_stream calls NPUStream::stream(), which can drain host queues.
    # Cache these handles BEFORE measurement; do not hash Stream objects either
    # (their Python __hash__ also reads npu_stream).
    stream_handles = {id(owner): str(owner.npu_stream), id(consumer): str(consumer.npu_stream)}
    stream_ids = {id(owner): str(owner.stream_id), id(consumer): str(consumer.stream_id)}

    @contextmanager
    def scope(trial, operation, stream):
        label = f"P32/{trial}/{operation}"
        record = dict(label=label, trial=trial, operation=operation,
                      thread_id=threading.get_native_id(), stream_handle=stream_handles[id(stream)],
                      start_wall_ns=time.time_ns(), start_mono_ns=time.monotonic_ns())
        with torch.profiler.record_function(label) if args.profile else nullcontext():
            yield record
        record.update(end_wall_ns=time.time_ns(), end_mono_ns=time.monotonic_ns())
        records.append(record)

    def progress(markers):
        before = time.monotonic_ns()
        status = {name: markers[name].query() for name in ("read_start", "read_end", "done")}
        status.update(query_start_ns=before, query_end_ns=time.monotonic_ns())
        return status

    def snapshots(trial, phase, address):
        snapshot = npu.memory._snapshot()
        filename = f"{trial}-{phase}.json.gz"
        # Serialization and I/O are postponed until the observation window ends.
        return filename, snapshot, locate(snapshot, address)

    def run_trial(index):
        trial = f"t{index:02d}"
        markers = trial_events[index]
        npu.synchronize()
        traces_before = len(npu.memory._snapshot()["device_traces"][0])
        before_stats = stats(npu)
        retained_snapshots = []
        with npu.stream(owner):
            with scope(trial, "create-a", owner):
                a = torch.empty((elements,), device="npu", dtype=torch.float32)
                a.fill_(index + 1)
                address = a.data_ptr()
                weak_a = weakref.ref(a)
            markers["ready"].record(owner)
        # Baseline: producer initialization finishes before starting backlog.
        # S1 still explicitly waits the producer event in all variants.
        markers["ready"].synchronize()
        release = {}

        def submit_and_release():
            # Fetch in this frame: passing A as a function argument would keep
            # an additional caller evaluation-stack reference until return.
            tensor = handoff.get()
            with npu.stream(consumer):
                markers["ready"].wait(consumer)
                with scope(trial, "backlog", consumer):
                    for _ in range(args.backlog):
                        torch.mm(left, right, out=scratch)
                with scope(trial, "read-start", consumer):
                    markers["read_start"].record(consumer)
                with scope(trial, "consumer-copy", consumer):
                    observed.copy_(tensor)
                with scope(trial, "read-end", consumer):
                    markers["read_end"].record(consumer)
                if args.mode == "record":
                    with scope(trial, "record-storage", consumer):
                        tensor.record_stream(consumer)
                with scope(trial, "done", consumer):
                    markers["done"].record(consumer)
            if args.mode == "join":
                with npu.stream(owner):
                    with scope(trial, "owner-joins-consumer", owner):
                        owner.wait_event(markers["done"])
            # This is the sole remaining strong Python reference to A.
            # Drop it on S1 even though its allocation owner is S0.
            with npu.stream(consumer):
                with scope(trial, "release-a", consumer):
                    release.update(thread_id=threading.get_native_id(), current_stream=stream_handles[id(consumer)])
                    del tensor

        handoff = queue.Queue(maxsize=1)
        handoff.put(a)
        del a
        errors = []

        def consume_handoff():
            try:
                # get() removes Queue ownership before registration/free.
                submit_and_release()
            except BaseException as exc:
                errors.append(exc)

        if args.threads == 2:
            worker = threading.Thread(target=consume_handoff, name="p32-consumer")
            worker.start()
            worker.join(timeout=60)
            if worker.is_alive():
                raise RuntimeError("Consumer host thread did not finish submitting")
        else:
            consume_handoff()
        if errors:
            raise errors[0]
        if weak_a() is not None:
            raise RuntimeError("A still has a Python owner after release")
        pending_after_release = progress(markers)
        released_snapshot = snapshots(trial, "released", address)
        retained_snapshots.append(released_snapshot[:2])
        observations, held = [], []
        with npu.stream(owner):
            for candidate_index in range(args.candidates):
                pre = progress(markers)
                with scope(trial, f"allocate-{candidate_index:03d}", owner) as allocation:
                    candidate = torch.empty((elements,), device="npu", dtype=torch.float32)
                    pointer = candidate.data_ptr()
                post = progress(markers)
                observations.append(dict(index=candidate_index, pointer=str(pointer),
                                         reused=pointer == address, before=pre, after=post,
                                         allocation_label=allocation["label"],
                                         allocation_start_ns=allocation["start_mono_ns"],
                                         allocation_end_ns=allocation["end_mono_ns"]))
                held.append(candidate)
        window_snapshot = snapshots(trial, "window", address)
        retained_snapshots.append(window_snapshot[:2])
        window_stats = stats(npu)
        # No synchronize, device value reads, candidate writes, logs or files
        # inside the address observation loop. End it before validation.
        with scope(trial, "completion-wait", consumer):
            consumer.synchronize()
            owner.synchronize()
        completed_snapshot = snapshots(trial, "synchronized", address)
        retained_snapshots.append(completed_snapshot[:2])
        later = []
        with npu.stream(owner):
            for _ in range(4):
                candidate = torch.empty((elements,), device="npu", dtype=torch.float32)
                later.append(dict(pointer=str(candidate.data_ptr()), reused=candidate.data_ptr() == address))
                held.append(candidate)
        polled_snapshot = snapshots(trial, "polled", address)
        retained_snapshots.append(polled_snapshot[:2])
        reclaim_trigger = None
        if not any(row["reused"] for row in observations + later):
            # Lazy reclaim can deliberately skip completed events while another
            # cached block satisfies the allocation. Force a bounded cache miss
            # AFTER device completion, without empty_cache or pressure/OOM.
            largest_free = max((block["size"] for segment in polled_snapshot[1]["segments"]
                                if str(segment["stream"]) == stream_handles[id(owner)]
                                for block in segment["blocks"] if block["state"] == "inactive"), default=0)
            trigger_bytes = max(64 << 20, largest_free + (16 << 20))
            if trigger_bytes > (192 << 20):
                reclaim_trigger = dict(requested_bytes=trigger_bytes, largest_cached_free_block=largest_free,
                                       reused=False, status="skipped_at_memory_bound")
            else:
                with npu.stream(owner):
                    trigger = torch.empty((trigger_bytes // 4,), device="npu", dtype=torch.float32)
                    held.append(trigger)
                    candidate = torch.empty((elements,), device="npu", dtype=torch.float32)
                    held.append(candidate)
                    reclaim_trigger = dict(requested_bytes=trigger_bytes, largest_cached_free_block=largest_free,
                                           reused=candidate.data_ptr() == address, pointer=str(candidate.data_ptr()),
                                           status="executed")
                    reclaim_trigger["statistics_after_trigger"] = stats(npu)
                trigger_snapshot = snapshots(trial, "miss-polled", address)
                retained_snapshots.append(trigger_snapshot[:2])
                reclaim_trigger["block_after_trigger"] = trigger_snapshot[2]
                polled_snapshot = trigger_snapshot
        with scope(trial, "validate", consumer):
            check = bool(torch.all(observed == index + 1).item())
        if not check:
            raise RuntimeError("Observed copy mismatch despite no candidate writes")
        traces = polled_snapshot[1]["device_traces"][0]
        relevant = [dict(trace_index=i, **event) for i, event in enumerate(traces)
                    if i >= traces_before and event.get("addr") == address]
        result = dict(id=trial, mode=args.mode, threads=args.threads, address_a=str(address),
                      tensor_python_owner_released=weak_a() is None, release=release,
                      submitting_main_thread=threading.get_native_id(),
                      progress_after_release=pending_after_release,
                      block_after_release=released_snapshot[2], block_after_window=window_snapshot[2],
                      block_after_synchronize=completed_snapshot[2], block_after_poll=polled_snapshot[2],
                      observations=observations, allocations_after_completion=later,
                      completion_cache_miss_trigger=reclaim_trigger,
                      statistics_before=before_stats, statistics_after_window=window_stats,
                      trace_entries_for_a=relevant, copy_all_elements_correct=check,
                      snapshot_files=[name for name, _ in retained_snapshots])
        # End-of-trial file work is outside the measured window and completion.
        for filename, snapshot in retained_snapshots:
            with gzip.open(out / filename, "wt") as stream:
                json.dump(snapshot, stream)
        del held, candidate
        return result

    profiler = None
    if args.profile:
        config = torch_npu.profiler._ExperimentalConfig(profiler_level=torch_npu.profiler.ProfilerLevel.Level1)
        profiler = torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
            record_shapes=False, profile_memory=False, with_stack=False,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(out / "profiler")),
            experimental_config=config)
    with profiler if profiler is not None else nullcontext():
        for index in range(args.repeats):
            trials.append(run_trial(index))
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
         streams={"owner": dict(handle=stream_handles[id(owner)], id=stream_ids[id(owner)]),
                  "consumer": dict(handle=stream_handles[id(consumer)], id=stream_ids[id(consumer)])},
         tensor_bytes=size_bytes, matrix=args.matrix, source_sha256=installed,
         memory_bound_description="96 MiB matrices at 4096 FP16; same-size candidates <=148 MiB; optional completed-window miss buffer <=192 MiB; older cached triggers and runtime/workspace additional",
         candidate_access="address only; no candidate value read, write, or raw dereference",
         records=records, trials=trials))
    npu.synchronize()
    print(f"Completed {len(trials)} {args.mode} trials, threads={args.threads}", flush=True)


if __name__ == "__main__":
    main()
