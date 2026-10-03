"""Graph capture/replay storage boundary probes: external tensors and private pools."""
import argparse
import datetime
import hashlib
import inspect
import json
import os
from pathlib import Path
import threading
import time
import weakref

SENTINEL = -777.0  # negative: external data is a non-negative arange multiple, so no collision
MODES = ("keep-alive", "external-free-ordered", "external-free-cross", "pool-release-pending")


def save(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def stats(npu):
    data = npu.memory_stats()
    return {key: data[key] for key in (
        "num_alloc_retries", "num_ooms", "allocated_bytes.all.current",
        "reserved_bytes.all.current", "active_bytes.all.current")}


def classify_observed(n_original, n_sentinel, n_other, elements):
    if n_other:
        return "unexpected_values"
    if n_sentinel == 0:
        return "intact"
    if n_original == 0:
        return "fully_consumed_replacement"
    return "mixed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--elements", type=int, default=1 << 20)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 10:
        parser.error("repeats 1..10 required")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)

    import torch
    import torch_npu
    npu = torch_npu.npu
    npu.set_device(0)
    if npu.get_allocator_backend() != "native":
        raise RuntimeError("This experiment requires the native allocator")
    if torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("Deterministic uninitialized-memory filling would confound outputs")
    npu.config.allow_internal_format = False
    owner = npu.current_stream()          # normal allocations, replacement writes, ordered replays
    capture_stream = npu.Stream()         # graph capture stream (non-default, required)
    replay_stream = npu.Stream()          # cross-stream replay lane
    elements = args.elements

    # Warm the exact kernels and pool paths before any capture.
    with npu.stream(owner):
        warm = torch.arange(elements, device="npu", dtype=torch.float32)
        warm_probe = torch.empty_like(warm)
    npu.synchronize()
    with npu.stream(owner):
        warm_probe.copy_(warm)
    npu.synchronize()
    if not bool(torch.equal(warm_probe, warm)):
        raise RuntimeError("Warm-up copy validation failed")
    del warm, warm_probe

    stream_handles = {id(owner): str(owner.npu_stream), id(capture_stream): str(capture_stream.npu_stream),
                      id(replay_stream): str(replay_stream.npu_stream)}
    npu.memory._record_memory_history(enabled="all", context=None, stacks="python", max_entries=10000)

    def run_trial(index):
        npu.synchronize()
        before_stats = stats(npu)
        trial = f"t{index:02d}"
        expected = (torch.arange(elements, dtype=torch.float32) * (index + 1)).to("npu")
        # X: external storage on the normal pool, allocated BEFORE capture.
        with npu.stream(owner):
            external = (torch.arange(elements, device="npu", dtype=torch.float32) * (index + 1))
        external_address = external.data_ptr()
        weak_external = weakref.ref(external)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=capture_stream, capture_error_mode="global"):
            static_out = torch.empty((elements,), device="npu", dtype=torch.float32)
            static_out.copy_(external)   # replay re-reads external's baked-in address
        static_address = static_out.data_ptr()
        # One ordered replay validates capture mechanics before the trial variable.
        with npu.stream(owner):
            graph.replay()
        owner.synchronize()
        if not bool(torch.equal(static_out, expected)):
            raise RuntimeError("Baseline replay output mismatch in " + trial)
        result = dict(id=trial, mode=args.mode, external_address=str(external_address),
                      static_address=str(static_address),
                      stream_handles=dict(owner=stream_handles[id(owner)],
                                          capture=stream_handles[id(capture_stream)],
                                          replay=stream_handles[id(replay_stream)]))

        if args.mode == "keep-alive":
            with npu.stream(owner):
                replacement = torch.empty((elements,), device="npu", dtype=torch.float32)
                replacement.fill_(SENTINEL)
                result["replacement_address"] = str(replacement.data_ptr())
                result["replacement_reused_external"] = False
                graph.replay()
            npu.synchronize()
            base_count = int((static_out == expected).sum().item())
            sentinel_count = int((static_out == SENTINEL).sum().item())
            other_count = elements - base_count - sentinel_count
            result.update(base_count=base_count, sentinel_count=sentinel_count,
                          other_count=other_count, elements=elements,
                          outcome=classify_observed(base_count, sentinel_count, other_count, elements))
            result["replacement_intact"] = bool((replacement == SENTINEL).all().item())
            result["external_kept_alive"] = True
        elif args.mode in ("external-free-ordered", "external-free-cross"):
            del external
            if weak_external() is not None:
                raise RuntimeError("External tensor still alive after release")
            # Several same-size blocks may sit in the pool (e.g. the arange
            # intermediate); allocate candidates until one lands exactly on the
            # freed external address, then write the sentinel into THAT buffer.
            candidates = []
            matched = None
            with npu.stream(owner):
                for attempt in range(32):
                    candidate = torch.empty((elements,), device="npu", dtype=torch.float32)
                    candidates.append(candidate)
                    if candidate.data_ptr() == external_address:
                        matched = candidate
                        break
            result["candidate_attempts"] = len(candidates)
            result["replacement_reused_external"] = matched is not None
            if matched is None:
                result["outcome"] = "external_address_not_returned"
                result["replacement_intact"] = True
                result["external_kept_alive"] = args.mode == "keep-alive"
            else:
                with npu.stream(owner):
                    matched.fill_(SENTINEL)
                if args.mode == "external-free-cross":
                    with npu.stream(replay_stream):
                        graph.replay()
                else:
                    with npu.stream(owner):
                        graph.replay()
                npu.synchronize()
                base_count = int((static_out == expected).sum().item())
                sentinel_count = int((static_out == SENTINEL).sum().item())
                other_count = elements - base_count - sentinel_count
                result.update(base_count=base_count, sentinel_count=sentinel_count,
                              other_count=other_count, elements=elements,
                              outcome=classify_observed(base_count, sentinel_count, other_count, elements))
                result["replacement_intact"] = bool((matched == SENTINEL).all().item())
                result["replacement_address"] = str(matched.data_ptr())
                result["external_kept_alive"] = args.mode == "keep-alive"
        else:  # pool-release-pending
            # Enqueue a replay, then destroy the graph and its static tensor while
            # the replay may still be executing. Observe addresses and survival.
            with npu.stream(owner):
                graph.replay()
            del static_out
            del graph
            with npu.stream(owner):
                replacement = torch.empty((elements,), device="npu", dtype=torch.float32)
                result["replacement_address"] = str(replacement.data_ptr())
                result["replacement_reused_static"] = replacement.data_ptr() == static_address
                replacement.fill_(SENTINEL)
            npu.synchronize()
            result["replacement_intact"] = bool((replacement == SENTINEL).all().item())
            result["outcome"] = "survived" if result["replacement_intact"] else "replacement_clobbered"
            result["external_kept_alive"] = True
        result.update(statistics_before=before_stats, statistics_after=stats(npu))
        # Local names (external/graph/static_out/replacement) die at return;
        # keep-alive trials intentionally held `external` only inside this frame.
        return result

    trials = []
    for index in range(args.repeats):
        result = run_trial(index)
        trials.append(result)
        save(out / "trials.json", dict(schema=1, mode=args.mode, trials=trials))
        print(result["id"], result.get("outcome"),
              "reuse_external=", result.get("replacement_reused_external"),
              "reuse_static=", result.get("replacement_reused_static"), flush=True)
        npu.synchronize()

    keys = ("ASCEND_HOME_PATH", "ASCEND_RT_VISIBLE_DEVICES", "TASK_QUEUE_ENABLE", "PER_STREAM_QUEUE",
            "PYTORCH_NPU_ALLOC_CONF", "PYTORCH_NO_NPU_MEMORY_CACHING", "ASCEND_LAUNCH_BLOCKING")
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
         environment={key: os.environ.get(key) for key in keys},
         sentinel=SENTINEL, elements=elements, source_sha256=installed,
         thread_id=threading.get_native_id(),
         duration_note="one warm replay precedes the trial variable in every repeat",
         trials=trials))
    print(f"Completed {len(trials)} {args.mode} trials", flush=True)


if __name__ == "__main__":
    main()
