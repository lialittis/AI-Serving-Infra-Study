"""P31a: one native HTTP service, independent closed-loop users, bounded cases."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "practice_17_vllm_multistream"))
from run_sampling_benchmark import clean_environment
sys.path.pop(0)


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def snapshot(command, path):
    r = subprocess.run(command, capture_output=True, text=True, timeout=45)
    save(path, dict(command=command, returncode=r.returncode, stdout=r.stdout, stderr=r.stderr))
    return r


def token_prompt(tokenizer, user, round_index, length):
    head = tokenizer.encode("User %02d round %02d. " % (user, round_index), add_special_tokens=False)
    body = tokenizer.encode("Explain how memory and computation work together in an inference service. ", add_special_tokens=False)
    result = (head + body * length)[:length]
    assert len(result) == length
    return result


def request(base, payload, user, round_index, repetition):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(base + "/v1/completions", data, {"Content-Type": "application/json"})
    start = time.time_ns()
    mono = time.monotonic_ns()
    record = dict(client_request_id=payload["request_id"], user=user, round=round_index,
                  repetition=repetition, start_ns=start, start_mono_ns=mono, chunks=[], token_ids=[],
                  response_ids=[], status="error", first_token_ns=None)
    try:
        done = False
        finishes = []
        with urllib.request.urlopen(req, timeout=240) as response:
            record["http_status"] = response.status
            for line in response:
                if not line.startswith(b"data: "):
                    continue
                raw = line[6:].strip()
                if raw == b"[DONE]":
                    done = True
                    break
                wall, tick = time.time_ns(), time.monotonic_ns()
                chunk = json.loads(raw)
                if chunk.get("error"):
                    raise ValueError(str(chunk["error"]))
                rid = chunk.get("id")
                if rid and rid not in record["response_ids"]:
                    record["response_ids"].append(rid)
                if chunk.get("usage"):
                    record["usage"] = chunk["usage"]
                tokens = []
                for choice in chunk.get("choices", []):
                    if choice["index"] != 0:
                        raise ValueError("Expected one completion per HTTP request")
                    tokens.extend(choice.get("token_ids") or [])
                    if choice.get("finish_reason"):
                        finishes.append(choice["finish_reason"])
                if tokens and record["first_token_ns"] is None:
                    record["first_token_ns"] = wall
                    record["ttft_ms"] = (tick - mono) / 1e6
                record["token_ids"].extend(tokens)
                record["chunks"].append(dict(time_ns=wall, mono_ns=tick, token_count=len(tokens)))
        if not done or finishes != ["length"] or len(record["token_ids"]) != payload["max_tokens"]:
            raise ValueError("Incomplete output: DONE=%s finish=%s tokens=%d" % (done, finishes, len(record["token_ids"])))
        usage = record.get("usage", {})
        if usage.get("completion_tokens") != payload["max_tokens"] or usage.get("prompt_tokens") != len(payload["prompt"]):
            raise ValueError("Usage/prompt length mismatch: " + str(usage))
        if len(record["response_ids"]) != 1:
            raise ValueError("Expected one response ID")
        record["status"] = "passed"
    except Exception as exc:
        record["error"] = repr(exc)
    finally:
        record.update(end_ns=time.time_ns(), end_mono_ns=time.monotonic_ns())
        record["latency_ms"] = (record["end_mono_ns"] - mono) / 1e6
    return record


def cycle(base, inputs, concurrency, rounds, tokens, tag, repetition, sampling=None):
    barrier = threading.Barrier(concurrency)
    def user_loop(user):
        records = []
        barrier.wait(timeout=30)
        for r in range(rounds):
            payload = dict(model="p31", prompt=inputs[str(user)][r], max_tokens=tokens,
                           temperature=0, ignore_eos=True, stream=True, n=1,
                           return_token_ids=True, stream_options={"include_usage": True},
                           request_id="%s-u%02d-r%02d" % (tag, user, r))
            if sampling:
                payload.update(sampling)
            records.append(request(base, payload, user, r, repetition))
            if records[-1]["status"] != "passed":
                break
        return records
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return [r for group in pool.map(user_loop, range(concurrency)) for r in group]


def service_command(args, directory, phase):
    command = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", args.model,
               "--served-model-name", "p31", "--host", "127.0.0.1", "--port", str(args.port),
               "--tensor-parallel-size", "1", "--dtype", "bfloat16", "--enforce-eager",
               "--max-model-len", "512", "--max-num-seqs", str(getattr(args, "max_num_seqs", 8)),
               "--max-num-batched-tokens", str(getattr(args, "max_num_batched_tokens", 1024)),
               "--gpu-memory-utilization", "0.3", "--block-size", "128", "--seed", "123",
               "--no-enable-prefix-caching", "--no-enable-chunked-prefill", "--no-async-scheduling",
               "--additional-config", json.dumps({"enable_async_exponential": getattr(args, "precompute", False)})]
    if phase == "diagnostic":
        command += ["--profiler-config", json.dumps(dict(profiler="torch", torch_profiler_dir=str(directory / "profiler"),
                                                       torch_profiler_with_stack=False, ignore_frontend=True))]
    return command


def stop_owned(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=40)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=15)
    # The leader can exit before its worker. The process group is owned by this run.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass


def run_case(args, out, inputs, concurrency, phase, case_name=None):
    directory = out / (case_name or "c%d-%s" % (concurrency, phase))
    directory.mkdir()
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", args.port))
    env = clean_environment(os.environ)
    env = {k: v for k, v in env.items() if not k.startswith("P31_") and k not in
           ("LD_AUDIT", "VLLM_ENABLE_V1_MULTIPROCESSING", "VLLM_DISABLE_REQUEST_ID_RANDOMIZATION")}
    if phase == "diagnostic":
        env["P31_OBSERVER_DIR"] = str(directory / "observer")
        env["PYTHONPATH"] = str(HERE) + os.pathsep + env.get("PYTHONPATH", "")
    command = service_command(args, directory, phase)
    save(directory / "command.json", dict(argv=command, phase=phase, concurrency=concurrency,
                                         sampling=getattr(args, "sampling", {"temperature": 0}),
                                         precompute=getattr(args, "precompute", False),
                                         environment={k: v for k, v in env.items() if k.startswith(("P31", "ASCEND", "VLLM"))
                                                      or k in ("PYTHONPATH", "LD_PRELOAD")},
                                         removed_environment_keys=sorted(set(os.environ) - set(env))))
    snapshot(["npu-smi", "info"], directory / "device_before.json")
    base = "http://127.0.0.1:%d" % args.port
    state = dict(status="running", phase=phase, concurrency=concurrency)
    save(directory / "status.json", state)
    records = []
    with (directory / "server.log").open("w") as log:
        process = subprocess.Popen(command, env=env, cwd=directory, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        save(directory / "launcher.json", dict(pid=process.pid, process_group=process.pid))
        try:
            deadline = time.monotonic() + 600
            while True:
                if process.poll() is not None:
                    raise RuntimeError("Service exited; inspect server.log")
                try:
                    with urllib.request.urlopen(base + "/health", timeout=2) as response:
                        if response.status == 200:
                            break
                except OSError:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError("Service startup timeout")
                time.sleep(1)
            warm = cycle(base, inputs, concurrency, args.warmup, args.output_tokens, "p31-warmup-" + directory.name, -1,
                         getattr(args, "sampling", None))
            save(directory / "warmup.json", warm)
            if any(r["status"] != "passed" for r in warm):
                raise RuntimeError("Warmup request failed")
            if phase == "diagnostic":
                with urllib.request.urlopen(urllib.request.Request(base + "/start_profile", data=b""), timeout=180) as response:
                    response.read()
            repeats = args.repeats if phase == "benchmark" else 1
            for rep in range(repeats):
                tag = "p31-measure-%s-%s-b%d" % (out.name, directory.name, rep)
                records += cycle(base, inputs, concurrency, args.rounds, args.output_tokens, tag, rep,
                                 getattr(args, "sampling", None))
                save(directory / "requests.json", records)
                if len(records) != (rep + 1) * concurrency * args.rounds or any(r["status"] != "passed" for r in records):
                    raise RuntimeError("Measured request failed")
            if phase == "diagnostic":
                with urllib.request.urlopen(urllib.request.Request(base + "/stop_profile", data=b""), timeout=600) as response:
                    response.read()
                if not list((directory / "profiler").rglob("trace_view.json")):
                    raise RuntimeError("Missing exported trace_view.json")
            state.update(status="passed", requests=len(records))
        except Exception as exc:
            state.update(status="failed", error=repr(exc))
            raise
        finally:
            save(directory / "status.json", state)
            stop_owned(process)
            snapshot(["npu-smi", "info"], directory / "device_after.json")
    print(directory.name + ": " + state["status"], flush=True)


def collect_sources(out, model):
    sources = {
        "vllm": (Path("/vllm-workspace/vllm"), ["vllm/entrypoints/openai/completion/serving.py", "vllm/v1/engine/async_llm.py",
                    "vllm/v1/engine/input_processor.py", "vllm/v1/core/sched/scheduler.py", "vllm/v1/request.py"]),
        "vllm_ascend": (Path("/vllm-workspace/vllm-ascend"), ["vllm_ascend/worker/worker.py", "vllm_ascend/worker/model_runner_v1.py",
                    "vllm_ascend/utils.py", "vllm_ascend/sample/sampler.py"]),
        "model": (Path(model), ["config.json", "generation_config.json"]),
        "experiment": (HERE, [p.name for p in HERE.glob("*.py")]),
    }
    manifest = {}
    for group, (root, names) in sources.items():
        if group.startswith("vllm"):
            snapshot(["git", "-C", str(root), "rev-parse", "HEAD"], out / (group + "_revision.json"))
            snapshot(["git", "-C", str(root), "status", "--short"], out / (group + "_status.json"))
        for name in names:
            data = (root / name).read_bytes()
            dst = out / "sources" / group / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(data)
            manifest[str(dst.relative_to(out))] = dict(original=str(root / name), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    for name in ("run_sampling_benchmark.py", "analyze_run.py"):
        data = (ROOT / "practice_17_vllm_multistream" / name).read_bytes()
        dst = out / "sources" / "p17" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
        manifest[str(dst.relative_to(out))] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    save(out / "source_manifest.json", manifest)
    save(out / "versions.json", {p: importlib.metadata.version(p) for p in ("vllm", "vllm-ascend", "torch", "torch-npu", "transformers")})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default="/data/huggingface_home/hub/Qwen2.5-0.5B-Instruct")
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--phase", choices=["both", "benchmark", "diagnostic"], default="both")
    p.add_argument("--port", type=int, default=8031)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--input-tokens", type=int, default=128)
    p.add_argument("--output-tokens", type=int, default=64)
    args = p.parse_args()
    if not set(args.concurrency) <= {1, 2, 4, 8} or min(args.rounds, args.repeats, args.warmup, args.input_tokens, args.output_tokens) < 1:
        p.error("Positive workload sizes and concurrency in 1,2,4,8 required")
    if args.input_tokens > 128 or args.output_tokens > 64:
        p.error("P31a is bounded to input <=128 and output <=64")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    snapshot(["npu-smi", "info", "-t", "health", "-i", "5", "-c", "0"], out / "device_health.json")
    npu = snapshot(["npu-smi", "info"], out / "preflight.json")
    if npu.returncode or "No running processes" not in npu.stdout:
        raise RuntimeError("NPU must have no running processes before the experiment")
    free = __import__("shutil").disk_usage(out).free
    if free < 4 * 1024**3:
        raise RuntimeError("Need at least 4 GiB free for bounded diagnostic traces")
    save(out / "plan.json", dict(**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                                 schema=1, host_clock="same host; wall ns and monotonic ns", start_ns=time.time_ns(), free_bytes=free))
    collect_sources(out, args.model)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    inputs = {str(u): [token_prompt(tokenizer, u, r, args.input_tokens) for r in range(max(args.rounds, args.warmup))]
              for u in range(max(args.concurrency))}
    save(out / "inputs.json", inputs)
    for c in args.concurrency:
        for phase in (["benchmark", "diagnostic"] if args.phase == "both" else [args.phase]):
            run_case(args, out, inputs, c, phase)
    print("Completed " + str(out), flush=True)


if __name__ == "__main__":
    main()
