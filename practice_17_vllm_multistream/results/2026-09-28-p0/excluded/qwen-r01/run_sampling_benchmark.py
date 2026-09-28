"""Unprofiled HTTP completion benchmark of async exponential enabled/disabled.

ABBA service blocks isolate the two configurations on a single NPU. Diagnostic
profiling is intentionally a separate program (run_vllm.py).
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def clean_environment(original):
    env = {k: v for k, v in original.items()
           if not k.endswith("TRACE_DIR") and "PROFILER" not in k
           and k not in {"PYTHONPATH", "PYTHONSTARTUP", "PYTHONPROFILEIMPORTTIME"}}
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               VLLM_WORKER_MULTIPROC_METHOD="spawn")
    return env


def server_command(model, mode, port, batch):
    return [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", str(model),
            "--served-model-name", "p17-benchmark", "--host", "127.0.0.1",
            "--port", str(port), "--tensor-parallel-size", "1", "--dtype", "bfloat16",
            "--max-model-len", "256", "--max-num-seqs", str(batch),
            "--max-num-batched-tokens", "1024", "--seed", "123",
            "--gpu-memory-utilization", "0.3", "--block-size", "128", "--enforce-eager",
            "--no-enable-prefix-caching", "--no-enable-chunked-prefill",
            "--no-async-scheduling", "--additional-config",
            json.dumps({"enable_async_exponential": mode == "enabled"})]


def validate_response(response, batch, tokens):
    if "error" in response:
        raise ValueError("API error: " + str(response["error"]))
    choices = response.get("choices", [])
    if len(choices) != batch or sorted(c["index"] for c in choices) != list(range(batch)):
        raise ValueError("response choices do not match batch")
    if response.get("usage", {}).get("completion_tokens") != batch * tokens:
        raise ValueError("completion token count mismatch")
    if any(c.get("finish_reason") != "length" for c in choices):
        raise ValueError("unexpected early completion")


def measure(base, payload, opener=urllib.request.urlopen, clock=time.perf_counter_ns):
    # Encoding and parsing are outside the timing window; full response reception
    # is inside. In particular this does not measure just asynchronous submission.
    data = json.dumps(payload).encode()
    request = urllib.request.Request(base + "/v1/completions", data=data,
                                     headers={"Content-Type": "application/json"})
    start = clock()
    with opener(request, timeout=240) as response:
        raw = response.read()
        status = response.status
    end = clock()
    if status != 200:
        raise ValueError("HTTP status %s" % status)
    return end - start, json.loads(raw)


def snapshot(command, path):
    result = subprocess.run(command, capture_output=True, text=True, timeout=40)
    save(path, {"argv": command, "exit_code": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", required=True)
    parser.add_argument("--output-tokens", type=int, nargs="+", default=[4, 64])
    parser.add_argument("--port", type=int, default=8017)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.batch_sizes + args.output_tokens + [args.warmup, args.repeats]) < 1:
        parser.error("batch, tokens, warmup and repeats must be positive")
    if max(args.output_tokens) > 128:
        parser.error("output tokens exceed supported 128")
    model, output = args.model.resolve(), args.output.resolve()
    if not (model / "config.json").exists():
        parser.error("model config missing")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", args.port))
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parent
    env = clean_environment(os.environ)
    cases = [(b, n) for b in sorted(set(args.batch_sizes))
             for n in sorted(set(args.output_tokens))]
    plan = {"model": str(model), "cases": cases, "warmup": args.warmup,
            "repeats_per_block": args.repeats,
            "blocks": ["enabled", "disabled", "disabled", "enabled"],
            "profiling": False, "python_observer": False,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "timing": "HTTP send through complete response reception; excludes encode/parse/file IO",
            "removed_environment_keys": sorted(set(os.environ) - set(env))}
    save(output / "plan.json", plan)
    data = Path(__file__).read_bytes()
    (output / "run_sampling_benchmark.py").write_bytes(data)
    save(output / "script_hash.json", {"sha256": hashlib.sha256(data).hexdigest()})
    subprocess.run([sys.executable, str(root.parent / "practice_03_ascend_start/collect_environment.py"),
                    "--model", str(model), "--output", str(output / "environment.json")],
                   env=env, cwd=output, check=True)
    from collect_sources import collect
    collect(output, model)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
    prompt = "Explain why the sky is blue in one short sentence."
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    save(output / "prompt_info.json", {"text": prompt, "token_ids": ids})
    snapshot(["npu-smi", "info", "-t", "health", "-i", "5", "-c", "0"],
             output / "device_health.json")
    base = "http://127.0.0.1:%d" % args.port
    for block, mode in enumerate(plan["blocks"]):
        directory = output / ("block-%02d-%s" % (block, mode))
        directory.mkdir()
        snapshot(["npu-smi", "info"], directory / "device_before.json")
        command = server_command(model, mode, args.port, max(args.batch_sizes))
        save(directory / "command.json", {"argv": command,
             "environment_overrides": {k: env[k] for k in
                ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "VLLM_WORKER_MULTIPROC_METHOD"]},
             "PYTHONPATH": env.get("PYTHONPATH"),
             "trace_environment_keys": [k for k in env if k.endswith("TRACE_DIR")]})
        with (directory / "server.log").open("w") as log:
            process = subprocess.Popen(command, env=env, cwd=directory, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            save(directory / "launcher.json", {"pid": process.pid})
            completed = False
            try:
                deadline = time.monotonic() + 600
                while True:
                    if process.poll() is not None:
                        raise RuntimeError("server exited; see " + str(directory / "server.log"))
                    try:
                        with urllib.request.urlopen(base + "/health", timeout=2) as response:
                            if response.status == 200:
                                break
                    except (urllib.error.URLError, TimeoutError):
                        pass
                    if time.monotonic() > deadline:
                        raise TimeoutError("server readiness exceeded 600 seconds")
                    time.sleep(1)
                # Identical order within each AB / BA pair; reverse between pairs.
                ordered = cases if block < 2 else list(reversed(cases))
                for batch, tokens in ordered:
                    payload = {"model": "p17-benchmark", "prompt": [ids] * batch,
                               "max_tokens": tokens, "temperature": .8, "top_p": .9,
                               "ignore_eos": True, "stream": False}
                    for phase, count in [("warmup", args.warmup), ("measure", args.repeats)]:
                        for repeat in range(count):
                            ident = "p17-bench-%d-%d-%d-%s-%d" % (block, batch, tokens, phase, repeat)
                            elapsed, response = measure(base, dict(payload, request_id=ident))
                            validate_response(response, batch, tokens)
                            record = {"id": ident, "block": block, "mode": mode,
                                      "batch_size": batch, "output_tokens": tokens,
                                      "phase": phase, "repeat": repeat, "elapsed_ns": elapsed,
                                      "usage": response["usage"], "validated": True,
                                      "response": response}
                            with (output / "samples.jsonl").open("a") as stream:
                                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    print(json.dumps({"block": block, "mode": mode, "batch": batch,
                                      "tokens": tokens, "status": "measured"}), flush=True)
                completed = True
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                save(directory / "shutdown.json", {"server_exit_code": process.returncode,
                                                    "completed": completed})
            if process.returncode != 0:
                raise RuntimeError("unclean server shutdown: %s" % process.returncode)
        snapshot(["npu-smi", "info"], directory / "device_after.json")
    save(output / "complete.json", {"completed_at": datetime.now(timezone.utc).isoformat(),
                                  "blocks": len(plan["blocks"])})
    print("Completed: " + str(output), flush=True)


if __name__ == "__main__":
    main()
