"""Capture a matched real-vLLM request with async exponential on or off."""
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mode", choices=("enabled", "disabled"), required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    model = args.model.resolve()
    output = args.output.resolve()
    if not (model / "config.json").is_file():
        parser.error("model config.json missing")
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", args.port))
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "instrumentation"
    snapshot.mkdir()
    hashes = {}
    for name in ("sitecustomize.py", "multistream_trace.py", "run_vllm.py", "collect_sources.py"):
        data = (root / name).read_bytes()
        (snapshot / name).write_bytes(data)
        hashes["instrumentation/" + name] = hashlib.sha256(data).hexdigest()
    save(output / "instrumentation_hashes.json", hashes)
    subprocess.run([sys.executable, str(root.parent / "practice_03_ascend_start/collect_environment.py"),
                    "--model", str(model), "--output", str(output / "environment.json")], check=True)
    enabled = args.mode == "enabled"
    served = "p17-" + model.name.lower().replace("-instruct", "") + "-" + args.mode
    profiler = {"profiler": "torch", "torch_profiler_dir": str(output / "profiler"),
                "torch_profiler_with_stack": False, "ignore_frontend": True}
    command = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", str(model),
               "--served-model-name", served, "--host", "127.0.0.1", "--port", str(args.port),
               "--tensor-parallel-size", "1", "--dtype", "bfloat16", "--max-model-len", "256",
               "--max-num-seqs", str(args.batch_size), "--max-num-batched-tokens", "1024",
               "--seed", "123",
               "--gpu-memory-utilization", "0.3", "--block-size", "128", "--enforce-eager",
               "--no-enable-prefix-caching", "--no-enable-chunked-prefill", "--no-async-scheduling",
               "--additional-config", json.dumps({"enable_async_exponential": enabled}),
               "--profiler-config", json.dumps(profiler)]
    env = os.environ.copy()
    for key in list(env):
        if key.endswith("TRACE_DIR") and (key.startswith("P0") or key.startswith("P1")):
            env.pop(key, None)
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               VLLM_WORKER_MULTIPROC_METHOD="spawn", P17_TRACE_DIR=str(output / "events"),
               PYTHONPATH=os.pathsep.join([str(root), env.get("PYTHONPATH", "")]))
    save(output / "command.json", {"argv": command, "mode": args.mode,
         "enable_async_exponential": enabled,
         "environment_overrides": {k: env[k] for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
             "VLLM_WORKER_MULTIPROC_METHOD", "P17_TRACE_DIR", "PYTHONPATH")}})
    from collect_sources import collect
    collect(output, model)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
    prompt = "Explain why the sky is blue in one short sentence."
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    save(output / "prompt_info.json", {"text": prompt, "token_ids": prompt_ids,
                                      "prompt_tokens": len(prompt_ids),
                                      "batch_size": args.batch_size})
    prompts = [prompt_ids] * args.batch_size if args.batch_size > 1 else prompt_ids
    body = {"model": served, "prompt": prompts, "max_tokens": 4,
            "temperature": 0.8, "top_p": 0.9, "ignore_eos": True,
            "stream": False, "request_id": "p17-profile"}
    save(output / "request.json", body)
    base_url = "http://127.0.0.1:%d" % args.port
    with (output / "server.log").open("w") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        save(output / "launcher.json", {"server_pid": process.pid,
                                       "started_at": datetime.now(timezone.utc).isoformat()})
        def post(endpoint, payload=None, timeout=240):
            data = json.dumps(payload).encode() if payload is not None else b""
            request = urllib.request.Request(base_url + endpoint, data=data,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode()
                return {"status": response.status, "body": json.loads(raw) if raw else None}
        try:
            deadline = time.monotonic() + 600
            while True:
                if process.poll() is not None:
                    raise RuntimeError("service exited %s; see server.log" % process.returncode)
                try:
                    with urllib.request.urlopen(base_url + "/health", timeout=2) as response:
                        if response.status == 200:
                            break
                except (urllib.error.URLError, TimeoutError):
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError("service readiness exceeded 600 seconds")
                time.sleep(1)
            save(output / "ready.json", {"time_ns": time.time_ns()})
            warmup = dict(body, request_id="p17-warmup")
            save(output / "warmup_response.json", post("/v1/completions", warmup)["body"])
            controls = [{"endpoint": "/start_profile", **post("/start_profile")}]
            save(output / "profile_control.json", controls)
            try:
                window = {"start_ns": time.time_ns()}
                save(output / "response.json", post("/v1/completions", body)["body"])
                window["end_ns"] = time.time_ns()
                save(output / "request_window.json", window)
            finally:
                controls.append({"endpoint": "/stop_profile", **post("/stop_profile")})
                save(output / "profile_control.json", controls)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            save(output / "shutdown.json", {"server_exit_code": process.returncode})
    subprocess.run(["npu-smi", "info"], stdout=(output / "device_after.txt").open("w"), check=True)
    print("Captured:", output, flush=True)


if __name__ == "__main__":
    main()
