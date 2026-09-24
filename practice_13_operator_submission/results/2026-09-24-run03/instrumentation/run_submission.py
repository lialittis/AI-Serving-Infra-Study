"""Capture cold compile/load plus a warm, ordinary eager inference submission trace."""

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
    parser.add_argument("--model", default="/data/huggingface_home/hub/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--port", type=int, default=8013)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = args.output.resolve()
    if not Path(args.model, "config.json").is_file():
        parser.error("model config.json missing")
    # Never attach this experiment's request to an unrelated preexisting server.
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", args.port))
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "instrumentation"
    snapshot.mkdir()
    hashes = {}
    for name in ("sitecustomize.py", "submission_trace.py", "run_submission.py", "collect_sources.py"):
        content = (root / name).read_bytes()
        (snapshot / name).write_bytes(content)
        hashes["instrumentation/" + name] = hashlib.sha256(content).hexdigest()
    base_hooks = root.parent / "practice_07_real_request_trace/trace_hooks.py"
    content = base_hooks.read_bytes()
    (snapshot / "trace_hooks.py").write_bytes(content)
    hashes["instrumentation/trace_hooks.py"] = hashlib.sha256(content).hexdigest()
    save(output / "instrumentation_hashes.json", hashes)
    subprocess.run([sys.executable, str(root.parent / "practice_03_ascend_start/collect_environment.py"),
                    "--model", args.model, "--output", str(output / "environment.json")], check=True)
    command = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", args.model,
               "--served-model-name", "qwen-submission", "--host", "127.0.0.1", "--port", str(args.port),
               "--tensor-parallel-size", "1", "--dtype", "bfloat16", "--max-model-len", "256",
               "--max-num-seqs", "1", "--max-num-batched-tokens", "256",
               "--gpu-memory-utilization", "0.3", "--block-size", "128", "--enforce-eager",
               "--no-enable-prefix-caching", "--no-enable-chunked-prefill", "--no-async-scheduling"]
    profiler_config = {"profiler": "torch", "torch_profiler_dir": str(output / "profiler"),
                       "torch_profiler_with_stack": False, "ignore_frontend": True}
    command += ["--profiler-config", json.dumps(profiler_config)]
    env = os.environ.copy()
    for key in ("P07_TRACE_DIR", "P08_TRACE_DIR", "P09_TRACE_DIR", "P10_GRAPH_DIR", "P11_TRACE_DIR", "P12_TRACE_DIR"):
        env.pop(key, None)
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               VLLM_WORKER_MULTIPROC_METHOD="spawn", P13_TRACE_DIR=str(output / "events"),
               TRITON_CACHE_DIR=str(output / "compiler_cache"),
               PYTHONPATH=os.pathsep.join([str(root), str(base_hooks.parent), env.get("PYTHONPATH", "")]))
    save(output / "command.json", {"argv": command,
         "environment_overrides": {key: env[key] for key in (
             "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "VLLM_WORKER_MULTIPROC_METHOD", "P13_TRACE_DIR", "TRITON_CACHE_DIR", "PYTHONPATH")}})
    from collect_sources import collect_sources
    collect_sources(output)
    save(output / "cache_before.json", {"triton_cache_exists": (output / "compiler_cache").exists(),
         "note": "new isolated Triton cache; no shared cache deleted"})
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    prompt = "请简要解释为什么天空是蓝色的。"
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    save(output / "prompt_info.json", {"text": prompt, "token_ids": prompt_ids,
                                      "prompt_tokens": len(prompt_ids)})
    request_body = {"model": "qwen-submission", "prompt": prompt_ids,
                    "max_tokens": 4, "temperature": 0, "ignore_eos": True,
                    "stream": False, "seed": 0, "request_id": "practice13-profile"}
    save(output / "request.json", request_body)
    base_url = f"http://127.0.0.1:{args.port}"
    with (output / "server.log").open("w") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        save(output / "launcher.json", {"server_pid": process.pid,
                                       "started_at": datetime.now(timezone.utc).isoformat()})
        try:
            deadline = time.monotonic() + 600
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"service exited {process.returncode}; see server.log")
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
            print("Service ready; warming up before profiler collection", flush=True)
            process_lines = subprocess.check_output(
                ["ps", "-eo", "pid,ppid,pgid,args"], text=True).splitlines()
            selected = [process_lines[0]] + [line for line in process_lines[1:]
                if int(line.split(None, 3)[2]) == process.pid]
            (output / "processes.txt").write_text("\n".join(selected) + "\n")
            def post(endpoint, body=None, timeout=180):
                data = json.dumps(body).encode() if body is not None else b""
                request = urllib.request.Request(base_url + endpoint, data=data,
                    headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    raw = response.read().decode()
                    return {"status": response.status, "body": json.loads(raw) if raw else None}

            warmup = dict(request_body, request_id="practice13-warmup")
            save(output / "warmup_request.json", warmup)
            save(output / "warmup_response.json", post("/v1/completions", warmup)["body"])
            controls = []
            controls.append({"endpoint": "/start_profile", **post("/start_profile")})
            save(output / "profile_control.json", controls)
            print("Profiler started; sending one natural-language prompt / 4-output request", flush=True)
            try:
                window={"start_ns":time.time_ns()}
                save(output / "response.json", post("/v1/completions", request_body)["body"])
                window["end_ns"]=time.time_ns()
                save(output / "request_window.json", window)
            finally:
                controls.append({"endpoint": "/stop_profile", **post("/stop_profile")})
                save(output / "profile_control.json", controls)
            print("Profiler stopped and exported", flush=True)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            save(output / "shutdown.json", {"server_exit_code": process.returncode})

    sources = {}
    for path in sorted((output / "events").glob("*.jsonl")):
        for line in path.read_text().splitlines():
            event = json.loads(line)
            if "source" in event:
                filename = event["source"]["file"]
                sources[filename] = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
    save(output / "source_hashes.json", sources)
    artifacts={}
    for directory in ('compiler_cache','launcher_sources'):
        for path in sorted((output/directory).rglob('*')):
            if path.is_file():
                artifacts[str(path.relative_to(output))]={"bytes":path.stat().st_size,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
    save(output/'compiler_artifacts.json',artifacts)
    subprocess.run(['npu-smi','info'],stdout=(output/'device_after.txt').open('w'),check=True)
    print("Archived:", output)


if __name__ == "__main__":
    main()
