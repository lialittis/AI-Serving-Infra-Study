"""Trace a real request crossing a 128-token KV block boundary on Ascend."""

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
    parser.add_argument("--port", type=int, default=8008)
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
    for name in ("sitecustomize.py", "kv_trace.py", "run_mapping.py", "summarize_mapping.py"):
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
               "--served-model-name", "qwen-kv-map", "--host", "127.0.0.1", "--port", str(args.port),
               "--tensor-parallel-size", "1", "--dtype", "bfloat16", "--max-model-len", "2048",
               "--max-num-seqs", "1", "--max-num-batched-tokens", "2048",
               "--gpu-memory-utilization", "0.3", "--block-size", "128", "--enforce-eager",
               "--no-enable-prefix-caching", "--no-enable-chunked-prefill", "--no-async-scheduling"]
    env = os.environ.copy()
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               VLLM_WORKER_MULTIPROC_METHOD="spawn", P08_TRACE_DIR=str(output / "events"),
               PYTHONPATH=os.pathsep.join([str(root), str(base_hooks.parent), env.get("PYTHONPATH", "")]))
    save(output / "command.json", {"argv": command,
         "environment_overrides": {key: env[key] for key in (
             "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "VLLM_WORKER_MULTIPROC_METHOD", "P08_TRACE_DIR", "PYTHONPATH")}})
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    token_ids = tokenizer.encode("hello", add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError("this exercise requires 'hello' to encode as one token")
    prompt_ids = token_ids * 126
    save(output / "prompt_info.json", {"token_text": "hello", "token_id": token_ids[0],
                                      "prompt_tokens": len(prompt_ids)})
    request_body = {"model": "qwen-kv-map", "prompt": prompt_ids,
                    "max_tokens": 8, "temperature": 0, "ignore_eos": True,
                    "stream": False, "seed": 0, "request_id": "practice08-cross-block"}
    save(output / "request.json", request_body)
    base_url = f"http://127.0.0.1:{args.port}"
    with (output / "server.log").open("w") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        save(output / "launcher.json", {"server_pid": process.pid,
                                       "started_at": datetime.now(timezone.utc).isoformat()})
        try:
            deadline = time.monotonic() + 300
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
                    raise TimeoutError("service readiness exceeded 300 seconds")
                time.sleep(1)
            print("Service ready; sending the single inference request", flush=True)
            process_lines = subprocess.check_output(
                ["ps", "-eo", "pid,ppid,pgid,args"], text=True).splitlines()
            selected = [process_lines[0]] + [line for line in process_lines[1:]
                if int(line.split(None, 3)[2]) == process.pid]
            (output / "processes.txt").write_text("\n".join(selected) + "\n")
            request = urllib.request.Request(base_url + "/v1/completions",
                data=json.dumps(request_body).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=120) as response:
                save(output / "response.json", json.load(response))
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
    subprocess.run([sys.executable, str(root / "summarize_mapping.py"), str(output)], check=True)
    print("Archived:", output)


if __name__ == "__main__":
    main()
