"""Start an isolated real vLLM-Ascend server and archive its model FX graph."""
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


def collect_sources(output, model):
    """Archive installed code that defines the graph boundary and model."""
    roots = {"vllm": Path("/vllm-workspace/vllm"),
             "vllm_ascend": Path("/vllm-workspace/vllm-ascend")}
    files = {
        "vllm": ["vllm/config/compilation.py", "vllm/compilation/backends.py",
                 "vllm/compilation/decorators.py", "vllm/compilation/wrapper.py",
                 "vllm/model_executor/models/qwen2.py", "vllm/envs.py",
                 "vllm/model_executor/layers/attention/attention.py"],
        "vllm_ascend": ["vllm_ascend/platform.py", "vllm_ascend/compilation/compiler_interface.py",
                        "vllm_ascend/compilation/acl_graph.py",
                        "vllm_ascend/attention/attention_v1.py",
                        "vllm_ascend/worker/model_runner_v1.py",
                        "vllm_ascend/ops/rotary_embedding.py", "vllm_ascend/ops/linear.py"]}
    manifest = {}
    for key, names in files.items():
        root = roots[key]
        manifest[key] = {"commit": subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(), "files": {}}
        for name in names:
            source = root / name
            if not source.is_file():
                raise FileNotFoundError(source)
            content = source.read_bytes()
            target = output / "sources" / key / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            manifest[key]["files"][name] = hashlib.sha256(content).hexdigest()
        manifest[key]["tracked_diff"] = subprocess.check_output(
            ["git", "-C", str(root), "diff", "HEAD", "--"] + names, text=True)
    for name in ("config.json", "generation_config.json"):
        target = output / "sources" / "model" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((Path(model) / name).read_bytes())
    save(output / "source_manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/data/huggingface_home/hub/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = args.output.resolve()
    with socket.socket() as sock:
        # A stopped HTTP server may leave TIME_WAIT sockets behind.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", args.port))
    output.mkdir(parents=True, exist_ok=False)
    (output / "instrumentation").mkdir()
    for name in ("sitecustomize.py", "graph_capture.py", "run_graph.py"):
        (output / "instrumentation" / name).write_bytes((root / name).read_bytes())
    collect_sources(output, args.model)
    subprocess.run([sys.executable, str(root.parent / "practice_03_ascend_start/collect_environment.py"),
                    "--model", args.model, "--output", str(output / "environment.json")], check=True)
    compilation = {"mode": 3, "cudagraph_mode": "PIECEWISE", "cudagraph_capture_sizes": [1],
                   "custom_ops": ["all"]}
    command = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", args.model,
               "--served-model-name", "qwen-graph", "--host", "127.0.0.1", "--port", str(args.port),
               "--tensor-parallel-size", "1", "--dtype", "bfloat16", "--max-model-len", "2048",
               "--max-num-seqs", "1", "--max-num-batched-tokens", "2048",
               "--gpu-memory-utilization", "0.3", "--block-size", "128",
               "--no-enable-prefix-caching", "--no-enable-chunked-prefill", "--no-async-scheduling",
               "--compilation-config", json.dumps(compilation)]
    env = os.environ.copy()
    for key in ("P07_TRACE_DIR", "P08_TRACE_DIR", "P09_TRACE_DIR"):
        env.pop(key, None)
    overrides = dict(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                     VLLM_WORKER_MULTIPROC_METHOD="spawn", VLLM_USE_AOT_COMPILE="0",
                     VLLM_DISABLE_COMPILE_CACHE="1", P10_GRAPH_DIR=str(output / "graphs"),
                     PYTHONPATH=os.pathsep.join([str(root), env.get("PYTHONPATH", "")]))
    env.update(overrides)
    save(output / "command.json", {"argv": command, "environment_overrides": overrides})
    from transformers import AutoTokenizer

    ids = AutoTokenizer.from_pretrained(args.model, local_files_only=True).encode(
        "hello", add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError("Expected hello to encode as one token")
    body = {"model": "qwen-graph", "prompt": ids * 126, "max_tokens": 2,
            "temperature": 0, "ignore_eos": True, "stream": False, "seed": 0,
            "request_id": "practice10-graph"}
    save(output / "request.json", body)
    base_url = "http://127.0.0.1:%s" % args.port
    with (output / "server.log").open("w") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        save(output / "launcher.json", {"server_pid": process.pid,
             "started_at": datetime.now(timezone.utc).isoformat()})
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
                    raise TimeoutError("readiness exceeded 600 seconds")
                time.sleep(1)
            print("Service ready; native graph capture completed during startup", flush=True)
            times = {"request_start_ns": time.time_ns()}
            request = urllib.request.Request(base_url + "/v1/completions",
                data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=180) as response:
                save(output / "response.json", json.loads(response.read()))
            times["request_end_ns"] = time.time_ns()
            save(output / "request_window.json", times)
            print("126-input / 2-output request completed", flush=True)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            save(output / "shutdown.json", {"server_exit_code": process.returncode})
    print("Archived:", output)


if __name__ == "__main__":
    main()
