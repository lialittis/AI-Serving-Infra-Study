"""Run two real requests through a one-usable-block KV pool and profile reuse."""

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
    files = {
        "vllm": ["vllm/config/cache.py", "vllm/v1/core/block_pool.py",
                 "vllm/v1/core/kv_cache_utils.py", "vllm/v1/core/kv_cache_manager.py",
                 "vllm/v1/core/kv_cache_coordinator.py", "vllm/v1/core/single_type_kv_cache_manager.py",
                 "vllm/v1/core/sched/scheduler.py", "vllm/v1/worker/gpu_model_runner.py",
                 "vllm/v1/engine/core.py", "vllm/model_executor/models/qwen2.py",
                 "vllm/compilation/backends.py", "vllm/compilation/piecewise_backend.py"],
        "vllm_ascend": ["vllm_ascend/worker/model_runner_v1.py",
                        "vllm_ascend/worker/block_table.py",
                        "vllm_ascend/attention/attention_v1.py", "vllm_ascend/device/device_op.py",
                        "vllm_ascend/compilation/acl_graph.py",
                        "vllm_ascend/compilation/compiler_interface.py"]}
    manifest = {}
    for key, names in files.items():
        repo = Path("/vllm-workspace/" + ("vllm" if key == "vllm" else "vllm-ascend"))
        manifest[key] = dict(commit=subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(), files={})
        diff = subprocess.check_output(["git", "-C", str(repo), "diff", "HEAD", "--"] + names, text=True)
        if diff:
            raise ValueError("Selected installed sources have local modifications: " + key)
        manifest[key]["tracked_diff"] = diff
        for name in names:
            content = (repo / name).read_bytes()
            target = output / "sources" / key / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            manifest[key]["files"][name] = hashlib.sha256(content).hexdigest()
    for name in ("config.json", "generation_config.json"):
        target = output / "sources/model" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(model, name).read_bytes())
    save(output / "source_manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/data/huggingface_home/hub/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--port", type=int, default=8012)
    parser.add_argument("--mode", choices=("eager", "graph"), default="eager",
                        help="graph uses PIECEWISE compilation and capture size [1]")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = args.output.resolve()
    if not Path(args.model, "config.json").is_file():
        parser.error("model config.json missing")
    # Never attach this experiment's request to an unrelated preexisting server.
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", args.port))
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "instrumentation"
    snapshot.mkdir()
    hashes = {}
    for name in ("sitecustomize.py", "lifetime_trace.py", "run_reuse.py"):
        content = (root / name).read_bytes()
        (snapshot / name).write_bytes(content)
        hashes["instrumentation/" + name] = hashlib.sha256(content).hexdigest()
    base_hooks = root.parent / "practice_07_real_request_trace/trace_hooks.py"
    for source in (base_hooks, root.parent / "practice_03_ascend_start/collect_environment.py"):
        content = source.read_bytes()
        (snapshot / source.name).write_bytes(content)
        hashes["instrumentation/" + source.name] = hashlib.sha256(content).hexdigest()
    save(output / "instrumentation_hashes.json", hashes)
    collect_sources(output, args.model)
    subprocess.run(["npu-smi", "info"], stdout=(output / "device_before.txt").open("w"), check=True)
    subprocess.run([sys.executable, str(root.parent / "practice_03_ascend_start/collect_environment.py"),
                    "--model", args.model, "--output", str(output / "environment.json")], check=True)
    command = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", args.model,
               "--served-model-name", "qwen-reuse", "--host", "127.0.0.1", "--port", str(args.port),
               "--tensor-parallel-size", "1", "--dtype", "bfloat16", "--max-model-len", "128",
               "--max-num-seqs", "1", "--max-num-batched-tokens", "128",
               "--gpu-memory-utilization", "0.3", "--block-size", "128",
               "--no-enable-prefix-caching", "--no-enable-chunked-prefill", "--no-async-scheduling"]
    command += ["--num-gpu-blocks-override", "2"]
    if args.mode == "eager":
        command += ["--enforce-eager"]
    else:
        compilation = {"mode": 3, "cudagraph_mode": "PIECEWISE",
                       "cudagraph_capture_sizes": [1], "custom_ops": ["all"]}
        command += ["--compilation-config", json.dumps(compilation)]
    profiler_config = {"profiler": "torch", "torch_profiler_dir": str(output / "profiler"),
                       "torch_profiler_with_stack": False, "ignore_frontend": True}
    command += ["--profiler-config", json.dumps(profiler_config)]
    env = os.environ.copy()
    for key in ("P07_TRACE_DIR", "P08_TRACE_DIR", "P09_TRACE_DIR", "P10_GRAPH_DIR", "P11_TRACE_DIR"):
        env.pop(key, None)
    overrides = dict(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                     VLLM_WORKER_MULTIPROC_METHOD="spawn", P12_TRACE_DIR=str(output / "events"),
                     VLLM_USE_AOT_COMPILE="0", VLLM_DISABLE_COMPILE_CACHE="1",
                     PYTHONPATH=os.pathsep.join([str(root), str(base_hooks.parent), env.get("PYTHONPATH", "")]))
    env.update(overrides)
    save(output / "command.json", {"mode": args.mode, "argv": command,
                                   "environment_overrides": overrides})
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    requests = {}
    prompts = {}
    for role, word in (("A", "hello"), ("B", "world")):
        ids = tokenizer.encode(word, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError("expected one token for " + word)
        prompts[role] = dict(text=word, token_id=ids[0], prompt_tokens=126)
        requests[role] = dict(model="qwen-reuse", prompt=ids * 126, max_tokens=2,
                              temperature=0, ignore_eos=True, stream=False, seed=0,
                              request_id="practice12-" + role)
        save(output / ("request_" + role + ".json"), requests[role])
    if prompts["A"]["token_id"] == prompts["B"]["token_id"]:
        raise ValueError("A and B must use different prompts")
    save(output / "prompt_info.json", prompts)
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

            warmup = dict(requests["A"], request_id="practice12-warmup")
            save(output / "warmup_request.json", warmup)
            save(output / "warmup_response.json", post("/v1/completions", warmup)["body"])
            controls = []
            controls.append({"endpoint": "/start_profile", **post("/start_profile")})
            save(output / "profile_control.json", controls)
            print("Profiler started; sending sequential A/B requests into the same KV pool", flush=True)
            try:
                windows = {}
                for role in ("A", "B"):
                    times = {"request_start_ns": time.time_ns()}
                    save(output / ("response_" + role + ".json"),
                         post("/v1/completions", requests[role])["body"])
                    times["request_end_ns"] = time.time_ns()
                    windows[role] = times
                    save(output / "request_windows.json", windows)
                    print("Request " + role + " completed", flush=True)
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

    subprocess.run(["npu-smi", "info"], stdout=(output / "device_after.txt").open("w"), check=True)
    sources = {}
    for path in sorted((output / "events").glob("*.jsonl")):
        for line in path.read_text().splitlines():
            event = json.loads(line)
            if "source" in event:
                filename = event["source"]["file"]
                sources[filename] = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
    save(output / "source_hashes.json", sources)
    print("Archived:", output)


if __name__ == "__main__":
    main()
