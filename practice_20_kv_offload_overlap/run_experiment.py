"""Real Ascend prefix offload/reload experiment, with independent diagnostics."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "practice_17_vllm_multistream"))
from run_sampling_benchmark import clean_environment, save
sys.path.pop(0)
from collect_sources import collect, device_snapshot


def request(base, payload, first=None):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(base + "/v1/completions", data,
                                 {"Content-Type": "application/json"})
    start = time.perf_counter_ns()
    chunks, ids, first_ns = [], [], None
    with urllib.request.urlopen(req, timeout=180) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            raw = line[6:].strip()
            if raw == b"[DONE]":
                break
            now = time.perf_counter_ns()
            chunk = json.loads(raw)
            chunks.append(chunk)
            for choice in chunk.get("choices", []):
                tokens = choice.get("token_ids") or []
                ids.extend(tokens)
                if tokens and first_ns is None:
                    first_ns = now
                    if first is not None:
                        first.set()
    finish = time.perf_counter_ns()
    if len(ids) != payload["max_tokens"]:
        raise ValueError("returned token count: %d != %d" % (len(ids), payload["max_tokens"]))
    if not any(c.get("finish_reason") == "length" for x in chunks for c in x.get("choices", [])):
        raise ValueError("missing length finish")
    return dict(start_ns=start, end_ns=finish, ttft_ms=(first_ns-start)/1e6,
                latency_ms=(finish-start)/1e6, token_ids=ids, chunks=chunks)


def prompt(tokenizer, length, trial, role):
    head = tokenizer.encode("Experiment %06d %s. " % (trial, role), add_special_tokens=False)
    body = tokenizer.encode("Explain the relationship between memory, computation and reliable software. ",
                            add_special_tokens=False)
    return (head + body * (length // len(body) + 1))[:length]


def cycle(base, tokenizer, length, trial, directory):
    directory.mkdir()
    common = dict(model="p20", temperature=0, ignore_eos=True, stream=True,
                  return_token_ids=True, stream_options={"include_usage": True})
    inputs = {"A": prompt(tokenizer, length + 32, trial, "A"),
              "B": prompt(tokenizer, 1024, trial, "B")}
    inputs.update({"C%d" % i: prompt(tokenizer, 3104, trial, "C%d" % i) for i in range(5)})
    save(directory / "inputs.json", inputs)
    results = {}
    for role in ["A"] + ["C%d" % i for i in range(5)]:
        payload = dict(common, prompt=inputs[role], max_tokens=64 if role == "A" else 4,
                       request_id="p20-%d-%s" % (trial, role))
        results[role] = request(base, payload)
        save(directory / (role + ".json"), results[role])
    first = threading.Event()
    with ThreadPoolExecutor(max_workers=1) as executor:
        b = executor.submit(request, base, dict(common, prompt=inputs["B"], max_tokens=256,
                            request_id="p20-%d-B" % trial), first)
        if not first.wait(120):
            raise TimeoutError("B did not produce first token")
        results["reload"] = request(base, dict(common, prompt=inputs["A"], max_tokens=64,
                                    request_id="p20-%d-reload" % trial))
        results["B"] = b.result()
    for role in ("B", "reload"):
        save(directory / (role + ".json"), results[role])
    summary = dict(trial=trial, prefix_tokens=length,
                   reload_ttft_ms=results["reload"]["ttft_ms"],
                   reload_latency_ms=results["reload"]["latency_ms"],
                   b_latency_ms=results["B"]["latency_ms"],
                   pair_window_ms=(max(results[r]["end_ns"] for r in ("B", "reload"))-
                                   results["B"]["start_ns"])/1e6,
                   a_reload_equal=results["A"]["token_ids"] == results["reload"]["token_ids"])
    save(directory / "summary.json", summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="/data/huggingface_home/hub/Qwen2.5-0.5B-Instruct")
    p.add_argument("--mode", choices=["native", "serialized", "recompute"], required=True)
    p.add_argument("--phase", choices=["diagnostic", "benchmark"], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--port", type=int, default=8020)
    p.add_argument("--prefix-lengths", type=int, nargs="+", default=[1024, 3072])
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--block", type=int, default=0, help="matched repetitions use the same block-pair ID")
    a = p.parse_args()
    if min(a.prefix_lengths) < 128 or max(a.prefix_lengths) > 3072 or a.warmup < 0 or a.repeats < 1:
        p.error("unsupported experiment size")
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    collect(out, a.model)
    device_snapshot(out/"device_before.txt")
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", a.port))
    env = clean_environment(os.environ)
    env = {k: v for k, v in env.items() if not k.startswith("P20_")}
    env.update(P20_MODE=a.mode, P20_PHASE=a.phase, P20_OUTPUT=str(out),
               PYTHONPATH=os.pathsep.join([str(HERE), env.get("PYTHONPATH", "")]))
    cmd = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", a.model,
           "--served-model-name", "p20", "--host", "127.0.0.1", "--port", str(a.port),
           "--tensor-parallel-size", "1", "--dtype", "bfloat16", "--enforce-eager",
           "--max-model-len", "4096", "--max-num-seqs", "4", "--max-num-batched-tokens", "4096",
           "--num-gpu-blocks-override", "97", "--block-size", "128", "--gpu-memory-utilization", "0.3",
           "--enable-prefix-caching", "--no-enable-chunked-prefill", "--no-async-scheduling",
           "--seed", "123", "--additional-config", '{"enable_async_exponential":false}']
    if a.mode != "recompute":
        cmd += ["--kv-transfer-config", json.dumps(dict(kv_connector="P20Connector",
                kv_connector_module_path="p20_connector", kv_role="kv_both",
                kv_connector_extra_config=dict(cpu_bytes_to_use=384*1024**2, lazy_offload=False)))]
    elif a.phase == "diagnostic":
        # A diagnostic-only sitecustomize installs the common compute observer.
        env["P20_RECOMPUTE_OBSERVER"] = "1"
    if a.phase == "diagnostic":
        cmd += ["--profiler-config", json.dumps(dict(profiler="torch", torch_profiler_dir=str(out/"profiler"),
                torch_profiler_with_stack=False, ignore_frontend=True))]
    save(out/"command.json", dict(argv=cmd, mode=a.mode, phase=a.phase, prefix_lengths=a.prefix_lengths,
         warmup=a.warmup, repeats=a.repeats, block=a.block,
         environment={k: env[k] for k in ("PYTHONPATH", "P20_MODE", "P20_PHASE", "P20_OUTPUT")}))
    (out/"instrumentation").mkdir()
    for path in HERE.glob("*.py"):
        (out/"instrumentation"/path.name).write_bytes(path.read_bytes())
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    base = "http://127.0.0.1:%d" % a.port
    def control(endpoint):
        with urllib.request.urlopen(urllib.request.Request(base+endpoint, data=b""), timeout=240) as r:
            return r.status
    proc = None
    try:
        with (out/"server.log").open("w") as log:
            proc = subprocess.Popen(cmd, env=env, cwd=out, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            save(out/"launcher.json", dict(pid=proc.pid))
            deadline = time.monotonic()+600
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError("server exited; see server.log")
                try:
                    with urllib.request.urlopen(base+"/health", timeout=2) as response:
                        if response.status == 200:
                            break
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(1)
            else:
                raise TimeoutError("server startup")
            # Short shape-independent initialization; excluded from every trial.
            request(base, dict(model="p20", prompt=[100, 200, 300], max_tokens=4,
                               temperature=0, ignore_eos=True, stream=True, return_token_ids=True))
            if a.phase == "diagnostic":
                save(out/"profile_start.json", dict(status=control("/start_profile")))
            for length in a.prefix_lengths:
                for phase, count in (("warmup", a.warmup), ("measure", a.repeats)):
                    for i in range(count):
                        trial = a.block*100000 + length*10 + (i if phase == "measure" else 100+i)
                        summary = cycle(base, tokenizer, length, trial,
                                        out/("%s-%d-%02d" % (phase, length, i)))
                        print(json.dumps(dict(phase=phase, **summary)), flush=True)
            if a.phase == "diagnostic":
                save(out/"profile_stop.json", dict(status=control("/stop_profile")))
    finally:
        if proc is not None:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGINT)
                try:
                    proc.wait(timeout=90)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGTERM)
                    proc.wait(timeout=30)
            save(out/"shutdown.json", dict(exit_code=proc.returncode))
            device_snapshot(out/"device_after.txt")


if __name__ == "__main__":
    main()
