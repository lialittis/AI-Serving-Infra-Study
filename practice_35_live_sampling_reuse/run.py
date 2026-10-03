"""Run live vLLM sampling-reuse cases in fresh processes on an idle Ascend device."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
CONFIGURATIONS = {
    "baseline": {"TASK_QUEUE_ENABLE": "1", "PER_STREAM_QUEUE": "0", "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:False,multi_stream_lazy_reclaim:False"},
}


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def snapshot(command, path):
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    save(path, dict(command=command, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--configs", nargs="+", choices=tuple(CONFIGURATIONS), default=["baseline"])
    parser.add_argument("--variants", nargs="+", choices=("default", "async", "protected"),
                        default=["default", "async", "protected"])
    parser.add_argument("--prompts", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    initial = snapshot(["npu-smi", "info"], out / "device_before.json")
    if initial.returncode or "No running processes" not in initial.stdout:
        raise RuntimeError("NPU must be idle before this experiment")
    snapshot(["npu-smi", "info", "-t", "health", "-i", "5", "-c", "0"], out / "health_before.json")
    sources = out / "sources"
    sources.mkdir()
    hashes = {}
    for path in sorted(HERE.glob("*.py")):
        data = path.read_bytes()
        (sources / path.name).write_bytes(data)
        hashes[path.name] = hashlib.sha256(data).hexdigest()
    save(out / "plan.json", dict(schema=1, start_ns=time.time_ns(),
         arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
         configurations={key: CONFIGURATIONS[key] for key in args.configs}, source_sha256=hashes))
    cases = []
    for configuration in args.configs:
        for variant in args.variants:
            case = f"{configuration}-{variant}"
            env = dict(os.environ)
            for key in ("PYTHONPATH", "PYTORCH_NO_NPU_MEMORY_CACHING", "ASCEND_LAUNCH_BLOCKING",
                        "PYTORCH_ALLOC_CONF", "MULTI_STREAM_LAZY_RECLAIM"):
                env.pop(key, None)
            env.update(CONFIGURATIONS[configuration])
            inherited = env.get("PYTHONPATH")
            cann_python = "/usr/local/Ascend/cann-9.0.0/python/site-packages"
            env.update(PYTHONPATH=os.pathsep.join(
                           part for part in (str(HERE), cann_python, inherited) if part),
                       PYTHONUNBUFFERED="1",
                       VLLM_WORKER_MULTIPROC_METHOD="spawn", MULTI_STREAM_MEMORY_REUSE="1")
            command = [sys.executable, str(HERE / "probe.py"), "--output", str(out / case),
                       "--variant", variant, "--prompts", str(args.prompts),
                       "--max-tokens", str(args.max_tokens), "--repeats", str(args.repeats)]
            start = time.time_ns()
            with (out / f"{case}.log").open("w") as log:
                result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1200)
            entry = dict(case=case, command=command, environment=CONFIGURATIONS[configuration],
                         start_ns=start, end_ns=time.time_ns(), returncode=result.returncode)
            cases.append(entry)
            save(out / "cases.json", cases)
            print(case, "passed" if result.returncode == 0 else f"exit={result.returncode}", flush=True)
            if result.returncode not in (0, 2):
                raise RuntimeError("Case failed; inspect " + str(out / f"{case}.log"))
    snapshot(["npu-smi", "info"], out / "device_after.json")
    snapshot(["npu-smi", "info", "-t", "health", "-i", "5", "-c", "0"], out / "health_after.json")
    print("Completed:", out, flush=True)


if __name__ == "__main__":
    main()
