"""Run the sampling-gap interval sweep in fresh processes on an idle Ascend device."""
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
    parser.add_argument("--modes", nargs="+", choices=("asis", "record", "leading-wait"),
                        default=["asis", "record", "leading-wait"])
    parser.add_argument("--gaps-us", type=int, nargs="+",
                        default=[0, 200, 1000, 5000, 25000, 50000])
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--backlogs", type=int, nargs="+", default=[0, 8, 32],
                        help="consumer-backlog values; record/leading-wait run at the first value")
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
        for mode in args.modes:
            backlogs = args.backlogs if mode == "asis" else args.backlogs[:1]
            for backlog in backlogs:
                case = f"{configuration}-{mode}-b{backlog}"
                env = dict(os.environ)
                for key in ("PYTHONPATH", "PYTORCH_NO_NPU_MEMORY_CACHING", "ASCEND_LAUNCH_BLOCKING",
                            "PYTORCH_ALLOC_CONF", "MULTI_STREAM_LAZY_RECLAIM"):
                    env.pop(key, None)
                env.update(CONFIGURATIONS[configuration])
                env.update(MULTI_STREAM_MEMORY_REUSE="1", PYTHONUNBUFFERED="1")
                command = [sys.executable, str(HERE / "probe.py"), "--output", str(out / case),
                           "--mode", mode, "--iterations", str(args.iterations),
                           "--consumer-backlog", str(backlog),
                           "--gaps-us", *[str(value) for value in args.gaps_us]]
                start = time.time_ns()
                with (out / f"{case}.log").open("w") as log:
                    result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=900)
                entry = dict(case=case, command=command, environment=CONFIGURATIONS[configuration],
                             start_ns=start, end_ns=time.time_ns(), returncode=result.returncode)
                cases.append(entry)
                save(out / "cases.json", cases)
                print(case, "passed" if result.returncode == 0 else "failed", flush=True)
                if result.returncode:
                    raise RuntimeError("Case failed; inspect " + str(out / f"{case}.log"))
    snapshot(["npu-smi", "info"], out / "device_after.json")
    snapshot(["npu-smi", "info", "-t", "health", "-i", "5", "-c", "0"], out / "health_after.json")
    print("Completed:", out, flush=True)


if __name__ == "__main__":
    main()
