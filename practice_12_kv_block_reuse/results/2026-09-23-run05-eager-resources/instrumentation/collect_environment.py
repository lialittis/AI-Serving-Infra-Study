"""Record selected software/device metadata and model hashes without credentials."""

import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def capture(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        return {"exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"error": str(error)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/data/huggingface_home/hub/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = Path(args.model).resolve()
    if not (model / "config.json").is_file():
        parser.error("Model config.json is missing")
    if args.output.exists():
        parser.error("Output already exists; choose a new file")

    data = {
        "captured_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python": sys.version,
        "executable": sys.executable,
        "machine": platform.machine(),
        "os_release": Path("/etc/os-release").read_text(),
        "packages": {},
        "model_path": str(model),
        "model_files": {},
        "environment": {key: os.environ.get(key) for key in (
            "ASCEND_HOME_PATH", "ASCEND_RT_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES",
            "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "VLLM_WORKER_MULTIPROC_METHOD",
        )},
        "npu_smi": capture(["npu-smi", "info"]),
        "npu_mapping": capture(["npu-smi", "info", "-m"]),
        "source_repositories": {},
    }
    for package in (
        "torch", "torch-npu", "vllm", "vllm-ascend", "triton", "triton-ascend",
        "transformers", "modelscope", "huggingface-hub",
    ):
        try:
            data["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            data["packages"][package] = None
    for path in sorted(model.iterdir()):
        if path.is_file():
            data["model_files"][path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    for name, repo in (
        ("vllm", "/vllm-workspace/vllm"),
        ("vllm_ascend", "/vllm-workspace/vllm-ascend"),
    ):
        data["source_repositories"][name] = {
            "head": capture(["git", "-C", repo, "rev-parse", "HEAD"]),
            "status": capture(["git", "-C", repo, "status", "--short"]),
        }
    # A git SHA alone cannot identify local patches to the benchmark semantics.
    data["source_files"] = {}
    for name in (
        "/vllm-workspace/vllm/vllm/benchmarks/serve.py",
        "/vllm-workspace/vllm/vllm/benchmarks/datasets/datasets.py",
        "/vllm-workspace/vllm/vllm/benchmarks/datasets/utils.py",
    ):
        path = Path(name)
        if path.is_file():
            data["source_files"][name] = sha256(path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print("Environment saved:", args.output)


if __name__ == "__main__":
    main()
