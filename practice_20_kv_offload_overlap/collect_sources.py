"""Archive exact installed contracts, including both offload implementations."""
import hashlib
import json
from pathlib import Path
import subprocess
import sysconfig


def collect(output, model):
    roots = {"vllm": Path("/vllm-workspace/vllm"),
             "ascend": Path("/vllm-workspace/vllm-ascend"),
             "packages": Path(sysconfig.get_paths()["purelib"]), "model": Path(model)}
    files = {
        "vllm": ["vllm/v1/simple_kv_offload/"+n+".py" for n in
                 ("manager", "worker", "metadata", "copy_backend")] + [
            "vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload_connector.py",
            "vllm/v1/core/block_pool.py", "vllm/v1/core/sched/scheduler.py",
            "vllm/v1/core/kv_cache_manager.py", "vllm/v1/worker/gpu_model_runner.py",
            "vllm/v1/worker/kv_connector_model_runner_mixin.py"],
        "ascend": ["vllm_ascend/simple_kv_offload/"+n+".py" for n in
                   ("worker", "copy_backend", "npu_mem_ops")] + [
            "vllm_ascend/distributed/kv_transfer/kv_pool/simple_cpu_offload/simple_cpu_offload_connector.py",
            "vllm_ascend/distributed/kv_transfer/__init__.py", "vllm_ascend/worker/model_runner_v1.py",
            "vllm_ascend/kv_offload/npu.py", "vllm_ascend/kv_offload/cpu_npu.py",
            "csrc/torch_binding.cpp"],
        "packages": ["torch_npu/npu/streams.py"], "model": ["config.json"]}
    manifest = {}
    for group, names in files.items():
        for name in names:
            source = roots[group]/name
            target = output/"sources"/group/name
            target.parent.mkdir(parents=True, exist_ok=True)
            data = source.read_bytes()
            target.write_bytes(data)
            manifest[str(target.relative_to(output))] = dict(original=str(source),
                sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
    (output/"source_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    versions = {g: subprocess.check_output(["git", "-C", str(roots[g]), "rev-parse", "HEAD"],
                                           text=True).strip() for g in ("vllm", "ascend")}
    (output/"revisions.json").write_text(json.dumps(versions, indent=2)+"\n")


def device_snapshot(path):
    result = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=30)
    path.write_text(result.stdout+result.stderr)
