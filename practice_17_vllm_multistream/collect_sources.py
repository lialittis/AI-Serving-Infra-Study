"""Snapshot exact installed sources used to explain the observed stream path."""
import hashlib
import json
from pathlib import Path
import subprocess
import sysconfig


def collect(output, model):
    packages = Path(sysconfig.get_paths()["purelib"])
    roots = {
        "vllm": Path("/vllm-workspace/vllm"),
        "vllm_ascend": Path("/vllm-workspace/vllm-ascend"),
        "packages": packages,
        "model": Path(model),
    }
    config = json.loads((Path(model) / "config.json").read_text())
    model_file = "vllm/model_executor/models/qwen2.py" if config["model_type"] == "qwen2" else "vllm/model_executor/models/llama.py"
    selected = {
        "vllm": ["vllm/v1/worker/gpu_model_runner.py", model_file],
        "vllm_ascend": ["vllm_ascend/worker/model_runner_v1.py",
                        "vllm_ascend/sample/sampler.py", "vllm_ascend/utils.py",
                        "vllm_ascend/ascend_config.py"],
        "packages": ["torch_npu/npu/streams.py"],
        "model": ["config.json", "generation_config.json"],
    }
    manifest = {}
    for group, names in selected.items():
        for name in names:
            source = roots[group] / name
            data = source.read_bytes()
            target = output / "sources" / group / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            manifest[str(target.relative_to(output))] = {
                "original": str(source), "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
    for group in ("vllm", "vllm_ascend"):
        manifest[group + "_revision"] = subprocess.check_output(
            ["git", "-C", str(roots[group]), "rev-parse", "HEAD"], text=True).strip()
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
