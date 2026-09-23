"""Archive a bounded set of installed sources for the full operator audit.

Run on the original Ascend host. No model, torch import or NPU work is needed.
The original run's recorded repository HEADs and existing source hashes must
still match, so a later source checkout cannot silently replace that evidence.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


FILES = {
    "vllm": [
        "vllm/model_executor/models/qwen2.py",
        "vllm/model_executor/layers/vocab_parallel_embedding.py",
        "vllm/model_executor/layers/logits_processor.py",
        "vllm/model_executor/layers/linear.py",
        "vllm/v1/worker/block_table.py",
        "vllm/v1/sample/sampler.py",
    ],
    "vllm_ascend": [
        "vllm_ascend/ops/linear.py", "vllm_ascend/ops/layernorm.py",
        "vllm_ascend/ops/activation.py", "vllm_ascend/ops/rotary_embedding.py",
        "vllm_ascend/ops/triton/rope.py", "vllm_ascend/ops/triton/bincount.py",
        "vllm_ascend/ops/triton/penalty.py", "vllm_ascend/sample/sampler.py",
        "vllm_ascend/sample/penalties.py",
        "vllm_ascend/worker/block_table.py", "vllm_ascend/worker/model_runner_v1.py",
        "vllm_ascend/attention/attention_v1.py", "vllm_ascend/device/device_op.py",
        "vllm_ascend/utils.py", "csrc/torch_binding.cpp",
        "csrc/moe/add_rms_norm_bias/add_rms_norm_bias_torch_adpt.h",
    ],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    environment = json.loads((run / "environment.json").read_text())
    roots = {"vllm": Path("/vllm-workspace/vllm"),
             "vllm_ascend": Path("/vllm-workspace/vllm-ascend")}
    for name, root in roots.items():
        head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        expected = environment["source_repositories"][name]["head"]["stdout"].strip()
        if head != expected:
            raise ValueError("source HEAD changed: " + name)
    for filename, digest in json.loads((run / "source_hashes.json").read_text()).items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != digest:
            raise ValueError("previously observed source changed: " + filename)
    output = run / "full_analysis" / "sources"
    output.mkdir(parents=True, exist_ok=False)
    entries = []
    for repo, names in FILES.items():
        for name in names:
            src = roots[repo] / name
            data = src.read_bytes()
            dst = output / repo / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(data)
            # Individual dirty-state evidence matters even when HEAD is equal.
            diff = subprocess.check_output(["git", "-C", str(roots[repo]),
                                            "diff", "HEAD", "--", name], text=True)
            if diff:
                raise ValueError("selected source has local changes: " + str(src))
            entries.append({"file": str(src), "snapshot": str(dst.relative_to(run)),
                            "sha256": hashlib.sha256(data).hexdigest()})
    model = Path(environment["model_path"])
    for name in ("config.json", "generation_config.json"):
        data = (model / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != environment["model_files"][name]["sha256"]:
            raise ValueError("model configuration changed: " + name)
        dst = output / "model" / name
        dst.parent.mkdir(exist_ok=True)
        dst.write_bytes(data)
        entries.append({"file": str(model / name), "snapshot": str(dst.relative_to(run)),
                        "sha256": hashlib.sha256(data).hexdigest()})
    (output.parent / "source_manifest.json").write_text(json.dumps({
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": "Collected after profiling; original repository HEADs and recorded hashes checked.",
        "files": entries}, ensure_ascii=False, indent=2) + "\n")
    print("Archived", len(entries), "source/config files at", output)


if __name__ == "__main__":
    main()
