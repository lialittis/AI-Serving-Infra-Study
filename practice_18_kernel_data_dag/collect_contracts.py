"""Archive installed Python sources used for explicit memory-access contracts."""
import argparse
import hashlib
import json
from pathlib import Path

SOURCES = {
    'rope.py': '/vllm-workspace/vllm-ascend/vllm_ascend/ops/triton/rope.py',
    'bincount.py': '/vllm-workspace/vllm-ascend/vllm_ascend/ops/triton/bincount.py',
    'penalty.py': '/vllm-workspace/vllm-ascend/vllm_ascend/ops/triton/penalty.py',
    'block_table.py': '/vllm-workspace/vllm/vllm/v1/worker/block_table.py',
}


def collect(output):
    target = output / 'contract_sources'
    target.mkdir(exist_ok=False)
    manifest = {}
    for name, source in SOURCES.items():
        data = Path(source).read_bytes()
        (target / name).write_bytes(data)
        manifest[name] = dict(original=source, sha256=hashlib.sha256(data).hexdigest())
    (target / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    collect(parser.parse_args().output)
