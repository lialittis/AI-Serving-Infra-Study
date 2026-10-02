import datetime
import hashlib
import importlib.metadata
import json
import os
import pathlib
import platform
import subprocess
import sys

root = pathlib.Path('/usr/local/python3.12.13/lib/python3.12/site-packages')
paths = [
    'torch_npu/version.py', 'torch/version.py',
    'torch_npu/include/torch_npu/csrc/core/npu/NPUCachingAllocator.h',
    'torch_npu/include/torch_npu/csrc/core/npu/NPUStream.h',
    'torch_npu/include/torch_npu/csrc/core/npu/NPUEvent.h',
    'torch_npu/include/torch_npu/csrc/core/npu/NPUEventManager.h',
    'torch_npu/include/third_party/op-plugin/op_plugin/utils/op_api_common.h',
    'torch_npu/include/third_party/op-plugin/op_plugin/utils/op_api_common_base.h',
    'torch/include/c10/core/StorageImpl.h', 'torch/include/c10/core/Allocator.h',
    'torch/include/c10/util/UniqueVoidPtr.h', 'torch/include/c10/core/TensorImpl.h',
    'torch/utils/data/_utils/pin_memory.py',
    'torch_npu/npu/memory.py', 'torch_npu/npu/streams.py',
]
files = []
for relative in paths:
    path = root / relative
    if path.is_file():
        data = path.read_bytes()
        files.append({'path': str(path), 'relative_path': relative,
                      'sha256': hashlib.sha256(data).hexdigest(),
                      'bytes': len(data), 'text': data.decode()})
binary = root / 'torch_npu/lib/libtorch_npu.so'
hasher = hashlib.sha256()
with binary.open('rb') as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
        hasher.update(chunk)
packages = {}
for name in ['torch', 'torch-npu', 'vllm', 'vllm-ascend']:
    try:
        packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        packages[name] = None
print(json.dumps({
    'captured_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'python': sys.version, 'executable': sys.executable, 'machine': platform.machine(),
    'packages': packages,
    'environment': {name: os.getenv(name) for name in [
        'ASCEND_HOME_PATH', 'PYTORCH_NPU_ALLOC_CONF', 'PYTORCH_NO_NPU_MEMORY_CACHING',
        'TASK_QUEUE_ENABLE', 'MULTI_STREAM_MEMORY_REUSE', 'MULTI_STREAM_LAZY_RECLAIM']},
    'binary': {'path': str(binary), 'bytes': binary.stat().st_size, 'sha256': hasher.hexdigest()},
    'files': files,
    'inspection': 'Read-only files and package metadata; did not import torch or initialize NPU.'
}, indent=2))
