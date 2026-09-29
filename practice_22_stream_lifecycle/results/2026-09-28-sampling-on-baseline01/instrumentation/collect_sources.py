"""Snapshot installed submission/compilation sources; never modify libraries."""
import hashlib
import json
from pathlib import Path
import subprocess
import sysconfig


def collect_sources(output):
    packages=Path(sysconfig.get_paths()['purelib'])
    roots={'vllm':Path('/vllm-workspace/vllm'),'vllm_ascend':Path('/vllm-workspace/vllm-ascend'),
           'packages':packages}
    selected={
     'vllm':['vllm/model_executor/models/qwen2.py','vllm/v1/worker/gpu_model_runner.py','vllm/v1/utils.py'],
     'vllm_ascend':['vllm_ascend/worker/model_runner_v1.py','vllm_ascend/worker/block_table.py',
                    'vllm_ascend/attention/attention_v1.py','vllm_ascend/device/device_op.py',
                    'vllm_ascend/ops/linear.py','vllm_ascend/ops/rotary_embedding.py'],
     'packages':['triton/runtime/jit.py','triton/runtime/cache.py','triton/compiler/compiler.py',
                 'triton/backends/ascend/driver.py','triton/backends/ascend/compiler.py',
                 'triton/backends/ascend/npu_utils.cpp','torch/_ops.py',
                 'torch_npu/npu/streams.py','torch_npu/op_plugin/atb/_atb_ops.py']}
    manifest={}
    for group,names in selected.items():
        for name in names:
            source=roots[group]/name;data=source.read_bytes()
            target=output/'sources'/group/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
            manifest[str(target.relative_to(output))]={'original':str(source),'sha256':hashlib.sha256(data).hexdigest()}
    (output/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for group in ('vllm','vllm_ascend'):
        (output/(group+'_revision.txt')).write_text(subprocess.check_output(['git','-C',str(roots[group]),'rev-parse','HEAD'],text=True))
