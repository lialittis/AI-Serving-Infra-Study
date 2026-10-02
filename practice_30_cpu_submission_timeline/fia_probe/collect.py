"""Read installed files after the probe; archive metadata, never device binaries."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
CANN = Path('/usr/local/Ascend/cann-9.0.0')
TBE = CANN / 'opp/built-in/op_impl/ai_core/tbe'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    evidence = ROOT / 'evidence'
    evidence.mkdir(exist_ok=False)
    sources = ROOT / 'sources'
    sources.mkdir(exist_ok=True)
    files = [
        CANN / 'include/acl/acl_rt.h',
        CANN / 'include/aclnnop/aclnn_fused_infer_attention_score_v3.h',
        TBE / 'impl/ops_transformer/dynamic/fused_infer_attention_score.py',
        TBE / 'impl/ops_transformer/ascendc/fused_infer_attention_score/fused_infer_attention_score.cpp',
        TBE / 'impl/ops_transformer/ascendc/fused_infer_attention_score/fused_infer_attention_score_tilingkey.h',
        Path('/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/include/third_party/op-plugin/op_plugin/utils/op_api_common.h'),
        Path('/vllm-workspace/vllm-ascend/vllm_ascend/attention/attention_v1.py'),
    ]
    config_path = TBE / 'kernel/config/ascend910b/ops_transformer/fused_infer_attention_score.json'
    config = json.loads(config_path.read_text())
    files.append(config_path)
    file_manifest = {}
    for index, path in enumerate(files):
        name = path.name if index < 2 else f'{index:02d}-{path.name}'
        target = sources / name
        if target.exists():
            assert sha(path) == sha(target), path
        else:
            shutil.copyfile(path, target)
        file_manifest[str(path)] = dict(local=str(target.relative_to(ROOT)), sha256=sha(path))
    cases = {}
    for case in ('bf16-l42', 'bf16-l43', 'fp16-l42'):
        rows = json.loads((ROOT / 'results' / case / 'api_calls.json').read_text())
        objects = sorted({r['text'] for r in rows if r['kind'] == 'file_open' and r['text'].endswith('.o')})
        assert len(objects) == 1, objects
        obj = Path(objects[0])
        metadata_path = obj.with_suffix('.json')
        metadata = json.loads(metadata_path.read_text())
        target = evidence / metadata_path.name
        shutil.copyfile(metadata_path, target)
        entries = [r['u'][2] for r in rows if r['kind'] == 'aclrtBinaryGetFunctionByEntry']
        matched = [k for k in metadata['kernelList'] if k['tilingKey'] in entries]
        assert len(matched) == 1, matched
        selected_config = [b for b in config['binList'] if Path(b['binInfo']['jsonFilePath']).name == metadata_path.name]
        assert selected_config, obj
        cases[case] = dict(
            object_path=str(obj), object_size=obj.stat().st_size, sha256=sha(obj),
            file_description=subprocess.check_output(['file', str(obj)], text=True).strip(),
            metadata_path=str(metadata_path), metadata_local=str(target.relative_to(ROOT)),
            metadata_sha256=sha(target), kernel_count=len(metadata['kernelList']),
            selected_entry=matched[0], config_entries=selected_config,
        )
    results = {c: json.loads((ROOT / 'results' / c / 'results.json').read_text())
               for c in ('plain-before', 'bf16-l42', 'bf16-l43', 'fp16-l42', 'plain-after')}
    assert all(r['status'] == 'passed' and r['integrity']['unchanged'] for r in results.values())
    assert results['plain-before']['output'] == results['bf16-l42']['output'] == results['plain-after']['output']
    integrity = results['plain-before']['integrity']['before']
    assert all(r['integrity']['before'] == integrity == r['integrity']['after'] for r in results.values())
    assert all(sha(path) == digest for path, digest in integrity.items())
    import vllm, vllm_ascend, torch, torch_npu
    environment = dict(python=sys.version, imports={m.__name__: m.__file__ for m in (vllm, vllm_ascend, torch, torch_npu)})
    environment['revisions'] = {p: subprocess.check_output(['git', '-C', p, 'rev-parse', 'HEAD'], text=True).strip()
                                for p in ('/vllm-workspace/vllm', '/vllm-workspace/vllm-ascend')}
    idle = subprocess.check_output(['npu-smi', 'info'], text=True)
    assert 'No running processes' in idle
    (evidence / 'device_after.txt').write_text(idle)
    # Capture exact collector and experiment versions used; no installed file is written.
    scripts = {n: sha(ROOT / n) for n in ('probe.py', 'hook.cpp', 'generated.inc', 'generate_hook.py', 'collect.py', 'hook.so')}
    payload = dict(cases=cases, source_manifest=file_manifest, environment=environment,
                   validation=dict(all_five_cases_passed=True, plain_before_hook_plain_after_exact=True,
                                   audited_files_unchanged=integrity, device_idle=True),
                   scripts_sha256=scripts)
    (evidence / 'selection.json').write_text(json.dumps(payload, indent=2) + '\n')


if __name__ == '__main__':
    main()
