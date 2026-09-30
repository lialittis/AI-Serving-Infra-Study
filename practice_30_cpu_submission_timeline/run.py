"""Freeze inputs/code; run reference, one diagnostic, then fresh native recovery."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parent
sys.path.insert(0,str(REPO/'practice_28_native_decode_streams'))
from common import save,sha256
from run_suite import FILES,require_idle
sys.path.insert(0,str(REPO/'practice_29_prequeued_streams'))
from run import launch

EXTRA=[
 '/vllm-workspace/vllm/vllm/v1/engine/core.py',
 '/vllm-workspace/vllm/vllm/v1/engine/core_client.py',
 '/vllm-workspace/vllm/vllm/v1/engine/llm_engine.py',
 '/vllm-workspace/vllm/vllm/v1/core/sched/scheduler.py',
 '/vllm-workspace/vllm/vllm/v1/executor/uniproc_executor.py',
 '/vllm-workspace/vllm/vllm/v1/outputs.py',
 '/vllm-workspace/vllm/vllm/v1/engine/output_processor.py',
 '/vllm-workspace/vllm/vllm/entrypoints/llm.py',
 '/vllm-workspace/vllm-ascend/vllm_ascend/ops/linear.py',
 '/vllm-workspace/vllm-ascend/vllm_ascend/device/device_op.py',
 '/vllm-workspace/vllm-ascend/vllm_ascend/patch/platform/patch_balance_schedule.py',
 '/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/npu/streams.py',
 '/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/npu/graphs.py',
]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode', choices=['eager','graph'], default='eager')
    a=p.parse_args();out=a.output.resolve()
    if os.environ.get('LD_AUDIT') or os.environ.get('ASCEND_LAUNCH_BLOCKING')=='1':
        raise RuntimeError('conflicting instrumentation environment')
    device=require_idle()
    out.mkdir(parents=True,exist_ok=False)
    (out/'device_before.txt').write_text(device)
    collector=out/'collector';collector.mkdir()
    for file in ROOT.glob('*.py'):shutil.copy2(file,collector/file.name)
    for name in ('native.py','common.py'):
        shutil.copy2(REPO/'practice_28_native_decode_streams'/name,collector/name)
    save(out/'collector_hashes.json',{f.name:sha256(f) for f in collector.glob('*.py')})
    files=sorted(set(FILES+EXTRA));before={f:sha256(f) for f in files}
    save(out/'sources_before.json',before)
    manifest={}
    for index,file in enumerate(files):
        rel=Path('sources')/(f'{index:02d}-'+Path(file).name)
        (out/rel).parent.mkdir(exist_ok=True)
        shutil.copy2(file,out/rel)
        manifest[file]=dict(path=str(rel),sha256=before[file])
    save(out/'source_manifest.json',manifest)
    save(out/'revisions.json',{repo:subprocess.check_output(['git','-C',repo,'rev-parse','HEAD'],text=True).strip()
         for repo in ('/vllm-workspace/vllm','/vllm-workspace/vllm-ascend')})
    (out/'cann_version.txt').write_text(Path('/usr/local/Ascend/cann-9.0.0/share/info/runtime/version.info').read_text())
    status=dict(status='running',stages=[],mode=a.mode)
    save(out/'status.json',status)
    def stage(name):
        launch(out,name,'child.py',('--stage',name,'--mode',a.mode),timeout=900 if a.mode=='graph' else 300)
        status['stages'].append(name);save(out/'status.json',status)
    try:
        stage('reference')
        stage('diagnostic')
        status['status']='passed'
    except BaseException as exc:
        status.update(status='failed',error=repr(exc))
        raise
    finally:
        try:
            (out/'device_after_diagnostic.txt').write_text(require_idle())
            stage('recovery')
            reference=json.loads((out/'reference/responses.json').read_text())[0]['output']
            checked=0
            for name in status['stages']:
                rows=json.loads((out/name/'responses.json').read_text())
                warmups=json.loads((out/name/'warmups.json').read_text())
                for value in warmups+[r['output'] for r in rows]:
                    assert value==reference,'exact output/logprob mismatch: '+name
                    checked+=1
            after={f:sha256(f) for f in files};save(out/'sources_after.json',after)
            assert after==before,'installed source changed'
            (out/'device_final.txt').write_text(require_idle())
            status['recovery']=dict(exact_responses_including_warmups=checked,
                source_files_unchanged=len(files),device_idle=True,native_inference_restored=True)
        except BaseException as exc:
            status.update(status='failed',recovery_error=repr(exc))
            raise
        finally:
            save(out/'status.json',status)


if __name__=='__main__':main()
