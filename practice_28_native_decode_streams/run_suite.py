"""Run disposable children; preserve evidence and verify native recovery."""
import argparse
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

from common import save, sha256

ROOT = Path(__file__).resolve().parent
FILES = [
    '/vllm-workspace/vllm/vllm/v1/worker/gpu_model_runner.py',
    '/vllm-workspace/vllm/vllm/v1/worker/workspace.py',
    '/vllm-workspace/vllm/vllm/config/vllm.py',
    '/vllm-workspace/vllm/vllm/model_executor/models/qwen2.py',
    '/vllm-workspace/vllm/vllm/model_executor/layers/rotary_embedding/__init__.py',
    '/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py',
    '/vllm-workspace/vllm-ascend/vllm_ascend/worker/worker.py',
    '/vllm-workspace/vllm-ascend/vllm_ascend/attention/attention_v1.py',
    '/vllm-workspace/vllm-ascend/vllm_ascend/compilation/acl_graph.py',
    '/vllm-workspace/vllm-ascend/vllm_ascend/ops/rotary_embedding.py',
    '/vllm-workspace/vllm-ascend/vllm_ascend/ops/triton/rope.py',
    '/vllm-workspace/vllm-ascend/vllm_ascend/sample/sampler.py',
    '/vllm-workspace/vllm-ascend/vllm_ascend/utils.py',
    '/data/huggingface_home/hub/Qwen2.5-0.5B-Instruct/config.json',
]


def device():
    return subprocess.check_output(['npu-smi', 'info'], text=True)


def require_idle():
    info = device()
    if 'No running processes found' not in info:
        raise RuntimeError('NPU occupied: leave other jobs untouched')
    return info


def launch(out, label, mode, stage, timeout=1800):
    destination = out / label
    command = [sys.executable, '-B', str(out / 'collector' / 'child.py'), '--mode', mode, '--stage', stage,
               '--output', str(destination)]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    if env.get('LD_AUDIT') or env.get('ASCEND_LAUNCH_BLOCKING') == '1':
        raise RuntimeError('incompatible external instrumentation environment')
    with (out / (label + '.log')).open('w') as log:
        proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        save(out / (label + '-launcher.json'), dict(pid=proc.pid, pgid=proc.pid, command=command))
        try:
            deadline = time.monotonic() + timeout
            failed_at = None
            while proc.poll() is None:
                try:
                    proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
                status_file = destination / 'status.json'
                if status_file.exists():
                    import json
                    try:
                        failed = json.loads(status_file.read_text()).get('status') == 'failed'
                    except json.JSONDecodeError:
                        failed = False
                    if failed and failed_at is None:
                        failed_at = time.monotonic()
                if failed_at is not None and time.monotonic() - failed_at > 5:
                    break  # Cleanup a failed child even if a native destructor hangs.
                if time.monotonic() > deadline:
                    raise subprocess.TimeoutExpired(command, timeout)
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
            save(out / (label + '-exit.json'), dict(returncode=proc.returncode, process_exited=True))
    return proc.returncode == 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--qualification-only', action='store_true')
    a = p.parse_args()
    out = a.output.resolve()
    before = require_idle()
    out.mkdir(parents=True, exist_ok=False)
    (out / 'collector').mkdir()
    for source in ROOT.glob('*.py'):
        shutil.copy2(source, out / 'collector' / source.name)
    (out / 'device_before.txt').write_text(before)
    sources = {str(f): sha256(f) for f in FILES}
    save(out / 'sources_before.json', sources)
    save(out / 'collector_hashes.json', {p.name: sha256(p) for p in ROOT.glob('*.py')})
    revisions = {repo: subprocess.check_output(['git', '-C', repo, 'rev-parse', 'HEAD'], text=True).strip()
                 for repo in ('/vllm-workspace/vllm', '/vllm-workspace/vllm-ascend')}
    save(out / 'revisions.json', revisions)
    overall = dict(status='running', stages=[])
    def stopped(signum, frame):
        raise InterruptedError('controller received signal ' + str(signum))
    signal.signal(signal.SIGTERM, stopped)
    save(out / 'status.json', overall)
    try:
        if not launch(out, 'native-before', 'eager', 'reference'):
            raise RuntimeError('unmodified native reference failed')
        for mode in ('eager', 'graph'):
            require_idle()
            passed = launch(out, mode + '-qualification', mode, 'qualify')
            overall['stages'].append(dict(mode=mode, stage='qualification', passed=passed))
            if not passed:
                raise RuntimeError(mode + ' qualification failed; no performance claims')
        if not a.qualification_only:
            for index, mode in enumerate(('eager', 'graph', 'graph', 'eager', 'eager', 'graph')):
                require_idle()
                if not launch(out, f'measure-{index}-{mode}', mode, 'measure'):
                    raise RuntimeError('measurement failed')
            for profile in ('plain', 'pipe'):
                for mode in ('eager', 'graph'):
                    require_idle()
                    if not launch(out, f'{mode}-{profile}', mode, profile):
                        raise RuntimeError('diagnostic failed')
        overall['status'] = 'passed'
    except BaseException as exc:
        overall.update(status='failed', reason=str(exc))
    finally:
        recovery = dict(sources_unchanged=False, device_has_no_running_processes=False,
                        original_request_verified=False, passed=False)
        try:
            after = {str(f): sha256(f) for f in FILES}
            save(out / 'sources_after.json', after)
            current = device()
            (out / 'device_after.txt').write_text(current)
            recovery.update(sources_unchanged=sources == after,
                            device_has_no_running_processes='No running processes found' in current)
            # No hardware reset. Retry native only after our processes exit.
            if recovery['device_has_no_running_processes'] and recovery['sources_unchanged']:
                if launch(out, 'native-after', 'eager', 'reference'):
                    import json
                    first = out / 'native-before/reference.json'
                    last = out / 'native-after/reference.json'
                    recovery['original_request_verified'] = first.exists() and json.loads(first.read_text()) == json.loads(last.read_text())
            final_sources = {str(f): sha256(f) for f in FILES}
            save(out / 'sources_final.json', final_sources)
            recovery['sources_unchanged'] = sources == final_sources
            final_device = device()
            (out / 'device_final.txt').write_text(final_device)
            recovery['device_has_no_running_processes'] = 'No running processes found' in final_device
            recovery['passed'] = all(value for key, value in recovery.items() if key != 'passed')
        except BaseException as exc:
            recovery.update(passed=False, error=str(exc))
        save(out / 'recovery.json', recovery)
        if not recovery['passed']:
            overall['status'] = 'failed'
        save(out / 'status.json', overall)
    return 0 if overall['status'] == 'passed' else 1


if __name__ == '__main__':
    sys.exit(main())
