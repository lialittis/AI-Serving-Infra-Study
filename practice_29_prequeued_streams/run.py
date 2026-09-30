"""Bounded disposable workers, frozen collector, source audit and native recovery."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
P28 = ROOT.parent / 'practice_28_native_decode_streams'
sys.path.insert(0, str(P28))
from common import save, sha256
from run_suite import FILES, require_idle


def launch(out, label, script, extra=(), timeout=300):
    command = [sys.executable, '-B', str(out / 'collector' / script),
               '--output', str(out / label), *extra]
    with (out / f'{label}.log').open('w') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
        save(out / f'{label}-process.json', dict(command=command, pid=child.pid, pgid=child.pid))
        expired = False
        try:
            child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            expired = True
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=10)
            save(out / f'{label}-exit.json', dict(returncode=child.returncode, timeout=expired))
    if expired or child.returncode:
        raise RuntimeError(f'{label} failed; see its log (timeout={expired})')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gate-only', action='store_true')
    a = p.parse_args()
    if os.environ.get('LD_AUDIT') or os.environ.get('ASCEND_LAUNCH_BLOCKING') == '1':
        raise RuntimeError('external instrumentation changes the experiment')
    before = require_idle()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    collector = out / 'collector'
    collector.mkdir()
    for file in ROOT.glob('*.py'):
        shutil.copy2(file, collector / file.name)
    for name in ('native.py', 'common.py', 'experiment.py'):
        shutil.copy2(P28 / name, collector / name)
    shutil.copy2(P28 / 'child.py', collector / 'reference_child.py')
    save(out / 'collector_hashes.json', {f.name: sha256(f) for f in collector.glob('*.py')})
    (out / 'device_before.txt').write_text(before)
    library = Path('/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu')
    files = FILES + [str(library / 'npu/streams.py'),
                    '/usr/local/Ascend/ascend-toolkit/latest/include/acl/acl_rt.h']
    sources = {f: sha256(f) for f in files}
    save(out / 'sources_before.json', sources)
    save(out / 'revisions.json', {repo: subprocess.check_output(
        ['git', '-C', repo, 'rev-parse', 'HEAD'], text=True).strip()
        for repo in ('/vllm-workspace/vllm', '/vllm-workspace/vllm-ascend')})
    (out / 'cann_version.txt').write_text(Path('/usr/local/Ascend/cann-9.0.0/share/info/runtime/version.info').read_text())
    evidence = out / 'installed_api'
    evidence.mkdir()
    for file in files[-2:]:
        shutil.copy2(file, evidence / Path(file).name)
    status = dict(status='running', stages=[])
    save(out / 'status.json', status)
    try:
        launch(out, 'native-before', 'reference_child.py', ('--mode', 'eager', '--stage', 'reference'))
        status['stages'].append('native-before')
        launch(out, 'gate-probe', 'probe.py', timeout=60)
        status['stages'].append('gate-probe')
        if not a.gate_only:
            launch(out, 'model', 'model.py')
            status['stages'].append('model')
        status['status'] = 'passed'
    except BaseException as exc:
        status.update(status='failed', error=repr(exc))
        raise
    finally:
        # Recovery is attempted even if the experiment failed. No device reset.
        try:
            (out / 'device_after_children.txt').write_text(require_idle())
            launch(out, 'native-after', 'reference_child.py', ('--mode', 'eager', '--stage', 'reference'))
            b, c = [json.loads((out / x / 'reference.json').read_text()) for x in ('native-before', 'native-after')]
            assert b == c, 'native inference changed after experiment'
            after = {f: sha256(f) for f in files}
            save(out / 'sources_after.json', after)
            assert after == sources, 'audited installed source changed'
            (out / 'device_final.txt').write_text(require_idle())
            status['recovery'] = dict(native_tokens_equal=True, audited_files_unchanged=True, device_idle=True)
        except BaseException as exc:
            status.update(status='failed', recovery_error=repr(exc))
            raise
        finally:
            save(out / 'status.json', status)


if __name__ == '__main__':
    main()
