"""Start the real service, observe KV pool initialization, then shut it down.

No inference request, graph comparison, allocator configuration change, or model
download. Native audit and Python observation apply only to this service tree.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import sysconfig
import time
import urllib.error
import urllib.request


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--port', type=int, default=8014)
    parser.add_argument('--model', default='/data/huggingface_home/hub/Qwen2.5-0.5B-Instruct')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = args.output.resolve()
    if not Path(args.model, 'config.json').is_file():
        parser.error('local model config missing')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', args.port))
    output.mkdir(parents=True, exist_ok=False)
    for name in ['events', 'native', 'instrumentation', 'sources']:
        (output/name).mkdir()
    for name in ['run_allocation.py', 'allocation_trace.py', 'sitecustomize.py', 'native_alloc_trace.c']:
        shutil.copyfile(root/name, output/'instrumentation'/name)
    include = Path(os.environ['ASCEND_HOME_PATH'])/'include'
    compile_command = ['gcc', '-shared', '-fPIC', '-O2', '-Wall', '-Wextra', '-Wl,-Bsymbolic',
                       '-I'+str(include), str(root/'native_alloc_trace.c'), '-ldl',
                       '-o', str(output/'native_alloc_trace.so')]
    build = subprocess.run(compile_command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    save(output/'native_build.json', dict(argv=compile_command, exit_code=build.returncode, output=build.stdout))
    build.check_returncode()
    subprocess.run([sys.executable, str(root.parent/'practice_03_ascend_start/collect_environment.py'),
                    '--model', args.model, '--output', str(output/'environment.json')], check=True)
    packages = Path(sysconfig.get_paths()['purelib'])
    sources = {
        'vllm': (Path('/vllm-workspace/vllm'), ['vllm/v1/core/kv_cache_utils.py', 'vllm/v1/core/block_pool.py',
                    'vllm/v1/kv_cache_interface.py', 'vllm/v1/worker/utils.py', 'vllm/utils/mem_utils.py']),
        'vllm_ascend': (Path('/vllm-workspace/vllm-ascend'), ['vllm_ascend/worker/worker.py',
                    'vllm_ascend/worker/model_runner_v1.py', 'vllm_ascend/platform.py',
                    'vllm_ascend/patch/worker/patch_qwen3_next_mtp.py']),
        'packages': (packages, ['torch_npu/npu/memory.py']),
        'cann_headers': (include, ['acl/acl_rt.h', 'acl/acl_base.h']),
    }
    manifest = {}
    for group, (base, names) in sources.items():
        for name in names:
            src = base/name
            dst = output/'sources'/group/name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            manifest[str(dst.relative_to(output))] = dict(original=str(src), sha256=hashlib.sha256(dst.read_bytes()).hexdigest())
    save(output/'source_manifest.json', manifest)
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(('P07_', 'P08_', 'P09_', 'P10_', 'P11_', 'P12_', 'P13_', 'P14_')):
            env.pop(key)
    if env.get('LD_AUDIT'):
        parser.error('another LD_AUDIT is active; refusing to replace it')
    # Extra static TLS space is needed by this container's jemalloc + audit namespace.
    # Preserve every other tunable and the existing CPU allocator/preload.
    tunables = [s for s in env.get('GLIBC_TUNABLES', '').split(':')
                if s and not s.startswith('glibc.rtld.optional_static_tls=')]
    tunables.append('glibc.rtld.optional_static_tls=65536')
    env.update(P14_OUTPUT=str(output), P14_NATIVE_DIR=str(output/'native'),
               LD_AUDIT=str(output/'native_alloc_trace.so'), GLIBC_TUNABLES=':'.join(tunables),
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', VLLM_WORKER_MULTIPROC_METHOD='spawn',
               PYTHONPATH=os.pathsep.join([str(root), env.get('PYTHONPATH', '')]))
    command = [sys.executable, '-m', 'vllm.entrypoints.cli.main', 'serve', args.model,
               '--served-model-name', 'qwen-allocation', '--host', '127.0.0.1', '--port', str(args.port),
               '--tensor-parallel-size', '1', '--dtype', 'bfloat16', '--max-model-len', '256',
               '--max-num-seqs', '1', '--max-num-batched-tokens', '256', '--block-size', '128',
               '--gpu-memory-utilization', '0.3', '--enforce-eager', '--no-enable-prefix-caching',
               '--no-enable-chunked-prefill', '--no-async-scheduling']
    save(output/'command.json', dict(argv=command, environment={k:env.get(k) for k in
        ['P14_OUTPUT', 'P14_NATIVE_DIR', 'LD_AUDIT', 'LD_PRELOAD', 'GLIBC_TUNABLES',
         'PYTHONPATH', 'PYTORCH_NPU_ALLOC_CONF', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE',
         'VLLM_WORKER_MULTIPROC_METHOD']}, inference_requests=0))
    with (output/'server.log').open('w') as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic()+600
            while True:
                if process.poll() is not None:
                    raise RuntimeError('service exited %s; see server.log' % process.returncode)
                try:
                    with urllib.request.urlopen('http://127.0.0.1:%s/health' % args.port, timeout=2) as response:
                        if response.status == 200:
                            save(output/'ready.json', dict(status=200, monotonic_ns=time.monotonic_ns(), inference_requests=0))
                            break
                except (urllib.error.URLError, TimeoutError):
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError('readiness deadline; see server.log')
                time.sleep(1)
            print('Service ready. KV initialization captured; no inference request sent.', flush=True)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            save(output/'shutdown.json', dict(exit_code=process.returncode))
    # Fingerprint the actual loaded provider libraries; do not copy large binaries.
    maps = (output/'process_maps.txt').read_text().splitlines()
    binaries = sorted({s.split()[-1] for s in maps if any(n in s for n in
                       ['/libtorch_npu.so', '/libascendcl.so', '/libacl_rt.so', '/libruntime.so'])})
    save(output/'provider_binaries.json', {p:dict(bytes=Path(p).stat().st_size,
         sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest()) for p in binaries})
    with (output/'device_after.txt').open('w') as stream:
        subprocess.run(['npu-smi', 'info'], stdout=stream, check=True)
    print('Saved:', output)


if __name__ == '__main__':
    main()
