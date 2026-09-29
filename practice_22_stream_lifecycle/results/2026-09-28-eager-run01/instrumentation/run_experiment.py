"""Capture process startup, graph stream resources and two measured requests."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import urllib.error


ROOT=Path(__file__).resolve().parent


def save(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',choices=['eager','graph','sampling-off','sampling-on'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',type=Path,default=Path('/data/huggingface_home/hub/Qwen2.5-0.5B-Instruct'))
    p.add_argument('--port',type=int,default=8022)
    p.add_argument('--baseline',action='store_true',help='Disable heavy observers/profiler; same model/request configuration')
    a=p.parse_args();out=a.output.resolve()
    if not (a.model/'config.json').is_file():p.error('model config missing')
    with socket.socket() as sock:sock.bind(('127.0.0.1',a.port))
    out.mkdir(parents=True,exist_ok=False)
    for name in ('native','stream_events','events','sources','instrumentation'): (out/name).mkdir()
    snap=out/'instrumentation'
    for path in ROOT.glob('*'):
        if path.suffix in ('.py','.c'):shutil.copy2(path,snap/path.name)
    for directory,names in [('practice_15_kernel_execution_graph',['graph_trace.py']),
                            ('practice_13_operator_submission',['submission_trace.py','collect_sources.py']),
                            ('practice_07_real_request_trace',['trace_hooks.py']),
                            ('practice_17_vllm_multistream',['multistream_trace.py'])]:
        for name in names:shutil.copy2(ROOT.parent/directory/name,snap/name)
    sys.path.insert(0,str(snap))
    from collect_sources import collect_sources
    collect_sources(out)
    # Snapshot actual installed sources without importing model code here.
    import torch_npu
    modules=['torch_npu.npu','torch_npu.npu.graphs','vllm_ascend.utils','vllm_ascend.ascend_config','vllm_ascend.sample.sampler','vllm_ascend.compilation.acl_graph','vllm.compilation.piecewise_backend']
    for module in modules:
        source=Path(importlib.util.find_spec(module).origin)
        shutil.copy2(source,out/'sources'/(module+'.py'))
    cann=Path(os.environ.get('ASCEND_HOME_PATH','/usr/local/Ascend/cann-9.0.0'))
    header=cann/'include/acl/acl_rt.h';shutil.copy2(header,out/'sources/acl_rt.h')
    from build_native import build
    if not a.baseline:build(out/'native_build',cann)
    subprocess.run([sys.executable,str(ROOT.parent/'practice_03_ascend_start/collect_environment.py'),'--model',str(a.model),'--output',str(out/'environment.json')],check=True)
    def status(name):
        result=subprocess.run(['npu-smi','info'],capture_output=True,text=True,check=True)
        (out/name).write_text(result.stdout)
    status('device_before.txt')
    sampling=a.case.startswith('sampling');batch=32 if sampling else 1
    command=[sys.executable,'-m','vllm.entrypoints.cli.main','serve',str(a.model),'--served-model-name','p22',
             '--host','127.0.0.1','--port',str(a.port),'--tensor-parallel-size','1','--dtype','bfloat16',
             '--max-model-len','256','--max-num-seqs',str(batch),'--max-num-batched-tokens','1024' if sampling else '256',
             '--seed','123','--gpu-memory-utilization','0.3','--block-size','128',
             '--no-enable-prefix-caching','--no-enable-chunked-prefill','--no-async-scheduling',
             '--additional-config',json.dumps({'enable_async_exponential':a.case=='sampling-on'})]
    if a.case=='graph':command+=['--compilation-config',json.dumps(dict(mode=3,cudagraph_mode='PIECEWISE',cudagraph_capture_sizes=[1],custom_ops=['all']))]
    else:command+=['--enforce-eager']
    if not a.baseline:command+=['--profiler-config',json.dumps(dict(profiler='torch',torch_profiler_dir=str(out/'profiler'),torch_profiler_with_stack=False,ignore_frontend=True))]
    env=os.environ.copy()
    for key in list(env):
        if key.startswith(tuple('P%02d_'%i for i in range(1,23))):env.pop(key,None)
    if env.get('LD_AUDIT'):raise RuntimeError('another LD_AUDIT is active')
    tunables=[s for s in env.get('GLIBC_TUNABLES','').split(':') if s and not s.startswith('glibc.rtld.optional_static_tls=')]
    tunables.append('glibc.rtld.optional_static_tls=65536')
    env['GLIBC_TUNABLES']=':'.join(tunables)
    env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_WORKER_MULTIPROC_METHOD='spawn',TRITON_CACHE_DIR=str(out/'compiler_cache'))
    if not a.baseline:
        env.update(P22_CASE=a.case,P22_TRACE_DIR=str(out/'stream_events'),P22_NATIVE_DIR=str(out/'native'),
                   P13_TRACE_DIR=str(out/'events'),P17_TRACE_DIR=str(out/'events'),P15_MODE_TRACE='1',
                   LD_AUDIT=str(out/'native_build/native_stream_trace.so'),
                   PYTHONPATH=str(snap)+os.pathsep+env.get('PYTHONPATH',''))
    save(out/'command.json',dict(case=a.case,baseline=a.baseline,argv=command,environment_overrides={k:v for k,v in env.items() if k.startswith('P22_') or k in ('LD_AUDIT','LD_PRELOAD','GLIBC_TUNABLES','PYTHONPATH','TRITON_CACHE_DIR')}))
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(str(a.model),local_files_only=True)
    prompt=tok.encode('请简要解释为什么天空是蓝色的。',add_special_tokens=False)
    body=dict(model='p22',prompt=[prompt]*batch if sampling else prompt,max_tokens=4,temperature=.8 if sampling else 0,
              ignore_eos=True,stream=False,return_tokens_as_token_ids=True,logprobs=1)
    if sampling:body['top_p']=.9
    save(out/'request.json',body)
    save(out/'model_fingerprint.json',{str(f.name):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(a.model.iterdir()) if f.is_file()})
    started=time.monotonic_ns();responses=[]
    with (out/'server.log').open('w') as log:
        process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        save(out/'launcher.json',dict(pid=process.pid,start_monotonic_ns=started))
        def post(endpoint,payload=None):
            request=urllib.request.Request('http://127.0.0.1:%d%s'%(a.port,endpoint),data=json.dumps(payload).encode() if payload is not None else b'',headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(request,timeout=300) as r:return json.loads(r.read() or b'null')
        try:
            deadline=time.monotonic()+900
            while True:
                if process.poll() is not None:raise RuntimeError('service exited; see server.log')
                try:
                    with urllib.request.urlopen('http://127.0.0.1:%d/health'%a.port,timeout=2) as r:
                        if r.status==200:break
                except (urllib.error.URLError,TimeoutError):pass
                if time.monotonic()>deadline:raise TimeoutError('service initialization exceeded 900 seconds')
                time.sleep(1)
            save(out/'ready.json',dict(monotonic_ns=time.monotonic_ns(),time_ns=time.time_ns()))
            # Snapshot loaded mappings while worker processes are alive.
            for proc in Path('/proc').iterdir():
                if proc.name.isdigit():
                    try:
                        if os.getpgid(int(proc.name))==process.pid:
                            (out/('maps-'+proc.name+'.txt')).write_text((proc/'maps').read_text())
                    except (OSError,ProcessLookupError):pass
            warm=post('/v1/completions',dict(body,request_id='p22-warmup'))
            save(out/'warmup_response.json',warm)
            if not a.baseline:post('/start_profile')
            try:
                for i in range(2):
                    rid=('p17-profile' if sampling else 'practice13-profile')+'-p22-'+str(i)
                    before=time.time_ns();mono=time.monotonic_ns()
                    response=post('/v1/completions',dict(body,request_id=rid))
                    if response['usage']['completion_tokens']!=batch*4:raise ValueError('incomplete model output')
                    responses.append(dict(request_id=rid,start_ns=before,end_ns=time.time_ns(),start_monotonic_ns=mono,end_monotonic_ns=time.monotonic_ns(),response=response))
                    save(out/'responses.json',responses)
            finally:
                if not a.baseline:post('/stop_profile')
        finally:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            save(out/'shutdown.json',dict(returncode=process.returncode,monotonic_ns=time.monotonic_ns()))
    status('device_after.txt')
    save(out/'source_hashes.json',{str(f.relative_to(out)):hashlib.sha256(f.read_bytes()).hexdigest() for folder in ('sources','instrumentation','native_build') for f in (out/folder).rglob('*') if f.is_file() and '__pycache__' not in f.parts})
    save(out/'complete.json',dict(case=a.case,baseline=a.baseline,requests=2,batch=batch))
    print('CAPTURED',out,flush=True)


if __name__=='__main__':main()
