"""Isolated vLLM eager/PIECEWISE 64-token controls and coarse diagnostic traces."""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT=Path(__file__).resolve().parent

def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['eager','graph'],required=True)
    p.add_argument('--profile',choices=['none','plain','pipe'],default='none')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',type=Path,default=Path('/data/huggingface_home/hub/Qwen2.5-0.5B-Instruct'))
    p.add_argument('--port',type=int,default=8026)
    p.add_argument('--requests',type=int,default=5)
    p.add_argument('--warmups',type=int,default=3)
    p.add_argument('--tokens',type=int,default=64)
    a=p.parse_args();out=a.output.resolve()
    if not 1<=a.requests<=20 or not 1<=a.warmups<=10 or not 4<=a.tokens<=128:p.error('invalid workload bounds')
    with socket.socket() as s:s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);s.bind(('127.0.0.1',a.port))
    before=subprocess.check_output(['npu-smi','info'],text=True)
    if 'No running processes found' not in before:raise RuntimeError('NPU occupied; wait for other experiments, never kill them')
    out.mkdir(parents=True,exist_ok=False);(out/'device_before.txt').write_text(before)
    snap=out/'instrumentation';snap.mkdir()
    for name in ('run.py','observer.py','sitecustomize.py'):
        shutil.copy2(ROOT/name,snap/name)
    sys.path.insert(0,str(ROOT.parent/'practice_13_operator_submission'))
    from collect_sources import collect_sources
    collect_sources(out)
    import torch_npu
    for module in ('vllm_ascend.profiler.torch_npu_profiler','vllm_ascend.compilation.acl_graph','torch_npu.npu.graphs','vllm.config.profiler'):
        source=Path(importlib.util.find_spec(module).origin);shutil.copy2(source,out/'sources'/(module+'.py'))
    subprocess.run([sys.executable,str(ROOT.parent/'practice_03_ascend_start/collect_environment.py'),'--model',str(a.model),'--output',str(out/'environment.json')],check=True)
    save(out/'model_hashes.json',{f.name:digest(f) for f in sorted(a.model.iterdir()) if f.is_file()})
    command=[sys.executable,'-m','vllm.entrypoints.cli.main','serve',str(a.model),'--served-model-name','p26',
             '--host','127.0.0.1','--port',str(a.port),'--tensor-parallel-size','1','--dtype','bfloat16',
             '--max-model-len','256','--max-num-seqs','1','--max-num-batched-tokens','256','--seed','123',
             '--gpu-memory-utilization','0.3','--block-size','128','--no-enable-prefix-caching',
             '--no-enable-chunked-prefill','--no-async-scheduling','--additional-config','{"enable_async_exponential":false}']
    if a.mode=='eager':command+=['--enforce-eager']
    else:command+=['--compilation-config',json.dumps(dict(mode=3,cudagraph_mode='PIECEWISE',cudagraph_capture_sizes=[1],custom_ops=['all']))]
    env={k:v for k,v in os.environ.items() if not re.match(r'P\d+_',k)}
    if env.get('LD_AUDIT') or env.get('ASCEND_LAUNCH_BLOCKING')=='1':raise RuntimeError('conflicting diagnostic environment')
    env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_WORKER_MULTIPROC_METHOD='spawn',TRITON_CACHE_DIR=str(out/'compiler_cache'))
    if a.profile!='none':
        command+=['--profiler-config',json.dumps(dict(profiler='torch',torch_profiler_dir=str(out/'profiler'),torch_profiler_with_stack=False,ignore_frontend=True))]
        env.update(P26_PROFILE=a.profile,P26_OUTPUT=str(out),PYTHONPATH=str(snap)+os.pathsep+env.get('PYTHONPATH',''))
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(str(a.model),local_files_only=True)
    prompt=tok.encode('请简要解释为什么天空是蓝色的。',add_special_tokens=False)
    body=dict(model='p26',prompt=prompt,max_tokens=a.tokens,temperature=0,ignore_eos=True,stream=False,return_tokens_as_token_ids=True,logprobs=1)
    save(out/'request.json',body)
    save(out/'command.json',dict(mode=a.mode,profile=a.profile,argv=command,requests=a.requests,warmups=a.warmups,tokens=a.tokens,date=datetime.datetime.now(datetime.timezone.utc).isoformat(),environment_overrides={k:env[k] for k in ('P26_PROFILE','P26_OUTPUT','PYTHONPATH','TRITON_CACHE_DIR','LD_PRELOAD','TASK_QUEUE_ENABLE') if k in env}))
    responses=[];controls=[]
    def post(endpoint,payload=None):
        req=urllib.request.Request('http://127.0.0.1:%d%s'%(a.port,endpoint),data=json.dumps(payload).encode() if payload is not None else b'',headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=600) as r:
            result=json.loads(r.read() or b'null')
            if endpoint!='/v1/completions':controls.append(dict(endpoint=endpoint,status=r.status,time_ns=time.time_ns()));save(out/'profile_control.json',controls)
            return result
    with (out/'server.log').open('w') as log:
        proc=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        save(out/'launcher.json',dict(pid=proc.pid,time_ns=time.time_ns()))
        try:
            deadline=time.monotonic()+900
            while True:
                if proc.poll() is not None:raise RuntimeError('server exited; see log')
                try:
                    with urllib.request.urlopen('http://127.0.0.1:%d/health'%a.port,timeout=2) as r:
                        if r.status==200:break
                except (urllib.error.URLError,TimeoutError):pass
                if time.monotonic()>deadline:raise TimeoutError('server startup timeout')
                time.sleep(1)
            for i in range(a.warmups):
                result=post('/v1/completions',dict(body,request_id='p26-warmup-%d'%i))
                if result['usage']['completion_tokens']!=a.tokens:raise ValueError('incomplete warmup')
                save(out/('warmup-%d.json'%i),result)
            if a.profile!='none':post('/start_profile')
            try:
                for i in range(a.requests):
                    start=time.monotonic_ns();wall=time.time_ns()
                    response=post('/v1/completions',dict(body,request_id='p26-measure-%d'%i))
                    finish=time.monotonic_ns()
                    if response['usage']['completion_tokens']!=a.tokens:raise ValueError('incomplete response')
                    responses.append(dict(index=i,start_ns=wall,end_ns=time.time_ns(),completed_ms=(finish-start)/1e6,response=response));save(out/'responses.json',responses)
            finally:
                if a.profile!='none':post('/stop_profile')
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            save(out/'shutdown.json',dict(returncode=proc.returncode,time_ns=time.time_ns()))
    (out/'device_after.txt').write_text(subprocess.check_output(['npu-smi','info'],text=True))
    save(out/'source_hashes.json',{str(f.relative_to(out)):digest(f) for directory in ('sources','instrumentation') for f in (out/directory).rglob('*') if f.is_file() and '__pycache__' not in f.parts})
    save(out/'complete.json',dict(mode=a.mode,profile=a.profile,requests=len(responses)))
    print('COMPLETE',out,flush=True)

if __name__=='__main__':main()
