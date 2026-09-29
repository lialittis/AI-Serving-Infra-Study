"""Coarse runner/sample and replay scopes; no sys.setprofile or device waits.

Capture dumps run at initialization. JSON records are buffered until stop_profile.
Only profiler construction changes for PipeUtilization; model execution is intact.
"""
import functools
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import sys
import threading
import time
import weakref

ROOT=Path(os.environ['P26_OUTPUT'])
records=[]
local=threading.local()
graphs=weakref.WeakKeyDictionary()
serial=0

def emit(kind,**values):
    records.append(dict(kind=kind,pid=os.getpid(),tid=threading.get_native_id(),time_ns=time.time_ns(),**values))

def flush():
    (ROOT/('observer-%d.json'%os.getpid())).write_text(json.dumps(records,ensure_ascii=False)+'\n')

def graph_hooks():
    import torch_npu
    cls=torch_npu.npu.NPUGraph
    if getattr(cls,'_p26',False):return
    cls._p26=True
    capture=cls.capture_end;replay=cls.replay
    @functools.wraps(capture)
    def captured(self,*a,**kw):
        global serial
        result=capture(self,*a,**kw);serial+=1
        uid='%d-g%d'%(os.getpid(),serial);graphs[self]=uid
        dest=ROOT/'graph_dumps';dest.mkdir(exist_ok=True)
        path=dest/(uid+'.json');self.debug_dump(str(path))
        emit('capture',uid=uid,path=str(path.relative_to(ROOT)))
        return result
    @functools.wraps(replay)
    def replayed(self,*a,**kw):
        step=getattr(local,'step',None)
        if step is None:return replay(self,*a,**kw)
        import torch
        uid=graphs[self];label='P26/replay/'+step+'/'+uid
        with torch.profiler.record_function(label):result=replay(self,*a,**kw)
        emit('replay',label=label,step=step,uid=uid)
        return result
    cls.capture_end=captured;cls.replay=replayed

def patch_runner(module):
    import torch
    graph_hooks()
    cls=module.NPUModelRunner
    execute=cls.execute_model;sample=cls.sample_tokens
    counts={}
    @functools.wraps(execute)
    def run(self,scheduler_output,*a,**kw):
        scheduled=dict(scheduler_output.num_scheduled_tokens)
        ids=list(scheduled)
        measured=len(ids)==1 and 'p26-measure' in ids[0]
        if not measured:
            local.step=None
            return execute(self,scheduler_output,*a,**kw)
        rid=ids[0];index=counts.get(rid,0);counts[rid]=index+1
        step=rid+'/step-%02d'%index;local.step=step
        label='P26/execute/'+step
        with torch.profiler.record_function(label):result=execute(self,scheduler_output,*a,**kw)
        emit('execute',step=step,label=label,index=index,request_id=rid,scheduled=scheduled)
        return result
    @functools.wraps(sample)
    def sampled(self,*a,**kw):
        step=getattr(local,'step',None)
        if step is None:return sample(self,*a,**kw)
        label='P26/sample/'+step
        with torch.profiler.record_function(label):result=sample(self,*a,**kw)
        emit('sample',step=step,label=label)
        local.step=None
        return result
    cls.execute_model=run;cls.sample_tokens=sampled
    emit('runner_installed')

def patch_profiler(module):
    import torch_npu
    cls=module.TorchNPUProfilerWrapper
    # Preserve all installed _create_profiler options except the metric selector.
    if os.environ['P26_PROFILE']=='pipe':
        original=cls._create_profiler
        def create(config,trace_name):
            factory=torch_npu.profiler._ExperimentalConfig
            def metric_config(*a,**kw):
                kw['aic_metrics']=torch_npu.profiler.AiCMetrics.PipeUtilization
                return factory(*a,**kw)
            torch_npu.profiler._ExperimentalConfig=metric_config
            try:return original(config,trace_name)
            finally:torch_npu.profiler._ExperimentalConfig=factory
        cls._create_profiler=staticmethod(create)
    stop=cls._stop
    @functools.wraps(stop)
    def stopped(self,*a,**kw):
        try:return stop(self,*a,**kw)
        finally:flush()
    cls._stop=stopped
    emit('profiler_installed',metric=os.environ['P26_PROFILE'])

PATCHES={'vllm_ascend.worker.model_runner_v1':patch_runner,
         'vllm_ascend.profiler.torch_npu_profiler':patch_profiler}
class Loader(importlib.abc.Loader):
    def __init__(self,base,callback):self.base=base;self.callback=callback
    def create_module(self,spec):return self.base.create_module(spec)
    def exec_module(self,module):self.base.exec_module(module);self.callback(module)
class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname not in PATCHES:return None
        spec=importlib.machinery.PathFinder.find_spec(fullname,path)
        if spec and spec.loader:spec.loader=Loader(spec.loader,PATCHES[fullname])
        return spec

def install():sys.meta_path.insert(0,Finder())
