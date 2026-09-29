"""Observe stream wrapper acquisition, context switches, graph capture and reuse.

Uses existing frame arguments and returned objects. Does not call current_stream
or global_stream to discover resources, and never reads device tensor contents.
Graph debug export runs after capture_end, outside the measured request window.
"""
import json
import ctypes
import os
from pathlib import Path
import sys
import threading
import time

_local=threading.local();_serial=0;_delegate=None
_dir=None;_query=None


def emit(kind, **fields):
    fields.update(kind=kind,pid=os.getpid(),tid=threading.get_native_id(),monotonic_ns=time.monotonic_ns(),time_ns=time.time_ns())
    with (_dir/('python-%d-%d.jsonl'%(os.getpid(),threading.get_native_id()))).open('a') as f:
        f.write(json.dumps(fields,ensure_ascii=False)+'\n')


def stack(frame):
    out=[]
    for _ in range(12):
        if frame is None:break
        out.append(dict(file=frame.f_code.co_filename,line=frame.f_lineno,function=frame.f_code.co_qualname))
        frame=frame.f_back
    return out


def describe(obj):
    global _query
    if obj is None or isinstance(obj,(str,int,float,bool)):return obj
    if isinstance(obj,dict):return {str(k):describe(v) for k,v in obj.items()}
    typ=type(obj).__module__+'.'+type(obj).__qualname__
    info=dict(object_id=str(id(obj)),type=typ)
    if type(obj).__module__=='torch_npu.npu.streams':
        if 'Stream' in type(obj).__name__:
            info.update(handle=str(obj.npu_stream),python_stream_id=str(obj.stream_id),device=str(obj.device))
            if _query is None:
                _query=ctypes.CDLL('libascendcl.so').aclrtStreamGetId
                _query.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_int32)];_query.restype=ctypes.c_int
            sid=ctypes.c_int32(-1)
            rc=_query(ctypes.c_void_p(obj.npu_stream),ctypes.byref(sid))
            info.update(runtime_stream_id=sid.value,stream_id_query_result=rc)
        elif 'Event' in type(obj).__name__:info['handle']=str(obj.npu_event)
    return info


def profile(frame,event,result):
    if _delegate is not None:_delegate(frame,event,result)
    if event not in ('call','return'):return
    mod=frame.f_globals.get('__name__','');name=frame.f_code.co_name
    qual=frame.f_code.co_qualname
    wanted=(mod=='torch_npu.npu.streams' and name in ('__new__','record','wait','wait_stream','wait_event','synchronize','query'))
    wanted|=(mod=='torch_npu.npu' and (name in ('set_stream','_set_stream_by_id') or qual.startswith('StreamContext.')))
    wanted|=(mod=='torch_npu.npu.graphs' and name in ('capture_begin','capture_end','replay','reset','__new__'))
    wanted|=(mod=='vllm_ascend.utils' and name in ('global_stream','current_stream','prefetch_stream','shared_experts_calculation_stream'))
    wanted|=(mod=='vllm_ascend.compilation.acl_graph' and name=='__call__')
    if not wanted:return
    try:
        if not hasattr(_local,'open'):_local.open={};_local.seq=0
        v=frame.f_locals
        if event=='call':
            _local.seq+=1;ident='%d:%d:%d'%(os.getpid(),threading.get_native_id(),_local.seq)
            parent=next(reversed(_local.open.values()))['id'] if _local.open else None
            fields=dict(id=ident,module=mod,function=qual,parent=parent,stack=stack(frame),arguments={k:describe(v[k]) for k in ('self','stream','event','kwargs','device','priority') if k in v})
            if mod=='vllm_ascend.compilation.acl_graph':
                fields['partition']=getattr(getattr(v.get('self'),'runnable',None),'submod_name',None)
            _local.open[id(frame)]=fields;emit('call',**fields)
        else:
            entry=_local.open.pop(id(frame),None)
            if entry is None:return
            emit('return',id=entry['id'],returned=describe(result),self_after=describe(v.get('self')))
            if mod=='torch_npu.npu.graphs' and name=='capture_end':
                obj=v['self'];dest=_dir.parent/'graph_dumps';dest.mkdir(exist_ok=True)
                path=dest/('graph-'+entry['id'].replace(':','-')+'.json')
                try:
                    obj.debug_dump(str(path))
                    emit('graph_dump',call=entry['id'],graph_id=str(id(obj)),path=str(path.relative_to(_dir.parent)),files=[p.name for p in dest.glob(path.stem+'*')])
                except Exception as exc:emit('graph_dump_unavailable',call=entry['id'],error=repr(exc))
    except Exception as exc:emit('observer_error',module=mod,function=qual,error=repr(exc))


def install():
    global _dir,_delegate
    _dir=Path(os.environ['P22_TRACE_DIR']);_dir.mkdir(parents=True,exist_ok=True)
    if os.environ['P22_CASE'].startswith('sampling'):
        import multistream_trace as base
    else:
        import graph_trace as base
    base.install();_delegate=base.profile
    emit('installed',python=sys.executable,extra_device_waits=False,device_values_read=False)
    sys.setprofile(profile);threading.setprofile(profile)
