"""Diagnostic-only scopes around original branch methods and event operations."""
from contextlib import contextmanager
import time


class Observer:
    def __init__(self,torch,layer):
        self.torch= torch;self.layer=layer;self.records=[];self.trial=None;self.restore=[]
        self.branch_outputs={}

    def tensor(self,t):
        return dict(ptr=str(t.data_ptr()),storage=str(t.untyped_storage().data_ptr()),
            offset=t.storage_offset(),shape=list(t.shape),stride=list(t.stride()),
            bytes=t.numel()*t.element_size(),dtype=str(t.dtype))

    @contextmanager
    def scope(self,kind,**fields):
        torch=self.torch;s=torch.npu.current_stream()
        record=dict(label='P21/%s/%05d/%s'%(self.trial,len(self.records),kind),
            trial=self.trial,kind=kind,stream_handle=str(s.npu_stream),
            logical_stream=s.stream_id,host_start_ns=time.monotonic_ns())
        record.update(fields)
        self.records.append(record)
        with torch.profiler.record_function(record['label']):
            yield record
        record['host_end_ns']=time.monotonic_ns()

    def patch(self,obj,name,new):
        old=getattr(obj,name);self.restore.append((obj,name,old));setattr(obj,name,new)
        return old

    def install(self):
        torch=self.torch
        for obj,name,kind in [(self.layer.gate,'forward','router'),
            (self.layer.experts,'forward_impl','routed'),
            (self.layer.experts,'_shared_experts_part1','shared_up'),
            (self.layer.experts,'_shared_experts_part2','shared_down')]:
            old=getattr(obj,name)
            def wrapped(*args,_old=old,_kind=kind,**kwargs):
                if self.trial is None:return _old(*args,**kwargs)
                tensors=[x for x in [*args,*kwargs.values()] if isinstance(x,torch.Tensor)]
                with self.scope(_kind,inputs=[self.tensor(t) for t in tensors]) as r:
                    result=_old(*args,**kwargs)
                    tensor=(result.routed_out if hasattr(result,'routed_out') else
                            result[0] if isinstance(result,tuple) else result)
                    r['output']=self.tensor(tensor)
                    if _kind in ('routed','shared_down'):self.branch_outputs[_kind]=tensor.data_ptr()
                    return result
            self.patch(obj,name,wrapped)
        for name in ('record','wait','synchronize'):
            old=getattr(torch.npu.Event,name)
            def event_api(event,*args,_name=name,_old=old,**kwargs):
                if self.trial is None:return _old(event,*args,**kwargs)
                stream=kwargs.get('stream',args[0] if args else None)
                stream=stream if stream is not None else torch.npu.current_stream()
                with self.scope('event_'+_name,event_object=id(event),
                    stream_handle=str(stream.npu_stream),logical_stream=stream.stream_id) as r:
                    result=_old(event,*args,**kwargs);r['event_handle']=str(event.npu_event)
                    return result
            self.patch(torch.npu.Event,name,event_api)
        # Aten add is intercepted only when its inputs are precisely the original
        # branch outputs. No numerical operation or synchronization is replaced.
        from torch.utils._python_dispatch import TorchDispatchMode
        observer=self
        class MergeMode(TorchDispatchMode):
            def __torch_dispatch__(self,func,types,args=(),kwargs=None):
                kwargs=kwargs or {}
                merge=(observer.trial is not None and func==torch.ops.aten.add.Tensor and
                    len(args)>=2 and all(isinstance(x,torch.Tensor) for x in args[:2]) and
                    len(observer.branch_outputs)==2 and
                    {x.data_ptr() for x in args[:2]}==set(observer.branch_outputs.values()))
                if merge:
                    with observer.scope('merge',inputs=[observer.tensor(t) for t in args[:2]]) as r:
                        result=func(*args,**kwargs);r['output']=observer.tensor(result);return result
                return func(*args,**kwargs)
        self.dispatch=MergeMode();self.dispatch.__enter__()

    def uninstall(self):
        self.dispatch.__exit__(None,None,None)
        for obj,name,old in reversed(self.restore):setattr(obj,name,old)
