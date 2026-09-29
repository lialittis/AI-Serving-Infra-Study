"""Observe cold Triton compile/load and measured host submissions, without waits.

Python metadata and existing frame locals only. No tensor values, event queries,
extra NPU copies, dispatch modes, or changes to the installed compiler/runtime.
"""
import hashlib
import os
from pathlib import Path
import sys
import threading
import trace_hooks as base

TARGETS=dict(base.TARGETS)
TARGETS.update({
 ('triton.compiler.compiler','compile'):'compile',
 ('triton.compiler.compiler','_init_handles'):'init_handles',
 ('triton.backends.ascend.driver','load_binary'):'load_binary',
 ('triton.backends.ascend.driver','__init__'):'launcher_init',
 ('triton.backends.ascend.driver','__call__'):'launch',
 # Triton loads this backend file under the dynamic module name `ascend`.
 ('ascend','load_binary'):'load_binary',
 ('ascend','__init__'):'launcher_init',
 ('ascend','__call__'):'launch',
 ('triton.runtime.jit','run'):'jit_run',
 ('vllm_ascend.ops.linear','unquantized_gemm'):'linear',
 ('vllm_ascend.device.device_op','reshape_and_cache'):'cache_write',
 ('vllm_ascend.attention.attention_v1','forward_fused_infer_attention'):'fia',
 ('vllm_ascend.worker.model_runner_v1','_prepare_inputs'):'prepare',
 ('vllm_ascend.worker.model_runner_v1','sample_tokens'):'sample',
 ('vllm.v1.utils','copy_to_gpu'):'buffer_copy',
 ('vllm.v1.worker.gpu_model_runner','_to_list'):'to_list',
 ('torch_npu.npu.streams','record'):'event_record',
 ('torch_npu.npu.streams','synchronize'):'synchronize',
 ('torch._ops','__call__'):'torch_api',
})
NAMES={name for _,name in TARGETS}
_codes={};_local=threading.local()


def describe(value):
    import torch
    if isinstance(value,torch.Tensor):
        return dict(kind='tensor',shape=list(value.shape),dtype=str(value.dtype),device=str(value.device),
                    data_ptr=value.data_ptr(),storage_ptr=value.untyped_storage().data_ptr(),
                    stride=list(value.stride()),storage_offset=value.storage_offset(),element_size=value.element_size())
    if isinstance(value,bytes):return dict(kind='bytes',size=len(value),sha256=hashlib.sha256(value).hexdigest())
    if isinstance(value,(tuple,list)):return [describe(x) for x in value]
    if isinstance(value,dict):return {str(k):describe(v) for k,v in value.items()}
    if value is None or isinstance(value,(str,int,float,bool)):return value
    return dict(kind=base.typename(value),contents='not inspected')


def emit(event,frame,**fields):
    base.emit(event,frame,request_id=getattr(_local,'rid',None),
              step=getattr(base._local,'step',None),measured=getattr(_local,'measured',False),**fields)


def start(frame,kind,**fields):
    _local.serial+=1
    label='P13/{}/{:05d}/{}'.format(os.getpid(),_local.serial,kind)
    parent=list(_local.open.values())[-1][1] if _local.open else None
    span=None
    if _local.measured:
        import torch
        span=torch.profiler.record_function(label);span.__enter__()
    _local.open[id(frame)]=(span,label,kind)
    emit('enter',frame,label=label,kind=kind,parent=parent,**fields)


def handle(kind,frame,event,result):
    if not hasattr(_local,'open'):
        _local.open={};_local.serial=0;_local.measured=False;_local.rid=None
    v=frame.f_locals;obj=v.get('self')
    if kind in base.TARGETS.values():base.handle(kind,frame,event,result)
    if kind=='runner' and event=='call':
        ids=list(v['scheduler_output'].num_scheduled_tokens)
        _local.rid=ids[0] if len(ids)==1 else None
        _local.measured=_local.rid is not None and 'practice13-profile' in _local.rid
    always=kind in ('compile','init_handles','load_binary','launcher_init')
    if kind=='launcher_init' and frame.f_code.co_qualname!='NPULauncher.__init__':return
    if kind=='init_handles' and event=='call' and obj.module is not None:return
    if event=='return':
        entry=_local.open.pop(id(frame),None)
        if not entry:return
        span,label,kind=entry;fields={}
        if kind=='compile' and result is not None:
            fields=dict(kernel_name=result.name,kernel_hash=result.hash,binary=describe(result.kernel),
                        ran_compiler_stages='stages' in v,metadata=describe(result.metadata._asdict()),
                        signature=describe(result.src.signature),constants=describe(result.src.constants),
                        cache_files=describe(v.get('metadata_group',{})))
        elif kind=='init_handles':
            fields=dict(module_handle=obj.module,function_handle=obj.function,kernel_hash=obj.hash)
        elif kind=='load_binary':fields=dict(returned_handles=describe(result))
        elif kind=='launcher_init':
            digest=hashlib.sha256(v['wrapper_src'].encode()).hexdigest()
            path=base._run_dir.parent/'launcher_sources'/ (digest+'.cpp')
            path.parent.mkdir(exist_ok=True);path.write_text(v['wrapper_src'])
            fields=dict(generated_launcher=str(path.relative_to(base._run_dir.parent)),sha256=digest,
                        compiled_launcher=v.get('so_launcher_path'))
        elif kind=='jit_run' and result is not None:
            fields=dict(kernel_name=result.name,kernel_hash=result.hash,grid=describe(v.get('grid')),
                        bound_arguments=describe(v.get('bound_args')),signature=describe(result.src.signature),
                        constants=describe(result.src.constants),function_handle=result.function,
                        stream=v.get('stream'))
        elif kind in ('linear','fia','torch_api','buffer_copy','forward'):
            fields=dict(returned=describe(result))
            if kind=='fia':
                fields['selected_parameters']={k:describe(v.get(k)) for k in ('query','key','value','block_size','block_table','actual_seq_lengths_kv','num_tokens')}
        elif kind=='to_list':fields=dict(token_ids=result)
        if span:span.__exit__(None,None,None)
        emit('exit',frame,label=label,kind=kind,**fields)
        return
    if not always and not _local.measured:return
    if kind=='compile':
        src=v['src'];start(frame,kind,source_name=getattr(src,'name',None),
                         options=describe(v.get('options')),source_signature=describe(getattr(src,'signature',None)))
    elif kind=='init_handles':start(frame,kind,kernel_name=obj.name,kernel_hash=obj.hash,binary=describe(obj.kernel))
    elif kind=='load_binary':start(frame,kind,kernel_name=v['name'],binary=describe(v['kernel']),shared=v['shared'],device=v['device'])
    elif kind=='launcher_init':start(frame,kind,metadata=describe(v['metadata']._asdict()))
    elif kind=='launch':
        args=v['args'];parent=frame.f_back
        caller=parent.f_locals if parent is not None else {}
        kernel=caller.get('kernel')
        start(frame,kind,grid=describe(args[:3]),runtime_stream=args[3],function_handle=args[4],
              packed_metadata=describe(args[5]),runtime_arguments=describe(args[9:]),
              named_arguments=describe(caller.get('bound_args')),
              signature=describe(kernel.src.signature) if kernel is not None else None,
              constants=describe(kernel.src.constants) if kernel is not None else None,
              abi_note='Python-to-generated-launcher boundary; opaque internal workspace pointers not read')
    elif kind=='jit_run':start(frame,kind,kernel_name=obj.__name__,warmup=v['warmup'],arguments=describe(v['args']),keyword_arguments=describe(v['kwargs']))
    elif kind in ('runner','prepare','sample'):start(frame,kind)
    elif kind=='forward':start(frame,kind,arguments={k:describe(v.get(k)) for k in ('input_ids','positions','inputs_embeds')})
    elif kind=='linear':start(frame,kind,arguments={k:describe(v.get(k)) for k in ('x','weight','bias')})
    elif kind=='attention':
        if not getattr(base._local,'active',False):return
        start(frame,kind,layer=v['layer'].layer_name,
              arguments={k:describe(v.get(k)) for k in ('query','key','value','output','kv_cache')})
    elif kind=='fia':
        md=v['attn_metadata']
        start(frame,kind,arguments={k:describe(v.get(k)) for k in ('query','key','value','output','kv_cache')},
              metadata=dict(num_heads=obj.num_heads,num_kv_heads=obj.num_kv_heads,scale=obj.scale,
                            causal=md.causal,sliding_window=obj.sliding_window,
                            seq_lens=list(md.seq_lens_list),actual_seq_lengths_q=list(md.actual_seq_lengths_q),
                            attn_mask=describe(md.attn_mask),state=str(md.attn_state)))
    elif kind=='cache_write':start(frame,kind,arguments={k:describe(v[k]) for k in ('key','value','key_cache','value_cache','slot_mapping') if k in v})
    elif kind=='buffer_copy':start(frame,kind,source_tensor=describe(obj.cpu),destination=describe(obj.gpu),rows=v.get('n'),non_blocking=True)
    elif kind=='to_list':start(frame,kind,source_tensor=describe(v['sampled_token_ids']),destination=describe(obj.sampled_token_ids_pinned_cpu),event_object_id=id(obj.transfer_event))
    elif kind in ('event_record','synchronize') and _local.open:
        start(frame,kind,object_type=base.typename(obj),object_id=id(obj))
    elif kind=='torch_api':
        # Record the actual kwargs at selected registered operator APIs. Other
        # CPU ops remain in the complete profiler inventory, without fake args.
        name=getattr(obj,'_qualified_op_name',None) or getattr(obj,'_name',None)
        if name in ('npu::npu_fused_infer_attention_score','atb::_npu_reshape_and_cache'):
            start(frame,kind,operator=name,args=describe(v.get('args')),kwargs=describe(v.get('kwargs')))


def profile(frame,event,result):
    if event not in ('call','return') or frame.f_code.co_name not in NAMES:return
    code=frame.f_code
    if code not in _codes:_codes[code]=TARGETS.get((frame.f_globals.get('__name__'),code.co_name))
    kind=_codes[code]
    if kind is not None:
        try:handle(kind,frame,event,result)
        except Exception as error:base.emit('trace_error',frame,kind=kind,error=repr(error))


def install():
    base._run_dir=Path(os.environ['P13_TRACE_DIR']);base._run_dir.mkdir(parents=True,exist_ok=True)
    base.emit('trace_installed',schema=1,extra_device_waits=False,device_values_read=False)
    sys.setprofile(profile);threading.setprofile(profile)
