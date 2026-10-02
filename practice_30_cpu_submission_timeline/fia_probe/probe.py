"""Isolated FIA matching P30 shapes; hooks and normal execution stay separate."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time


def save(path,data):path.write_text(json.dumps(data,indent=2)+'\n')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dtype',choices=['bf16','fp16'],default='bf16')
    p.add_argument('--length',type=int,default=42)
    p.add_argument('--hook',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    device=subprocess.check_output(['npu-smi','info'],text=True)
    (a.output/'device_before.txt').write_text(device)
    assert 'No running processes' in device
    import torch,torch_npu
    torch.set_num_threads(1);torch.manual_seed(123)
    dtype=torch.bfloat16 if a.dtype=='bf16' else torch.float16
    hook=ctypes.CDLL(None)
    if a.hook:
        hook.fia_scope.argtypes=[ctypes.c_uint];hook.fia_dump.argtypes=[ctypes.c_char_p]
    else:assert not hasattr(hook,'fia_scope')
    root=Path('/usr/local/Ascend/cann-9.0.0')
    config=root/'opp/built-in/op_impl/ai_core/tbe/kernel/config/ascend910b/ops_transformer/fused_infer_attention_score.json'
    tracked=[Path(torch_npu.__file__).parent/'lib/libtorch_npu.so',config,
             root/'include/acl/acl_rt.h',root/'include/aclnnop/aclnn_fused_infer_attention_score_v3.h']
    before={str(x):sha(x) for x in tracked}
    qcpu=torch.randn(1,14,64).to(dtype)
    kcpu=torch.randn(10,128,2,64).to(dtype)
    vcpu=torch.randn(10,128,2,64).to(dtype)
    # Map logical block 0 to physical B3. Only that block contains the valid sequence.
    kref=kcpu[3,:a.length].float().repeat_interleave(7,dim=1)
    vref=vcpu[3,:a.length].float().repeat_interleave(7,dim=1)
    scores=torch.einsum('thd,shd->hts',qcpu.float(),kref)*0.125
    expected=torch.einsum('hts,shd->thd',scores.softmax(-1),vref)
    q=qcpu.npu();k=kcpu.npu().view(10,128,128);v=vcpu.npu().view(10,128,128)
    table=torch.tensor([[3,0]],dtype=torch.int32).npu()
    mask=torch.triu(torch.ones((2048,2048),dtype=torch.int8),diagonal=1).npu()
    args=dict(query=q,key=k,value=v,atten_mask=mask,block_table=table,input_layout='TND',block_size=128,
              actual_seq_lengths=[1],actual_seq_lengths_kv=[a.length],num_key_value_heads=2,
              num_heads=14,scale=0.125,sparse_mode=3)
    torch.npu.synchronize()
    layout=lambda x:dict(shape=list(x.shape),stride=list(x.stride()),dtype=str(x.dtype),
                         device=str(x.device),data_ptr=x.data_ptr(),storage_ptr=x.untyped_storage().data_ptr(),offset=x.storage_offset())
    save(a.output/'parameters.json',{key:layout(val) if isinstance(val,torch.Tensor) else val for key,val in args.items()})
    rows=[];outputs=[]
    try:
        for i in range(2):
            if a.hook:hook.fia_scope(i+1)
            start=time.monotonic_ns()
            y,lse=torch_npu.npu_fused_infer_attention_score(**args)
            returned=time.monotonic_ns();torch.npu.synchronize();joined=time.monotonic_ns()
            if a.hook:hook.fia_scope(0)
            actual=y.cpu().view(1,14,64).float()
            error=float((actual-expected).abs().max())
            atol=0.005 if a.dtype=='bf16' else 0.001
            assert torch.allclose(actual,expected,atol=atol,rtol=0.01),(error,a.dtype)
            outputs.append(actual.tolist())
            rows.append(dict(scope=i+1,begin_ns=start,return_ns=returned,joined_ns=joined,
                             max_abs_error=error,atol=atol,rtol=0.01,passed=True,output=layout(y)))
        assert outputs[0]==outputs[1]
    finally:
        if a.hook:
            hook.fia_scope(0);hook.fia_dump(os.fsencode(a.output/'api_calls.json'))
    after={str(x):sha(x) for x in tracked};assert before==after
    save(a.output/'results.json',dict(status='passed',rows=rows,output=outputs[0],
         main_tid=threading.get_native_id(),parameters_match='P30 shape/dtype/layout for baseline; synthetic values, block table B3, independent eager call; not original model invocation',
         torch=torch.__version__,torch_npu=torch_npu.__version__,npu_git=torch_npu.version.git_version,
         environment={k:os.getenv(k) for k in ['LD_PRELOAD','TASK_QUEUE_ENABLE','ASCEND_LAUNCH_BLOCKING']},
         integrity=dict(before=before,after=after,unchanged=True)))


if __name__=='__main__':main()
