"""Correctness-only Ascend experiment. Intentionally synchronizes and reads back."""
import argparse
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from spec import make_cases,checks


def save(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    root=Path(__file__).resolve().parent;out=args.output;out.mkdir(parents=True,exist_ok=False)
    status=subprocess.check_output(['npu-smi','info'],text=True);(out/'device_before.txt').write_text(status)
    assert 'No running processes' in status,'Device busy; stop before NPU init'
    for name in ('probe.py','spec.py','preparation.json','resolved_config.json'):shutil.copy2(root/name,out/name)
    import torch
    import torch_npu
    torch.set_num_threads(1)
    paths=[Path(torch.__file__),Path(torch_npu.__file__),Path(torch_npu.__file__).parent/'lib/libtorch_npu.so']
    hashes={str(p):sha(p) for p in paths}
    stream=torch.npu.Stream();cases=make_cases(json.loads((root/'preparation.json').read_text()))
    def layout(t):return dict(ptr=t.data_ptr(),base=t.untyped_storage().data_ptr(),offset=t.storage_offset(),
        stride=list(t.stride()),shape=list(t.shape),itemsize=t.element_size(),storage_bytes=t.untyped_storage().nbytes())
    rows=[]
    with torch.npu.stream(stream):
        for c in cases:
            n=len(c['boundaries']);pad=c['pad'];step=c['stride']
            raw=[-777]*(pad+(n-1)*step+1+3)
            for i,v in enumerate(c['boundaries']):raw[pad+i*step]=v
            # Explicit wide->narrow path, including deliberately invalid cases.
            wide=torch.tensor(raw,dtype=torch.int64,device='npu')
            parent=wide.to(torch.int32);boundaries=parent[pad:pad+n*step:step]
            left=boundaries[1:];right=boundaries[:-1]
            before=parent.cpu().tolist();device_bounds=boundaries.cpu().tolist()
            y=right-left if c['operation']=='swap' else left-right
            stream.synchronize();computed=y.cpu().tolist();observed=list(computed)
            if c['mutation']=='reverse':observed=observed[::-1]
            if c['mutation']=='increment':observed[0]+=1
            if c['mutation']=='redistribute':observed[0]+=1;observed[1]-=1
            if c['mutation']=='bit':observed[0]^=1
            if c['mutation']=='merge':observed=[observed[0]+observed[1]]+observed[2:]
            # Injection affects only this isolated output tensor; never raw memory writes.
            if c['mutation']:y=torch.tensor(observed,dtype=torch.int32,device='npu')
            stream.synchronize();observed=y.cpu().tolist();after=parent.cpu().tolist()
            ls={k:layout(v) for k,v in [('parent',parent),('boundaries',boundaries),('left',left),('right',right),('output',y)]}
            ok=True
            for name,offset in [('boundaries',pad),('left',pad+step),('right',pad)]:
                l=ls[name];last=l['offset']+(l['shape'][0]-1)*step
                ok &= l['base']==ls['parent']['base'] and l['offset']==offset and l['stride']==[step]
                ok &= l['ptr']==l['base']+offset*4 and 0<=offset<=last and (last+1)*4<=l['storage_bytes']
            l=ls['output'];p=ls['parent']
            mem=dict(input_unchanged=before==after,layout=bool(ok),no_output_alias=
                l['base']+l['storage_bytes']<=p['base'] or l['base']>=p['base']+p['storage_bytes'])
            split={'status':'skipped','reason':'序列长度不在 0..4096 内；不分配巨大数据 tensor'}
            if 0<=c['seq']<=4096:
                data=torch.arange(c['seq'],dtype=torch.int32,device='npu').reshape(1,1,c['seq'],1)
                try:
                    chunks=torch.split(data,observed,dim=2);stream.synchronize()
                    split=dict(status='accepted',sizes=[x.shape[2] for x in chunks],
                        chunks=[x.flatten().cpu().tolist() for x in chunks])
                except (RuntimeError,ValueError) as e:
                    split=dict(status='rejected',error_type=type(e).__name__,error=str(e)[:3500]);stream.synchronize()
            row=dict(case=c,computed_before_injection=computed,observed=observed,device_boundaries=device_bounds,
                memory=mem,layout=ls,parent_before=before,parent_after=after,split=split,
                checks=checks(c,observed,device_bounds,mem))
            rows.append(row);save(out/'cases.json',rows)
        # Framework argument errors, observed separately from semantic value checks.
        def trial(name,fn):
            try:
                value=fn();stream.synchronize()
                return dict(id=name,status='accepted',value=value.cpu().tolist() if isinstance(value,torch.Tensor) else str(value))
            except (RuntimeError,ValueError,TypeError) as e:
                stream.synchronize();return dict(id=name,status='rejected',error_type=type(e).__name__,error=str(e)[:3500])
        a=torch.tensor([1,2,3],dtype=torch.int32,device='npu');b=torch.tensor([0,0,0],dtype=torch.int32,device='npu')
        framework=[trial('alpha_float',lambda:torch.sub(a,b,alpha=1.5)),
            trial('shape_mismatch',lambda:a-torch.tensor([0,0],dtype=torch.int32,device='npu')),
            trial('constructor_int32_overflow',lambda:torch.tensor([1<<31],dtype=torch.int32,device='npu')),
            trial('cast_int64_to_int32',lambda:torch.tensor([1<<31],dtype=torch.int64,device='npu').to(torch.int32))]
        save(out/'framework.json',framework)
        # Exercise the installed get_window_index without constructing a model or loading weights.
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel as V
        from types import SimpleNamespace
        cfg=json.loads((root/'resolved_config.json').read_text())['vision_config']
        dummy=SimpleNamespace(window_size=cfg['window_size'],spatial_merge_size=cfg['spatial_merge_size'],
            patch_size=cfg['patch_size'],spatial_merge_unit=cfg['spatial_merge_size']**2)
        model_source=Path(inspect.getfile(V));shutil.copy2(model_source,out/'installed_model.py')
        hf=[]
        for c in cases:
            if 'grid' not in c:continue
            grid=torch.tensor(c['grid'],dtype=torch.int64)
            if c['full']:
                cumulative=torch.nn.functional.pad(torch.repeat_interleave(grid[:,1]*grid[:,2],grid[:,0]).cumsum(0,dtype=torch.int32),(1,0))
                before_unique=cumulative.tolist()
            else:
                _,raw_bounds=V.get_window_index(dummy,grid)
                before_unique=[int(x) for x in raw_bounds]
                cumulative=torch.unique_consecutive(torch.tensor(raw_bounds,dtype=torch.int32))
            hf.append(dict(id=c['id'],raw_boundaries=before_unique,unique_boundaries=cumulative.tolist(),
                reference=c['reference'],exact=cumulative.tolist()==c['reference']))
        save(out/'model_reference.json',hf)
        # Wide intermediates: demonstrate where information can be lost before Sub.
        product=torch.tensor([46342],dtype=torch.int32,device='npu')
        addends=torch.tensor([(1<<31)-1,2],dtype=torch.int32,device='npu')
        save(out/'upstream.json',dict(product_reference=46342*46342,product_int32=(product*product).cpu().tolist(),
            cumulative_reference=[(1<<31)-1,(1<<31)+1],cumsum_int32=addends.cumsum(0,dtype=torch.int32).cpu().tolist(),
            cumsum_int64=addends.cumsum(0,dtype=torch.int64).cpu().tolist()))
        save(out/'environment.json',dict(torch=torch.__version__,torch_git=torch.version.git_version,
            torch_npu=torch_npu.__version__,torch_npu_git=torch_npu.version.git_version,python=os.sys.version,
            sha256_before=hashes,model_source_sha256=sha(model_source),vision_config=cfg,stream_handle=stream.npu_stream,
            controls={k:os.environ.get(k) for k in ('TASK_QUEUE_ENABLE','ASCEND_LAUNCH_BLOCKING')},
            dispatch={op:torch._C._dispatch_dump_table(op) for op in ('aten::sub.Tensor','aten::split_with_sizes')},
            mode='eager correctness only, one stream, no performance measurement'))
        stream.synchronize()
    after={str(p):sha(p) for p in paths};assert after==hashes
    assert all(r['exact'] for r in hf)
    assert all(all(r['checks'].values())==r['case']['expect_valid'] for r in rows)
    assert all(all(r['memory'].values()) for r in rows)
    save(out/'integrity.json',dict(unchanged=True,sha256_after=after))
    save(out/'completed.json',dict(status='passed',cases=len(rows)))
    manifest={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
    save(out/'checksums.json',manifest)
    print('Completed',len(rows),'correctness cases')


if __name__=='__main__':main()
