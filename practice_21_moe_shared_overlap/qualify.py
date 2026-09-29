import argparse,json,traceback
from pathlib import Path
from model import setup,cleanup
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
try:
    torch,torch_npu,config,ctx,layer,weights=setup(a.output)
    from vllm_ascend.ascend_forward_context import set_ascend_forward_context
    print(type(layer),type(layer.experts),[(n,list(p.shape)) for n,p in layer.named_parameters()],flush=True)
    results=[]
    with torch.inference_mode():
        for n in (1,32,256):
            x=torch.randn((n,1024),dtype=torch.bfloat16,device='npu')
            outputs=[]
            for enabled in (False,True):
                layer.experts.multistream_overlap_shared_expert=enabled
                with set_ascend_forward_context(None,config,num_tokens=n):
                    y=layer(x)
                torch.npu.synchronize();outputs.append(y.cpu().float())
                print(n,enabled,y.shape,torch.isfinite(y).all().item(),flush=True)
            results.append(dict(tokens=n,max_abs=(outputs[0]-outputs[1]).abs().max().item(),equal=torch.equal(*outputs)))
    (a.output/'qualification.json').write_text(json.dumps({'status':'passed','checks':results},indent=2)+'\n')
    cleanup(ctx)
except Exception:
    (a.output/'failure.txt').write_text(traceback.format_exc());raise
