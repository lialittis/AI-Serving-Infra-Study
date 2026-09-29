"""Profiler control without per-method observers or TorchDispatchMode."""
import argparse,hashlib,json,subprocess
from pathlib import Path
from model import setup,cleanup


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    torch,torch_npu,config,ctx,layer,weights=setup(out)
    from vllm_ascend.ascend_forward_context import set_ascend_forward_context
    g=torch.Generator(device='cpu').manual_seed(8721)
    inputs={}
    for n in (1,32,256,1024):
        xs=[torch.randn((n,1024),generator=g,dtype=torch.float32).bfloat16() for _ in range(3)]
        inputs[n]=xs[0].npu()
    def call(x):
        with set_ascend_forward_context(None,config,num_tokens=x.shape[0]):return layer(x)
    records=[];golden={}
    with torch.inference_mode():
        for n,x in inputs.items():
            for mode in ('serial','parallel'):
                layer.experts.multistream_overlap_shared_expert=(mode=='parallel')
                for _ in range(20):y=call(x)
                torch.npu.synchronize()
                if mode=='serial':golden[n]=y.cpu()
                else:assert torch.equal(y.cpu(),golden[n])
        cfg=torch_npu.profiler._ExperimentalConfig(profiler_level=torch_npu.profiler.ProfilerLevel.Level1)
        with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],
            schedule=torch_npu.profiler.schedule(wait=0,warmup=1,active=1,repeat=1),record_shapes=True,
            experimental_config=cfg,on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(out/'profiler'))) as prof:
            prof.step()
            for n,x in inputs.items():
                for r in range(3):
                    for mode in (['serial','parallel'] if r%2==0 else ['parallel','serial']):
                        torch.npu.synchronize();layer.experts.multistream_overlap_shared_expert=(mode=='parallel')
                        label=f'P21Light/n{n}-r{r}-{mode}'
                        with torch.profiler.record_function(label):
                            y=call(x*1.0)
                            torch.npu.current_stream().record_event().synchronize()
                        assert torch.equal(y.cpu(),golden[n])
                        records.append(dict(label=label,tokens=n,repeat=r,mode=mode,correct=True))
            prof.step()
        cleanup(ctx)
    meta=dict(status='passed',trials=records,observer_installed=False,torch_dispatch_mode=False,
        weights={n:hashlib.sha256(w.view(torch.uint8).numpy().tobytes()).hexdigest() for n,w in weights.items()})
    (out/'run.json').write_text(json.dumps(meta,indent=2)+'\n')
    (out/'sources').mkdir()
    for name in ('capture_light.py','model.py'):
        (out/'sources'/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    print('COMPLETED',out,flush=True)


if __name__=='__main__':main()
