"""Installed Qwen2 MoE layer: numerical checks, unprofiled timings, native DAG evidence."""
import argparse,datetime,hashlib,inspect,json,os,platform,subprocess,time,traceback
from pathlib import Path
from model import setup,cleanup


def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def cpu_reference(torch,x,w,ids=None):
    """Independent dense per-expert calculation, with BF16 stage rounding."""
    import torch.nn.functional as F
    def linear(x,weight):return F.linear(x.float(),weight.float()).bfloat16()
    def swiglu(z):
        a,b=z.float().chunk(2,-1)
        return (F.silu(a)*b).bfloat16()
    router=linear(x,w['gate.weight']).float()
    probs=router.softmax(-1)
    if ids is None:_,ids=probs.topk(2,-1)
    scores=probs.gather(1,ids.long());scores=scores/scores.sum(-1,keepdim=True)
    routed=torch.zeros(x.shape,dtype=torch.float32)
    for expert in range(8):
        rows,slots=torch.where(ids==expert)
        if len(rows):
            z=linear(x[rows],w['experts.w13_weight'][expert])
            y=linear(swiglu(z),w['experts.w2_weight'][expert])
            routed.index_add_(0,rows,y.float()*scores[rows,slots,None])
    shared=linear(swiglu(linear(x,w['shared_expert.gate_up_proj.weight'])),w['shared_expert.down_proj.weight'])
    gate=linear(x,w['shared_expert_gate.weight']).float().sigmoid().bfloat16()
    shared=(shared.float()*gate.float()).bfloat16()
    return (routed.bfloat16().float()+shared.float()).bfloat16()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tokens',default='1,32,256,1024')
    p.add_argument('--rounds',type=int,default=12);p.add_argument('--iterations',type=int,default=20)
    p.add_argument('--warmup',type=int,default=20);p.add_argument('--profile-repeats',type=int,default=3)
    a=p.parse_args();tokens=[int(x) for x in a.tokens.split(',')]
    assert len(set(tokens))==len(tokens) and 1<=min(tokens)<=max(tokens)<=2048
    assert 1<=a.rounds<=30 and 1<=a.iterations<=100 and 1<=a.warmup<=100 and 1<=a.profile_repeats<=5
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    save(out/'plan.json',dict(tokens=tokens,rounds=a.rounds,iterations=a.iterations,warmup=a.warmup,
        profile_repeats=a.profile_repeats,dtype='bfloat16',cross_mode_atol=0,cross_mode_rtol=0,
        cpu_reference_atol=.003,cpu_reference_rtol=.02,routing_weight_rtol=1/128,
        routing_weight_atol=0,router_logits_rtol=1/128,router_logits_atol=1e-4,routing_topk='legal BF16 softmax top-k, permitting ties',scope='scaled real Qwen2 MoE layer; deterministic synthetic weights; no checkpoint',
        performance='batched eager layer calls including per-forward context, native stream/event submission and terminal completion'))
    def device(name):
        r=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=30,check=True)
        (out/name).write_text(r.stdout)
    device('device_before.txt')
    try:
        torch,torch_npu,config,ctx,layer,weights=setup(out)
        from vllm_ascend.ascend_forward_context import set_ascend_forward_context
        from vllm_ascend.utils import shared_experts_calculation_stream
        from vllm_ascend.ops.fused_moe.experts_selector import select_experts
        sources={}
        roots={'ascend':Path('/vllm-workspace/vllm-ascend/vllm_ascend'),
               'vllm':Path('/vllm-workspace/vllm/vllm')}
        paths=list(roots['ascend'].joinpath('ops/fused_moe').rglob('*.py'))
        paths += [roots['ascend']/s for s in ['utils.py','ascend_config.py','ascend_forward_context.py',
            'device/device_op.py','ops/activation.py','ops/linear.py','ops/weight_prefetch.py','worker/v2/model_runner.py']]
        paths += [roots['vllm']/s for s in ['model_executor/models/qwen2_moe.py','model_executor/layers/fused_moe/runner/moe_runner.py']]
        paths += list(Path(__file__).parent.glob('*.py'))+[Path(inspect.getfile(torch.npu.Stream))]
        for f in paths:
            tag=next((k+'/'+str(f.relative_to(v)) for k,v in roots.items() if f.is_relative_to(v)), 'instrumentation/'+f.name)
            dest=out/'sources'/tag;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(f.read_bytes())
            sources[str(dest.relative_to(out))]={'original':str(f),'sha256':sha(dest)}
        save(out/'sources.json',sources)
        env=dict(captured_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),torch=torch.__version__,
            torch_npu=torch_npu.__version__,python=platform.python_version(),device=str(torch.npu.get_device_properties(0)),
            classes={'layer':str(type(layer)),'experts':str(type(layer.experts))},
            revisions={k:subprocess.check_output(['git','-C',str(v.parent),'rev-parse','HEAD'],text=True).strip() for k,v in roots.items()},
            flags={k:os.environ.get(k) for k in ['ASCEND_LAUNCH_BLOCKING','TASK_QUEUE_ENABLE','VLLM_BATCH_INVARIANT','PYTORCH_NPU_ALLOC_CONF']},
            logical_streams={'main':dict(id=torch.npu.current_stream().stream_id,handle=str(torch.npu.current_stream().npu_stream)),
                             'shared':dict(id=shared_experts_calculation_stream().stream_id,handle=str(shared_experts_calculation_stream().npu_stream))},
            weights={n:dict(shape=list(w.shape),sha256=hashlib.sha256(w.view(torch.uint8).numpy().tobytes()).hexdigest()) for n,w in weights.items()})
        def weight_hashes():
            return {name:hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                    for name,t in layer.named_parameters()}
        env['processed_weight_sha256']=weight_hashes()
        save(out/'environment.json',env)
        generator=torch.Generator(device='cpu').manual_seed(8721)
        inputs={n:[torch.randn((n,1024),generator=generator,dtype=torch.float32).bfloat16() for _ in range(3)] for n in tokens}
        npu_inputs={n:[x.npu() for x in xs] for n,xs in inputs.items()};torch.npu.synchronize()
        def call(x,mode):
            with set_ascend_forward_context(None,config,num_tokens=x.shape[0]):
                if mode in ('serial','parallel'):
                    return layer(x)
                if mode=='shared_only':return layer.shared_expert(x)
                return layer.experts.forward_impl(x,layer.gate(x)[0])
        def mode(m):layer.experts.multistream_overlap_shared_expert=(m=='parallel')
        checks=[];baseline={};samples=[];trials=[]
        with torch.inference_mode():
            for n in tokens:
                for seed,x in enumerate(npu_inputs[n]):
                    outputs={}
                    for m in ('serial','parallel'):
                        mode(m)
                        with set_ascend_forward_context(None,config,num_tokens=n):y=call(x,m)
                        torch.npu.synchronize();outputs[m]=y.cpu()
                    # BF16 router logits can tie at the top-k cutoff. Verify the native
                    # selected scores against CPU top-k values, then evaluate those
                    # valid expert choices independently on CPU (no relaxed output tolerance).
                    with set_ascend_forward_context(None,config,num_tokens=n):
                        logits=layer.gate(x)[0]
                        native_scores,native_ids=select_experts(x,logits,2,False,True)
                    logits_cpu=logits.cpu().float();ids=native_ids.cpu().long();native_scores=native_scores.cpu().float()
                    cpu_logits=torch.nn.functional.linear(inputs[n][seed].float(),weights['gate.weight'].float()).bfloat16().float()
                    probs=cpu_logits.softmax(-1).bfloat16();top_scores,top_ids=probs.topk(2,-1)
                    selected=probs.gather(1,ids)
                    fp32_selected=cpu_logits.softmax(-1).gather(1,ids)
                    expected_scores=fp32_selected/fp32_selected.sum(-1,keepdim=True)
                    routing=dict(logits_exact=torch.equal(logits_cpu,cpu_logits),
                        logits_valid=bool(torch.allclose(logits_cpu,cpu_logits,atol=1e-4,rtol=1/128)),
                        logits_max_abs=(logits_cpu-cpu_logits).abs().max().item(),
                        selected_scores_valid=bool(torch.allclose(selected.float().sort(-1,descending=True).values,top_scores.float(),atol=1e-7,rtol=1e-6)),
                        weights_valid=bool(torch.allclose(native_scores,expected_scores,atol=0,rtol=1/128)),
                        weights_max_abs=(native_scores-expected_scores).abs().max().item(),
                        native_weights=native_scores[:3].tolist(),reference_weights=expected_scores[:3].tolist(),
                        distinct_ids=bool((ids[:,0]!=ids[:,1]).all()),
                        different_topk_rows=int((ids.sort(-1).values!=top_ids.sort(-1).values).any(-1).sum()))
                    assert all(routing[k] for k in ('logits_valid','selected_scores_valid','weights_valid','distinct_ids')),routing
                    reference=cpu_reference(torch,inputs[n][seed],weights,ids)
                    delta=(outputs['serial'].float()-reference.float()).abs()
                    check=dict(tokens=n,seed=seed,routing=routing,exact_modes=torch.equal(outputs['serial'],outputs['parallel']),
                        finite=bool(torch.isfinite(outputs['serial']).all()),cpu_max_abs=delta.max().item(),
                        cpu_rmse=delta.square().mean().sqrt().item(),
                        cpu_reference_pass=bool(torch.allclose(outputs['serial'].float(),reference.float(),atol=.003,rtol=.02)))
                    checks.append(check);save(out/'correctness.json',checks)
                    assert check['exact_modes'] and check['finite'] and check['cpu_reference_pass'],check
                    baseline[n,seed]=outputs['serial']
                print('NUMERICS',n,checks[-1],flush=True)
            # Inputs are overwritten between queued calls; native shared->main join
            # must finish old shared reads before the next main-stream copy.
            stress=[]
            for n in tokens:
                scratch=npu_inputs[n][0].clone();outputs=[]
                mode('parallel')
                with set_ascend_forward_context(None,config,num_tokens=n):
                    for i in range(18):
                        scratch.copy_(npu_inputs[n][i%3]);outputs.append(call(scratch,'parallel'))
                torch.npu.synchronize()
                ok=all(torch.equal(y.cpu(),baseline[n,i%3]) for i,y in enumerate(outputs))
                stress.append(dict(tokens=n,calls=18,changing_input_storage=True,all_outputs_equal=ok));assert ok
            save(out/'lifetime_checks.json',stress)
            # No observer / profiler is installed during any performance sample.
            modes=['serial','parallel','shared_only','routed_only']
            for n in tokens:
                x=npu_inputs[n][0]
                with set_ascend_forward_context(None,config,num_tokens=n):
                    for m in modes:
                        mode(m)
                        for _ in range(a.warmup):y=call(x,m)
                        torch.npu.synchronize()
                    for r in range(a.rounds):
                        order=modes if r%2==0 else list(reversed(modes))
                        for m in order:
                            mode(m);torch.npu.synchronize()
                            start=torch.npu.Event(enable_timing=True);done=torch.npu.Event(enable_timing=True)
                            begin=time.perf_counter_ns();start.record()
                            for _ in range(a.iterations):y=call(x,m)
                            done.record();done.synchronize();finish=time.perf_counter_ns()
                            sample=dict(tokens=n,round=r,mode=m,iterations=a.iterations,
                                wall_us_per_call=(finish-begin)/1000/a.iterations,
                                device_interval_us_per_call=start.elapsed_time(done)*1000/a.iterations)
                            samples.append(sample)
                save(out/'measurements.json',samples);print('BENCHMARK',n,flush=True)
            from observer import Observer
            observer=Observer(torch,layer);observer.install()
            experimental=torch_npu.profiler._ExperimentalConfig(profiler_level=torch_npu.profiler.ProfilerLevel.Level1)
            with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],
                schedule=torch_npu.profiler.schedule(wait=0,warmup=1,active=1,repeat=1),record_shapes=True,
                experimental_config=experimental,on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(out/'profiler'))) as prof:
                prof.step()
                for n in tokens:
                    for r in range(a.profile_repeats):
                        for m in (['serial','parallel'] if r%2==0 else ['parallel','serial']):
                            torch.npu.synchronize();mode(m)
                            ident=f'n{n}-r{r}-{m}';observer.trial=ident;observer.branch_outputs={}
                            with observer.scope('input') as rec:
                                x=npu_inputs[n][0]*1.0;rec['output']=observer.tensor(x)
                            with set_ascend_forward_context(None,config,num_tokens=n):y=layer(x)
                            done=torch.npu.current_stream().record_event();done.synchronize()
                            with observer.scope('validation'):
                                ok=torch.equal(y.cpu(),baseline[n,0]);assert ok
                            observer.trial=None
                            trials.append(dict(id=ident,tokens=n,repeat=r,mode=m,correct=ok))
                prof.step()
            observer.uninstall()
            save(out/'observations.json',observer.records);save(out/'trials.json',trials)
            torch.npu.synchronize()
            unchanged=weight_hashes()==env['processed_weight_sha256']
            save(out/'weight_check.json',dict(unchanged=unchanged));assert unchanged
            cleanup(ctx)
        save(out/'completed.json',dict(status='passed',correctness_checks=len(checks),performance_samples=len(samples),diagnostic_trials=len(trials)))
        device('device_after.txt');print('COMPLETED',out,flush=True)
    except Exception:
        (out/'failure.txt').write_text(traceback.format_exc());raise


if __name__=='__main__':main()
