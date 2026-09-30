"""One native request at a time, full growing-KV generation; eager or PIECEWISE."""
import argparse
from contextlib import nullcontext, ExitStack
import inspect
import math
import os
from pathlib import Path
import time
import traceback

from common import save


def output_record(result):
    item = result.outputs[0]
    logprobs = [{str(k):dict(logprob=v.logprob, rank=v.rank) for k,v in row.items()} for row in item.logprobs]
    assert len(item.token_ids)==len(logprobs)==64
    assert all(math.isfinite(v['logprob']) for row in logprobs for v in row.values())
    return dict(tokens=list(item.token_ids), logprobs=logprobs, finish_reason=item.finish_reason)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stage',choices=['reference','diagnostic','recovery'],required=True)
    p.add_argument('--mode',choices=['eager','graph'],default='eager')
    p.add_argument('--forward-detail',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    if a.forward_detail and a.mode!='eager':p.error('RoPE detail is a warmed eager-only probe')
    os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        TRITON_CACHE_DIR=str(a.output/'triton_cache'),VLLM_CACHE_ROOT=str(a.output/'vllm_cache'))
    status=dict(status='running',stage=a.stage,mode=a.mode,pid=os.getpid())
    save(a.output/'status.json',status)
    observer=None; graph_observer=None; detail=None; hooks=ExitStack()
    try:
        from native import make_engine, prompts, generate, settings
        import torch, torch_npu
        if a.mode=='graph' and a.stage=='diagnostic':
            from graph_observer import GraphObserver
            graph_observer=hooks.enter_context(GraphObserver(torch,a.output).installed())
        init_wall=time.perf_counter_ns();init_cpu=time.thread_time_ns()
        torch,llm,runner=make_engine(a.mode)
        save(a.output/'initialization.json',dict(wall_us=(time.perf_counter_ns()-init_wall)/1000,
            thread_cpu_us=(time.thread_time_ns()-init_cpu)/1000,
            includes='engine construction, profiling, compilation/capture where enabled; not steady request time'))
        import torch_npu,vllm,vllm_ascend
        save(a.output/'imports.json',{m.__name__:dict(path=m.__file__,version=getattr(m,'__version__',None))
                                     for m in (torch,torch_npu,vllm,vllm_ascend)})
        save(a.output/'settings.json',settings(a.mode))
        save(a.output/'effective_compilation.json',dict(config=str(runner.vllm_config.compilation_config)))
        ids=prompts(llm)['A'];save(a.output/'request.json',dict(prompt_token_ids=ids,max_tokens=64,
            temperature=0,ignore_eos=True,logprobs=1,stage=a.stage))
        warmups=[output_record(generate(llm,ids)) for _ in range(2)]
        save(a.output/'warmups.json',warmups)
        responses=[]
        count=3 if a.stage=='reference' else 1
        if a.stage=='diagnostic':
            from observer import Observer
            observer=Observer(torch,llm,runner)
            if graph_observer:observer.sources.update(graph_observer.sources)
            if a.forward_detail:
                from forward_observer import ForwardObserver
                detail=ForwardObserver(observer)
            save(a.output/'phase_sources.json',observer.sources)
            profiler=torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],
                record_shapes=True,with_stack=False,profile_memory=False,
                experimental_config=torch_npu.profiler._ExperimentalConfig(
                    profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                    aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(a.output/'profiler')))
        else:
            profiler=nullcontext()
        try:
            with profiler, (observer.installed() if observer else nullcontext()), (detail.installed() if detail else nullcontext()):
                if graph_observer:graph_observer.active=observer
                for _ in range(count):
                    wall=time.perf_counter_ns();cpu=time.thread_time_ns()
                    with observer.phase('request') if observer else nullcontext():
                        result=generate(llm,ids)
                    cpu_end=time.thread_time_ns();wall_end=time.perf_counter_ns()
                    responses.append(dict(wall_us=(wall_end-wall)/1000,
                        thread_cpu_us=(cpu_end-cpu)/1000,output=output_record(result)))
        finally:
            if graph_observer:graph_observer.active=None
            if observer:
                save(a.output/'observer.json',observer.records)
                save(a.output/'binding_recovery.json',dict(restored=observer.restored))
            if detail:save(a.output/'forward_detail.json',detail.metadata)
        save(a.output/'responses.json',responses)
        assert all(r['output']==warmups[0] for r in responses) and warmups[0]==warmups[1]
        if observer:
            executed=[r for r in observer.records if r['kind']=='execute']
            assert [r['index'] for r in executed]==list(range(64))
            assert [list(r['scheduled'].values()) for r in executed]==[[10]]+[[1]]*63
        if detail:
            assert detail.selected==1
            assert not any(r['kind']=='rope_compile' for r in observer.records),'unexpected compilation during selected call'
        status.update(status='passed',requests=count,exact_warmup_output_match=True)
    except BaseException as exc:
        status.update(status='failed',reason=repr(exc),traceback=traceback.format_exc())
        raise
    finally:
        try:hooks.close()
        finally:save(a.output/'status.json',status)


if __name__=='__main__':main()
