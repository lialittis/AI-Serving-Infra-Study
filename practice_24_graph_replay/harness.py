"""Fixed-step full-model NPUGraph harness; separate live inputs and graph pools."""
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'practice_23_independent_inference'))
from model import setup as setup_model,make_case,prepare,describe,cache_layout,state_hash
from run_experiment import snapshot,compare


def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def setup():
    torch,tn,tr,model,tokenizer=setup_model();model.float();assert not torch.npu.matmul.allow_hf32
    return torch,tn,tr,model,tokenizer


def variant(case,swap):
    if not swap:return case
    return {**case,'ids':{'A':case['ids']['B'],'B':case['ids']['A']},
        'base':{'A':case['base']['B'],'B':case['base']['A']} if case['base'] else {}}


class Captured:
    def __init__(self,torch,model,case,name,stream,out):
        self.torch=torch;self.name=name;self.stream=stream
        mode='batch' if name=='AB' else 'serial'
        for _ in range(3):
            warm=prepare(torch,model,case,mode)[name]
            with torch.npu.stream(stream):y=model(**warm)
            torch.npu.synchronize()
        del warm,y
        self.job=prepare(torch,model,case,mode)[name]
        self.job['input_ids']=self.job['input_ids'].clone();torch.npu.synchronize()
        # DynamicCache replaces .keys/.values during capture. Keep the old
        # input tensors alive explicitly; graph replay still reads their addresses.
        self.input_kv=[(layer.keys,layer.values) for layer in self.job['past_key_values'].layers if layer.keys is not None]
        self.graph=torch.npu.NPUGraph();self.graph.enable_debug_mode()
        before=torch.npu.memory_allocated();begin=time.perf_counter_ns()
        with torch.npu.graph(self.graph,stream=stream,capture_error_mode='global'):
            self.output=model(**self.job)
        torch.npu.synchronize();capture_ms=(time.perf_counter_ns()-begin)/1e6
        begin=time.perf_counter_ns()
        with torch.npu.stream(stream):self.graph.replay()
        stream.synchronize();first_ms=(time.perf_counter_ns()-begin)/1e6
        dump=out/f'{case["id"]}-{name}.json';self.graph.debug_dump(str(dump))
        self.metadata=dict(name=name,case=case['id'],pool=list(self.graph.pool()),capture_ms=capture_ms,first_replay_ms=first_ms,
            capture_allocated_before=before,capture_allocated_after=torch.npu.memory_allocated(),
            capture_stream=dict(id=stream.stream_id,handle=str(stream.npu_stream)),dump=str(dump.name),
            input=describe(torch,self.job['input_ids']),input_kv=[dict(key=describe(torch,k),value=describe(torch,v)) for k,v in self.input_kv],
            output_logits=describe(torch,self.output.logits),output_kv=cache_layout(torch,self.output.past_key_values))

    def load(self,case):
        # Called only after the previous replay's terminal events have completed.
        torch=self.torch;names=('A','B') if self.name=='AB' else (self.name,)
        source=torch.cat([case['ids'][n] for n in names],0) if self.name=='AB' else case['ids'][self.name]
        self.job['input_ids'].copy_(source)
        for layer,(k,v) in enumerate(self.input_kv):
            for index,dest in enumerate((k,v)):
                source=torch.cat([case['base'][n][layer][index] for n in names],0) if self.name=='AB' else case['base'][self.name][layer][index]
                dest.copy_(source)


def run_pair(torch,model,case,mode,backend,streams,captures,order='AB',observer=None):
    names=['AB'] if mode=='batch' else list(order)
    if backend=='eager':jobs=prepare(torch,model,case,mode)
    else:
        for name in names:captures[name].load(case)
        torch.npu.synchronize();jobs=None
    def scope(kind,**kw):
        from contextlib import nullcontext
        return observer.scope(kind,**kw) if observer else nullcontext()
    def event(e,action,role):
        if observer:observer.event(e,action,role)
        else:getattr(e,action)()
    origin=torch.npu.Event(enable_timing=True);ends={n:torch.npu.Event(enable_timing=True) for n in names}
    torch.npu.reset_peak_memory_stats();before=torch.npu.memory_allocated();result={};submit={}
    begin=time.perf_counter_ns();event(origin,'record','origin')
    for name in names:
        stream=streams['B'] if mode=='parallel' and name=='B' else streams['A']
        with torch.npu.stream(stream):
            event(origin,'wait',name+'_ready')
            with scope('forward' if backend=='eager' else 'replay',task=name):
                if backend=='eager':result[name]=model(**jobs[name])
                else:captures[name].graph.replay();result[name]=captures[name].output
            event(ends[name],'record',name+'_done')
        submit[name]=(time.perf_counter_ns()-begin)/1000
    for name in names:event(ends[name],'synchronize',name+'_join')
    wall=(time.perf_counter_ns()-begin)/1000
    ready={n:origin.elapsed_time(e)*1000 for n,e in ends.items()}
    metric=dict(wall_us=wall,ready_us={n:ready.get(n,ready.get('AB')) for n in ('A','B')},host_submission_us=submit,
        allocated_before=before,allocated_peak=torch.npu.max_memory_allocated(),reserved_peak=torch.npu.max_memory_reserved())
    return result,metric
