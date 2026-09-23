"""Archived real evidence plus deliberately damaged ownership/correlation data."""
import copy
import json
from pathlib import Path
import unittest
from analyze_reuse import analyze, read_trace

RUN=Path(__file__).resolve().parent/'results/2026-09-23-run02'

class ReuseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records=[json.loads(line) for p in (RUN/'events').glob('*.jsonl') for line in p.read_text().splitlines()]
        cls.trace=read_trace(next((RUN/'profiler').rglob('trace_view.json')))

    def test_real_reuse_and_all_layers(self):
        result,chains,_=analyze(RUN,self.records,self.trace)
        self.assertEqual(result['verified_kv_operator_chains'],192)
        self.assertEqual(result['reused_block'],1)
        self.assertGreater(float(result['gap_last_A_access_to_free_us']),0)

    def test_nonzero_refcount_after_free_is_rejected(self):
        events=copy.deepcopy(self.records)
        e=next(e for e in events if e['event']=='scope_exit' and e.get('kind')=='pool_free' and e.get('affected_blocks'))
        e['after']['blocks'][1]['ref_cnt']=1
        with self.assertRaisesRegex(ValueError,'reference count'):analyze(RUN,events,self.trace)

    def test_missing_layer_is_rejected(self):
        events=copy.deepcopy(self.records)
        e=next(e for e in events if e['event']=='scope_enter' and e.get('kind')=='attention')
        e['layer']='unknown'
        with self.assertRaisesRegex(ValueError,'layer coverage'):analyze(RUN,events,self.trace)

    def test_wrong_reuse_block_is_rejected(self):
        events=copy.deepcopy(self.records)
        e=next(e for e in events if e['event']=='scope_exit' and e.get('kind')=='pool_allocate' and e.get('affected_blocks'))
        e['affected_blocks']=[0]
        with self.assertRaisesRegex(ValueError,'reused block'):analyze(RUN,events,self.trace)

    def test_wrong_decode_cache_pointer_is_rejected(self):
        events=copy.deepcopy(self.records)
        e=next(e for e in events if e['event']=='fia_inputs' and e['lengths']==[127])
        e['key']['data_ptr']+=256
        with self.assertRaisesRegex(ValueError,'cache input'):analyze(RUN,events,self.trace)

    def test_missing_native_wait_is_rejected(self):
        trace=[e for e in self.trace if e.get('name')!='Event::synchronize']
        with self.assertRaisesRegex(ValueError,'Event::synchronize'):analyze(RUN,self.records,trace)

    def test_release_before_native_wait_is_rejected(self):
        trace=copy.deepcopy(self.trace)
        for e in trace:
            if e.get('name','').startswith('P12/A/') and e['name'].endswith(('/pool_free','/manager_free')):
                e['ts']=str(float(e['ts'])-10000)
        with self.assertRaisesRegex(ValueError,'release precedes native result wait'):
            analyze(RUN,self.records,trace)

    def test_missing_device_flow_is_rejected(self):
        result=analyze(RUN,self.records,self.trace)
        flow=result[1][0]['torch_flow_id']
        trace=[e for e in self.trace if not(e.get('cat')=='async_npu' and e.get('ph')=='f' and str(e.get('id'))==flow)]
        with self.assertRaisesRegex(ValueError,'torch flow end'):analyze(RUN,self.records,trace)

    def test_instrumentation_error_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'instrumentation errors'):
            analyze(RUN,self.records+[{'event':'trace_error'}],self.trace)

if __name__=='__main__':unittest.main()
