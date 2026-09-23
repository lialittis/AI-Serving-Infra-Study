"""Validate real PIECEWISE evidence and reject false mode/comparison claims."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import compare_modes
from analyze_reuse import analyze,read_trace

ROOT=Path(__file__).resolve().parent
EAGER=ROOT/'results/2026-09-23-run03-eager'
GRAPH=ROOT/'results/2026-09-23-run04-graph'

class ModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records=[json.loads(line) for p in (GRAPH/'events').glob('*.jsonl') for line in p.read_text().splitlines()]
        cls.trace=read_trace(next((GRAPH/'profiler').rglob('trace_view.json')))
        cls.graph_result,cls.chains,_=analyze(GRAPH,cls.records,cls.trace)
        cls.eager_result,_,_=analyze(EAGER)

    def test_real_graph_retains_all_direct_kv_chains(self):
        d=self.graph_result
        self.assertEqual(d['verified_kv_operator_chains'],192)
        self.assertEqual(d['device_task_counts']['MODEL_EXECUTE'],50)
        self.assertEqual(d['replayed_tasks'],488)
        self.assertEqual(d['replayed_tasks_without_torch_flow'],488)
        self.assertEqual(d['requests']['A']['steps'][0]['execution'],'compiled_callable')
        self.assertEqual(d['requests']['A']['steps'][1]['execution'],'graph_replay')
        self.assertEqual(d['requests']['A']['steps'][1]['partition_body_calls'],0)

    def test_missing_captured_graph_is_rejected(self):
        events=copy.deepcopy(self.records)
        e=next(e for e in events if e['event']=='scope_enter' and e.get('kind')=='acl_dispatch' and e['runtime_mode']=='PIECEWISE')
        e['has_captured_graph']=False
        with self.assertRaisesRegex(ValueError,'decode replay evidence'):analyze(GRAPH,events,self.trace)

    def test_decode_python_body_claim_is_rejected(self):
        events=self.records+[dict(event='partition_body_call',role='A',step=4,partition='submod_0')]
        with self.assertRaisesRegex(ValueError,'decode unexpectedly called'):analyze(GRAPH,events,self.trace)

    def test_graph_reuse_identity_mismatch_is_rejected(self):
        events=copy.deepcopy(self.records)
        e=next(e for e in events if e['event']=='scope_enter' and e.get('kind')=='acl_dispatch' and e['role']=='B' and e['runtime_mode']=='PIECEWISE')
        e['graph_object_id']+=1
        with self.assertRaisesRegex(ValueError,'same captured graph objects'):analyze(GRAPH,events,self.trace)

    def test_graph_kv_flow_still_required(self):
        flow=next(c['torch_flow_id'] for c in self.chains if c['phase']=='decode')
        trace=[e for e in self.trace if not(e.get('cat')=='async_npu' and e.get('ph')=='f' and str(e.get('id'))==flow)]
        with self.assertRaisesRegex(ValueError,'torch flow end'):analyze(GRAPH,self.records,trace)

    def _cached_analyze(self,run):
        return (self.eager_result if run==EAGER else self.graph_result),[],[]

    def test_pair_matches_configuration_and_output_tokens(self):
        # The analyses above are real; reuse their results while exercising provenance checks.
        with patch.object(compare_modes,'analyze',side_effect=self._cached_analyze):
            d=compare_modes.compare(EAGER,GRAPH)
        self.assertTrue(d['same_output_tokens'])
        self.assertFalse(d['performance_comparison'])

    def test_different_input_request_is_not_comparable(self):
        original=compare_modes.load
        def changed(p):
            d=original(p)
            if p==GRAPH/'request_B.json':d['prompt'][0]+=1
            return d
        with patch.object(compare_modes,'analyze',side_effect=self._cached_analyze),patch.object(compare_modes,'load',side_effect=changed):
            with self.assertRaisesRegex(ValueError,'request bodies differ'):compare_modes.compare(EAGER,GRAPH)

    def test_different_instrumentation_is_not_comparable(self):
        original=compare_modes.load
        def changed(p):
            d=original(p)
            if p==GRAPH/'instrumentation_hashes.json':d['instrumentation/lifetime_trace.py']='different'
            return d
        with patch.object(compare_modes,'analyze',side_effect=self._cached_analyze),patch.object(compare_modes,'load',side_effect=changed):
            with self.assertRaisesRegex(ValueError,'identical runtime observation'):compare_modes.compare(EAGER,GRAPH)

if __name__=='__main__':unittest.main()
