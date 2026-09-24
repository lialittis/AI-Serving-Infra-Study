"""Check real evidence and reject false overlap or corrupted cross-layer links."""
import copy
from decimal import Decimal
import json
from pathlib import Path
import unittest

from analyze_parallel import analyze, overlap_metrics

RUN=Path(__file__).resolve().parent/'results/2026-09-24-run02'


class ParallelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.meta=json.loads((RUN/'run.json').read_text())
        cls.events=json.loads(next((RUN/'profiler').rglob('trace_view.json')).read_text(),parse_float=Decimal)
        if isinstance(cls.events,dict):
            cls.events=cls.events['traceEvents']
        cls.result=analyze(RUN,cls.meta,cls.events)

    def test_real_serial_and_parallel_trials(self):
        self.assertEqual(self.result['summary'],dict(compute_tasks=72,trial_count=6,
                         parallel_trials_with_overlap=3,all_outputs_correct=True,terminal_event_waits=9))
        for trial in self.result['trials']:
            if trial['mode']=='serial':
                self.assertEqual(Decimal(trial['overlap_us']),0)
            else:
                self.assertGreater(Decimal(trial['overlap_us']),3000)
                self.assertEqual(len(trial['physical_streams']),2)

    def test_overlapping_envelopes_do_not_prove_kernel_overlap(self):
        # Both branch envelopes overlap, but their actual computation alternates.
        tasks=[dict(label=str(i),branch=b,start_us=str(a),end_us=str(z))
               for i,(b,a,z) in enumerate([('mm',0,2),('mul',2,4),('mm',4,6),('mul',6,8)])]
        self.assertEqual(Decimal(overlap_metrics(tasks)['overlap_us']),0)

    def test_failed_output_is_rejected(self):
        meta=copy.deepcopy(self.meta)
        meta['checks'][0]['matrix_all_one']=False
        with self.assertRaisesRegex(ValueError,'numerical validation'):
            analyze(RUN,meta,self.events)

    def test_cross_branch_alias_is_rejected(self):
        meta=copy.deepcopy(self.meta)
        meta['resources']['vector_out']['data_ptr']=meta['resources']['matrix_out']['data_ptr']
        with self.assertRaisesRegex(ValueError,'storage alias'):
            analyze(RUN,meta,self.events)

    def test_missing_flow_cannot_be_replaced_by_nearest_time(self):
        task=self.result['tasks'][0]
        events=[e for e in self.events if not (e.get('ph')=='s' and e.get('cat')=='async_npu'
                                             and str(e.get('id'))==task['torch_flow'])]
        with self.assertRaisesRegex(ValueError,'async_npu start'):
            analyze(RUN,self.meta,events)

    def test_wrong_terminal_event_is_rejected(self):
        meta=copy.deepcopy(self.meta)
        next(r for r in meta['records'] if r['kind']=='host_wait')['event_handle']='wrong'
        with self.assertRaisesRegex(ValueError,'event identity'):
            analyze(RUN,meta,self.events)

    def test_wrong_stream_assignment_is_rejected(self):
        meta=copy.deepcopy(self.meta)
        next(r for r in meta['records'] if r['trial']=='01-parallel' and r.get('branch')=='mul')['stream_name']='A'
        with self.assertRaisesRegex(ValueError,'stream contract'):
            analyze(RUN,meta,self.events)

    def test_trace_duration_must_match_kernel_csv(self):
        events=copy.deepcopy(self.events)
        task=next(t for t in self.result['tasks'] if t['kind']=='compute')
        events[task['trace_index']]['dur']+=Decimal('1')
        with self.assertRaisesRegex(ValueError,'kernel CSV identity'):
            analyze(RUN,self.meta,events)


if __name__=='__main__':
    unittest.main()
