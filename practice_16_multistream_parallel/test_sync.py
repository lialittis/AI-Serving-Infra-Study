"""Tests of real sync-ablation evidence, not timings of a synthetic mock."""
import copy
from decimal import Decimal
import json
from pathlib import Path
import unittest

from analyze_sync import analyze

RUN=Path(__file__).resolve().parent/'results/2026-09-24-sync-run01'


class SynchronizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.meta=json.loads((RUN/'sync_run.json').read_text())
        cls.events=json.loads(next((RUN/'profiler').rglob('trace_view.json')).read_text(),parse_float=Decimal)
        if isinstance(cls.events,dict):cls.events=cls.events['traceEvents']
        cls.result=analyze(RUN,cls.meta,cls.events)

    def test_actual_completion_and_early_read_results(self):
        r=self.result
        self.assertEqual(r['summary']['profiled_compute_tasks'],144)
        self.assertEqual(r['summary']['profiled_host_waits'],12)
        self.assertEqual(r['summary']['full_tensor_checks'],192)
        self.assertTrue(r['summary']['all_final_outputs_correct'])
        self.assertEqual(r['diagnostic']['incorrect_reads'],56)
        self.assertEqual(r['diagnostic']['not_ready_after_read'],56)
        self.assertLess(r['performance']['final_only']['issue_median_ms'],
                        r['performance']['final_only']['completed_median_ms'])

    def test_enqueue_time_cannot_replace_complete_time(self):
        m=copy.deepcopy(self.meta)
        b=next(b for b in m['performance'] if b['mode']=='final_only')
        b['completed_ns']=b['issue_phase_ns']
        with self.assertRaisesRegex(ValueError,'timing arithmetic'):analyze(RUN,m,self.events)

    def test_profiled_data_cannot_enter_performance_group(self):
        m=copy.deepcopy(self.meta);m['performance'][0]['profiled']=True
        with self.assertRaisesRegex(ValueError,'profile/performance'):analyze(RUN,m,self.events)

    def test_bad_final_output_is_rejected(self):
        m=copy.deepcopy(self.meta);m['diagnostics'][0]['checks'][0]['full_output_correct']['mm']=False
        with self.assertRaisesRegex(ValueError,'final arithmetic'):analyze(RUN,m,self.events)

    def test_reused_output_cannot_be_claimed_independent(self):
        m=copy.deepcopy(self.meta);m['outputs'][1]['mm']['address']=m['outputs'][0]['mm']['address']
        with self.assertRaisesRegex(ValueError,'storage alias'):analyze(RUN,m,self.events)

    def test_pageable_destination_is_rejected(self):
        m=copy.deepcopy(self.meta);m['samples'][0]['mm']['pinned']=False
        with self.assertRaisesRegex(ValueError,'unpinned'):analyze(RUN,m,self.events)

    def test_missing_flow_is_rejected(self):
        t=self.result['tasks'][0]
        e=[x for x in self.events if not(x.get('ph')=='s' and x.get('cat')=='HostToDevice' and str(x.get('id'))==t['cann_flow'])]
        with self.assertRaisesRegex(ValueError,'HostToDevice start'):analyze(RUN,self.meta,e)

    def test_copy_assigned_to_wrong_round_is_rejected(self):
        m=copy.deepcopy(self.meta);next(r for r in m['records'] if r['kind']=='copy')['round']=1
        with self.assertRaisesRegex(ValueError,'branch submission order'):analyze(RUN,m,self.events)


if __name__=='__main__':unittest.main()
