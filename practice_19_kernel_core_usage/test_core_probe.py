"""Reject broken provenance, incorrect output and incomplete timing evidence."""
import copy
import csv
from decimal import Decimal
import json
from pathlib import Path
import unittest

from analyze_core_probe import analyze


RUN = Path(__file__).parent/'results/2026-09-28-run01'


class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.meta=json.loads((RUN/'run.json').read_text())
        cls.traces={}
        cls.rows={}
        for profile in ('plain','pipe'):
            root=RUN/('profiler_'+profile)
            trace=json.loads(next(root.rglob('trace_view.json')).read_text(),parse_float=Decimal)
            cls.traces[profile]=trace['traceEvents'] if isinstance(trace,dict) else trace
            with next(root.rglob('kernel_details.csv')).open() as f:
                cls.rows[profile]=list(csv.DictReader(f))

    def check(self,meta=None,traces=None,rows=None):
        return analyze(RUN,meta if meta is not None else self.meta,
                       traces if traces is not None else self.traces,
                       rows if rows is not None else self.rows)

    def test_real_evidence(self):
        data=self.check()
        self.assertEqual(data['correctness_checks'],100)
        self.assertEqual(len(data['tasks']),60)
        self.assertEqual(data['hardware']['vector_core_num'],48)

    def test_invalid_pool_result(self):
        meta=copy.deepcopy(self.meta);meta['checks'][0]['value_pool_equal']=False
        with self.assertRaisesRegex(ValueError,'correctness failure'):self.check(meta=meta)

    def test_missing_final_join(self):
        meta=copy.deepcopy(self.meta);meta['trials'][0]['final_wait']=False
        with self.assertRaisesRegex(ValueError,'completed-work timing'):self.check(meta=meta)

    def test_wrong_slot_contract(self):
        meta=copy.deepcopy(self.meta);meta['checks'][0]['first_slot']=0
        with self.assertRaisesRegex(ValueError,'correctness scope'):self.check(meta=meta)

    def test_excess_core_count(self):
        rows=copy.deepcopy(self.rows)
        next(r for r in rows['plain'] if r['Name']=='ReshapeAndCacheNdKernel')['Block Num']='49'
        with self.assertRaisesRegex(ValueError,'core count'):self.check(rows=rows)

    def test_unknown_zero_not_valid_core_count(self):
        rows=copy.deepcopy(self.rows)
        next(r for r in rows['plain'] if r['Name']=='ReshapeAndCacheNdKernel')['Block Num']='0'
        with self.assertRaisesRegex(ValueError,'core count'):self.check(rows=rows)

    def test_mismatched_task_identity(self):
        rows=copy.deepcopy(self.rows)
        next(r for r in rows['plain'] if r['Name']=='ReshapeAndCacheNdKernel')['Task ID']='999999'
        with self.assertRaisesRegex(ValueError,'CSV identity'):self.check(rows=rows)

    def test_missing_pipeline_metric(self):
        rows=copy.deepcopy(self.rows)
        next(r for r in rows['pipe'] if r['Name']=='ReshapeAndCacheNdKernel')['aiv_scalar_ratio']='N/A'
        with self.assertRaisesRegex(ValueError,'pipeline metrics absent'):self.check(rows=rows)

    def test_missing_flow(self):
        traces=copy.deepcopy(self.traces)
        traces['plain']=[e for e in traces['plain'] if not(e.get('ph')=='f' and e.get('cat')=='async_npu')]
        with self.assertRaisesRegex(ValueError,'async_npu endpoint'):self.check(traces=traces)

    def test_wrong_connection(self):
        traces=copy.deepcopy(self.traces)
        next(e for e in traces['plain'] if e.get('ph')=='X' and e.get('name')=='ReshapeAndCacheNdKernel')['args']['connection_id']=-1
        with self.assertRaisesRegex(ValueError,'connection mismatch'):self.check(traces=traces)


if __name__=='__main__':unittest.main()
