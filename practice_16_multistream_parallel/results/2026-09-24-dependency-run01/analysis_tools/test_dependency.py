"""Real three-stream evidence: reject broken event/data/completion assumptions."""
import copy
from decimal import Decimal
import json
from pathlib import Path
import unittest

from analyze_dependency import analyze

RUN=Path(__file__).resolve().parent/'results/2026-09-24-dependency-run01'


class DependencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.meta=json.loads((RUN/'dependency_run.json').read_text())
        cls.events=json.loads(next((RUN/'profiler').rglob('trace_view.json')).read_text(),parse_float=Decimal)
        if isinstance(cls.events,dict):cls.events=cls.events['traceEvents']
        cls.result=analyze(RUN,cls.meta,cls.events)

    def test_actual_wait_order_stale_results_and_independent_overlap(self):
        r=self.result
        self.assertEqual(r['summary']['profiled_compute_tasks'],78)
        self.assertEqual(r['summary']['profiled_device_waits'],3)
        self.assertEqual(r['summary']['profiled_host_joins'],18)
        self.assertEqual(r['performance']['event_wait']['correct_trials'],7)
        self.assertEqual(r['performance']['no_wait']['stale_trials'],7)
        for t in r['trials']:
            if t['mode']=='event_wait':
                self.assertEqual(t['producer_consumer_relation'],'B_started_after_A_producer_finished')
                self.assertGreater(Decimal(t['C_while_B_waiting_us']),3000)
            else:
                self.assertEqual(t['producer_consumer_relation'],'B_finished_before_A_producer_started')

    def test_wait_for_wrong_event_is_rejected(self):
        m=copy.deepcopy(self.meta);next(r for r in m['records'] if r['kind']=='device_wait')['event_handle']='wrong'
        with self.assertRaisesRegex(ValueError,'dependency event identity'):analyze(RUN,m,self.events)

    def test_wrong_input_contract_is_rejected(self):
        m=copy.deepcopy(self.meta);next(r for r in m['records'] if r['name']=='consume-Y')['reads']=['V']
        with self.assertRaisesRegex(ValueError,'data resource contract'):analyze(RUN,m,self.events)

    def test_missing_final_join_is_rejected(self):
        m=copy.deepcopy(self.meta);m['measurements'][0]['host_join_calls']=2
        with self.assertRaisesRegex(ValueError,'final joins missing'):analyze(RUN,m,self.events)

    def test_unsafe_stale_count_cannot_be_reported_as_correct(self):
        m=copy.deepcopy(self.meta);t=next(t for t in m['measurements'] if t['mode']=='no_wait')
        t['Z_correct_elements']=t['Z_elements'];t['Z_old_value_elements']=0
        with self.assertRaisesRegex(ValueError,'sample/count mismatch'):analyze(RUN,m,self.events)

    def test_independent_storage_must_not_alias_shared_Y(self):
        m=copy.deepcopy(self.meta);m['resources']['W']['address']=m['resources']['Y']['address']
        with self.assertRaisesRegex(ValueError,'storage alias'):analyze(RUN,m,self.events)

    def test_missing_event_flow_is_rejected(self):
        t=next(t for t in self.result['tasks'] if t['kind']=='device_wait')
        e=[x for x in self.events if not(x.get('ph')=='s' and x.get('cat')=='HostToDevice' and str(x.get('id'))==t['cann_flow'])]
        with self.assertRaisesRegex(ValueError,'HostToDevice start'):analyze(RUN,self.meta,e)

    def test_enqueue_timing_cannot_replace_complete_timing(self):
        m=copy.deepcopy(self.meta);m['measurements'][0]['elapsed_ns']=m['measurements'][0]['submission_ns']
        with self.assertRaisesRegex(ValueError,'completed timing arithmetic'):analyze(RUN,m,self.events)


if __name__=='__main__':unittest.main()
