import unittest
import copy
import os
import json
from pathlib import Path
from decimal import Decimal as D
from analyze import difference,length,overlap,core_count,union,require,validate_csv,validate_graph_chunk

class IntervalTests(unittest.TestCase):
    def test_union_not_duration_sum(self):
        self.assertEqual(length([(D(0),D(10)),(D(5),D(15))]),D(15))
    def test_gaps_not_branch_envelope(self):
        self.assertEqual(difference((D(0),D(20)),[(D(0),D(5)),(D(15),D(20))]),[(D(5),D(15))])
    def test_difference_clips_outside_window(self):
        self.assertEqual(difference((D(0),D(20)),[(D(-5),D(4)),(D(18),D(25))]),[(D(4),D(18))])
    def test_wait_not_compute_overlap(self):
        tasks=[dict(is_compute=True,stream='a',start_us='0',end_us='10'),dict(is_compute=False,stream='b',start_us='0',end_us='10')]
        self.assertEqual(overlap(tasks),dict(overlap_us='0',peak_compute_streams=1))
    def test_cross_stream_actual_intersection(self):
        tasks=[dict(is_compute=True,stream='a',start_us='0',end_us='5'),dict(is_compute=True,stream='a',start_us='15',end_us='20'),dict(is_compute=True,stream='b',start_us='4',end_us='16')]
        self.assertEqual(overlap(tasks),dict(overlap_us='2',peak_compute_streams=2))
    def test_touching_intervals_not_parallel(self):
        tasks=[dict(is_compute=True,stream='a',start_us='0',end_us='5'),dict(is_compute=True,stream='b',start_us='5',end_us='10')]
        self.assertEqual(overlap(tasks),dict(overlap_us='0',peak_compute_streams=1))
    def test_zero_core_is_unknown(self):
        for x in (0,'0','N/A','',None):self.assertIsNone(core_count(x))
        self.assertEqual(core_count('48'),48)

class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=Path(os.environ.get('P26_REPLAY_ROOT',str(Path(__file__).parent/'results/2026-09-29-run01')))
        cls.data={k:json.loads((cls.root/k/'analysis/evidence.json').read_text()) for k in ('eager-plain','graph-plain','eager-pipe','graph-pipe')}
    def test_all_steps_and_csv_covered_once(self):
        for d in self.data.values():
            self.assertEqual([s['index'] for s in d['steps']],list(range(64)))
            rows=[t['csv_row'] for t in d['tasks'] if t['is_compute']]
            self.assertEqual(sorted(rows),list(range(len(d['kernel_rows']))))
            assigned=[tid for s in d['steps'] for tid in s['tasks']]
            self.assertEqual(len(assigned),len(set(assigned)))
            self.assertFalse(d['unattributed_runtime'])
    def test_wrong_csv_identity_rejected(self):
        d=self.data['eager-plain'];t=next(t for t in d['tasks'] if t['is_compute']);row=copy.deepcopy(d['kernel_rows'][t['csv_row']]);row['Task ID']='999999'
        with self.assertRaisesRegex(ValueError,'identity'):validate_csv(t,row)
    def graph_example(self):
        d=self.data['graph-plain'];r=d['replays'][0];ts={t['id']:t for t in d['tasks']};chunk=[copy.deepcopy(ts[k]) for k in r['internal_tasks']]
        dump=json.loads((self.root/'graph-plain/graph_dumps'/(r['uid']+'.json')).read_text())
        return chunk,dump,copy.deepcopy(ts[r['launch']]),copy.deepcopy(ts[r['wait']])
    def test_wrong_graph_stream_rejected(self):
        chunk,dump,launch,wait=self.graph_example();chunk[0]['stream']='999'
        with self.assertRaisesRegex(ValueError,'sequence'):validate_graph_chunk(chunk,dump,launch,wait)
    def test_missing_graph_task_rejected(self):
        chunk,dump,launch,wait=self.graph_example();chunk.pop(0)
        with self.assertRaisesRegex(ValueError,'size'):validate_graph_chunk(chunk,dump,launch,wait)
    def test_graph_completion_before_kernel_rejected(self):
        chunk,dump,launch,wait=self.graph_example();wait['end_us']=chunk[0]['start_us']
        with self.assertRaisesRegex(ValueError,'enclosure'):validate_graph_chunk(chunk,dump,launch,wait)
    def test_gap_categories_conserve_uncovered_time(self):
        for d in self.data.values():
            for s in d['steps']:
                self.assertEqual(sum((D(x) for x in s['uncovered_by_next_submission'].values()),D(0)),D(s['uncovered_us']))
                self.assertEqual(D(s['compute_union_us'])+D(s['no_compute_us']),D(s['device_span_us']))
    def test_pipe_metrics_present(self):
        for case in ('eager-pipe','graph-pipe'):
            self.assertTrue(any(c['pipeline'] for c in self.data[case]['cores']))

if __name__=='__main__':unittest.main()
