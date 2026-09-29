"""Protect lifecycle identities and the difference between ownership and ordering."""
import json
from pathlib import Path
import unittest

from analyze_streams import live_graph,check_handle,validate_dag,compute_overlap


class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.graphs=[dict(id='old',model_ids=[42],capture_time_ns=100,destroy_time_ns=200),
                     dict(id='new',model_ids=[42],capture_time_ns=300,destroy_time_ns=None)]

    def test_model_id_reuse_requires_lifetime(self):
        self.assertEqual(live_graph(self.graphs,42,150)['id'],'old')
        self.assertEqual(live_graph(self.graphs,42,350)['id'],'new')
        self.assertIsNone(live_graph(self.graphs,42,250))

    def test_overlapping_model_lifetimes_rejected(self):
        self.graphs[0]['destroy_time_ns']=None
        with self.assertRaisesRegex(ValueError,'ambiguous'):live_graph(self.graphs,42,350)

    def test_unknown_id_not_assigned_to_nearby_graph(self):
        self.assertIsNone(live_graph(self.graphs,43,350))

    def test_wrong_handle_namespace_rejected(self):
        with self.assertRaisesRegex(ValueError,'physical stream'):check_handle([dict(id='resource',runtime_ids=[46])],96)

    def test_missing_creation_cannot_be_exact(self):
        with self.assertRaisesRegex(ValueError,'physical stream'):check_handle([],46)

    def test_ambiguous_resource_rejected(self):
        with self.assertRaisesRegex(ValueError,'physical stream'):check_handle([dict(id='a',runtime_ids=[46]),dict(id='b',runtime_ids=[46])],46)

    def test_dependency_cycle_rejected(self):
        with self.assertRaisesRegex(ValueError,'cycle'):validate_dag([dict(id='a'),dict(id='b')],[dict(source='a',target='b'),dict(source='b',target='a')])

    def test_missing_dependency_endpoint_rejected(self):
        with self.assertRaisesRegex(ValueError,'endpoint'):validate_dag([dict(id='a')],[dict(source='a',target='missing')])

    def test_wait_interval_is_not_compute_overlap(self):
        tasks=[dict(csv=0,stream='main',start_us=0,end_us=10),
               dict(csv=None,stream='aux',start_us=0,end_us=20)]
        self.assertEqual(compute_overlap(tasks),dict(cross_stream_compute_overlap_us='0',peak_observed_compute_streams=1))

    def test_compute_overlap_counts_streams_and_union(self):
        tasks=[dict(csv=0,stream='main',start_us=0,end_us=10),
               dict(csv=1,stream='main',start_us=1,end_us=9),
               dict(csv=2,stream='aux',start_us=5,end_us=15),
               dict(csv=3,stream='third',start_us=15,end_us=20)]
        self.assertEqual(compute_overlap(tasks),dict(cross_stream_compute_overlap_us='5',peak_observed_compute_streams=2))

    def test_real_report_coverage_and_replay_instances(self):
        runs=Path(__file__).parent/'results'
        paths=list(runs.glob('*-run01/analysis/stream_evidence.json'))
        self.assertEqual(len(paths),4,'analyze all four real cases before testing')
        for path in paths:
            data=json.loads(path.read_text());validate_dag(data['tasks']+data['completion_nodes'],data['edges'])
            self.assertEqual(sum(s['tasks'] for s in data['streams']),len(data['tasks']))
            self.assertFalse(data['gaps']['unmatched_native_profiler_apis'])
            if data['case']=='graph':
                self.assertEqual(len(data['replays']),150)
                selected=[t for t in data['tasks'] if t['graph']]
                assigned=[k for r in data['replays'] for k in r['internal_tasks']]
                self.assertEqual(set(assigned),{t['id'] for t in selected})
                self.assertEqual(len(assigned),len(set(assigned)))
                self.assertTrue(all(len(r['boundary_tasks'])==2 for r in data['replays']))


if __name__=='__main__':unittest.main()
