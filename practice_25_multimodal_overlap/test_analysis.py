"""Counterfactuals against the recorded P3b data handoff."""
import gzip,json,unittest
from pathlib import Path
from trace_graph import Graph,number
from analyze import overlap,bounds


class DependencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with gzip.open(Path(__file__).with_name('results')/'published/execution_graph.json.gz','rt') as f:cls.data=json.load(f)
        cls.nodes={n['id']:n for n in cls.data['nodes']}

    def rebuild(self,drop=lambda e:False):
        g=Graph()
        for n in self.data['nodes']:g.node(n['id'],**{k:v for k,v in n.items() if k!='id'})
        for e in self.data['edges']:
            if not drop(e):g.edge(**e)
        return g

    def test_requirements_are_not_proof_edges(self):
        g=self.rebuild();self.assertTrue(all(r['satisfied'] for r in g.verify(self.data['requirements'])))
        self.assertFalse(any(e['kind'] in ('input_readiness','output_completion','visual_feature_ready','merged_embedding_ready') for e in g.edges))

    def test_missing_feature_event_breaks_cross_stream_handoff(self):
        def drop(e):
            target=self.nodes[e['target']]
            return e['kind'] in ('event_wait','event_wait_api') and target.get('trial','').endswith('-parallel') and '/event_wait' in target.get('scope','') and target.get('scope') in {
                r['label'] for r in self.data['records'] if r.get('role')=='B_features_ready'}
        g=self.rebuild(drop)
        rs=[r for r in self.data['requirements'] if r['kind']=='visual_feature_ready' and r['trial'].endswith('-parallel')]
        self.assertEqual(len(rs),16);self.assertTrue(all(not r['satisfied'] for r in g.verify(rs)))

    def test_missing_origin_wait_breaks_readiness(self):
        labels={r['label'] for r in self.data['records'] if r.get('role') in ('L_ready','V_ready')}
        g=self.rebuild(lambda e:e['kind'] in ('event_wait','event_wait_api') and self.nodes[e['target']].get('scope') in labels)
        rs=[r for r in self.data['requirements'] if r['kind']=='input_readiness']
        self.assertTrue(all(not r['satisfied'] for r in g.verify(rs)))

    def test_missing_host_join_breaks_completion(self):
        g=self.rebuild(lambda e:e['kind']=='host_sync')
        rs=[r for r in self.data['requirements'] if r['kind']=='output_completion']
        self.assertTrue(all(not r['satisfied'] for r in g.verify(rs)))

    def test_serial_order_is_not_a_parallel_data_dependency(self):
        rs=self.rebuild().verify(self.data['cross_task_order'])
        self.assertTrue(all(r['satisfied']==r['expected'] for r in rs))

    def test_overlap_excludes_gaps_and_duplicate_intervals(self):
        a=[(number(0),number(1)),(number(9),number(10))]
        self.assertEqual(overlap(a,[(number(2),number(8))]),0)
        self.assertEqual(overlap(a+a,[(number(0),number(10))]),2)

    def test_strided_storage_extent_includes_holes(self):
        self.assertEqual(bounds(dict(ptr='1000',shape=[2,3],stride=[8,1],bytes=24)),(1000,1044))


if __name__=='__main__':unittest.main()
