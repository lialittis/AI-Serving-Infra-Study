"""Counterfactual tests against the recorded execution graph."""
import gzip,json,unittest
from pathlib import Path
from trace_graph import Graph,number
from analyze import overlap


class DependencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with gzip.open(Path(__file__).with_name('results')/'published/execution_graph.json.gz','rt') as f:cls.data=json.load(f)
        cls.nodes={n['id']:n for n in cls.data['nodes']}
        cls.trial=next(t for t in cls.data['summary']['trials'] if t['backend']=='graph' and t['mode']=='parallel')

    def rebuild(self,drop=lambda e:False):
        g=Graph()
        for n in self.data['nodes']:g.node(n['id'],**{k:v for k,v in n.items() if k!='id'})
        for e in self.data['edges']:
            if not drop(e):g.edge(**e)
        return g

    def test_requirements_are_separate_from_proofs(self):
        g=self.rebuild()
        self.assertTrue(all(r['satisfied'] for r in g.verify(self.data['requirements'])))
        self.assertFalse(any(e['kind'] in ('input_readiness','output_completion') for e in g.edges))

    def test_missing_origin_wait_breaks_readiness(self):
        g=self.rebuild(lambda e:e['kind'] in ('event_wait','event_wait_api') and self.nodes[e['target']].get('trial')==self.trial['id'])
        rs=[r for r in self.data['requirements'] if r['trial']==self.trial['id'] and r['kind']=='input_readiness']
        self.assertEqual(len(rs),2);self.assertTrue(all(not r['satisfied'] for r in g.verify(rs)))

    def test_missing_graph_launch_breaks_readiness(self):
        g=self.rebuild(lambda e:e['kind']=='graph_launch')
        rs=[r for r in self.data['requirements'] if '-graph-' in r['trial'] and r['kind']=='input_readiness']
        self.assertEqual(len(rs),40);self.assertTrue(all(not r['satisfied'] for r in g.verify(rs)))

    def test_missing_graph_completion_breaks_output_proof(self):
        g=self.rebuild(lambda e:e['kind']=='graph_completion')
        rs=[r for r in self.data['requirements'] if '-graph-' in r['trial'] and r['kind']=='output_completion']
        self.assertEqual(len(rs),40);self.assertTrue(all(not r['satisfied'] for r in g.verify(rs)))

    def test_missing_host_join_breaks_output_proof(self):
        g=self.rebuild(lambda e:e['kind']=='host_sync')
        rs=[r for r in self.data['requirements'] if r['kind']=='output_completion']
        self.assertTrue(all(not r['satisfied'] for r in g.verify(rs)))

    def test_internal_stream_count_does_not_decide_concurrency(self):
        checks=self.rebuild().verify(self.data['cross_task_order'])
        self.assertTrue(all(r['satisfied']==r['expected'] for r in checks))
        for t in self.data['summary']['trials']:
            if t['backend']=='graph' and t['mode']=='serial':
                self.assertEqual(len(t['streams']),2);self.assertEqual(len(t['caller_streams']),1);self.assertEqual(t['overlap_us'],0)

    def test_compute_overlap_is_union_not_envelope(self):
        a=[(number(0),number(1)),(number(9),number(10))];b=[(number(2),number(8))]
        self.assertEqual(overlap(a,b),0);self.assertEqual(overlap(a+a,[(number(0),number(10))]),2)


if __name__=='__main__':unittest.main()
