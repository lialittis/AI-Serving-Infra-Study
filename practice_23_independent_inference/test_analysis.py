"""Captured-graph counterfactuals: removing readiness or joins must break proofs."""
import gzip,json,unittest
from pathlib import Path
from trace_graph import Graph,intersection,number
from analyze import duration


class DependencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p=Path(__file__).with_name('results')/'published/execution_graph.json.gz'
        with gzip.open(p,'rt') as f:cls.data=json.load(f)
        cls.nodes={n['id']:n for n in cls.data['nodes']}
        cls.trial=next(t for t in cls.data['summary']['trials'] if t['mode']=='parallel')

    def rebuild(self,drop=lambda e:False):
        g=Graph()
        for n in self.data['nodes']:g.node(n['id'],**{k:v for k,v in n.items() if k!='id'})
        for e in self.data['edges']:
            if not drop(e):g.edge(**e)
        return g

    def test_requirements_do_not_self_prove(self):
        g=self.rebuild()
        self.assertTrue(all(r['satisfied'] for r in g.verify(self.data['requirements'])))
        self.assertFalse(any(e['kind'] in ('input_readiness','output_completion') for e in g.edges))

    def test_missing_origin_wait_breaks_readiness(self):
        g=self.rebuild(lambda e:e['kind'] in ('event_wait','event_wait_api') and self.nodes[e['target']].get('trial')==self.trial['id'])
        rs=[r for r in self.data['requirements'] if r['trial']==self.trial['id'] and r['kind']=='input_readiness']
        self.assertEqual(len(rs),2)
        self.assertTrue(all(not r['satisfied'] for r in g.verify(rs)))

    def test_missing_host_join_breaks_output_completion(self):
        g=self.rebuild(lambda e:e['kind']=='host_sync')
        rs=[r for r in self.data['requirements'] if r['kind']=='output_completion']
        self.assertTrue(all(not r['satisfied'] for r in g.verify(rs)))

    def test_serial_orders_independent_tasks_parallel_does_not(self):
        checks=self.rebuild().verify(self.data['cross_task_order'])
        self.assertTrue(all(r['satisfied']==r['expected'] for r in checks))
        self.assertTrue(any(not r['satisfied'] for r in checks))

    def test_failed_batch_is_not_reported_as_valid_speedup(self):
        s=self.data['summary']
        self.assertFalse(s['all_comparisons_passed'])
        self.assertEqual(s['invalid_cells'],['prefill-1024/batch'])
        case=next(r for r in s['performance'] if r['case']=='prefill-1024')
        self.assertFalse(case['modes']['batch']['numerically_valid'])
        self.assertTrue(case['modes']['parallel']['numerically_valid'])

    def test_compute_overlap_not_envelopes_or_double_counted(self):
        a=[(number(0),number(1)),(number(9),number(10))];b=[(number(2),number(8))]
        self.assertEqual(duration(intersection(a,b)),0)
        self.assertEqual(duration(intersection(a+a,[(number(0),number(10))])),2)


if __name__=='__main__':unittest.main()
