"""Counterfactual checks on the captured graph: missing joins must fail HB proofs."""
import json,unittest
from pathlib import Path
from analyze import Graph,intersection,length,number


class DependencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=json.loads(Path(__file__).with_name('results').joinpath('published/execution_graph.json').read_text())
        cls.nodes={n['id']:n for n in cls.data['nodes']}
        cls.trial=next(t for t in cls.data['summary']['trials'] if t['mode']=='parallel')
        cls.main=cls.nodes[cls.trial['stages']['merge'][0]]['stream']
        cls.shared=cls.nodes[cls.trial['stages']['shared_down'][0]]['stream']

    def rebuild(self,drop=lambda e:False):
        g=Graph()
        for n in self.data['nodes']:g.node(n['id'],**{k:v for k,v in n.items() if k!='id'})
        for e in self.data['edges']:
            if not drop(e):g.edge(**e)
        return g

    def test_all_observed_dependencies_have_independent_hb(self):
        self.assertTrue(all(e['satisfied'] for e in self.rebuild().verify(self.data['requirements'])))

    def test_remove_shared_completion_join_breaks_merge(self):
        def drop(e):
            a,b=self.nodes[e['source']],self.nodes[e['target']]
            return e['kind'] in ('event_wait','event_wait_api') and b.get('trial')==self.trial['id'] and a.get('stream')==self.shared and b.get('stream')==self.main
        g=self.rebuild(drop)
        requirements=[e for e in self.data['requirements'] if e['trial']==self.trial['id'] and e['producer']=='shared_down' and e['consumer']=='merge']
        self.assertEqual(len(requirements),1)
        self.assertFalse(g.verify(requirements)[0]['satisfied'])

    def test_remove_input_ready_waits_breaks_shared_input(self):
        def drop(e):
            a,b=self.nodes[e['source']],self.nodes[e['target']]
            return e['kind'] in ('event_wait','event_wait_api') and b.get('trial')==self.trial['id'] and a.get('stream')==self.main and b.get('stream')==self.shared
        g=self.rebuild(drop)
        requirements=[e for e in self.data['requirements'] if e['trial']==self.trial['id'] and e['producer']=='input' and e['consumer']=='shared_up']
        self.assertFalse(g.verify(requirements)[0]['satisfied'])

    def test_api_only_barriers_are_needed_without_inventing_device_tasks(self):
        barriers=[n for n in self.data['nodes'] if n['kind']=='runtime_wait']
        self.assertEqual(len(barriers),2)
        self.assertTrue(all('start_us' not in n for n in barriers))
        trial=barriers[0]['trial']
        g=self.rebuild(lambda e:e['kind']=='event_wait_api')
        requirements=[e for e in self.data['requirements'] if e['trial']==trial and e['producer']=='input' and e['consumer']=='shared_up']
        self.assertFalse(g.verify(requirements)[0]['satisfied'])

    def test_overlap_uses_actual_intervals_not_branch_envelopes(self):
        a=[(number(0),number(1)),(number(9),number(10))]
        b=[(number(2),number(8))]
        self.assertEqual(length(intersection(a,b)),0)
        self.assertEqual(length(intersection(a+a,[(number(0),number(10))])),2)


if __name__=='__main__':unittest.main()
