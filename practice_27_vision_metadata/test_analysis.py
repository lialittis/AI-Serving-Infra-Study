"""Run the P25 counterfactual dependency tests on every P27 variant graph."""
import gzip
import importlib.util
import json
from pathlib import Path
import sys
import unittest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'practice_25_multimodal_overlap'))
spec=importlib.util.spec_from_file_location('p25_dependency_tests',HERE.parent/'practice_25_multimodal_overlap/test_analysis.py')
p25=importlib.util.module_from_spec(spec);spec.loader.exec_module(p25)
from sync_audit import Intervals


class SyncAssociationTests(unittest.TestCase):
    def test_nested_and_disjoint_intervals(self):
        tree=Intervals([(0,dict(ts='100.000',dur='20')),(1,dict(ts='105',dur='5')),(2,dict(ts='121',dur='2'))])
        self.assertEqual([i for i,_ in tree.enclosing(dict(ts='106',dur='2'))],[1,0])
        self.assertEqual(tree.enclosing(dict(ts='120.001',dur='0.001')),[])
        self.assertEqual([i for i,_ in tree.enclosing(dict(ts='109',dur='2'))],[0])

    def test_interval_tree_against_exhaustive_containment(self):
        import random
        rng=random.Random(27)
        rows=[(i,dict(ts=rng.randrange(1000),dur=rng.randrange(1000))) for i in range(400)]
        tree=Intervals(rows)
        for _ in range(400):
            e=dict(ts=rng.randrange(2000),dur=rng.randrange(100))
            expected={i for i,x in rows if x['ts']<=e['ts'] and x['ts']+x['dur']>=e['ts']+e['dur']}
            self.assertEqual({i for i,_ in tree.enclosing(e)},expected)


def load_tests(loader,tests,pattern):
    suite=unittest.TestSuite([loader.loadTestsFromTestCase(SyncAssociationTests)])
    for variant in ('native','lengths','cached'):
        def setup(cls,variant=variant):
            with gzip.open(HERE/'results/published'/f'{variant}-graph.json.gz','rt') as f:cls.data=json.load(f)
            cls.nodes={n['id']:n for n in cls.data['nodes']}
        cls=type(variant.title()+'DependencyTests',(p25.DependencyTests,),dict(setUpClass=classmethod(setup)))
        suite.addTests(loader.loadTestsFromTestCase(cls))
    return suite


if __name__=='__main__':unittest.main()
