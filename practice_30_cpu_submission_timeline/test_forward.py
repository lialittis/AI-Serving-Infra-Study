"""Reject CPU attribution that duplicates nesting, threads or crossing scopes."""
import unittest
from forward_analysis import forest


class ForwardTests(unittest.TestCase):
    def event(self,name,start,duration,tid=1):
        return dict(id=name,name=name,ts=str(start),dur=str(duration),tid=tid)

    def test_parent_children_not_double_counted(self):
        tree=forest([self.event('attention',10,60),self.event('copy',20,10)],0,100,1)
        self.assertEqual(tree['covered_us'],'60')
        self.assertEqual(tree['unattributed_us'],'40')
        self.assertEqual(tree['nodes'][0]['self_us'],'50')

    def test_crossing_is_not_nesting(self):
        with self.assertRaisesRegex(ValueError,'overlapping'):
            forest([self.event('a',10,50),self.event('b',20,60)],0,100,1)

    def test_other_thread_must_not_be_added(self):
        with self.assertRaisesRegex(ValueError,'thread'):
            forest([self.event('worker',10,20,tid=2)],0,100,1)

    def test_wrong_boundary_rejected(self):
        with self.assertRaisesRegex(ValueError,'outside'):
            forest([self.event('outside',90,20)],0,100,1)

    def test_absolute_microsecond_precision_is_preserved(self):
        start='1790776278328015.424'
        from decimal import Decimal as D
        tree=forest([self.event('event',start,'.001')],start,D(start)+D('.003'),1)
        self.assertEqual(tree['covered_us'],'0.001')
        self.assertEqual(tree['unattributed_us'],'0.002')


if __name__=='__main__':unittest.main()
