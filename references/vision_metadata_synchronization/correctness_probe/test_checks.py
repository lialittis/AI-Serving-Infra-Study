import copy
import unittest
from analyze import ROOT,read,validate_row
from spec import checks,grid_lengths,prefix,MAX32


class ChecksTests(unittest.TestCase):
    def setUp(self):
        self.rows={r['case']['id']:r for r in read(ROOT/'results/run-r01/cases.json')}

    def test_all_recorded_rows_recompute(self):
        for row in self.rows.values():validate_row(row)

    def test_same_sum_requires_elementwise_reference(self):
        r=self.rows['same_sum'];self.assertTrue(r['checks']['sum']);self.assertTrue(r['checks']['positive'])
        self.assertFalse(r['checks']['exact']);self.assertEqual(r['split']['status'],'accepted')

    def test_narrowing_can_preserve_correct_difference(self):
        r=self.rows['overflow_cast'];self.assertTrue(r['checks']['exact'])
        self.assertFalse(r['checks']['pre_int32']);self.assertFalse(r['checks']['device_boundary_exact'])

    def test_wrong_origin_not_detectable_from_differences(self):
        r=self.rows['wrong_origin'];self.assertTrue(r['checks']['exact']);self.assertFalse(r['checks']['pre_origin'])

    def test_zero_length_accepted_by_split(self):
        r=self.rows['duplicate'];self.assertEqual(r['split']['status'],'accepted');self.assertFalse(r['checks']['positive'])

    def test_exact_python_sum_does_not_wrap(self):
        self.assertEqual(prefix([MAX32,2]),[0,MAX32,MAX32+2])

    def test_edge_window_reference(self):
        self.assertEqual(grid_lengths([[1,10,10]]),[64,16,16,4])
        self.assertEqual(grid_lengths([[2,10,10]],full=True),[100,100])

    def test_forged_offset_rejected(self):
        r=copy.deepcopy(self.rows['offset']);r['layout']['left']['offset']-=1
        with self.assertRaises(AssertionError):validate_row(r)

    def test_corrupted_guard_rejected(self):
        r=copy.deepcopy(self.rows['offset']);r['parent_after'][0]+=1
        with self.assertRaises(AssertionError):validate_row(r)

    def test_guard_already_corrupt_before_sub(self):
        r=copy.deepcopy(self.rows['offset']);r['parent_before'][0]+=1;r['parent_after'][0]+=1
        with self.assertRaises(AssertionError):validate_row(r)

    def test_effective_pointer_mismatch(self):
        r=copy.deepcopy(self.rows['baseline']);r['layout']['output']['ptr']+=4
        with self.assertRaises(AssertionError):validate_row(r)

    def test_computed_value_before_injection_corrupted(self):
        r=copy.deepcopy(self.rows['same_sum']);r['computed_before_injection'][0]+=1
        with self.assertRaises(AssertionError):validate_row(r)

    def test_output_alias_rejected(self):
        r=copy.deepcopy(self.rows['baseline']);r['layout']['output']['base']=r['layout']['parent']['base']
        with self.assertRaises(AssertionError):validate_row(r)

    def test_forged_verdict_rejected(self):
        r=copy.deepcopy(self.rows['same_sum']);r['checks']['exact']=True
        with self.assertRaises(AssertionError):validate_row(r)

    def test_split_chunk_boundary_corruption_rejected(self):
        r=copy.deepcopy(self.rows['baseline']);r['split']['chunks'][0][0]=999
        with self.assertRaises(AssertionError):validate_row(r)


if __name__=='__main__':unittest.main()
