"""Reject broken causal order or mismatched copy identities in the evidence."""
import json
from pathlib import Path
import unittest
from analyze import validate


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.d=json.loads((Path(__file__).parent/'results/instrument-r02/diagnostic.json').read_text())

    def test_actual(self): validate(self.d)

    def test_missing_wait(self):
        self.d['acl'].pop(0)
        with self.assertRaises(AssertionError): validate(self.d)

    def test_wrong_cpu_destination(self):
        self.d['dispatch'][0]['output']['ptr']+=4
        with self.assertRaises(AssertionError): validate(self.d)

    def test_copy_before_wait_completed(self):
        self.d['acl'][1]['begin_ns']=self.d['acl'][0]['begin_ns']
        with self.assertRaises(AssertionError): validate(self.d)

    def test_other_stream(self):
        self.d['acl'][0]['stream']+=1
        with self.assertRaises(AssertionError): validate(self.d)

    def test_wrong_direction(self):
        self.d['acl'][1]['kind']=1
        with self.assertRaises(AssertionError): validate(self.d)


if __name__=='__main__': unittest.main()
