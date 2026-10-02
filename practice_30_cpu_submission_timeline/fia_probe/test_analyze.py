"""Negative tests prevent a plausible but incorrect binary/timeline association."""
import copy
import json
from pathlib import Path
import unittest

from analyze import verify_case, verify_model


ROOT = Path(__file__).resolve().parent


class AssociationTests(unittest.TestCase):
    def setUp(self):
        self.info = json.loads((ROOT / 'evidence/selection.json').read_text())['cases']['bf16-l42']
        self.rows = json.loads((ROOT / 'results/bf16-l42/api_calls.json').read_text())
        self.result = json.loads((ROOT / 'results/bf16-l42/results.json').read_text())
        self.metadata = json.loads((ROOT / self.info['metadata_local']).read_text())
        self.model = json.loads((ROOT / 'evidence/model_launch.json').read_text())

    def check_case(self):
        return verify_case(self.rows, self.info, self.result, self.metadata)

    def row(self, kind):
        return next(r for r in self.rows if r['kind'] == kind)

    def test_valid(self):
        self.check_case()
        verify_model(self.model)

    def test_loaded_bytes_must_match_file(self):
        self.row('aclrtBinaryLoadFromData')['text'] = '0' * 64
        with self.assertRaises(AssertionError): self.check_case()

    def test_function_handle_must_match_launch(self):
        self.row('aclrtLaunchKernelWithHostArgs')['u'][0] += 1
        with self.assertRaises(AssertionError): self.check_case()

    def test_wrong_binary_handle(self):
        self.row('aclrtBinaryGetFunctionByEntry')['u'][0] += 1
        with self.assertRaises(AssertionError): self.check_case()

    def test_executor_must_match_preparation(self):
        self.row('aclnnFusedInferAttentionScoreV3')['u'][2] += 1
        with self.assertRaises(AssertionError): self.check_case()

    def test_duplicate_launch_is_ambiguous(self):
        self.rows.append(copy.deepcopy(self.row('aclrtLaunchKernelWithHostArgs')))
        with self.assertRaises(ValueError): self.check_case()

    def test_same_name_different_connection(self):
        self.model['link']['task']['connection_id'] = '99999'
        with self.assertRaises(AssertionError): verify_model(self.model)

    def test_same_time_different_worker(self):
        self.model['launch']['tid'] += 1
        with self.assertRaises(AssertionError): verify_model(self.model)

    def test_launch_outside_node(self):
        self.model['launch']['dur'] = '9999'
        with self.assertRaises(AssertionError): verify_model(self.model)


if __name__ == '__main__':
    unittest.main()
