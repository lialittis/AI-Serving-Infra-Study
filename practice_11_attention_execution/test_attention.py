"""Real archived evidence plus corruption tests; no NPU required."""
import copy
from pathlib import Path
import unittest

from analyze_attention import analyze, load, read_trace
import json

RUN = Path(__file__).parent / "results/2026-09-23-run02"


class AttentionEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trace = read_trace(next((RUN / "profiler").rglob("trace_view.json")))
        cls.records = [json.loads(l) for p in (RUN / "events").glob("*.jsonl") for l in p.read_text().splitlines()]

    def test_real_storage_and_kernel_chains(self):
        result, excerpt, replay = analyze(RUN, self.trace, self.records)
        self.assertEqual(len(result["operator_chains"]), 5)
        self.assertEqual(result["scopes"], 23)
        self.assertEqual([p["consumer_mode"] for p in result["phases"]], ["NONE", "PIECEWISE"])
        self.assertTrue(all(p["output_alias_verified"] for p in result["phases"]))
        self.assertTrue(excerpt)
        self.assertEqual(len(replay), 244)

    def test_wrong_decode_output_storage_is_rejected(self):
        records = copy.deepcopy(self.records)
        e = next(e for e in records if e.get("label") == "P11/step=4/acl/submod_2" and e["event"] == "scope_enter")
        e["argument_mapping"]["output_2"]["storage_ptr"] += 512
        with self.assertRaisesRegex(ValueError, "output/post-partition storage mismatch"):
            analyze(RUN, self.trace, records)

    def test_decode_using_current_key_instead_of_cache_is_rejected(self):
        records = copy.deepcopy(self.records)
        fx = next(e for e in records if e.get("label") == "P11/step=4/fx_attention" and e["event"] == "scope_enter")
        fia = next(e for e in records if e["event"] == "fia_inputs" and e["step"] == 4)
        fia["key"] = copy.deepcopy(fx["tensors"]["key"])
        with self.assertRaisesRegex(ValueError, "decode FIA must read cached key"):
            analyze(RUN, self.trace, records)

    def test_broken_torch_flow_is_rejected(self):
        events = copy.deepcopy(self.trace)
        for e in events:
            if e.get("cat") == "async_npu" and e.get("ph") == "f":
                e["id"] = "broken-" + str(e["id"])
        with self.assertRaisesRegex(ValueError, "torch flow end"):
            analyze(RUN, events, self.records)

    def test_missing_cann_flow_is_rejected(self):
        events = [e for e in self.trace if e.get("cat") != "HostToDevice"]
        with self.assertRaisesRegex(ValueError, "CANN device endpoint"):
            analyze(RUN, events, self.records)

    def test_wrong_kernel_task_is_rejected(self):
        events = copy.deepcopy(self.trace)
        e = next(e for e in events if e.get("name") == "ReshapeAndCacheNdKernel" and e.get("ph") == "X")
        e["args"]["Task Id"] = -1
        with self.assertRaisesRegex(ValueError, "kernel CSV match"):
            analyze(RUN, events, self.records)

    def test_python_scope_missing_is_rejected(self):
        events = [e for e in self.trace if e.get("name") != "P11/step=4/acl/submod_2"]
        with self.assertRaisesRegex(ValueError, "matching Python/profiler scopes"):
            analyze(RUN, events, self.records)

    def test_replay_kernels_are_not_given_invented_fx_mapping(self):
        result, _, replay = analyze(RUN, self.trace, self.records)
        self.assertEqual(result["replayed_tasks_without_torch_flow"], len(replay))
        self.assertEqual(result["replayed_tasks_without_cann_start"], len(replay))
        self.assertTrue(all(r["fx_node_attribution"] == "not established" for r in replay))


if __name__ == "__main__":
    unittest.main()
