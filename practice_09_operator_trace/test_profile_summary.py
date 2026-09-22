"""Reject broken evidence instead of inferring device execution from host names."""
import copy
from pathlib import Path
import unittest

from summarize_profile import analyze, read_trace


RUN = Path(__file__).resolve().parent / "results/2026-09-22-run01"


class ProfileSummaryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trace = read_trace(next(RUN.rglob("trace_view.json")))

    def setUp(self):
        self.events = copy.deepcopy(self.trace)

    def test_real_prefill_decode_chains(self):
        result, excerpt = analyze(RUN, self.events)
        self.assertEqual(len(result["examples"]), 4)
        self.assertEqual(result["device_task_counts"]["ReshapeAndCacheNdKernel"], 48)
        self.assertGreater(float(result["examples"][0]["kernel_start_after_python_return_us"]), 0)
        self.assertTrue(any(e.get("cat") == "HostToDevice" for e in excerpt))

    def test_missing_device_events_rejected(self):
        self.events = [e for e in self.events if "Task Type" not in e.get("args", {})]
        with self.assertRaisesRegex(ValueError, "missing device events"):
            analyze(RUN, self.events)

    def test_broken_torch_flow_id_rejected(self):
        for event in self.events:
            if event.get("cat") == "async_npu" and event.get("ph") == "f":
                event["id"] = "broken-" + str(event["id"])
        with self.assertRaisesRegex(ValueError, "torch flow end"):
            analyze(RUN, self.events)

    def test_missing_cann_flows_rejected(self):
        self.events = [e for e in self.events if e.get("cat") != "HostToDevice"]
        with self.assertRaisesRegex(ValueError, "CANN device endpoint"):
            analyze(RUN, self.events)

    def test_wrong_kernel_task_id_rejected(self):
        event = next(e for e in self.events if e.get("name") == "ReshapeAndCacheNdKernel"
                     and e.get("ph") == "X")
        event["args"]["Task Id"] = -123
        with self.assertRaisesRegex(ValueError, "kernel CSV match"):
            analyze(RUN, self.events)

    def test_wrong_device_duration_rejected(self):
        event = next(e for e in self.events if e.get("name") == "ReshapeAndCacheNdKernel"
                     and e.get("ph") == "X")
        event["dur"] += 100
        with self.assertRaisesRegex(ValueError, "kernel duration mismatch"):
            analyze(RUN, self.events)

    def test_missing_python_scope_rejected(self):
        event = next(e for e in self.events if e.get("name", "").startswith("P09/"))
        self.events.remove(event)
        with self.assertRaisesRegex(ValueError, "matching Python/profiler scopes"):
            analyze(RUN, self.events)


if __name__ == "__main__":
    unittest.main()
