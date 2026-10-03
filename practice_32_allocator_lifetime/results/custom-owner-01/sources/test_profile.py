import json
from pathlib import Path
import tempfile
import unittest
from analyze_profile import analyze


class FlowEvidenceTests(unittest.TestCase):
    def fixture(self, directory, ambiguous=False):
        labels = ("P32/t00/consumer-copy", "P32/t00/allocate-000")
        meta = dict(arguments=dict(mode="omit"), records=[dict(label=labels[0], operation="consumer-copy", trial="t00"),
                                                        dict(label=labels[1], operation="allocate-000", trial="t00")],
                    trials=[dict(id="t00", observations=[dict(index=0, reused=True, allocation_label=labels[1])])])
        events = [dict(ph="X", pid=1, tid=1, ts=100, dur=10, name=labels[0]),
                  dict(ph="X", pid=1, tid=1, ts=160, dur=10, name=labels[1]),
                  dict(ph="X", pid=1, tid=1, ts=102, dur=6, name="aten::copy_"),
                  dict(ph="X", pid=1, tid=2, ts=200, dur=20, name="AscendCL@aclnnInplaceCopy", args=dict(connection_id=7)),
                  dict(ph="X", pid=2, tid=3, ts=500, dur=30, name="inplace_copy", args={"Task Type": "AI_VECTOR_CORE", "Physic Stream Id": 3, "Task Id": 9, "connection_id": 7}),
                  dict(ph="s", pid=1, tid=1, ts=102, cat="async_npu", id=11),
                  dict(ph="f", pid=2, tid=3, ts=500, cat="async_npu", id=11),
                  dict(ph="s", pid=1, tid=2, ts=200, cat="HostToDevice", id=12),
                  dict(ph="f", pid=2, tid=3, ts=500, cat="HostToDevice", id=12)]
        if ambiguous:
            events.append(dict(ph="s", pid=1, tid=1, ts=103, cat="async_npu", id=11))
        (directory / "run.json").write_text(json.dumps(meta))
        (directory / "trace_view.json").write_text(json.dumps(events))

    def test_exact_flow_proves_reported_order_without_candidate_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.fixture(directory)
            result = analyze(directory)
            before = result["trials"][0]["reused_before_copy_tasks_start"]
            self.assertEqual(before[0]["gap_us"], "330")
            self.assertIn("no candidate device accesses", result["limit"])
            self.assertEqual(result["trials"][0]["copy_tasks"][0]["connection_id"], "7")

    def test_ambiguous_flow_must_not_use_nearest_time(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.fixture(directory, ambiguous=True)
            with self.assertRaisesRegex(ValueError, "No device copy task"):
                analyze(directory)


if __name__ == "__main__":
    unittest.main()
