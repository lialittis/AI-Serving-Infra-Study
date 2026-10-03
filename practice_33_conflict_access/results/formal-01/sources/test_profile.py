import json
from pathlib import Path
import tempfile
import unittest
from analyze_profile import analyze


class FlowEvidenceTests(unittest.TestCase):
    def fixture(self, directory, write_first=True, ambiguous=False):
        labels = ("P33/t00/consumer-copy", "P33/t00/write-candidate", "P33/t00/create-a")
        meta = dict(arguments=dict(mode="omit"),
                    records=[dict(label=labels[0], operation="consumer-copy", trial="t00"),
                             dict(label=labels[1], operation="write-candidate", trial="t00"),
                             dict(label=labels[2], operation="create-a", trial="t00")],
                    trials=[dict(id="t00", observations=[dict(index=0, reused=True)],
                                 write_address_reused=True, observed_outcome="fully_overwritten",
                                 observed_sentinel_count=8)])
        copy_ts, write_ts = (500, 100) if write_first else (100, 500)
        events = [dict(ph="X", pid=1, tid=1, ts=10, dur=5, name=labels[2]),
                  dict(ph="X", pid=1, tid=1, ts=60, dur=8, name=labels[0]),
                  dict(ph="X", pid=1, tid=1, ts=80, dur=8, name=labels[1]),
                  dict(ph="X", pid=1, tid=1, ts=62, dur=6, name="aten::copy_"),
                  dict(ph="X", pid=1, tid=1, ts=82, dur=6, name="aten::fill_"),
                  dict(ph="X", pid=1, tid=2, ts=300, dur=10, name="AscendCL@aclnnInplaceCopy", args=dict(connection_id=7)),
                  dict(ph="X", pid=1, tid=2, ts=150, dur=10, name="AscendCL@aclnnInplaceFill", args=dict(connection_id=8)),
                  dict(ph="X", pid=2, tid=3, ts=copy_ts, dur=30, name="inplace_copy",
                       args={"Task Type": "AI_VECTOR_CORE", "Physic Stream Id": 3, "Task Id": 9, "connection_id": 7}),
                  dict(ph="X", pid=2, tid=3, ts=write_ts, dur=30, name="fill",
                       args={"Task Type": "AI_VECTOR_CORE", "Physic Stream Id": 2, "Task Id": 10, "connection_id": 8}),
                  dict(ph="s", pid=1, tid=1, ts=62, cat="async_npu", id=11),
                  dict(ph="f", pid=2, tid=3, ts=copy_ts, cat="async_npu", id=11),
                  dict(ph="s", pid=1, tid=1, ts=82, cat="async_npu", id=13),
                  dict(ph="f", pid=2, tid=3, ts=write_ts, cat="async_npu", id=13),
                  dict(ph="s", pid=1, tid=2, ts=300, cat="HostToDevice", id=12),
                  dict(ph="f", pid=2, tid=3, ts=copy_ts, cat="HostToDevice", id=12),
                  dict(ph="s", pid=1, tid=2, ts=150, cat="HostToDevice", id=14),
                  dict(ph="f", pid=2, tid=3, ts=write_ts, cat="HostToDevice", id=14)]
        if ambiguous:
            events.append(dict(ph="s", pid=1, tid=1, ts=63, cat="async_npu", id=11))
        (directory / "run.json").write_text(json.dumps(meta))
        (directory / "trace_view.json").write_text(json.dumps(events))

    def test_write_before_copy_is_reported_with_outcome(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.fixture(directory, write_first=True)
            result = analyze(directory)
            trial = result["trials"][0]
            self.assertEqual(trial["device_order"], "write_completed_before_copy_started")
            self.assertEqual(trial["observed_outcome"], "fully_overwritten")
            self.assertEqual(len(trial["write_tasks"]), 1)
            self.assertEqual(trial["write_tasks"][0]["connection_id"], "8")
            self.assertIn("element", result["limit"])

    def test_copy_before_write_order_is_distinguished(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.fixture(directory, write_first=False)
            trial = analyze(directory)["trials"][0]
            self.assertEqual(trial["device_order"], "copy_completed_before_write_started")

    def test_missing_write_tasks_are_an_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.fixture(directory)
            meta = json.loads((directory / "run.json").read_text())
            meta["records"] = [r for r in meta["records"] if r["operation"] != "write-candidate"]
            (directory / "run.json").write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, "No device write task"):
                analyze(directory)

    def test_ambiguous_flow_must_not_use_nearest_time(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.fixture(directory, ambiguous=True)
            with self.assertRaisesRegex(ValueError, "No device copy task"):
                analyze(directory)


if __name__ == "__main__":
    unittest.main()
