"""Validate full-trace coverage and reject broken attribution evidence."""
import copy
import csv
from pathlib import Path
import unittest

from analyze_all_ops import audit
from summarize_profile import read_trace


RUN = Path(__file__).resolve().parent / "results/2026-09-22-run01"


class AllOperatorsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trace = read_trace(next((RUN / "profiler").rglob("trace_view.json")))
        with next((RUN / "profiler").rglob("kernel_details.csv")).open() as stream:
            cls.kernels = list(csv.DictReader(stream))

    def setUp(self):
        self.events = copy.deepcopy(self.trace)

    def test_complete_device_and_host_coverage(self):
        summary, rows, hosts, inventory, examples, queues = audit(self.events, self.kernels)
        self.assertEqual(summary["device_task_count"], 772)
        self.assertEqual(summary["correlated_device_tasks"], 770)
        self.assertEqual(summary["kernel_csv_rows_verified"], 687)
        self.assertEqual(summary["kernel_name_count"], 33)
        self.assertEqual(summary["queue_pairs_verified"], 748)
        self.assertEqual(sum(summary["route_counts"].values()), len(rows))
        self.assertEqual(summary["route_counts"]["triton"], 55)
        self.assertEqual(len(examples), len(inventory))
        view = next(e for e in hosts if e["name"] == "aten::view")
        self.assertEqual(view["unique_descendant_device_tasks"], 0)
        matmul = next(e for e in rows if e["kernel"] == "aclnnAddmm_MatMulCommon_MatMulV2")
        self.assertIn("vllm::unquantized_gemm", matmul["host_ancestry"])
        self.assertEqual(matmul["route"], "aten_to_aclnn")

    def test_missing_triton_flow_fails(self):
        task = next(e for e in self.events if e.get("ph") == "X" and e.get("name") == "_triton_rope"
                    and "Task Type" in e.get("args", {}))
        self.events = [e for e in self.events if not (e.get("ph") == "f" and e.get("cat") == "async_npu"
                       and e.get("pid") == task["pid"] and e.get("ts") == task["ts"])]
        with self.assertRaisesRegex(ValueError, "async_npu device endpoint"):
            audit(self.events, self.kernels)

    def test_wrong_cann_connection_fails(self):
        task = next(e for e in self.events if e.get("ph") == "X" and e.get("name") == "_triton_rope"
                    and "Task Type" in e.get("args", {}))
        task["args"]["connection_id"] = -1
        with self.assertRaisesRegex(ValueError, "CANN connection ID"):
            audit(self.events, self.kernels)

    def test_missing_device_kernel_fails_csv_coverage(self):
        task = next(e for e in self.events if e.get("name") == "SwiGlu" and e.get("ph") == "X")
        self.events.remove(task)
        with self.assertRaisesRegex(ValueError, "unmatched kernel CSV"):
            audit(self.events, self.kernels)

    def test_orphan_queue_flow_fails(self):
        event = next(e for e in self.events if e.get("cat") == "async_task_queue" and e.get("ph") == "f")
        self.events.remove(event)
        with self.assertRaisesRegex(ValueError, "unpaired queue flows"):
            audit(self.events, self.kernels)

    def test_wrong_queue_correlation_fails(self):
        event = next(e for e in self.events if e.get("cat") == "dequeue")
        event["args"]["correlation_id"] = -1
        with self.assertRaisesRegex(ValueError, "queue correlation ID"):
            audit(self.events, self.kernels)


if __name__ == "__main__":
    unittest.main()
