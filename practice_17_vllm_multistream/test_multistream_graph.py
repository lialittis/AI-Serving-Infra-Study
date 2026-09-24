import json
from pathlib import Path
import unittest

from analyze_run import analyze
from verify_results import verify


ROOT = Path(__file__).resolve().parent / "results"
QWEN_SEEDED = ROOT / "2026-09-24-qwen-enabled-b32-r02"
QWEN_ASYNC = ROOT / "2026-09-24-qwen-enabled-b32-r03"
QWEN_INLINE = ROOT / "2026-09-24-qwen-disabled-b32-r02"
LLAMA_B32 = ROOT / "2026-09-24-llama-enabled-b32"
LLAMA_ASYNC = ROOT / "2026-09-24-llama-enabled-b64"
LLAMA_INLINE = ROOT / "2026-09-24-llama-disabled-b64"
RUNS = (QWEN_SEEDED, QWEN_ASYNC, QWEN_INLINE, LLAMA_B32, LLAMA_ASYNC, LLAMA_INLINE)


def summary(path):
    return json.loads((path / "analysis" / "summary.json").read_text())["summary"]


class PublishedSummaryTests(unittest.TestCase):
    def test_all_device_tasks_are_reconciled(self):
        for path in RUNS:
            result = summary(path)
            self.assertEqual(result["physical_streams"], ["44", "46"])
            self.assertEqual(result["kernel_csv_rows_verified"], result["compute_tasks"])
            self.assertEqual(result["outside_scope_device_tasks"], 0)

    def test_qwen_trigger_and_matched_control(self):
        seeded, enabled, disabled = map(summary, (QWEN_SEEDED, QWEN_ASYNC, QWEN_INLINE))
        self.assertEqual(seeded["overlap_steps"], 0)
        self.assertEqual(enabled["overlap_steps"], 4)
        self.assertEqual(enabled["total_model_random_overlap_us"], "105.385")
        self.assertEqual(disabled["overlap_steps"], 0)
        self.assertEqual(enabled["compute_tasks"], disabled["compute_tasks"])

    def test_llama_batch_threshold_and_submission_modes(self):
        threshold, enabled, disabled = map(summary, (LLAMA_B32, LLAMA_ASYNC, LLAMA_INLINE))
        self.assertEqual(threshold["overlap_steps"], 0)
        self.assertEqual(enabled["overlap_steps"], 4)
        self.assertEqual(enabled["total_model_random_overlap_us"], "1082.181")
        self.assertEqual(disabled["overlap_steps"], 1)
        self.assertEqual(disabled["total_model_random_overlap_us"], "1052.263")
        self.assertEqual(enabled["edges_by_kind"]["event_sync"], 5)
        self.assertEqual(disabled["edges_by_kind"]["event_wait"], 5)

    def test_request_seed_changes_branch_shape(self):
        comparison = json.loads((ROOT / "comparison.json").read_text())["runs"]
        seeded = comparison["qwen_seeded_calibration"]["branch_observations"]
        enabled = comparison["qwen_async"]["branch_observations"]
        self.assertEqual([row["generator_count"] for row in seeded], [1, 32, 32, 32, 31])
        self.assertEqual([row["generator_count"] for row in enabled], [0, 0, 0, 0, 0])


@unittest.skipUnless(all((path / "artifact_manifest.json").exists() for path in RUNS),
                         "full private evidence bundle is not present")
class FullEvidenceReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = {path.name: analyze(path) for path in RUNS}

    def test_precomputed_q_identity_is_recorded(self):
        for path in (QWEN_ASYNC, LLAMA_ASYNC):
            result = self.results[path.name]
            entered = {(row["step"], row["kind"]): row for row in result["observations"]}
            returned = {(row["step"], row["kind"]): row for row in result["returns"]}
            for step in range(1, 6):
                produced = returned[step, "async_exponential"]
                consumed = entered[step, "sampler"]
                self.assertEqual(produced["q"], consumed["precomputed_q"])
                self.assertEqual(produced["event_handle"], consumed["async_event_handle"])

    def test_export_manifests_and_offline_replay(self):
        for path in RUNS:
            self.assertEqual(verify(path), summary(path))


if __name__ == "__main__":
    unittest.main()
