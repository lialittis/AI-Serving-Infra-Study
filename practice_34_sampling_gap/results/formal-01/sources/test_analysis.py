"""Check that gap-curve claims stay tied to element checks and mode controls."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from analyze import analyze, summarize_interval


def interval(gap_us=0, corrupt_indices=(0,), iterations=4, elements=1024):
    rows = []
    for index in range(iterations):
        outcome = "overwritten" if index in corrupt_indices else "intact"
        row = dict(index=index, expected=index + 1, address=str(1000 + (0 if index % 2 == 0 else 1)),
                   all_elements_expected=outcome == "intact", outcome=outcome,
                   next_address_reused=None)
        if outcome == "overwritten":
            row["mismatch_count"] = elements if index == corrupt_indices[0] else 10
            row["foreign_values"] = [float(index + 2)]
        rows.append(row)
    for i in range(len(rows) - 1):
        rows[i]["next_address_reused"] = rows[i]["address"] == rows[i + 1]["address"]
    return dict(gap_us=gap_us, rows=rows)


class IntervalTests(unittest.TestCase):
    def test_corrupt_counts_and_full_overwrite_detection(self):
        result = summarize_interval(interval(), 4, 1024)
        self.assertEqual(result["corrupt_iterations"], 1)
        self.assertEqual(result["fully_overwritten_iterations"], 1)
        self.assertEqual(result["first_corrupt_index"], 0)

    def test_intact_interval_has_no_corruption(self):
        result = summarize_interval(interval(corrupt_indices=()), 4, 1024)
        self.assertEqual(result["corrupt_iterations"], 0)
        self.assertIsNone(result["first_corrupt_index"])

    def test_outcome_must_match_element_check(self):
        sample = interval()
        sample["rows"][0]["all_elements_expected"] = True
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            summarize_interval(sample, 4, 1024)


def build_run(temporary, mode_corrupt=None):
    root = Path(temporary) / "run"
    for case in ("baseline-asis-b0", "baseline-asis-b8", "baseline-record-b0"):
        (root / case).mkdir(parents=True)
    plan = dict(arguments=dict(configs=["baseline"], modes=["asis", "record"], backlogs=[0, 8],
                               gaps_us=[0, 1000], iterations=4, elements=1024))
    cases = [dict(case="baseline-asis-b0", returncode=0),
             dict(case="baseline-asis-b8", returncode=0),
             dict(case="baseline-record-b0", returncode=0)]
    (root / "cases.json").write_text(json.dumps(cases))
    (root / "plan.json").write_text(json.dumps(plan))
    for case, mode, backlog in (("baseline-asis-b0", "asis", 0),
                                ("baseline-asis-b8", "asis", 8),
                                ("baseline-record-b0", "record", 0)):
        metadata = dict(
            arguments=dict(mode=mode, consumer_backlog=backlog, iterations=4, elements=1024,
                           gaps_us=[0, 1000]),
            allocator_backend="native",
            environment={}, intervals=[interval(0, (0,) if mode == "asis" and mode_corrupt != "none" else ()),
                                       interval(1000, ())],
            statistics_before=dict(num_alloc_retries=0, num_ooms=0),
            statistics_after=dict(num_alloc_retries=0, num_ooms=0),
            trace_entries_for_first_address=[dict(action="alloc"), dict(action="free_requested")],
            source_sha256={})
        (root / case / "run.json").write_text(json.dumps(metadata))
    return root


class CaseTests(unittest.TestCase):
    def test_asis_corruption_is_reported_without_failing(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = analyze(build_run(temporary))
            asis = next(case for case in result["cases"] if case["case"] == "asis-b0")
            self.assertEqual(asis["total_corrupt"], 1)
            self.assertEqual(asis["consumer_backlog"], 0)

    def test_record_corruption_is_a_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = build_run(temporary)
            metadata = json.loads((root / "baseline-record-b0" / "run.json").read_text())
            metadata["intervals"][0] = interval(0, (0,))
            (root / "baseline-record-b0" / "run.json").write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "record protection"):
                analyze(root)

    def test_gap_coverage_must_match_plan(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = build_run(temporary)
            metadata = json.loads((root / "baseline-asis-b0" / "run.json").read_text())
            metadata["intervals"] = metadata["intervals"][:1]
            (root / "baseline-asis-b0" / "run.json").write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "gap coverage"):
                analyze(root)


if __name__ == "__main__":
    unittest.main()
