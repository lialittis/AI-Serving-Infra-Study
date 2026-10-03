"""Check graph-boundary claims stay tied to element checks and mode controls."""
import json
from pathlib import Path
import tempfile
import unittest
from analyze import analyze, summarize_trial


def trial(mode="external-free-ordered", outcome="fully_consumed_replacement", reused=True,
          elements=1024):
    base, sentinel, other = elements, 0, 0
    if outcome == "fully_consumed_replacement":
        base, sentinel = 0, elements
    elif outcome == "mixed":
        base, sentinel = elements // 2, elements // 2
    elif outcome == "unexpected_values":
        base, sentinel, other = 0, 0, elements
    row = dict(id="t00", mode=mode, elements=elements, base_count=base,
               sentinel_count=sentinel, other_count=other, outcome=outcome,
               replacement_intact=True,
               statistics_before=dict(num_alloc_retries=0, num_ooms=0),
               statistics_after=dict(num_alloc_retries=0, num_ooms=0))
    if mode == "pool-release-pending":
        row = dict(id="t00", mode=mode, outcome=outcome,
                   replacement_reused_static=reused, replacement_intact=True,
                   statistics_before=dict(num_alloc_retries=0, num_ooms=0),
                   statistics_after=dict(num_alloc_retries=0, num_ooms=0))
    else:
        row["replacement_reused_external"] = reused
    return row


class TrialTests(unittest.TestCase):
    def test_consumed_replacement_is_classified(self):
        result = summarize_trial(trial())
        self.assertEqual(result["outcome"], "fully_consumed_replacement")

    def test_keep_alive_corruption_is_a_failure(self):
        with self.assertRaisesRegex(ValueError, "keep-alive"):
            summarize_trial(trial("keep-alive", "fully_consumed_replacement"))

    def test_pool_release_shape(self):
        result = summarize_trial(trial("pool-release-pending", outcome="survived", reused=False))
        self.assertTrue(result["replacement_intact"])
        self.assertFalse(result["replacement_reused_static"])

    def test_counts_must_cover(self):
        row = trial()
        row["base_count"] -= 1
        with self.assertRaisesRegex(ValueError, "cover output"):
            summarize_trial(row)


def build_run(temporary):
    root = Path(temporary) / "run"
    root.mkdir(parents=True)
    plan = dict(arguments=dict(repeats=1))
    (root / "plan.json").write_text(json.dumps(plan))
    cases, rows = [], {}
    for mode in ("keep-alive", "external-free-ordered", "external-free-cross", "pool-release-pending"):
        case = f"baseline-{mode}"
        (root / case).mkdir(parents=True)
        outcome = ("intact" if mode == "keep-alive" else
                   "fully_consumed_replacement" if mode.startswith("external-free") else "survived")
        rows[case] = trial(mode, outcome)
        cases.append(dict(case=case, returncode=0))
        (root / case / "run.json").write_text(json.dumps(
            dict(arguments=dict(mode=mode), allocator_backend="native",
                 source_sha256={}, trials=[rows[case]])))
    (root / "cases.json").write_text(json.dumps(cases))
    return root


class RunTests(unittest.TestCase):
    def test_full_matrix_passes_and_counts_reuse(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = analyze(build_run(temporary))
            by_mode = {case["mode"]: case for case in result["cases"]}
            self.assertEqual(by_mode["external-free-ordered"]["outcomes"],
                             {"fully_consumed_replacement": 1})
            self.assertEqual(by_mode["keep-alive"]["outcomes"], {"intact": 1})
            self.assertEqual(by_mode["pool-release-pending"]["outcomes"], {"survived": 1})

    def test_keep_alive_failure_surfaces_at_case_level(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = build_run(temporary)
            metadata = json.loads((root / "baseline-keep-alive" / "run.json").read_text())
            metadata["trials"][0].update(outcome="fully_consumed_replacement", base_count=0,
                                         sentinel_count=1024)
            (root / "baseline-keep-alive" / "run.json").write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "keep-alive"):
                analyze(root)


if __name__ == "__main__":
    unittest.main()
