"""Check that precondition counting stays boolean and protected stays call-free."""
import json
from pathlib import Path
import tempfile
import unittest
from analyze import analyze


def build(temporary, *, protected_calls=0, precondition=0, unsupported=False):
    root = Path(temporary) / "run"
    for case in ("baseline-default", "baseline-async", "baseline-protected"):
        (root / case).mkdir(parents=True)
    cases = []
    for case, variant in (("baseline-default", "default"), ("baseline-async", "async"),
                          ("baseline-protected", "protected")):
        code = 2 if (unsupported and variant == "async") else 0
        cases.append(dict(case=case, returncode=code))
        directory = root / case
        if code == 2:
            (directory / "engine_error.json").write_text(json.dumps(
                dict(variant=variant, error="TypeError: async_scheduling")))
            continue
        records = [dict(index=0, rows=4, address=111, reused_prev=None, prev_div_pending=None)]
        for i in range(1, 4):
            records.append(dict(index=i, rows=4, address=111, reused_prev=True,
                                prev_div_pending=(i <= precondition), prev_gap_ms=10.0 + i))
        if variant == "protected":
            records = records[:protected_calls]
        (directory / f"records-{7000 + len(records)}.json").write_text(json.dumps(
            dict(pid=7000 + len(records), calls=len(records), records=records,
                 original_source=dict(file="sampler.py", text="def...", sha256="deadbeef"))))
        (directory / f"records-9999.json").write_text(json.dumps(
            dict(pid=9999, calls=0, records=[],
                 original_source=dict(file="sampler.py", text="def...", sha256="deadbeef"))))
        (directory / "generation.json").write_text(json.dumps(
            dict(identical_across_repeats=True, generations=[])))
    (root / "cases.json").write_text(json.dumps(cases))
    return root


class AnalysisTests(unittest.TestCase):
    def test_precondition_counting(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = analyze(build(temporary, precondition=2))
            case = next(c for c in result["cases"] if c["variant"] == "default")
            self.assertEqual(case["address_reuse"], 3)
            self.assertEqual(case["prev_div_pending_while_reused"], 2)
            self.assertEqual(case["precondition_indices"], [1, 2])

    def test_protected_must_not_call_random_sample(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "flag not effective"):
                analyze(build(temporary, protected_calls=2))

    def test_unsupported_variant_is_recorded_not_fatal(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = analyze(build(temporary, unsupported=True))
            case = next(c for c in result["cases"] if c["variant"] == "async")
            self.assertEqual(case["status"], "unsupported")


if __name__ == "__main__":
    unittest.main()
