"""Check evidence validation against a real trace and deliberately damaged copies."""

import json
from pathlib import Path
import shutil
import tempfile
import unittest

from summarize_trace import summarize


ARCHIVE = Path(__file__).resolve().parent / "results/2026-09-20-run02"


class TraceSummaryTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.run = Path(directory.name)
        shutil.copytree(str(ARCHIVE / "events"), str(self.run / "events"))
        for name in ("request.json", "response.json"):
            shutil.copyfile(str(ARCHIVE / name), str(self.run / name))

    def edit_events(self, transform):
        for path in (self.run / "events").glob("*.jsonl"):
            events = [json.loads(line) for line in path.read_text().splitlines()]
            path.write_text("".join(json.dumps(row) + "\n" for row in transform(events)))

    def test_real_archive_passes(self):
        result = summarize(self.run)
        self.assertIn("输入 / 输出 token：5 / 8", result)
        self.assertIn("UniProcExecutor", result)

    def test_missing_attention_evidence_is_rejected(self):
        self.edit_events(lambda events: [row for row in events
            if not (row["event"] == "attention_first_layer" and row["step"] == 4)])
        with self.assertRaisesRegex(ValueError, "does not cover every scheduled step"):
            summarize(self.run)

    def test_wrong_step_association_is_rejected(self):
        def change(events):
            for row in events:
                if row["event"] == "runner_call" and row["step"] == 4:
                    row["step"] = 3
            return events
        self.edit_events(change)
        with self.assertRaisesRegex(ValueError, "step correlation"):
            summarize(self.run)

    def test_mismatched_http_count_is_rejected(self):
        path = self.run / "response.json"
        response = json.loads(path.read_text())
        response["usage"]["completion_tokens"] = 7
        path.write_text(json.dumps(response))
        with self.assertRaisesRegex(ValueError, "unexpected output token count"):
            summarize(self.run)

    def test_instrumentation_errors_are_not_silently_accepted(self):
        self.edit_events(lambda rows: rows + [{"event": "trace_error", "monotonic_ns": 0,
                                              "error": "missing source attribute"}])
        with self.assertRaisesRegex(ValueError, "instrumentation errors"):
            summarize(self.run)


if __name__ == "__main__":
    unittest.main()
