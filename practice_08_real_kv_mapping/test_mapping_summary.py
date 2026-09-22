"""Validate a real archive and reject deliberately corrupted evidence."""

import csv
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from summarize_mapping import analyze


ARCHIVE = Path(__file__).resolve().parent / "results/2026-09-22-run01"


class MappingSummaryTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        shutil.copytree(str(ARCHIVE / "events"), str(self.root / "events"))
        for name in ("request.json", "response.json"):
            shutil.copyfile(str(ARCHIVE / name), str(self.root / name))

    def change(self, event_name, step, modify):
        for path in (self.root / "events").glob("*.jsonl"):
            events = [json.loads(line) for line in path.read_text().splitlines()]
            for event in events:
                if event["event"] == event_name and event.get("step") == step:
                    modify(event)
            path.write_text("".join(json.dumps(event) + "\n" for event in events))

    def test_real_cross_block_mapping(self):
        summary, table = analyze(self.root)
        rows = list(csv.DictReader(io.StringIO(table)))
        self.assertEqual(len(rows), 133)
        self.assertEqual(rows[127]["offset"], "127")
        self.assertEqual(rows[128]["offset"], "0")
        self.assertNotEqual(rows[127]["physical_block"], rows[128]["physical_block"])
        self.assertIn("126 / 8", summary)

    def test_wrong_device_slot_is_rejected(self):
        self.change("attention_kv_metadata", 4, lambda row: row["slot_mapping"].__setitem__(0, 255))
        with self.assertRaisesRegex(ValueError, "attention slot mismatch"):
            analyze(self.root)

    def test_wrong_cpu_table_is_rejected(self):
        self.change("runner_block_table", 4,
                    lambda row: row["groups"][0]["rows"][0].__setitem__(1, 999))
        with self.assertRaisesRegex(ValueError, "CPU block table mismatch"):
            analyze(self.root)

    def test_wrong_operator_slot_is_rejected(self):
        self.change("kv_write_call", 4, lambda row: row["slot_mapping"].__setitem__(0, 999))
        with self.assertRaisesRegex(ValueError, "write slot mismatch"):
            analyze(self.root)

    def test_wrong_actual_cache_data_is_rejected(self):
        self.change("kv_write_check", 4, lambda row: row.update(key_equal=False))
        with self.assertRaisesRegex(ValueError, "actual KV write verification failed"):
            analyze(self.root)

    def test_missing_observation_is_rejected(self):
        self.change("kv_write_check", 4, lambda row: row.update(event="removed"))
        with self.assertRaisesRegex(ValueError, "incomplete kv_write_check"):
            analyze(self.root)


if __name__ == "__main__":
    unittest.main()
