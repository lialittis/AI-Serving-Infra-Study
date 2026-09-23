"""Integrity checks against the real archived graph; CPU-only, Python 3.7+."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from analyze_graph import attention_effects, load, validate_nodes, validate_run


RUN = Path(__file__).parent / "results/2026-09-23-run02"


class GraphEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary, cls.nodes, cls.effects, cls.parts = validate_run(RUN)

    def test_real_graph_and_request(self):
        self.assertEqual(self.summary["nodes"], 852)
        self.assertEqual(self.summary["partitions"], 49)
        self.assertEqual(self.summary["request_forward_tokens"], [126, 1])
        self.assertEqual(self.summary["response_text"], " syntax,")

    def test_missing_producer_is_rejected(self):
        nodes = copy.deepcopy(self.nodes)
        node = next(n for n in nodes if n["inputs"])
        node["inputs"].append("nonexistent_node")
        with self.assertRaisesRegex(ValueError, "missing producer"):
            validate_nodes(nodes)

    def test_argument_edge_disagreement_is_rejected(self):
        nodes = copy.deepcopy(self.nodes)
        node = next(n for n in nodes if n["inputs"])
        node["args"] = []
        node["kwargs"] = {}
        with self.assertRaisesRegex(ValueError, "argument/edge mismatch"):
            validate_nodes(nodes)

    def test_missing_mutation_schema_is_rejected(self):
        nodes = copy.deepcopy(self.nodes)
        node = next(n for n in nodes if n["target"] == "vllm.unified_attention_with_output")
        node["schema"] = ""
        node["schemas"] = {}
        with self.assertRaisesRegex(ValueError, "mutation schema missing"):
            attention_effects(nodes, {n["name"]: n for n in nodes})

    def test_attention_has_effect_even_without_fx_users(self):
        attention = [n for n in self.nodes if n["target"] == "vllm.unified_attention_with_output"]
        self.assertTrue(all(n["value"] is None and not n["users"] for n in attention))
        self.assertEqual(len(self.effects), 24)
        self.assertEqual(self.effects[0]["reader"], "attn_output")

    def test_startup_capture_alone_does_not_prove_request_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            shutil.copytree(str(RUN / "graphs"), str(run / "graphs"))
            for name in ("request_window.json", "response.json"):
                shutil.copy2(str(RUN / name), str(run / name))
            for path in (run / "graphs").glob("events-*.jsonl"):
                events = [json.loads(line) for line in path.read_text().splitlines()]
                path.write_text("\n".join(json.dumps(e) for e in events
                                           if e["event"] != "compiled_wrapper_call") + "\n")
            with self.assertRaisesRegex(ValueError, "compiled execution evidence missing"):
                validate_run(run)


if __name__ == "__main__":
    unittest.main()
