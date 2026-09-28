import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_sampling_benchmark import clean_environment, measure, server_command, validate_response
from analyze_sampling_benchmark import summarize


def response(batch=1, tokens=4):
    return {"choices": [{"index": i, "finish_reason": "length"} for i in range(batch)],
            "usage": {"completion_tokens": batch * tokens}}


class BenchmarkTests(unittest.TestCase):
    def test_no_observation_in_performance_process(self):
        env = clean_environment({"PATH": "/bin", "P17_TRACE_DIR": "trace", "P18_TRACE_DIR": "trace",
                                 "PYTHONPATH": "observer", "VLLM_TORCH_PROFILER_DIR": "profile"})
        self.assertEqual(env["PATH"], "/bin")
        self.assertNotIn("PYTHONPATH", env)
        self.assertFalse(any(k.endswith("TRACE_DIR") or "PROFILER" in k for k in env))
        self.assertFalse(any("profiler" in x for x in server_command("model", "enabled", 8017, 32)))

    def test_ascend_runtime_pythonpath_retained(self):
        env = clean_environment({"PYTHONPATH": "/data/tianchi/practice_17_vllm_multistream:/usr/local/Ascend/cann-9.0.0/python/site-packages:"})
        self.assertEqual(env["PYTHONPATH"], "/usr/local/Ascend/cann-9.0.0/python/site-packages")

    def test_timer_includes_response_read(self):
        events = []
        class Reply:
            status = 200
            def __enter__(self): return self
            def __exit__(self, *args): events.append("close")
            def read(self):
                events.append("read")
                return json.dumps(response()).encode()
        def clock():
            events.append("clock")
            return 10 if len(events) == 1 else 110
        def opener(request, timeout):
            events.append("send")
            self.assertEqual(json.loads(request.data), {"test": 1})
            return Reply()
        elapsed, value = measure("http://local", {"test": 1}, opener, clock)
        self.assertEqual(elapsed, 100)
        self.assertEqual(value, response())
        self.assertEqual(events, ["clock", "send", "read", "close", "clock"])

    def test_incorrect_response_rejected(self):
        for value in [response(2), response(tokens=3), {"error": "bad"}]:
            with self.assertRaises(ValueError): validate_response(value, 1, 4)
        value = response()
        value["choices"][0]["finish_reason"] = "stop"
        with self.assertRaises(ValueError): validate_response(value, 1, 4)

    def fixture(self):
        plan = {"blocks": ["enabled", "disabled", "disabled", "enabled"],
                "cases": [[1, 4]], "warmup": 1, "repeats_per_block": 5,
                "profiling": False, "python_observer": False, "model": "test"}
        rows = []
        for block, mode in enumerate(plan["blocks"]):
            for phase, count in [("warmup", 1), ("measure", 5)]:
                for repeat in range(count):
                    rows.append({"block": block, "mode": mode, "batch_size": 1,
                                 "output_tokens": 4, "phase": phase, "repeat": repeat,
                                 "elapsed_ns": 999999999 if phase == "warmup" else
                                     (1000000 if mode == "enabled" else 2000000),
                                 "validated": True, "usage": response()["usage"], "response": response()})
        return plan, rows

    def test_warmup_excluded_and_pair_direction_checked(self):
        plan, rows = self.fixture()
        result = summarize(plan, rows)
        self.assertEqual(result["measurement_samples"], 20)
        case = result["cases"][0]
        self.assertEqual(case["enabled"]["median_ms"], 1)
        self.assertEqual(case["pair_latency_reduction_percent"], [50, 50])
        for row in rows:
            if row["block"] == 3: row["elapsed_ns"] = 3000000
        self.assertEqual(summarize(plan, rows)["cases"][0]["classification"], "uncertain")

    def test_incomplete_or_duplicate_pair_rejected(self):
        plan, rows = self.fixture()
        with self.assertRaises(ValueError): summarize(plan, rows[:-1])
        with self.assertRaises(ValueError): summarize(plan, rows + [rows[0]])


if __name__ == "__main__":
    unittest.main()
