import unittest
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace

from common import Bindings, disjoint, overlap, union


class RecoveryTests(unittest.TestCase):
    def test_exception_restores_exact_reference(self):
        old = object()
        state = SimpleNamespace(value=old)
        with self.assertRaisesRegex(RuntimeError, "injected"):
            with Bindings([(state, "value", object()), (state, "temporary", 1)]):
                raise RuntimeError("injected")
        self.assertIs(state.value, old)
        self.assertFalse(hasattr(state, "temporary"))

    def test_nested_binding(self):
        state = SimpleNamespace(value=1)
        with Bindings([(state, "value", 2)]):
            with Bindings([(state, "value", 3)]):
                self.assertEqual(state.value, 3)
            self.assertEqual(state.value, 2)
        self.assertEqual(state.value, 1)

    def test_partial_install_failure_restores_prior_bindings(self):
        state = SimpleNamespace(value=1)
        class ReadOnly:
            @property
            def value(self):
                return 2
        with self.assertRaises(AttributeError):
            with Bindings([(state, 'value', 3), (ReadOnly(), 'value', 4)]):
                self.fail('partial installation should fail')
        self.assertEqual(state.value, 1)

    def test_inherited_attribute_restored_without_shadow(self):
        class Base:
            value = object()
        class Child(Base):
            pass
        with Bindings([(Child, "value", 42)]):
            self.assertEqual(Child.value, 42)
        self.assertNotIn("value", vars(Child))
        self.assertIs(Child.value, Base.value)


class AnalysisTests(unittest.TestCase):
    def test_overlap_is_not_envelope_overlap(self):
        self.assertEqual(overlap([(0, 1), (9, 10)], [(2, 8)]), 0)
        self.assertEqual(overlap([(0, 5), (2, 6)], [(3, 7)]), 3)

    def test_negative_interval_rejected(self):
        with self.assertRaises(ValueError):
            union([(2, 1)])

    def test_storage_alias_detected(self):
        self.assertTrue(disjoint([('A', 'kv', 0, 10), ('B', 'out', 9, 12)]))
        self.assertFalse(disjoint([('A', 'kv', 0, 10), ('B', 'out', 10, 12)]))


class SubmissionTests(unittest.TestCase):
    def setup_runtime(self, fail=None):
        log, current = [], ['default']
        class Stream:
            def __init__(self, name): self.name = name
            def wait_event(self, event): log.append(('wait', self.name, event.recorded))
            def synchronize(self): log.append(('drain', self.name))
        class Event:
            def __init__(self, **kwargs): self.recorded = None
            def record(self):
                self.recorded = current[0]
                log.append(('record', self.recorded))
            def synchronize(self):
                assert self.recorded is not None, 'unrecorded event waited'
                log.append(('join', self.recorded))
            def elapsed_time(self, other): return 0
        @contextmanager
        def stream(value):
            old, current[0] = current[0], value.name
            try: yield
            finally: current[0] = old
        class Capsule:
            def __init__(self, role): self.role = role
            def submit(self):
                log.append(('submit', self.role, current[0]))
                if self.role == fail: raise RuntimeError('failed_after_submission')
                return self.role
        runtime = SimpleNamespace(npu=SimpleNamespace(Event=Event, stream=stream))
        return runtime, {r:Capsule(r) for r in 'AB'}, {r:Stream(r) for r in 'AB'}, log

    def test_no_cpu_join_between_submissions(self):
        from native import pair
        torch, caps, streams, log = self.setup_runtime()
        pair(torch, caps, streams, 'parallel')
        self.assertLess(log.index(('submit','B','B')), log.index(('join','A')))
        self.assertEqual([x for x in log if x[0]=='wait'], [('wait','A','default'),('wait','B','default')])

    def test_partial_submission_drains_stream_without_terminal_event(self):
        from native import pair
        torch, caps, streams, log = self.setup_runtime(fail='B')
        with self.assertRaisesRegex(RuntimeError,'failed_after_submission'):
            pair(torch, caps, streams, 'parallel')
        self.assertIn(('join','A'),log)
        self.assertIn(('drain','B'),log)
        self.assertNotIn(('join','B'),log)

    def test_injected_first_submit_never_waits_unsubmitted_branch(self):
        from native import pair
        torch, caps, streams, log = self.setup_runtime()
        with self.assertRaisesRegex(RuntimeError,'injected_after_first_submit'):
            pair(torch, caps, streams, 'parallel', inject=True)
        self.assertIn(('join','A'),log)
        self.assertFalse(any('B' in x for x in log))


class ControllerTests(unittest.TestCase):
    def test_timeout_terminates_only_owned_child_and_records_exit(self):
        from run_suite import launch
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out / 'collector').mkdir()
            (out / 'collector/child.py').write_text('import time\ntime.sleep(120)\n')
            with self.assertRaises(subprocess.TimeoutExpired):
                launch(out, 'test', 'eager', 'qualify', timeout=.01)
            exit_record = json.loads((out / 'test-exit.json').read_text())
            self.assertTrue(exit_record['process_exited'])
            self.assertEqual(exit_record['returncode'], -15)


if __name__ == '__main__':
    unittest.main()
