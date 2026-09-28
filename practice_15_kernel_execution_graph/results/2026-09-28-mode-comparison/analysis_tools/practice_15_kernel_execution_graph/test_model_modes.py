"""Evidence corruption tests for real eager / graph captures."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from build_mode_graph import analyze, read_trace
import compare_model_modes

ROOT = Path(__file__).resolve().parent
GRAPH = ROOT / 'results/2026-09-28-graph-run01'
EAGER = ROOT / 'results/2026-09-28-eager-run01'


class ModelModesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = [json.loads(s) for p in (GRAPH / 'events').glob('*.jsonl') for s in p.read_text().splitlines()]
        cls.events = read_trace(next((GRAPH / 'profiler').rglob('trace_view.json')))
        cls.graph = analyze(GRAPH, cls.records, cls.events)

    def test_replays_and_unknown_tasks_are_preserved(self):
        g = self.graph
        self.assertEqual(g['summary']['replay_calls'], 75)
        self.assertEqual(g['summary']['native_completion_boundaries'], 4)
        self.assertEqual(g['summary']['replay_boundary_tasks_with_runtime_connection'], 150)
        self.assertEqual(g['summary']['model_tasks_without_both_flows'], 732)
        internal = {n['id'] for n in g['nodes'] if n['kind'] == 'kernel' and n['model_id'] not in (None, 4294967295)}
        self.assertFalse(any(e['kind'] in ('launch', 'runtime_connection', 'replay_dispatch') and e['target'] in internal for e in g['edges']))

    def test_matched_eager_output(self):
        c = compare_model_modes.compare(EAGER, GRAPH)
        self.assertTrue(c['same_output_token_ids'])
        self.assertEqual(c['runs'][0]['kernel_tasks'], 1444)
        self.assertEqual(c['runs'][0]['replay_calls'], 0)
        self.assertEqual(c['runs'][0]['compute_csv_rows'], c['runs'][1]['compute_csv_rows'])

    def test_stale_replay_address_rejected(self):
        r = copy.deepcopy(self.records)
        x = next(e for e in r if e['event'] == 'enter' and e.get('kind') == 'acl_dispatch' and e['resources']['runtime_mode'] == 'PIECEWISE')
        x['resources']['input_addresses'][0] += 64
        with self.assertRaisesRegex(ValueError, 'input address mismatch'):
            analyze(GRAPH, r, self.events)

    def test_capture_identity_rejected(self):
        r = copy.deepcopy(self.records)
        x = next(e for e in r if e['event'] == 'graph_capture')
        # Corrupt every capture baseline, so the latest same-ID baseline cannot mask it.
        for x in r:
            if x['event'] == 'graph_capture':
                x['resources']['graph_id'] = -1
        with self.assertRaisesRegex(ValueError, 'capture baseline'):
            analyze(GRAPH, r, self.events)

    def test_direct_flow_loss_is_not_treated_as_replay(self):
        events = copy.deepcopy(self.events)
        task = next(e for e in events if e.get('name') == 'ReshapeAndCacheNdKernel' and 'Task Type' in e.get('args', {}))
        events = [e for e in events if not (e.get('ph') == 'f' and e.get('cat') == 'async_npu' and
                  (e.get('pid'),e.get('tid'),e.get('ts')) == (task['pid'],task['tid'],task['ts']))]
        with self.assertRaisesRegex(ValueError, 'unexpected missing direct task flow'):
            analyze(GRAPH, self.records, events)

    def test_replay_runtime_connection_rejected(self):
        events = copy.deepcopy(self.events)
        task = next(e for e in events if e.get('name') == 'MODEL_EXECUTE')
        task['args']['connection_id'] = -1
        with self.assertRaisesRegex(ValueError, 'boundary tasks missing'):
            analyze(GRAPH, self.records, events)

    def test_wrong_completion_event_rejected(self):
        r = copy.deepcopy(self.records)
        x = next(e for e in r if e['event'] == 'enter' and e.get('kind') == 'synchronize' and e['measured'])
        x['object_id'] = -1
        with self.assertRaisesRegex(ValueError, 'completion event mismatch'):
            analyze(GRAPH, r, self.events)

    def test_request_mismatch_rejected(self):
        original = compare_model_modes.load
        def changed(path):
            data = original(path)
            if path == GRAPH / 'request.json':
                data['max_tokens'] = 8
            return data
        with patch.object(compare_model_modes, 'load', side_effect=changed):
            with self.assertRaisesRegex(ValueError, 'comparison mismatch: request'):
                compare_model_modes.compare(EAGER, GRAPH)


if __name__ == '__main__':
    unittest.main()
