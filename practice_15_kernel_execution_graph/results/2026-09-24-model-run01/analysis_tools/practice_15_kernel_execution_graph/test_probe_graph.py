import copy
import json
from pathlib import Path
import unittest

from build_probe_graph import build_probe, read_trace

RUN = Path(__file__).resolve().parent / 'results/2026-09-24-stream-run02'


class ProbeGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.meta = json.loads((RUN/'probe.json').read_text())
        cls.events = read_trace(next((RUN/'profiler').rglob('trace_view.json')))

    def test_real_two_stream_graph(self):
        g = build_probe(RUN)
        self.assertEqual(len(g['streams']), 2)
        self.assertEqual(g['summary']['edges_by_kind']['event_wait'], 2)
        self.assertEqual(g['summary']['edges_by_kind']['data_contract'], 2)
        self.assertEqual(g['summary']['edges_by_kind']['event_sync'], 1)
        self.assertTrue(g['summary']['output_equals_five'])

    def test_stale_event_generation_rejected(self):
        meta = copy.deepcopy(self.meta)
        meta['records'][5]['generation'] = 1
        with self.assertRaisesRegex(ValueError, 'stale or unknown'):
            build_probe(RUN, meta, self.events)

    def test_wrong_event_handle_rejected(self):
        meta = copy.deepcopy(self.meta)
        meta['records'][2]['event_handle'] = '123'
        with self.assertRaisesRegex(ValueError, 'stale or unknown'):
            build_probe(RUN, meta, self.events)

    def test_missing_flow_rejected(self):
        events = copy.deepcopy(self.events)
        events.remove(next(e for e in events if e.get('cat') == 'HostToDevice' and e.get('ph') == 'f'))
        with self.assertRaisesRegex(ValueError, 'HostToDevice endpoint'):
            build_probe(RUN, self.meta, events)

    def test_wrong_tensor_role_rejected(self):
        meta = copy.deepcopy(self.meta)
        meta['records'][3]['reads'] = ['x']
        with self.assertRaisesRegex(ValueError, 'resource role mismatch'):
            build_probe(RUN, meta, self.events)

    def test_alias_lifetime_assumption_rejected(self):
        meta = copy.deepcopy(self.meta)
        meta['resources']['y']['data_ptr'] = meta['resources']['x']['data_ptr']
        with self.assertRaisesRegex(ValueError, 'unexpectedly alias'):
            build_probe(RUN, meta, self.events)


if __name__ == '__main__':
    unittest.main()
