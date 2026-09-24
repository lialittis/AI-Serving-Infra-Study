"""Graph semantics tests plus an independent re-audit of the archived model run."""
import copy
from pathlib import Path
import unittest

from build_model_graph import analyze, build, same_view, tensor_span, validate_dag

BASE = Path(__file__).resolve().parent / 'results/2026-09-24-model-run01'


class ExecutionGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = analyze(BASE)
        cls.graph = build(cls.data)

    def test_real_model_graph(self):
        g = self.graph
        self.assertEqual(g['summary']['kernel_tasks'], 1444)
        self.assertEqual(g['summary']['edges_by_kind']['stream_order'], 1443)
        self.assertEqual(g['summary']['edges_by_kind']['event_sync'], 4)
        self.assertEqual(g['summary']['edges_by_kind']['data_contract'], 192)
        self.assertEqual(g['summary']['edges_by_kind']['storage_candidate'], 144)
        self.assertFalse(g['summary']['complete_data_graph'])
        self.assertEqual(g['summary']['raw_stream_observations'], 111)

    def test_prefill_has_no_cache_to_fia_data_edge(self):
        nodes = {n['id']: n for n in self.graph['nodes']}
        edges = [e for e in self.graph['edges'] if e['kind'] == 'storage_candidate']
        self.assertTrue(all(nodes[e['target']]['phase'].startswith('decode') for e in edges))

    def test_reused_event_gets_distinct_generations(self):
        waits = [n for n in self.graph['nodes'] if n['kind'] == 'wait_return']
        self.assertEqual(len({n['event_object'] for n in waits}), 1)
        self.assertEqual(len({n['generation'] for n in waits}), 4)

    def test_wrong_rope_pointer_rejected(self):
        data = copy.deepcopy(self.data)
        ob = next(o for o in data['observations'] if o.get('named_arguments', {}).get('q_ptr'))
        ob['named_arguments']['q_ptr']['data_ptr'] += 8
        with self.assertRaisesRegex(ValueError, 'pointer/offset mismatch'):
            build(data)

    def test_no_stream_edge_between_distinct_lanes(self):
        data = copy.deepcopy(self.data)
        data['device_tasks'][0]['task']['stream_id'] = 99
        data['device_tasks'][0]['device_event']['args']['Physic Stream Id'] = 99
        g = build(data)
        nodes = {n['id']: n for n in g['nodes']}
        for e in g['edges']:
            if e['kind'] == 'stream_order':
                self.assertEqual(nodes[e['source']]['stream'], nodes[e['target']]['stream'])

    def test_overlapping_stream_tasks_rejected(self):
        data = copy.deepcopy(self.data)
        data['device_tasks'][0]['device_event']['dur'] = 1000000
        with self.assertRaisesRegex(ValueError, 'overlapping same-stream'):
            build(data)

    def test_aliases_are_not_identical_views(self):
        a = dict(kind='tensor', shape=[2], stride=[2], device='npu:0', data_ptr=100,
                 storage_ptr=100, storage_offset=0, element_size=4)
        b = dict(a, data_ptr=104, storage_offset=1)
        self.assertEqual(tensor_span(a), (100, 112))
        self.assertEqual(tensor_span(b), (104, 116))
        self.assertFalse(same_view(a, b))  # Bounding spans overlap, actual elements do not.

    def test_cycle_and_dangling_edge_rejected(self):
        nodes = [dict(id='a'), dict(id='b')]
        with self.assertRaisesRegex(ValueError, 'cycle'):
            validate_dag(nodes, [dict(source='a', target='b'), dict(source='b', target='a')])
        with self.assertRaisesRegex(ValueError, 'dangling'):
            validate_dag(nodes, [dict(source='a', target='missing')])


if __name__ == '__main__':
    unittest.main()
