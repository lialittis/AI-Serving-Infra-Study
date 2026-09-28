"""Ensure SVG detail slices preserve evidence and never invent replay internals."""
import copy
import json
from pathlib import Path
import unittest
from xml.etree import ElementTree

from export_mode_focus import PHASES, focus, dot_text

ROOT = Path(__file__).resolve().parent


class ModeFocusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graphs = {mode:json.loads((ROOT / ('results/2026-09-28-' + mode + '-run01/analysis/execution_graph.json')).read_text()) for mode in ('eager', 'graph')}

    def test_all_eight_slices_preserve_exact_execution_edges(self):
        for mode, graph in self.graphs.items():
            for phase in PHASES:
                with self.subTest(mode=mode, phase=phase):
                    result = focus(graph, phase)
                    for edge in result['edges']:
                        index = edge['original_edge_index']
                        self.assertEqual({k:v for k,v in edge.items() if k != 'original_edge_index'}, graph['edges'][index])
                    self.assertTrue(any(n['name'] == 'ReshapeAndCacheNdKernel' for n in result['nodes']))
                    self.assertTrue(any(n['name'] == 'FusedInferAttentionScore' for n in result['nodes']))
                    self.assertTrue(all(n.get('phase') == phase for n in result['nodes'] if n['kind'] == 'kernel'))

    def test_unknown_replay_tasks_not_assigned_to_first_layer(self):
        result = focus(self.graphs['graph'], 'decode-1')
        self.assertEqual(len(result['unknown_phase_tasks_not_assigned_to_layer']), 244)
        self.assertFalse({n['id'] for n in result['nodes']} & set(result['unknown_phase_tasks_not_assigned_to_layer']))
        self.assertEqual(len(result['tensor_bindings']), 5)
        self.assertFalse(any(e['kind'] == 'data_contract' for e in result['edges']))
        self.assertNotIn('binding-', dot_text(result))  # metadata is a separate panel, never a dependency edge

    def test_core_fields_preserve_unknown_zero(self):
        graph = self.graphs['graph']
        kernels = [n for n in graph['nodes'] if n['kind'] == 'kernel' and n.get('core_usage')]
        unknown = [n for n in kernels if n['core_usage']['block_num'] == '0']
        self.assertTrue(unknown)
        self.assertTrue(all(n['core_usage']['status'] == 'unknown_or_not_reported' for n in unknown))
        result = focus(graph, 'decode-1')
        kv = next(n for n in result['nodes'] if n['name'] == 'ReshapeAndCacheNdKernel')
        self.assertEqual(kv['core_usage']['block_num'], '1')
        self.assertIn('AI_VECTOR_CORE / Block 1 / Mix 0', dot_text(result))

    def test_changed_attention_tensor_rejected(self):
        g = copy.deepcopy(self.graphs['graph'])
        r = focus(g, 'decode-1')
        tensor = g['parameter_scopes'][r['attention_scope']]['entry']['arguments']['key']
        tensor['data_ptr'] += 2
        tensor['storage_offset'] += 1
        with self.assertRaisesRegex(ValueError, 'tensor view mismatch'):
            focus(g, 'decode-1')

    def test_wrong_parent_forward_rejected(self):
        g = copy.deepcopy(self.graphs['graph'])
        r = next(r for r in g['replays'] if r['phase'] == 'decode-1' and r['partition'] == 'submod_0')
        parent = g['parameter_scopes'][r['label']]['entry']['parent']
        g['parameter_scopes'][parent]['entry']['parent'] = 'another-forward'
        with self.assertRaisesRegex(ValueError, 'scope identity mismatch'):
            focus(g, 'decode-1')

    def test_late_producer_scope_rejected(self):
        g = copy.deepcopy(self.graphs['graph'])
        r = next(r for r in g['replays'] if r['phase'] == 'decode-1' and r['partition'] == 'submod_0')
        parent = g['parameter_scopes'][r['label']]['entry']['parent']
        g['parameter_scopes'][parent]['exit']['monotonic_ns'] = 10**30
        with self.assertRaisesRegex(ValueError, 'producer dispatch must precede'):
            focus(g, 'decode-1')

    def test_svg_nodes_and_edge_ids_match_sidecar(self):
        ns = {'s':'http://www.w3.org/2000/svg'}
        for mode in self.graphs:
            for phase in PHASES:
                stem = ROOT / ('results/2026-09-28-' + mode + '-run01/analysis/first_attention_' + phase)
                sidecar = json.loads(stem.with_suffix('.json').read_text())
                svg = ElementTree.parse(str(stem.with_suffix('.svg')))
                ids = {e.attrib['id'] for e in svg.findall('.//s:g', ns) if 'id' in e.attrib}
                self.assertTrue({n['id'] for n in sidecar['nodes']}.issubset(ids))
                self.assertTrue({'evidence-edge-' + str(i) for i in range(len(sidecar['edges']))}.issubset(ids))


if __name__ == '__main__':
    unittest.main()
