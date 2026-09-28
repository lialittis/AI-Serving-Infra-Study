import copy
import unittest

from dag import MemoryFrontier, metrics, schedule, schedule_scopes, topological, validate_schedule
from kv_ranges import pool_ranges
from build_dag import view_ranges, contract


def node(name, duration):
    return dict(id=name, duration_ns=duration)


def edge(a, b):
    return dict(source=a, target=b, kind='RAW')


class DependencyTests(unittest.TestCase):
    def setUp(self):
        self.memory = MemoryFrontier()

    def access(self, node, mode, lo, hi, gen=1):
        self.memory.access(node, dict(device='npu:0', lo=lo, hi=hi,
                                      generation=gen, mode=mode))

    def edges(self):
        return {(e['source'], e['target'], e['kind']) for e in self.memory.edges}

    def test_partial_overwrite_preserves_old_writer(self):
        self.access('a', 'W', 0, 100)
        self.access('b', 'W', 30, 70)
        self.access('c', 'R', 0, 100)
        self.assertEqual(self.edges(), {('a', 'b', 'WAW'), ('a', 'c', 'RAW'), ('b', 'c', 'RAW')})

    def test_readers_do_not_depend_on_each_other_but_precede_write(self):
        self.access('a', 'W', 0, 4)
        self.access('b', 'R', 0, 4)
        self.access('c', 'R', 0, 4)
        self.access('d', 'W', 0, 4)
        self.assertEqual(self.edges(), {('a', 'b', 'RAW'), ('a', 'c', 'RAW'),
                                       ('a', 'd', 'WAW'), ('b', 'd', 'WAR'), ('c', 'd', 'WAR')})

    def test_distinct_allocation_same_address_retains_lifetime_hazard(self):
        self.access('old', 'R', 0, 4, gen=1)
        self.access('new', 'R', 0, 4, gen=2)
        self.assertEqual(self.edges(), {('old', 'new', 'allocation_reuse')})

    def test_no_false_dependency_for_disjoint_views_or_empty_access(self):
        self.access('a', 'W', 0, 4)
        self.access('b', 'R', 4, 8)
        self.access('c', 'W', 0, 0)
        self.assertFalse(self.edges())

    def test_inplace_does_not_make_self_cycle(self):
        self.access('a', 'W', 0, 8)
        self.access('b', 'R', 0, 8)
        self.access('b', 'W', 0, 8)
        topological([node('a', 1), node('b', 1)], self.memory.edges)
        self.assertNotIn(('b', 'b', 'WAR'), self.edges())

    def test_byte_reference_oracle_with_aliases_and_reuse(self):
        import random
        rng = random.Random(180)
        cells, expected = {}, set()
        for index in range(80):
            n = str(index)
            lo = rng.randrange(16)
            hi = rng.randrange(lo + 1, 17)
            mode = rng.choice(('R', 'W'))
            generation = 1 + index // 20
            self.access(n, mode, lo, hi, generation)
            for address in range(lo, hi):
                gen, writer, readers = cells.get(address, (generation, None, set()))
                readers = set(readers)
                if gen != generation:
                    expected |= {(p, n, 'allocation_reuse') for p in readers | ({writer} if writer else set())}
                    writer, readers = None, set()
                if writer is not None:
                    expected.add((writer, n, 'RAW' if mode == 'R' else 'WAW'))
                if mode == 'W':
                    expected |= {(p, n, 'WAR') for p in readers}
                    writer, readers = n, set()
                else:
                    readers.add(n)
                cells[address] = (generation, writer, readers)
        self.assertEqual(self.edges(), expected)


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.nodes = [node('a', 2), node('b', 5), node('c', 3), node('d', 1)]
        self.edges = [edge('a', 'b'), edge('a', 'c'), edge('b', 'd'), edge('c', 'd')]

    def test_diamond_path_slack_and_parallelism(self):
        m = metrics(self.nodes, self.edges)
        self.assertEqual(m['critical_path'], ['a', 'b', 'd'])
        self.assertEqual(m['critical_path_ns'], 8)
        self.assertEqual(m['work_ns'], 11)
        self.assertEqual(m['slack_ns']['c'], 2)
        self.assertEqual(m['asap_peak_active'], 2)
        self.assertEqual(schedule(self.nodes, self.edges, 1)['makespan_ns'], 11)
        self.assertEqual(schedule(self.nodes, self.edges, 2)['makespan_ns'], 8)

    def test_event_cost_and_missing_wait(self):
        plan = schedule(self.nodes, self.edges, 2, event_cost_ns=1)
        self.assertEqual(plan['makespan_ns'], 8)
        self.assertTrue(plan['events'])
        plan['events'] = []
        with self.assertRaisesRegex(ValueError, 'missing cross-stream'):
            validate_schedule(self.nodes, self.edges, plan)

    def test_tampered_dependency_rejected(self):
        plan = schedule(self.nodes, self.edges, 2)
        bad = copy.deepcopy(plan)
        row = next(r for r in bad['rows'] if r['id'] == 'c')
        row.update(start_ns=0, end_ns=3)
        with self.assertRaisesRegex(ValueError, 'violation'):
            validate_schedule(self.nodes, self.edges, bad)

    def test_transitive_dependencies_need_no_duplicate_wait(self):
        plan = schedule(self.nodes, self.edges + [edge('a', 'd')], 2)
        self.assertEqual(len(plan['events']), 2)
        validate_schedule(self.nodes, self.edges + [edge('a', 'd')], plan)
        plan['events'][0]['source_stream'] = 99
        with self.assertRaisesRegex(ValueError, 'stream mismatch'):
            validate_schedule(self.nodes, self.edges, plan)

    def test_invalid_graphs_fail_closed(self):
        for edges in (self.edges + [edge('d', 'a')], [edge('a', 'missing')], [edge('a', 'a')]):
            with self.assertRaises(ValueError):
                metrics(self.nodes, edges)

    def test_chain_has_no_parallelism(self):
        edges = [edge('a', 'b'), edge('b', 'c'), edge('c', 'd')]
        self.assertEqual(metrics(self.nodes, edges)['work_over_span'], 1)
        for count in (1, 2, 4, 8):
            self.assertEqual(schedule(self.nodes, edges, count)['makespan_ns'], 11)

    def test_native_scope_is_kept_on_one_stream(self):
        nodes = [dict(node('cast', 2), owner='gemm'), dict(node('matmul', 5), owner='gemm'), node('independent', 6)]
        edges = [edge('cast', 'matmul')]
        plan = schedule_scopes(nodes, edges, 2)
        rows = {r['id']: r for r in plan['rows']}
        self.assertEqual(rows['cast']['stream'], rows['matmul']['stream'])
        self.assertEqual(rows['cast']['end_ns'], rows['matmul']['start_ns'])
        self.assertEqual(plan['makespan_ns'], 7)


class KVTests(unittest.TestCase):
    def test_native_redispatch_positional_binding_matches_outer_kwargs(self):
        class NoIndices:
            def pool_slots(self, entry):
                return None
        t = dict(kind='tensor', shape=[2], stride=[1], element_size=2,
                 storage_ptr=100, data_ptr=100, storage_offset=0,
                 storage_generation=1, storage_bytes=4, device='npu:0')
        keyword = dict(event='enter', kind='torch_api', operator='npu::npu_fused_infer_attention_score',
                       args=[], kwargs=dict(query=t, key=t, value=t, block_table=None))
        positional = dict(keyword, args=[t, t, t], kwargs={})
        returned = dict(returned=[t])
        self.assertEqual(contract(keyword, returned, ['FusedInferAttentionScore'], NoIndices()),
                         contract(positional, returned, ['FusedInferAttentionScore'], NoIndices()))

    def test_slot_ranges_and_padding(self):
        tensor = dict(shape=[8, 4, 2, 2], stride=[16, 4, 2, 1], element_size=2,
                      data_ptr=1000, device='npu:0', storage_generation=1)
        result = pool_ranges(tensor, [-1, 4, 5, 9], 'W')
        self.assertEqual([(r['lo'], r['hi']) for r in result], [(1032, 1048), (1072, 1080)])
        with self.assertRaises(ValueError):
            pool_ranges(tensor, [32], 'W')

    def test_packed_qkv_views_do_not_fill_stride_holes(self):
        tensor = dict(kind='tensor', shape=[3, 2], stride=[6, 1],
                      data_ptr=100, storage_ptr=100, storage_offset=0, element_size=2)
        intervals, precision = view_ranges(tensor)
        self.assertEqual(intervals, [(100, 104), (112, 116), (124, 128)])
        self.assertEqual(precision, 'view_bytes')
        self.assertEqual(view_ranges(tensor, limit=2), ([(100, 128)], 'bounding_fallback'))

    def test_transposed_dense_view_and_broadcast(self):
        tensor = dict(kind='tensor', shape=[3, 2], stride=[1, 3],
                      data_ptr=100, storage_ptr=100, storage_offset=0, element_size=2)
        self.assertEqual(view_ranges(tensor)[0], [(100, 112)])
        tensor.update(stride=[0, 1])
        self.assertEqual(view_ranges(tensor)[0], [(100, 104)])


if __name__ == '__main__':
    unittest.main()
