"""Check real initialization evidence and reject broken cross-layer links."""
import copy
from pathlib import Path
import unittest

from analyze_allocation import analyze, json_lines, load

RUN = Path(__file__).resolve().parent/'results/2026-09-24-run02'


class AllocationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.events = json_lines(RUN/'events')
        cls.native = json_lines(RUN/'native')
        cls.after = load(RUN/'allocator_after.json')
        cls.pool_begin = next(e['monotonic_ns'] for e in cls.events if e['event']=='enter' and e.get('kind')=='pool')

    def test_real_pool_matches_budget_allocator_and_native_calls(self):
        result = analyze(RUN, self.events, self.native, self.after)
        summary = result['summary']
        self.assertEqual(summary['num_blocks'], 11824)
        self.assertEqual(summary['tensors'], 48)
        self.assertEqual(summary['physical_calls'], 887)
        self.assertEqual(summary['native_api_counts'], {'aclrtMallocPhysical':887,'aclrtMapMem':887})
        self.assertEqual(summary['kv_tensor_bytes'], 18597543936)
        self.assertEqual(summary['allocated_delta_bytes']-summary['kv_tensor_bytes'], 48*512)
        self.assertEqual(summary['new_allocations_during_reshape_bind'], 0)
        self.assertGreater(result['rows'][0]['reused_mapped_bytes'], 0)

    def test_wrong_physical_handle_is_rejected(self):
        native = copy.deepcopy(self.native)
        mapping = next(e for e in native if e['api']=='aclrtMapMem' and e['address']==20699527774208 and e['begin_ns']>self.pool_begin)
        mapping['extra'] = -1
        with self.assertRaisesRegex(ValueError, 'physical handle link'):
            analyze(RUN, self.events, native, self.after)

    def test_missing_native_map_is_rejected(self):
        native = [e for e in self.native if not (e['api']=='aclrtMapMem' and e['address']==20699527774208)]
        with self.assertRaisesRegex(ValueError, 'physical/map coverage'):
            analyze(RUN, self.events, native, self.after)

    def test_wrong_tensor_pointer_is_rejected(self):
        events = copy.deepcopy(self.events)
        event = next(e for e in events if e['event']=='exit' and e.get('kind')=='raw_tensor')
        event['returned']['data_ptr'] += 1
        with self.assertRaisesRegex(ValueError, 'tensor pointer not in raw pool'):
            analyze(RUN, events, self.native, self.after)

    def test_missing_virtual_reservation_is_rejected(self):
        native = [e for e in self.native if e['api']!='aclrtReserveMemAddress']
        with self.assertRaisesRegex(ValueError, 'virtual reservation identity'):
            analyze(RUN, self.events, native, self.after)

    def test_wrong_allocator_size_is_rejected(self):
        after = copy.deepcopy(self.after)
        event = next(e for e in after['device_traces'][0] if e['action']=='alloc')
        event['size'] += 512
        with self.assertRaisesRegex(ValueError, 'allocator requested bytes mismatch'):
            analyze(RUN, self.events, self.native, after)

    def test_changed_view_storage_is_rejected(self):
        events = copy.deepcopy(self.events)
        event = next(e for e in events if e['event']=='exit' and e.get('kind')=='views')
        next(iter(event['returned'].values()))[0]['storage_ptr'] += 512
        with self.assertRaisesRegex(ValueError, 'view/bind storage mismatch'):
            analyze(RUN, events, self.native, self.after)

    def test_incorrect_budget_is_rejected(self):
        events = copy.deepcopy(self.events)
        event = next(e for e in events if e['event']=='exit' and e.get('kind')=='budget')
        event['available_bytes'] += 1
        with self.assertRaisesRegex(ValueError, 'KV budget arithmetic'):
            analyze(RUN, events, self.native, self.after)


if __name__=='__main__':
    unittest.main()
