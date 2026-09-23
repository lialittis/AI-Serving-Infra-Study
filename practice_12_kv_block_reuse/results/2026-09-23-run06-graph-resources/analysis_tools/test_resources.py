"""Actual resource captures and negative tests of graph identity/flow claims."""
import copy
import json
from pathlib import Path
import unittest
from audit_resources import audit, build_inventory, check_replay, tensors
from analyze_reuse import read_trace

ROOT=Path(__file__).resolve().parent/'results'
EAGER=ROOT/'2026-09-23-run05-eager-resources'
GRAPH=ROOT/'2026-09-23-run06-graph-resources'

class ResourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.eager=audit(EAGER)
        cls.graph=audit(GRAPH)
        cls.replay=cls.graph['replays'][0]

    def check(self,capture=None,before=None,after=None):
        r=self.replay
        return check_replay(capture or r['capture']['resources'],before or r['resources_before'],after or r['resources_after'])

    def test_real_pair_has_all_events_and_replays(self):
        self.assertEqual(self.eager['summary']['observed_scopes'],362)
        self.assertEqual(self.graph['summary']['observed_scopes'],512)
        self.assertEqual(len(self.graph['replays']),50)
        self.assertEqual(len(self.eager['replays']),0)
        for d in (self.eager,self.graph):
            self.assertEqual(len(d['completion_boundaries']),4)
            self.assertEqual(d['summary']['weight_tensors'],170)
            self.assertEqual(d['summary']['verified_slot_preparation_chains'],4)
            self.assertEqual(d['summary']['zero_length_copies'],4)
            self.assertEqual(d['summary']['kv_operator_chains'],192)
            self.assertFalse(d['summary']['hidden_workspace_lifetime_proven'])
            self.assertEqual(len(d['cpu_events']),d['summary']['cpu_events'])
            self.assertTrue(all('resources' in e and 'preparation_events' in e for e in d['events']))

    def test_allocator_before_schedule_has_explicit_step_attribution(self):
        for d in (self.eager,self.graph):
            alloc=next(e for e in d['events'] if e['kind']=='pool_allocate' and e['role']=='A')
            self.assertEqual(alloc['host_trace_step'],2)
            self.assertEqual(alloc['step'],3)
            self.assertEqual(alloc['phase'],'prefill')
            self.assertIn('computed_tokens',alloc['step_assignment'])

    def test_replays_reference_capture_occurrences_not_just_object_ids(self):
        records=[json.loads(s) for p in (GRAPH/'events').glob('*.jsonl') for s in p.read_text().splitlines()]
        captures=[e for e in records if e['event']=='graph_capture_resources']
        ids=[(e['pid'],e['resources']['graph_object_id']) for e in captures]
        self.assertLess(len(set(ids)),len(ids))  # actual startup object-ID reuse
        self.assertEqual(len(set(r['capture_occurrence'] for r in self.graph['replays'])),25)
        for r in self.graph['replays']:
            self.assertEqual(r['capture']['resources']['wrapper_object_id'],r['resources_before']['wrapper_object_id'])
            self.assertEqual(r['checks']['complete_resource_safety'],'not_proven')

    def test_changed_input_pointer_rejected(self):
        before=copy.deepcopy(self.replay['resources_before']);before['input_addresses'][0]+=64
        with self.assertRaisesRegex(ValueError,'input addresses'):self.check(before=before)

    def test_same_pointer_wrong_shape_rejected(self):
        before=copy.deepcopy(self.replay['resources_before'])
        _,t=next(tensors(before['arguments']));t['shape'][0]+=1
        with self.assertRaisesRegex(ValueError,'layout/storage'):self.check(before=before)

    def test_changed_graph_pool_rejected(self):
        before=copy.deepcopy(self.replay['resources_before']);before['graph_pool']=['different']
        with self.assertRaisesRegex(ValueError,'graph pool'):self.check(before=before)

    def test_changed_batch_descriptor_rejected(self):
        before=copy.deepcopy(self.replay['resources_before']);before['batch_descriptor']['num_tokens']+=1
        with self.assertRaisesRegex(ValueError,'batch descriptor'):self.check(before=before)

    def test_changed_output_storage_rejected(self):
        after=copy.deepcopy(self.replay['resources_after']);_,t=next(tensors(after['persistent_output']));t['data_ptr']+=256
        with self.assertRaisesRegex(ValueError,'output storage'):self.check(after=after)

    def test_wrong_wrapper_with_same_graph_id_rejected(self):
        before=copy.deepcopy(self.replay['resources_before']);before['wrapper_object_id']+=1
        with self.assertRaisesRegex(ValueError,'wrapper identity'):self.check(before=before)

    def test_missing_flow_stays_unassociated(self):
        # Use a real task and its exact endpoints; removing the torch start must
        # not invent a replacement from nearby host events.
        trace=read_trace(next((EAGER/'profiler').rglob('trace_view.json')))
        original=next(t for t in self.eager['device_tasks'] if any(a['category']=='async_npu' for a in t['associations']))
        starts={int(a['start'].split(':')[1]) for a in original['associations'] if a['category']=='async_npu'}
        edited=[e for i,e in enumerate(trace) if i not in starts]
        _,tasks=build_inventory(edited,{})
        task=next(t for t in tasks if t['trace']==original['trace'])
        self.assertFalse(any(a['category']=='async_npu' for a in task['associations']))

    def damaged_records(self):
        return [json.loads(s) for p in (EAGER/'events').glob('*.jsonl') for s in p.read_text().splitlines()]

    def test_event_record_wait_identity_mismatch_rejected(self):
        records=self.damaged_records()
        e=next(e for e in records if e['event']=='scope_enter' and e['kind']=='event_record')
        e['object_id']+=1
        with self.assertRaisesRegex(ValueError,'event object identity'):audit(EAGER,records)

    def test_slot_producer_address_mismatch_rejected(self):
        records=self.damaged_records()
        e=next(e for e in records if e['event']=='scope_enter' and e['kind']=='slot_prepare')
        e['slot_mapping']['data_ptr']+=256
        with self.assertRaisesRegex(ValueError,'slot producer/consumer'):audit(EAGER,records)

    def test_nonempty_copy_cannot_be_labeled_zero_length(self):
        records=self.damaged_records()
        e=next(e for e in records if e['event']=='scope_enter' and e['kind']=='buffer_copy' and e['copied_rows'])
        e['copied_rows']=0
        with self.assertRaisesRegex(ValueError,'zero-length buffer copy'):audit(EAGER,records)

    def test_unassociated_graph_tasks_remain_visible(self):
        tasks=[t for t in self.graph['device_tasks'] if t['trace']['name']=='MODEL_EXECUTE']
        self.assertEqual(len(tasks),50)
        self.assertTrue(all(t['role'] is None and not t['associations'] for t in tasks))
        self.assertTrue(all(not r['direct_device_tasks'] for r in self.graph['replays']))

if __name__=='__main__':unittest.main()
