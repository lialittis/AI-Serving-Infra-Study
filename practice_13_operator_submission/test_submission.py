"""Verify real archived submissions and reject broken evidence associations."""
import copy
import json
from pathlib import Path
import unittest

from analyze_submission import analyze, read_trace

RUN = Path(__file__).resolve().parent / 'results/2026-09-24-run03'


class SubmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = [json.loads(line) for p in (RUN / 'events').glob('*.jsonl')
                       for line in p.read_text().splitlines()]
        cls.trace = read_trace(next((RUN / 'profiler').rglob('trace_view.json')))
        cls.result = analyze(RUN, cls.records, cls.trace)

    def test_real_compile_submission_and_completion(self):
        result = self.result
        self.assertEqual(result['summary']['compile_stages'], {'startup': 1, 'warmup': 7})
        self.assertEqual(result['summary']['measured_compiles'], 0)
        self.assertEqual(result['summary']['measured_binary_loads'], 0)
        self.assertEqual(result['summary']['binary_loads'], 8)
        self.assertEqual(len(result['triton_launches']), 111)
        self.assertEqual(len(result['device_tasks']), 1444)
        self.assertTrue(all(x['queue'] for x in result['device_tasks']))
        self.assertEqual(len(result['completions']), 4)
        self.assertEqual(len(result['stream_order']), 1)
        self.assertEqual(result['stream_order'][0]['overlapping_neighbors'], [])
        self.assertTrue(all(x['entry']['source_tensor']['device'] == 'npu:0'
                            and x['entry']['destination']['device'] == 'cpu'
                            for x in result['completions']))
        self.assertFalse(result['summary']['code_dma_timing_observed'])

    def test_prefill_and_decode_use_different_attention_inputs(self):
        examples = self.result['examples']
        def kwargs(phase):
            example = next(e for e in examples if e['task']['kernel'] == 'FusedInferAttentionScore'
                           and e['phase'] == phase)
            return next(p['entry']['kwargs'] for p in example['parameter_scopes']
                        if p['entry']['kind'] == 'torch_api')
        prefill, decode = kwargs('prefill'), kwargs('decode-1')
        self.assertIsNone(prefill['block_table'])
        self.assertEqual(prefill['key']['shape'], [10, 2, 64])
        self.assertEqual(decode['block_table']['device'], 'npu:0')
        self.assertEqual(decode['actual_seq_lengths_kv'], [11])
        self.assertGreater(decode['key']['shape'][0], 2)

    def test_missing_copy_source_is_rejected(self):
        records = copy.deepcopy(self.records)
        entry = next(e for e in records if e['event'] == 'enter' and e['kind'] == 'buffer_copy')
        del entry['source_tensor']
        with self.assertRaisesRegex(ValueError, 'copy source tensor'):
            analyze(RUN, records, self.trace)

    def test_different_compiled_binary_is_rejected(self):
        records = copy.deepcopy(self.records)
        entry = next(e for e in records if e['event'] == 'exit' and e['kind'] == 'compile')
        entry['binary']['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'compiled binary not archived'):
            analyze(RUN, records, self.trace)

    def test_wrong_registered_handle_is_rejected(self):
        records = copy.deepcopy(self.records)
        entry = next(e for e in records if e['event'] == 'exit' and e['kind'] == 'load_binary')
        entry['returned_handles'][1] += 8
        with self.assertRaisesRegex(ValueError, 'registration handle mismatch'):
            analyze(RUN, records, self.trace)

    def test_stale_or_uninitialized_launch_handle_is_rejected(self):
        records = copy.deepcopy(self.records)
        entry = next(e for e in records if e['event'] == 'enter' and e['kind'] == 'launch')
        entry['function_handle'] = -1
        with self.assertRaisesRegex(ValueError, 'function initialization link'):
            analyze(RUN, records, self.trace)

    def test_missing_device_flow_is_rejected(self):
        trace = copy.deepcopy(self.trace)
        target = next(e for e in trace if e.get('cat') == 'async_npu' and e.get('ph') == 'f')
        trace.remove(target)
        with self.assertRaisesRegex(ValueError, 'async_npu device endpoint'):
            analyze(RUN, self.records, trace)

    def test_wrong_queue_identity_is_rejected(self):
        trace = copy.deepcopy(self.trace)
        target = next(e for e in trace if e.get('cat') == 'dequeue' and e.get('ph') == 'X')
        target['args']['correlation_id'] = -1
        with self.assertRaisesRegex(ValueError, 'queue correlation ID mismatch'):
            analyze(RUN, self.records, trace)

    def test_missing_native_result_wait_is_rejected(self):
        trace = [e for e in self.trace if e.get('name') != 'Event::synchronize']
        with self.assertRaisesRegex(ValueError, 'native Event::synchronize'):
            analyze(RUN, self.records, trace)

    def test_instrumentation_error_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'instrumentation errors'):
            analyze(RUN, self.records + [{'event': 'trace_error'}], self.trace)


if __name__ == '__main__':
    unittest.main()
