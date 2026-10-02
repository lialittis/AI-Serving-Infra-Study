import csv
from decimal import Decimal
import json
from pathlib import Path
import tempfile
import unittest
from analyze import overlap, intersect_length, attach_requests, analyze_case, export


def task(i, stream, start, stop, compute=True):
    return dict(id=str(i), stream=str(stream), start_us=str(start), end_us=str(stop), is_compute=compute)


class IntervalTests(unittest.TestCase):
    def test_serial_and_touching(self):
        r = overlap([task(1, 0, 0, 3), task(2, 1, 3, 6)])
        self.assertEqual(r['compute_overlap_us'], '0')
        self.assertEqual(r['peak_compute_streams'], 1)

    def test_three_stream_union_not_pair_sum(self):
        r = overlap([task(1, 0, 0, 10), task(2, 1, 2, 8), task(3, 2, 4, 6)])
        self.assertEqual(r['compute_overlap_us'], '6')
        self.assertEqual(r['compute_union_us'], '10')
        self.assertEqual(r['peak_compute_streams'], 3)
        self.assertEqual(len(r['evidence']), 3)

    def test_copy_excluded_and_same_stream_not_parallel(self):
        r = overlap([task(1, 0, 0, 10), task(2, 0, 2, 5), task(3, 1, 0, 10, False)])
        self.assertEqual(r['compute_overlap_us'], '0')
        self.assertEqual(r['peak_compute_streams'], 1)
        self.assertEqual(intersect_length([(0, 5), (3, 9)], [(4, 12)]), 5)

    def test_large_epoch_and_fractional_time(self):
        base = Decimal('1790000000000000')
        r = overlap([task(1, 0, base, base + Decimal('.002')), task(2, 1, base + Decimal('.001'), base + 1)])
        self.assertEqual(Decimal(r['compute_overlap_us']), Decimal('.001'))

    def test_missing_identity_does_not_substring_match(self):
        requests = [dict(client_request_id='a', response_ids=['cmpl-a'])]
        mapping, issues = attach_requests(requests, [dict(kind='frontend_map', client_request_id='aa', response_id='cmpl-aa', external_id='aa-0')])
        self.assertEqual(mapping, {})
        self.assertEqual(len(issues), 1)


def fixture(case, remove_flow=False):
    case.mkdir()
    def write(path, data):
        path.write_text(json.dumps(data))
    base = 1790000000000000
    stamp = lambda offset: (base + offset) * 1000
    reqs = [dict(client_request_id='r'+str(i), response_ids=['cmpl-r'+str(i)], user=i, round=0, repetition=0,
                 start_ns=stamp(0), end_ns=stamp(100), start_mono_ns=0, end_mono_ns=100000,
                 token_ids=[1], status='passed', ttft_ms=.05, latency_ms=.1,
                 usage=dict(prompt_tokens=128, completion_tokens=1)) for i in range(2)]
    write(case / 'requests.json', reqs)
    write(case / 'command.json', dict(phase='diagnostic', concurrency=2))
    write(case / 'status.json', dict(status='passed'))
    scheduled = {'internal0-x': 128, 'internal1-y': 128}
    records = []
    for i, internal in enumerate(scheduled):
        rid='r'+str(i)
        records += [dict(kind='frontend_map', client_request_id=rid, response_id='cmpl-'+rid, external_id=rid+'-0'),
                    dict(kind='engine_map', external_id=rid+'-0', internal_id=internal),
                    dict(kind='admitted', internal_id=internal, wall_ns=stamp(2)),
                    dict(kind='received', client_request_id=rid, wall_ns=stamp(1))]
    records += [dict(kind='schedule', key='k', scheduled=scheduled, start_ns=stamp(3), end_ns=stamp(4),
                     requests={r: dict(phase='prefill', computed_before=0, prompt_tokens=128) for r in scheduled}),
                dict(kind='execute', key='k', scheduled=scheduled),
                dict(kind='scope', key='k', label='P31/k/execute', stage='execute', scheduled=scheduled)]
    (case / 'observer').mkdir()
    (case / 'observer' / 'observer-1.jsonl').write_text('\n'.join(json.dumps(r) for r in records))
    events = [dict(ph='X', name='P31/k/execute', pid=1, tid=22, ts=base+5, dur=70)]
    rows = []
    for i in range(3):
        name = 'MatMul%d' % i if i < 2 else 'MEMCPY_ASYNC'
        sid = '46' if i < 2 else '44'
        begin = base + 40 + i*8 if i < 2 else base + 42
        args = {'Physic Stream Id': sid, 'Task Id': str(i), 'Task Type': 'AI_CORE' if i < 2 else 'SDMA_SQE', 'connection_id': i}
        host = dict(ph='X', name='aten::mm', pid=1, tid=22, ts=base+10+i*10, dur=8)
        native = dict(ph='X', name='AscendCL@launch', pid=2, tid=22, ts=base+11+i*10, dur=4, args={'connection_id': i, 'Thread Id': 22})
        kernel = dict(ph='X', name=name, pid=3, tid=int(sid), ts=begin, dur=6, args=args)
        events += [host, native, kernel]
        if not remove_flow:
            for cat, source in [('async_npu', host), ('HostToDevice', native)]:
                events += [dict(ph='s', cat=cat, id=i, pid=source['pid'], tid=source['tid'], ts=source['ts']),
                           dict(ph='f', cat=cat, id=i, pid=kernel['pid'], tid=kernel['tid'], ts=begin)]
        if i < 2:
            rows.append({'Name': name, 'Stream ID': sid, 'Task ID': str(i), 'Start Time(us)': str(begin), 'Duration(us)': '6', 'Accelerator Core': 'AI_CORE'})
    (case / 'profiler').mkdir()
    write(case / 'profiler' / 'trace_view.json', events)
    with (case / 'profiler' / 'kernel_details.csv').open('w') as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


class JoinTests(unittest.TestCase):
    def test_shared_batch_exact_flows_and_copy(self):
        with tempfile.TemporaryDirectory() as d:
            case=Path(d)/'case'; fixture(case)
            result=analyze_case(case)
            self.assertEqual(result['analysis_status'], 'passed')
            self.assertEqual(result['coverage']['csv_matched'], 2)
            self.assertTrue(all(t['requests'] == ['r0', 'r1'] for t in result['tasks']))
            self.assertEqual(result['device_metrics']['compute_overlap_us'], '0')
            self.assertEqual(result['device_metrics']['copy_compute_overlap_us'], '4')
            export(case,result)
            self.assertTrue((case/'analysis/report.html').exists())

    def test_missing_flows_are_incomplete_not_zero_overlap_proof(self):
        with tempfile.TemporaryDirectory() as d:
            case=Path(d)/'case'; fixture(case, remove_flow=True)
            result=analyze_case(case)
            self.assertEqual(result['analysis_status'], 'incomplete')
            self.assertEqual(result['coverage']['compute_attributed'], 0)


if __name__ == '__main__':
    unittest.main()
