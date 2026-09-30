"""Verify immutable evidence and join Sub by flow IDs (never nearest timestamps)."""
import csv
from decimal import Decimal as D
import gzip
import hashlib
import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent


def read(p):
    return json.loads(p.read_text())


def one(items):
    items = list(items)
    assert len(items) == 1, f'Expected one match, got {len(items)}'
    return items[0]


def verify_manifest(directory):
    for name, digest in read(directory / 'checksums.json').items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest, name


def api_analysis(calls, obs, env):
    desc = {r['u'][0]: r for r in calls if r['kind'] == 1}
    phase1 = one(r for r in calls if r['kind'] == 2)
    assert phase1['result'] == 0 and phase1['tid'] == env['main_tid']
    assert len(desc) == 3
    tensors = [obs['inputs']['left'], obs['inputs']['right'], obs['rows'][0]['output']]
    for handle, t in zip([phase1['u'][0], phase1['u'][1], phase1['u'][3]], tensors):
        u = desc[handle]['u']
        assert u[1] == t['storage_ptr'] and u[2] == t['offset']
        assert u[1] + u[2] * t['itemsize'] == t['ptr']
        assert u[3:6] == [1, 20, 1] and u[6] == 3  # ACL_INT32
    parent = obs['inputs']['parent']
    for t in tensors[:2]:
        assert parent['ptr'] <= t['ptr'] < t['ptr'] + t['bytes'] <= parent['ptr'] + parent['bytes']
    rows = []
    hashes = set()
    for row in obs['rows']:
        scoped = [r for r in calls if r['scope'] == row['scope_id']]
        cache = one(r for r in scoped if r['kind'] == 7)
        launch = one(r for r in scoped if r['kind'] == 3)
        addr = [r['u'][0] for r in scoped if r['kind'] == 8]
        hashes.add(cache['u'][2])
        assert addr == [tensors[0]['storage_ptr'], tensors[1]['storage_ptr'], row['output']['storage_ptr']]
        assert launch['result'] == 0 and launch['u'][:2] == [0, 0]
        assert launch['u'][3] == env['stream_handle']
        assert cache['tid'] == env['main_tid'] != launch['tid']
        if row['index'] == 0:
            assert cache['u'][0] == 0 and phase1['u'][4] == 0
            assert launch['u'][2] == phase1['u'][5]
        else:
            assert cache['u'][0] != 0 and launch['u'][2] == cache['u'][0]
            assert not any(r['kind'] in (1, 2) for r in scoped)
        o = row['output']
        assert o['ptr'] + o['bytes'] <= parent['ptr'] or o['ptr'] >= parent['ptr'] + parent['bytes']
        assert row['exact']
        rows.append(dict(index=row['index'],cache_hit=bool(cache['u'][0]),
            host_sub_us=(row['sub_return_ns']-row['sub_begin_ns'])/1000,
            phase2_start_after_python_return_us=(launch['begin_ns']-row['sub_return_ns'])/1000,
            phase2_tid=launch['tid'], output_ptr=o['ptr']))
    assert len(rows) == 6 and len(hashes) == 1
    assert len({r['output_ptr'] for r in rows}) == 6
    return dict(rows=rows,phase1_us=(phase1['end_ns']-phase1['begin_ns'])/1000,
                main_tid=env['main_tid'],stream_handle=env['stream_handle'],lifetime=obs['lifetime'])


def contains(outer, inner, same_pid=True):
    return (outer.get('ph') == 'X' and outer['tid'] == inner['tid'] and
            (not same_pid or outer['pid'] == inner['pid']) and
            D(outer['ts']) <= D(inner['ts']) and
            D(inner['ts']) + D(str(inner.get('dur', 0))) <= D(outer['ts']) + D(str(outer['dur'])))


def profile_analysis(events, csv_rows, mode):
    for i, e in enumerate(events):
        e['_index'] = i
    def backward(category, kernel):
        end = one(e for e in events if e.get('cat') == category and e.get('ph') == 'f'
                  and e['pid'] == kernel['pid'] and e['tid'] == kernel['tid'] and D(e['ts']) == D(kernel['ts']))
        return one(e for e in events if e.get('cat') == category and e.get('ph') == 's' and e['id'] == end['id'])
    kernels = [e for e in events if e.get('ph') == 'X' and e.get('name') == 'aclnnSub_SubAiCore_Sub']
    assert len(kernels) == len(csv_rows) == 6
    out = []
    for kernel in kernels:
        torch_flow = backward('async_npu', kernel)
        host = one(e for e in events if e.get('name') == 'aclnnSub' and contains(e, torch_flow))
        aten = one(e for e in events if e.get('name') == 'aten::sub' and contains(e, host))
        scope = one(e for e in events if e.get('name', '').startswith('SUB/') and contains(e, aten))
        launch_flow = backward('HostToDevice', kernel)
        launch = one(e for e in events if e.get('name') == 'Node@launch' and contains(e, launch_flow))
        assert launch['args']['connection_id'] == kernel['args']['connection_id']
        dequeue = one(e for e in events if e.get('name') == 'Dequeue@aclnnSub' and contains(e, launch, same_pid=False))
        qend = one(e for e in events if e.get('cat') == 'async_task_queue' and e.get('ph') == 'f' and contains(dequeue, e))
        qstart = one(e for e in events if e.get('cat') == 'async_task_queue' and e.get('ph') == 's' and e['id'] == qend['id'])
        assert contains(host, qstart) and host['tid'] != launch['tid']
        csvrow = one(r for r in csv_rows if r['Name'] == kernel['name'] and D(r['Start Time(us)'].strip()) == D(kernel['ts']))
        assert D(csvrow['Duration(us)']) == D(str(kernel['dur']))
        assert int(csvrow['Stream ID']) == kernel['args']['Physic Stream Id']
        assert int(csvrow['Task ID']) == kernel['args']['Task Id']
        runtime = [e for e in events if e.get('name', '').startswith('AscendCL@') and contains(scope, e, same_pid=False)]
        sync = [e for e in runtime if 'Synchronize' in e['name']]
        copies = [e for e in runtime if e['name'] == 'AscendCL@aclrtMemcpy']
        assert len(sync) == (0 if mode == 'sub' else 1)
        assert len(copies) == (1 if mode == 'tolist' else 0)
        if mode == 'tolist':
            assert sync[0]['name'] == 'AscendCL@aclrtSynchronizeStream'
            assert D(sync[0]['ts']) + D(str(sync[0]['dur'])) <= D(copies[0]['ts'])
        if mode == 'sync':
            assert sync[0]['name'] == 'AscendCL@aclrtSynchronizeStreamWithTimeout'
        def brief(e):
            return dict(trace_index=e['_index'],name=e['name'],tid=e['tid'],ts_us=e['ts'],dur_us=e['dur'])
        out.append(dict(label=scope['name'],aten=brief(aten),launch=brief(launch),dequeue=brief(dequeue),
            queue_flow_id=qend['id'],kernel=brief(kernel),stream=kernel['args']['Physic Stream Id'],
            task_id=kernel['args']['Task Id'],block_num=int(csvrow['Block Num']),
            aten_return_before_kernel_start=D(aten['ts'])+D(str(aten['dur']))<D(kernel['ts']),
            scope_runtime=[brief(e) for e in sync+copies]))
    return sorted(out,key=lambda r:int(r['label'].split('/')[-1]))


def main():
    r02=ROOT/'results/delivery-r02';r03=ROOT/'results/delivery-r03'
    for p in (r02,r03):
        verify_manifest(p)
        for run in p.iterdir():
            if run.is_dir():
                assert read(run/'completed.json')['status']=='passed'
                env=read(run/'environment.json');integrity=read(run/'integrity.json')
                assert integrity['unchanged'] and env['sha256_before']==integrity['sha256_after']
                assert all(r['exact'] for r in read(run/'observations.json')['rows'])
    for entry in read(ROOT/'sources/manifest.json'):
        assert hashlib.sha256((ROOT/'sources'/entry['file']).read_bytes()).hexdigest()==entry['sha256']
    api=r02/'api-r02'
    result={'api':api_analysis(read(api/'api_calls.json'),read(api/'observations.json'),read(api/'environment.json')),
            'timings':{},'profiles':{}}
    artifacts=read(api/'opened_sub_artifacts.json')
    binary=one(a for a in artifacts if a['path'].endswith('.o'))
    metadata=one(a for a in artifacts if a['path'].endswith('_high_performance.json'))
    assert binary['sha256'] in json.dumps(metadata['metadata'])
    result['loaded_binary']={k:v for k,v in binary.items() if k!='metadata'}
    for mode in ('sub','tolist','sync'):
        rows=read(r02/f'plain-{mode}-r02/observations.json')['rows']
        assert len(rows)==101
        summary={}
        for name,a,b in [('sub','sub_begin_ns','sub_return_ns'),('consume','sub_begin_ns','consume_return_ns'),
                         ('outer_join','submit_done_ns','joined_ns'),('total','sub_begin_ns','joined_ns')]:
            vals=[(r[b]-r[a])/1000 for r in rows]
            summary[name]={'first_us':vals[0],'warm_median_us':median(vals[1:]),'warm_n':100}
        result['timings'][mode]=summary
        p=r03/f'profile-{mode}-r03'
        events=json.loads(gzip.decompress((p/'trace_view.json.gz').read_bytes()))
        with (p/'kernel_details.csv').open() as f: csv_rows=list(csv.DictReader(f))
        result['profiles'][mode]=profile_analysis(events,csv_rows,mode)
    (ROOT/'results/summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Verified: 10 completed runs, source/results hashes, 18 Level1 kernel joins, 303 plain outputs, API descriptors/cache.')


if __name__=='__main__': main()
