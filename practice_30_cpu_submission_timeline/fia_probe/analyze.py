"""Verify file → loaded bytes → binary → entry → function → launch identities."""
from decimal import Decimal
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
D = lambda x: Decimal(str(x))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_case(rows, info, result, metadata):
    assert result['status'] == 'passed'
    assert info['sha256'] == metadata['sha256']
    assert len(metadata['kernelList']) == info['kernel_count']
    assert {r['scope'] for r in rows} == {1, 2}
    assert all(r['result'] == 0 for r in rows if r['kind'] != 'file_open')
    opened = [r for r in rows if r['kind'] == 'file_open' and r['text'] == info['object_path']]
    assert opened and all(r['result'] >= 0 for r in opened)
    loads = [r for r in rows if r['kind'] == 'aclrtBinaryLoadFromData']
    load, = loads
    assert load['scope'] == 1 and load['text'] == info['sha256']
    assert load['u'][1] == info['object_size']
    summaries = []
    for scope in (1, 2):
        def one(kind):
            match, = [r for r in rows if r['scope'] == scope and r['kind'] == kind]
            return match
        prepare = one('aclnnFusedInferAttentionScoreV3GetWorkspaceSize')
        execute = one('aclnnFusedInferAttentionScoreV3')
        entry = one('aclrtBinaryGetFunctionByEntry')
        launch = one('aclrtLaunchKernelWithHostArgs')
        assert prepare['tid'] == result['main_tid'] != execute['tid']
        assert entry['tid'] == launch['tid'] == execute['tid']
        assert prepare['end_ns'] <= execute['begin_ns']
        assert execute['begin_ns'] <= entry['begin_ns'] <= entry['end_ns'] <= launch['begin_ns']
        assert launch['end_ns'] <= execute['end_ns']
        assert load['end_ns'] <= entry['begin_ns']
        if scope == 1:
            assert prepare['begin_ns'] <= load['begin_ns'] <= load['end_ns'] <= prepare['end_ns']
        assert load['u'][0] == entry['u'][0]
        assert entry['u'][1] == launch['u'][0]
        selected, = [k for k in metadata['kernelList'] if k['tilingKey'] == entry['u'][2]]
        assert selected == info['selected_entry']
        assert execute['u'][2] == prepare['u'][1]
        assert execute['u'][1] == prepare['u'][0]
        assert execute['u'][3] == launch['u'][2]
        assert prepare['text'] == 'TND' and prepare['scalar'] == 0.125
        summaries.append(dict(scope=scope, entry=entry['u'][2], num_blocks=launch['u'][1],
                              args_size=launch['u'][3], placeholder_count=launch['u'][4],
                              workspace_bytes=prepare['u'][0],
                              prepare_tid=prepare['tid'], execute_tid=execute['tid'],
                              prepare_wall_us=(prepare['end_ns']-prepare['begin_ns'])/1000,
                              execute_wall_us=(execute['end_ns']-execute['begin_ns'])/1000))
    assert summaries[0]['entry'] == summaries[1]['entry']
    return dict(object_path=info['object_path'], sha256=info['sha256'],
                kernel_count=info['kernel_count'], calls=summaries,
                max_abs_error=result['rows'][0]['max_abs_error'])


def verify_model(evidence):
    link, launch = evidence['link'], evidence['launch']
    q, node, task = link['queue'], link['cann'], link['task']
    enqueue, dequeue = q['enqueue'], q['dequeue']
    assert enqueue['args']['correlation_id'] == dequeue['args']['correlation_id']
    assert q['id'] == task['queue_id'] == task['submission_queue_id']
    assert str(node['args']['connection_id']) == task['connection_id']
    assert str(dequeue['tid']) == str(node['tid']) == str(launch['tid'])
    assert launch['name'] == 'AscendCL@aclrtLaunchKernelWithHostArgs'
    assert task['name'] == 'FusedInferAttentionScore' and task['replay_launch_id'] is None
    def contained(child, parent):
        return D(parent['ts']) <= D(child['ts']) and D(child['ts']) + D(child['dur']) <= D(parent['ts']) + D(parent['dur'])
    assert contained(launch, node) and contained(node, dequeue)
    base = D(enqueue['ts'])
    timeline = []
    for label, event in [('enqueue', enqueue), ('dequeue', dequeue), ('Node@launch', node), ('aclrtLaunchKernelWithHostArgs', launch)]:
        start = D(event['ts']) - base
        timeline.append(dict(event=label, start_us=str(start), end_us=str(start+D(event['dur'])), tid=event['tid']))
    timeline.append(dict(event='NPU FIA', start_us=str(D(task['start_us'])-base),
                         end_us=str(D(task['end_us'])-base), stream=task['stream'], task=task['task_id']))
    return timeline


def main():
    selection = json.loads((ROOT / 'evidence/selection.json').read_text())
    for entry in selection['source_manifest'].values():
        assert sha(ROOT / entry['local']) == entry['sha256']
    for name, digest in selection['scripts_sha256'].items():
        if name != 'hook.so':  # Device-side build artifact is deliberately not distributed.
            assert sha(ROOT / name) == digest, name
    cases = {}
    for name, info in selection['cases'].items():
        assert sha(ROOT / info['metadata_local']) == info['metadata_sha256']
        cases[name] = verify_case(
            json.loads((ROOT / 'results' / name / 'api_calls.json').read_text()), info,
            json.loads((ROOT / 'results' / name / 'results.json').read_text()),
            json.loads((ROOT / info['metadata_local']).read_text()))
    outputs = [json.loads((ROOT / 'results' / name / 'results.json').read_text())['output']
               for name in ('plain-before', 'bf16-l42', 'plain-after')]
    assert outputs[0] == outputs[1] == outputs[2]
    model = json.loads((ROOT / 'evidence/model_launch.json').read_text())
    assert sha(ROOT.parent / 'report/attention/attention.json') == model['attention_report_sha256']
    summary = dict(model_timeline=verify_model(model), synthetic_cases=cases,
                   limits=['Model timing and synthetic binary selection are separate experiments.',
                           'The hook does not measure device execution or binary/argument DMA completion.',
                           'Binary hashing and wrappers perturb CPU time; no performance claim.'])
    (ROOT / 'evidence/summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('Verified three selection chains, model launch identity, source hashes, and recovery outputs.')


if __name__ == '__main__':
    main()
