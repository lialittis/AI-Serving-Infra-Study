"""Join block ownership, native completion waits and real KV device tasks.

CPU only. Reject missing ownership transitions, layers, waits or flow endpoints.
This validates a serial observation, not a general race detector.
"""
import argparse
from collections import Counter
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'practice_11_attention_execution'))
from analyze_attention import DeviceLinks, inside, number, only, read_trace, require


def load(p):
    return json.loads(p.read_text())


def end(event):
    return number(event['ts']) + number(event.get('dur', 0))


def analyze(run, records=None, trace=None):
    if records is None:
        records = [json.loads(line) for p in (run / 'events').glob('*.jsonl')
                   for line in p.read_text().splitlines()]
    require(not any(e['event'] == 'trace_error' for e in records), 'instrumentation errors')
    require(load(run / 'shutdown.json')['server_exit_code'] == 0, 'unclean service shutdown')
    require([(c['endpoint'], c['status']) for c in load(run / 'profile_control.json')] ==
            [('/start_profile', 200), ('/stop_profile', 200)], 'profiler control failed')
    configs = [e for e in records if e['event'] == 'scheduler_config']
    require(len(configs) == 1 and not configs[0]['async_scheduling'] and
            not configs[0]['prefix_caching'] and not configs[0]['chunked_prefill'],
            'expected serial scheduler without caching/chunked prefill')
    require(configs[0]['block_size'] == 128 and configs[0]['max_num_seqs'] == 1, 'wrong scheduler sizes')
    command_info = load(run / 'command.json')
    command = command_info['argv']
    mode = 'eager' if '--enforce-eager' in command else 'graph'
    require(command_info.get('mode', mode) == mode, 'mode/command mismatch')
    require(command[command.index('--num-gpu-blocks-override')+1] == '2',
            'small pool configuration missing')
    if mode == 'graph':
        require('--compilation-config' in command, 'graph compilation configuration missing')
        config = json.loads(command[command.index('--compilation-config')+1])
        require(config == {'mode': 3, 'cudagraph_mode': 'PIECEWISE',
                           'cudagraph_capture_sizes': [1], 'custom_ops': ['all']},
                'unexpected graph configuration')
    init = only([e for e in records if e['event'] == 'pool_init'], 'pool initialization')['state']
    require(init['num_blocks'] == 2 and init['free_queue'] == [1] and
            init['blocks'][0]['is_null'], 'expected B0 null and B1 usable')
    trace_path = only(list((run / 'profiler').rglob('trace_view.json')), 'trace file')
    events = read_trace(trace_path) if trace is None else trace
    kernel_path = only(list((run / 'profiler').rglob('kernel_details.csv')), 'kernel CSV')
    with kernel_path.open() as stream:
        kernels = list(csv.DictReader(stream))
    links = DeviceLinks(events, kernels)
    scopes = {e['name']: e for e in events if e.get('ph') == 'X' and e.get('name', '').startswith('P12/')}
    entries = [e for e in records if e['event'] == 'scope_enter']
    exits_list = [e for e in records if e['event'] == 'scope_exit']
    require(Counter(e['label'] for e in entries) == Counter(scopes.keys()) ==
            Counter(e['label'] for e in exits_list), 'missing, duplicate or unbalanced profiler scopes')
    exits = {e['label']: e for e in exits_list}
    selected, chains, requests = [], [], {}
    layers = {'model.layers.{}.self_attn.attn'.format(i) for i in range(24)}
    pool_identity = None
    for role in ('A', 'B'):
        body, response = load(run / ('request_'+role+'.json')), load(run / ('response_'+role+'.json'))
        require(response['usage']['prompt_tokens'] == 126 and response['usage']['completion_tokens'] == 2,
                'unexpected token counts')
        require(response['choices'][0]['finish_reason'] == 'length', 'unexpected stopping reason')
        own = [e for e in entries if e['role'] == role]
        rid = only(list({e['request_id'] for e in own}), 'request ID')
        require(rid.startswith(response['id']+'-') and body['request_id'] == 'practice12-'+role,
                'HTTP/request ID mismatch')
        schedule = sorted([e for e in records if e['event'] == 'schedule' and rid in e['scheduled_tokens']],
                          key=lambda e: e['step'])
        require([e['scheduled_tokens'][rid] for e in schedule] == [126, 1], 'wrong scheduled sequence')
        finish = only([e for e in records if e['event'] == 'request_finish' and e['request_id'] == rid], 'finish')
        require(finish['computed_tokens'] == 127 and finish['output_tokens'] == 2, 'wrong finish state')
        alloc = only([e for e in own if e['kind'] == 'pool_allocate' and
                      exits[e['label']]['affected_blocks']], 'nonempty pool allocation')
        freed = only([e for e in own if e['kind'] == 'pool_free' and
                      exits[e['label']]['affected_blocks']], 'nonempty pool free')
        for entry, before_ref, after_ref, before_queue, after_queue in (
            (alloc, 0, 1, [1], []), (freed, 1, 0, [], [1])):
            before, after = entry['before'], exits[entry['label']]['after']
            require(before['free_queue'] == before_queue and after['free_queue'] == after_queue,
                    'free queue transition mismatch')
            require(before['blocks'][1]['ref_cnt'] == before_ref and
                    after['blocks'][1]['ref_cnt'] == after_ref, 'reference count transition mismatch')
            require(exits[entry['label']]['affected_blocks'] == [1], 'wrong reused block')
            identity = (before['pool_id'], before['blocks'][1]['object_id'])
            if pool_identity is None:
                pool_identity = identity
            require(identity == pool_identity and identity == (after['pool_id'], after['blocks'][1]['object_id']),
                    'pool/block CPU identity changed')
        manager_free = only([e for e in own if e['kind'] == 'manager_free'], 'manager free')
        require(manager_free['blocks'] == [[1]], 'request released wrong blocks')
        alloc_scope, free_scope = scopes[alloc['label']], scopes[freed['label']]
        manager_scope = scopes[manager_free['label']]
        require(inside(manager_scope, free_scope), 'pool free not inside request release')
        steps = []
        for sched, phase in zip(schedule, ('prefill', 'decode')):
            step_id = sched['step']
            current = [e for e in own if e['step'] == step_id]
            dispatch = [e for e in current if e['kind'] == 'acl_dispatch']
            bodies = [e for e in records if e['event'] == 'partition_body_call' and
                      e['role'] == role and e['step'] == step_id]
            if mode == 'graph':
                partitions = {'submod_{}'.format(i) for i in range(0, 49, 2)}
                require(len(dispatch) == 25 and {e['partition'] for e in dispatch} == partitions,
                        'expected 25 observed ACL partition dispatches')
                require(all(e['wrapper_mode'] == 'PIECEWISE' for e in dispatch),
                        'unexpected ACL wrapper mode')
                if phase == 'prefill':
                    require(all(e['runtime_mode'] == 'NONE' for e in dispatch),
                            '126-token prefill unexpectedly replayed')
                    require(len(bodies) == 25 and {e['partition'] for e in bodies} == partitions,
                            'prefill compiled callable body coverage missing')
                else:
                    require(all(e['runtime_mode'] == 'PIECEWISE' and e['has_captured_graph'] and
                                e['graph_object_id'] is not None for e in dispatch),
                            'decode replay evidence missing')
                    require(not bodies, 'decode unexpectedly called Python partition bodies')
            else:
                require(not dispatch and not bodies, 'eager unexpectedly dispatched compiled partitions')
            attention = [e for e in current if e['kind'] == 'attention']
            require(len(attention) == 24 and {e['layer'] for e in attention} == layers,
                    'missing attention layer coverage')
            host = only([e for e in records if e['event'] == 'host_block_table' and
                         e['role'] == role and e['step'] == step_id], 'host block table')
            require(host['groups'] == [{'block_size': 128, 'rows': [[1]]}], 'host block table mismatch')
            task_chains = []
            storage = {}
            for attn in attention:
                layer = attn['layer']
                cache = attn['kv_cache']
                require(all(c['shape'] == [2,128,2,64] and c['device'] == 'npu:0' and
                            c['dtype'] == 'torch.bfloat16' for c in cache), 'unexpected cache layout')
                require(attn['seq_lens'] == [126 if phase == 'prefill' else 127], 'KV length mismatch')
                storage[layer] = cache
                fia = only([e for e in records if e['event'] == 'fia_inputs' and
                            e['role'] == role and e['step'] == step_id and e['layer'] == layer], 'FIA input')
                if phase == 'decode':
                    for k, c in zip(('key', 'value'), cache):
                        require(fia[k]['data_ptr'] == c['data_ptr'] and
                                fia[k]['storage_offset'] == c['storage_offset'] and
                                fia[k]['shape'] == [2,128,128], 'decode cache input mismatch')
                    require(fia['block_table'] is not None and fia['lengths'] == [127], 'decode metadata missing')
                else:
                    require(fia['block_table'] is None and fia['lengths'] == [126], 'wrong prefill source')
                pair = []
                for kind, op, kernel in (
                    ('cache_write', 'atb::_npu_reshape_and_cache', 'ReshapeAndCacheNdKernel'),
                    ('fia', 'npu::npu_fused_infer_attention_score', 'FusedInferAttentionScore')):
                    entry = only([e for e in current if e['kind'] == kind and e['layer'] == layer], kind)
                    if kind == 'cache_write':
                        for k, c in zip(('key_cache', 'value_cache'), cache):
                            require(entry['tensors'][k] == c, 'writer/cache storage mismatch')
                        require(entry['tensors']['slot_mapping']['data_ptr'] == attn['slot_mapping']['data_ptr'],
                                'writer slot metadata mismatch')
                    chain = links.core(scopes[entry['label']], op, kernel)
                    chain.update(role=role, phase=phase, step=step_id, layer=layer, access=kind)
                    chains.append(chain); task_chains.append(chain); pair.append(chain['kernel'])
                    selected.extend(chain['trace_events'])
                require(pair[0]['args']['Physic Stream Id'] == pair[1]['args']['Physic Stream Id'] and
                        end(pair[0]) <= number(pair[1]['ts']), 'KV write/read device order missing')
            to_list = only([e for e in current if e['kind'] == 'to_list'], 'native sampled ID transfer')
            wait_scope = scopes[to_list['label']]
            waits = [e for e in links.cpu if e['name'] == 'Event::synchronize' and inside(wait_scope, e)]
            wait = only(waits, 'native Event::synchronize')
            transfer = links.in_scope(wait_scope)
            copies = [c for c in transfer if c['kernel']['name'] == 'MEMCPY_ASYNC']
            record = only([c['kernel'] for c in transfer if c['kernel']['name'] == 'EVENT_RECORD'],
                          'native device event record')
            require(len(copies) == 1, 'expected one native D2H copy device task')
            last = max((c['kernel'] for c in task_chains), key=end)
            stream = last['args']['Physic Stream Id']
            require(all(c['kernel']['args']['Physic Stream Id'] == stream for c in task_chains),
                    'KV tasks span multiple streams; cannot apply this single-stream argument')
            require(all(c['kernel']['args']['Physic Stream Id'] == stream and
                        end(last) <= number(c['kernel']['ts']) and end(c['kernel']) <= end(wait) for c in copies),
                    'KV -> native transfer -> wait completion order missing')
            require(record['args']['Physic Stream Id'] == stream and
                    end(copies[0]['kernel']) <= number(record['ts']) and end(record) <= end(wait),
                    'native copy/event/wait completion order missing')
            require(end(last) <= end(wait) <= end(wait_scope), 'native wait/last KV access order mismatch')
            for c in transfer:
                selected.extend(c['trace_events'])
            selected.extend([wait,wait_scope])
            steps.append(dict(phase=phase, step=step_id, storage=storage, last_kv_access=last,
                              native_wait=wait, native_transfer_scope=wait_scope,
                              transfer_tasks=[c['kernel'] for c in transfer],
                              token_ids=exits[to_list['label']]['token_ids'],
                              execution='eager' if mode == 'eager' else
                                        ('compiled_callable' if phase == 'prefill' else 'graph_replay'),
                              graph_dispatches=[{k:e[k] for k in ('partition','runtime_mode','wrapper_mode',
                                                 'has_captured_graph','graph_object_id')} for e in dispatch],
                              partition_body_calls=len(bodies),
                              kv_accesses=len(task_chains), host_blocks=[1],
                              expected_slots=[128,253] if phase=='prefill' else [254,254]))
        require(end(steps[-1]['native_transfer_scope']) <= number(manager_scope['ts']),
                'request release precedes native result wait')
        require(end(alloc_scope) <= min(number(c['kernel']['ts']) for c in chains if c['role']==role),
                'device access precedes block allocation')
        selected.extend([alloc_scope,free_scope,manager_scope])
        requests[role] = dict(request_id=rid, allocation=alloc_scope, pool_free=free_scope,
                              manager_free=manager_scope, allocation_before=alloc['before'],
                              allocation_after=exits[alloc['label']]['after'],
                              free_before=freed['before'], free_after=exits[freed['label']]['after'],
                              steps=steps, response_text=response['choices'][0]['text'])
    a,b=requests['A'],requests['B']
    require(end(a['pool_free']) <= number(b['allocation']['ts']), 'B reallocated before A release finished')
    all_storage=[step['storage'] for req in requests.values() for step in req['steps']]
    require(all(s == all_storage[0] for s in all_storage), 'K/V storage changed between phases or requests')
    counts=Counter(e['name'] for e in links.device)
    require(counts['ReshapeAndCacheNdKernel']==counts['FusedInferAttentionScore']==96,
            'expected 24 layers x four forwards KV/FIA kernels')
    replay = links.replay_coverage()
    if mode == 'eager':
        require(not replay and not counts['MODEL_EXECUTE'], 'unexpected graph replay')
    else:
        require(replay and counts['MODEL_EXECUTE'] == 50, 'device graph replay tasks missing')
        a_graphs = {d['partition']:d['graph_object_id'] for d in a['steps'][1]['graph_dispatches']}
        b_graphs = {d['partition']:d['graph_object_id'] for d in b['steps'][1]['graph_dispatches']}
        require(a_graphs == b_graphs, 'A/B did not reuse the same captured graph objects')
        require(all(c['kernel']['args'].get('Model Id') in (None,4294967295) for c in chains),
                'KV tasks entered replay; current direct-call lifetime proof no longer applies')
    first_b=min((c['kernel'] for c in chains if c['role']=='B'),key=lambda k:number(k['ts']))
    last_a=a['steps'][-1]['last_kv_access']
    result=dict(mode=mode, requests=requests, reused_block=1, layers=24, verified_kv_operator_chains=len(chains),
                scopes=len(scopes), device_tasks=len(links.device), kernel_csv_rows=len(kernels),
                device_task_counts=dict(counts),
                replayed_tasks=len(replay),
                replayed_tasks_without_torch_flow=sum(not e['has_torch_flow_end'] for e in replay),
                replayed_tasks_without_cann_start=sum(not e['has_cann_flow_start'] for e in replay),
                replay_coverage=replay,
                kv_execution='direct calls outside replay in both modes',
                gap_last_A_access_to_free_us=str(number(a['pool_free']['ts'])-end(last_a)),
                gap_free_to_B_allocation_us=str(number(b['allocation']['ts'])-end(a['pool_free'])),
                gap_last_A_access_to_first_B_access_us=str(number(first_b['ts'])-end(last_a)),
                trace_file=str(trace_path.relative_to(run)),trace_sha256=hashlib.sha256(trace_path.read_bytes()).hexdigest(),
                limits=['serial requests, prefix caching and async scheduling disabled',
                        'graph mode is PIECEWISE capture [1], not full-graph or 126-token prefill replay',
                        'replay-internal FX/kernel attribution is not inferred from temporal proximity',
                        'KV access attribution is per kernel and layer, not instrumented device memory instructions',
                        'slot values are CPU-derived, no diagnostic NPU reads or extra device waits',
                        'observed ordering is not a proof for all workloads or a race detector'])
    excerpt=list({json.dumps(e,sort_keys=True):e for e in selected+list(scopes.values())+
                  [e for e in events if e.get('ph')=='M']}.values())
    return result,chains,excerpt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);args=p.parse_args()
    result,chains,excerpt=analyze(args.run)
    dest=args.run/'analysis';dest.mkdir(exist_ok=True)
    for name,data in [('reuse_evidence.json',result),('operator_links.json',chains),('lifetime_trace.json',excerpt)]:
        (dest/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    print('Verified A/B reuse of B1, all 24 layers, {} kernel chains and native completion waits.'.format(len(chains)))
    print('A last KV access end -> pool free: {} us'.format(result['gap_last_A_access_to_free_us']))


if __name__=='__main__':
    main()
