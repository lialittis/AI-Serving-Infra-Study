"""Join KV tensor storage, allocator history, and actual CANN memory calls.

This audit deliberately targets the recorded single-card expandable-segment
initialization. It rejects incomplete native coverage instead of inferring APIs.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def only(items, message):
    require(len(items) == 1, message + ': expected 1, got %s' % len(items))
    return items[0]


def load(path):
    return json.loads(path.read_text())


def json_lines(directory):
    return [json.loads(line) for p in sorted(directory.glob('*.jsonl')) for line in p.read_text().splitlines()]


def covered(start, end, intervals):
    cursor = start
    for left, right in sorted(intervals):
        if left <= cursor < right:
            cursor = right
        if cursor >= end:
            return True
    return False


def analyze(run, events=None, native=None, after=None):
    events = json_lines(run/'events') if events is None else events
    native = json_lines(run/'native') if native is None else native
    native = sorted(native, key=lambda e: (e['pid'], e['begin_ns']))
    before = load(run/'allocator_before.json')
    after = load(run/'allocator_after.json') if after is None else after
    require(not any(e['event']=='trace_error' for e in events), 'instrumentation error')
    require(load(run/'ready.json')['inference_requests']==0, 'unexpected inference request')
    require(load(run/'shutdown.json')['exit_code']==0, 'unclean shutdown')
    for name, meta in load(run/'source_manifest.json').items():
        require(hashlib.sha256((run/name).read_bytes()).hexdigest()==meta['sha256'], 'source snapshot mismatch')
    entries = {e['label']:e for e in events if e['event']=='enter'}
    exits = {e['label']:e for e in events if e['event']=='exit'}
    require(set(entries)==set(exits), 'unbalanced observations')
    for event, items in [('enter', entries), ('exit', exits)]:
        require(Counter(e['label'] for e in events if e['event']==event)==Counter(items.keys()), 'duplicate observation')
    def pair(kind):
        entry = only([e for e in entries.values() if e['kind']==kind], kind)
        return entry, exits[entry['label']]
    def calls(entry, out):
        return [e for e in native if (e['pid'],e['tid'])==(entry['pid'],entry['tid']) and
                entry['monotonic_ns']<=e['begin_ns']<=e['end_ns']<=out['monotonic_ns']]

    budget_entry, budget = pair('budget')
    config_entry, config_out = pair('config')
    config = only(config_out['returned'], 'single-worker config')
    worker_entry, _ = pair('worker_initialize')
    pool_entry, pool = pair('pool')
    _, raw_pool = pair('raw_pool')
    view_entry, views = pair('views')
    bind_entry, bound = pair('bind')
    _, block_pool = pair('block_pool')
    require(worker_entry['sleep_mode'] is False, 'unexpected custom sleep allocator')
    require('expandable_segments:True' in pool_entry['effective_allocation_config'], 'expected expandable-segment baseline')
    require(budget['available_bytes']==budget['requested_memory']-budget['profile_result']['non_kv_cache_memory']-
            budget['graph_estimate_applied'], 'KV budget arithmetic')
    planned = sum(t['size'] for t in config['kv_cache_tensors'])
    blocks = config['num_blocks']
    page_bytes = planned//blocks
    require(planned%blocks==0 and blocks==budget['available_bytes']//page_bytes, 'block capacity arithmetic')
    require(config==worker_entry['config']==pool_entry['config'], 'KV config propagation')
    require(raw_pool['returned']==view_entry['raw_tensors'], 'raw tensor handoff')
    require(views['returned']==pool['returned']==pool['bound']==bound['kv_caches']==bound['bound'], 'view/bind storage mismatch')
    require(not calls(view_entry, views) and not calls(bind_entry, bound), 'reshape/bind unexpectedly allocates via CANN')
    require(block_pool['num_blocks']==blocks and block_pool['null_block']==0 and
            block_pool['free_blocks']==blocks-1, 'CPU block pool accounting')

    history = only([trace for trace in after['device_traces'] if trace], 'single device history')
    require(len(history)<100000, 'history ring may have overflowed')
    require(set(e['action'] for e in history)<= {'snapshot','segment_map','alloc'}, 'unexpected KV history action')
    snapshots = {e['name']:e for e in events if e['event']=='snapshot'}
    mapped = [(s['address'],s['address']+s['total_size']) for s in before['segments']]
    tensor_roles = {}
    for layer, tensors in raw_pool['returned'].items():
        require(len(tensors)==2, 'expected separate K and V')
        for role, tensor in zip(('K','V'), tensors):
            require(tensor['data_ptr'] not in tensor_roles, 'aliased raw KV storage')
            tensor_roles[tensor['data_ptr']] = (layer,role)
    raw_entries = sorted([e for e in entries.values() if e['kind']=='raw_tensor'], key=lambda e:e['monotonic_ns'])
    require(len(raw_entries)==len(tensor_roles)==48, '24-layer K/V coverage')
    rows = []
    for entry in raw_entries:
        out = exits[entry['label']]
        tensor = out['returned']
        address, requested = tensor['data_ptr'], entry['requested_bytes']
        require(address in tensor_roles, 'tensor pointer not in raw pool')
        layer, role = tensor_roles[address]
        require(layer==entry['layer'] and tensor['nbytes']==requested and tensor['storage_ptr']==address,
                'raw storage metadata mismatch')
        alloc = only([e for e in history if e['action']=='alloc' and e['addr']==address], 'allocator alloc identity')
        require(alloc['size']==requested, 'allocator requested bytes mismatch')
        segment = only([s for s in after['segments'] if s['address']<=address<s['address']+s['total_size']], 'allocator segment')
        require(segment['is_expandable'], 'non-expandable segment')
        block = only([b for b in segment['blocks'] if b['address']==address], 'allocator block')
        require(block['state']=='active_allocated' and block['requested_size']==requested and block['size']>=requested,
                'allocator block size/state mismatch')
        native_calls = calls(entry, out)
        require(all(e['result']==0 for e in native_calls), 'failed native allocation')
        require(set(e['api'] for e in native_calls)=={'aclrtMallocPhysical','aclrtMapMem'}, 'native allocation path/coverage')
        physical = [e for e in native_calls if e['api']=='aclrtMallocPhysical']
        maps = sorted([e for e in native_calls if e['api']=='aclrtMapMem'], key=lambda e:e['address'])
        require(len(maps)==len(physical)>0, 'physical/map coverage')
        require(len({e['extra'] for e in maps})==len(maps) and
                {e['extra'] for e in maps}=={e['address'] for e in physical}, 'physical handle link bijection')
        for mapping in maps:
            allocation = only([e for e in physical if e['address']==mapping['extra']], 'physical handle link')
            require(allocation['bytes']==mapping['bytes'] and allocation['end_ns']<=mapping['begin_ns'], 'physical/map ordering or bytes')
        start, size = maps[0]['address'], sum(e['bytes'] for e in maps)
        require(all(a['address']+a['bytes']==b['address'] for a,b in zip(maps,maps[1:])), 'noncontiguous native mapping')
        history_map = only([e for e in history if e['action']=='segment_map' and e['addr']==start], 'allocator/native map address')
        require(history_map['size']==size, 'allocator/native mapped bytes mismatch')
        require(history.index(history_map)<history.index(alloc), 'allocator map/alloc ordering')
        require(not any(max(start,a)<min(start+size,b) for a,b in mapped), 'new mapping overlaps existing mapping')
        reused = sum(max(0,min(address+requested,b)-max(address,a)) for a,b in mapped)
        mapped.append((start,start+size))
        require(covered(address,address+block['size'],mapped), 'tensor storage not fully backed by mappings')
        reservation = only([e for e in native if e['api']=='aclrtReserveMemAddress' and e['result']==0 and
                            e['pid']==entry['pid'] and e['end_ns']<=entry['monotonic_ns'] and
                            e['address']<=address and address+block['size']<=e['address']+e['bytes']], 'virtual reservation identity')
        view = views['returned'][layer][0 if role=='K' else 1]
        require(view['data_ptr']==address and view['storage_ptr']==address and view['nbytes']==requested and
                view['dtype']=='torch.bfloat16' and view['shape']==[blocks,128,2,64], 'raw-to-BF16 view mismatch')
        rows.append(dict(label=entry['label'],layer=layer,role=role,address=address,requested_bytes=requested,
                         allocator_block_bytes=block['size'],new_mapped_bytes=size,reused_mapped_bytes=reused,
                         physical_calls=len(physical),physical_chunk_bytes=sorted({e['bytes'] for e in physical}),
                         shape=view['shape'],reservation=reservation,allocator_allocation=alloc,
                         allocator_map=history_map,native_calls=native_calls,
                         host_begin_ns=entry['monotonic_ns'],host_end_ns=out['monotonic_ns']))
    require(sum(r['requested_bytes'] for r in rows)==planned, 'planned/actual KV bytes mismatch')
    require(len([e for e in history if e['action']=='alloc'])==len(rows), 'unaccounted allocator allocations')
    before_stats = snapshots['allocator_before']['statistics']
    after_stats = snapshots['allocator_after']['statistics']
    allocated_delta = after_stats['allocated_bytes.all.current']-before_stats['allocated_bytes.all.current']
    reserved_delta = after_stats['reserved_bytes.all.current']-before_stats['reserved_bytes.all.current']
    require(allocated_delta==sum(r['allocator_block_bytes'] for r in rows), 'allocated-byte reconciliation')
    require(reserved_delta==sum(r['new_mapped_bytes'] for r in rows), 'reserved-byte reconciliation')
    pool_calls = calls(pool_entry,pool)
    require(len(pool_calls)==sum(len(r['native_calls']) for r in rows), 'unattributed pool native calls')
    summary = dict(run=run.name,num_blocks=blocks,free_blocks=block_pool['free_blocks'],layers=len(config['kv_cache_tensors']),
                   tensors=len(rows),kv_budget_bytes=budget['available_bytes'],kv_tensor_bytes=planned,
                   bytes_per_block_all_layers=page_bytes,budget_remainder_bytes=budget['available_bytes']-planned,
                   allocated_delta_bytes=allocated_delta,reserved_delta_bytes=reserved_delta,
                   physical_calls=sum(r['physical_calls'] for r in rows),
                   native_api_counts=dict(Counter(e['api'] for e in pool_calls)),
                   expandable_segments=True,new_allocations_during_reshape_bind=0,inference_requests=0)
    return dict(summary=summary,budget=budget,config=config,cpu_block_pool=block_pool,
                rows=rows,native_before_pool=[e for e in native if e['end_ns']<pool_entry['monotonic_ns']],
                before_statistics=before_stats,after_statistics=after_stats,
                limits=['Device virtual addresses and physical-memory handles, not HBM physical page addresses.',
                        'CANN call boundaries are host monotonic times, not device DMA timestamps.',
                        'Audit covers selected torch-npu-to-CANN allocation bindings; driver/firmware internals are opaque.',
                        'Initialization with observation overhead; no inference or performance claim.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    result = analyze(args.run)
    output = args.run/'analysis'
    output.mkdir(exist_ok=True)
    (output/'allocation_evidence.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (output/'summary.json').write_text(json.dumps(result['summary'],ensure_ascii=False,indent=2)+'\n')
    fields = ['layer','role','address','requested_bytes','allocator_block_bytes','reused_mapped_bytes',
              'new_mapped_bytes','physical_calls']
    with (output/'allocations.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        writer.writerows({k:row[k] for k in fields} for row in result['rows'])
    print(json.dumps(result['summary'],ensure_ascii=False))


if __name__=='__main__':
    main()
