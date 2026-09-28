"""Reconstruct single-card KV indices from observed CPU staging copies.

These are source-contract indices, not memory-instruction observations. Missing
evidence returns None so callers keep whole-pool conservative bounds.
"""
import math


def flatten(values):
    if isinstance(values, list):
        return [x for child in values for x in flatten(child)]
    return [values]


class KVIndices:
    def __init__(self, records, entries, returns):
        host_positions = {e['label']: e for e in records if e['event'] == 'slot_host_contract'}
        self.copies = []
        for e in records:
            if e['event'] != 'host_copy_values':
                continue
            label = e['label']
            if label not in entries or label not in returns:
                raise ValueError('unpaired host staging copy')
            self.copies.append(dict(tensor=e['destination'], values=flatten(e['values']),
                                    time=returns[label]['monotonic_ns'], evidence=label))
        self.slots = []
        for e in sorted(entries.values(), key=lambda e: e['monotonic_ns']):
            args = e.get('named_arguments')
            if e.get('kind') != 'launch' or not args or 'slot_mapping_ptr' not in args:
                continue
            if args['TOTAL_CP_WORLD_SIZE'] != 1 or args['TOTAL_CP_RANK'] != 0:
                continue
            positions = self.values(args['positions_ptr'], e['monotonic_ns'])
            host = host_positions.get(e['label'])
            if host is not None:
                if host['positions_tensor'] != args['positions_ptr']:
                    raise ValueError('position contract tensor mismatch')
                positions = host['positions']
            queries = self.values(args['query_start_loc_ptr'], e['monotonic_ns'])
            blocks = self.values(args['block_table_ptr'], e['monotonic_ns'])
            if positions is None or queries is None or blocks is None:
                continue
            count = args['num_tokens']
            if queries[0] != 0 or queries[-1] != count or len(positions) != count:
                raise ValueError('slot mapping CPU input shape mismatch')
            if any(a > b for a, b in zip(queries, queries[1:])):
                raise ValueError('nonmonotonic query starts')
            values = []
            for row, (start, end) in enumerate(zip(queries, queries[1:])):
                for position in positions[start:end]:
                    if position < 0:
                        raise ValueError('negative position')
                    index = row * args['block_table_stride'] + position // args['block_size']
                    if index >= len(blocks):
                        raise ValueError('block table index out of bounds')
                    values.append(blocks[index] * args['block_size'] + position % args['block_size'])
            self.slots.append(dict(tensor=args['slot_mapping_ptr'], values=values,
                                   time=returns[e['label']]['monotonic_ns'], evidence=e['label']))

    def values(self, tensor, before, slots=False):
        if not isinstance(tensor, dict) or tensor.get('kind') != 'tensor':
            return None
        # Only contiguous integer input views in this single-card adapter.
        stride = 1
        for size, actual in reversed(list(zip(tensor['shape'], tensor['stride']))):
            if size > 1 and actual != stride:
                return None
            stride *= size
        count = math.prod(tensor['shape'])
        for record in sorted(self.slots if slots else self.copies,
                             key=lambda x: x['time'], reverse=True):
            source = record['tensor']
            if (record['time'] >= before or source['device'] != tensor['device']
                    or source['storage_generation'] != tensor['storage_generation']):
                continue
            delta = tensor['data_ptr'] - source['data_ptr']
            if delta < 0 or delta % tensor['element_size']:
                continue
            offset = delta // tensor['element_size']
            if offset + count <= len(record['values']):
                return record['values'][offset:offset + count]
        return None

    def pool_slots(self, entry):
        args = entry['kwargs']
        if entry['operator'] == 'atb::_npu_reshape_and_cache':
            return self.values(args['slot_indices'], entry['monotonic_ns'], slots=True)
        if entry['operator'] == 'npu::npu_fused_infer_attention_score' and args['block_table'] is not None:
            table = args['block_table']
            values = self.values(table, entry['monotonic_ns'])
            if values is None:
                return None
            lengths = args['actual_seq_lengths_kv']
            if table['shape'][0] != len(lengths):
                raise ValueError('attention batch/block table mismatch')
            slots = []
            for row, length in enumerate(lengths):
                for pos in range(length):
                    column = pos // args['block_size']
                    if column >= table['shape'][1]:
                        raise ValueError('attention block table overflow')
                    block = values[row * table['shape'][1] + column]
                    slots.append(block * args['block_size'] + pos % args['block_size'])
            return slots
        return None


def pool_ranges(tensor, slots, mode):
    shape, strides = tensor['shape'], tensor['stride']
    expected = [math.prod(shape[i + 1:]) for i in range(len(shape))]
    if len(shape) not in (3, 4) or strides != expected:
        raise ValueError('unsupported KV pool layout')
    row_bytes = math.prod(shape[2:]) * tensor['element_size']
    ranges = []
    for slot in sorted(set(slots)):
        if slot == -1:
            continue  # PAD_SLOT_ID
        if not 0 <= slot < shape[0] * shape[1]:
            raise ValueError('KV slot out of bounds')
        start = tensor['data_ptr'] + slot * row_bytes
        if ranges and ranges[-1]['hi'] == start:
            ranges[-1]['hi'] += row_bytes
        else:
            ranges.append(dict(device=tensor['device'], lo=start, hi=start + row_bytes,
                               generation=tensor['storage_generation'], mode=mode))
    return ranges
