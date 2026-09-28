import copy
import unittest

from kv_ranges import KVIndices


def tensor(address, shape, generation):
    stride = []
    size = 1
    for n in reversed(shape):
        stride.insert(0, size)
        size *= n
    return dict(kind='tensor', device='npu:0', data_ptr=address,
                storage_generation=generation, element_size=4, shape=shape, stride=stride)


class KVInputTests(unittest.TestCase):
    def setUp(self):
        self.block = tensor(1000, [1, 2], 1)
        self.query = tensor(2000, [2], 2)
        self.positions = tensor(3000, [1], 3)
        self.slot = tensor(4000, [256], 4)
        self.entries = {'block-copy': {'monotonic_ns': 1}, 'query-copy': {'monotonic_ns': 3},
            'launch': dict(kind='launch', label='launch', monotonic_ns=10, named_arguments={
                'positions_ptr': self.positions, 'query_start_loc_ptr': self.query,
                'block_table_ptr': self.block, 'slot_mapping_ptr': self.slot,
                'TOTAL_CP_WORLD_SIZE': 1, 'TOTAL_CP_RANK': 0, 'num_tokens': 1,
                'block_size': 128, 'block_table_stride': 2})}
        self.returns = {'block-copy': {'monotonic_ns': 2}, 'query-copy': {'monotonic_ns': 4},
                        'launch': {'monotonic_ns': 11}}
        self.records = [dict(event='host_copy_values', label='block-copy', destination=self.block, values=[[2, 0]]),
                        dict(event='host_copy_values', label='query-copy', destination=self.query, values=[0, 1])]

    def test_final_position_contract_reconstructs_decode_slot(self):
        self.records.append(dict(event='slot_host_contract', label='launch',
                                 positions_tensor=self.positions, positions=[10]))
        kv = KVIndices(self.records, self.entries, self.returns)
        self.assertEqual(kv.slots[0]['values'], [266])

    def test_relative_query_position_cannot_replace_final_position(self):
        relative = tensor(5000, [1], 5)
        self.records.append(dict(event='host_copy_values', label='relative-copy', destination=relative, values=[0]))
        self.entries['relative-copy'] = {'monotonic_ns': 5}
        self.returns['relative-copy'] = {'monotonic_ns': 6}
        self.assertEqual(KVIndices(self.records, self.entries, self.returns).slots, [])

    def test_wrong_position_allocation_is_rejected(self):
        wrong = copy.deepcopy(self.positions)
        wrong['storage_generation'] = 99
        self.records.append(dict(event='slot_host_contract', label='launch', positions_tensor=wrong, positions=[10]))
        with self.assertRaisesRegex(ValueError, 'tensor mismatch'):
            KVIndices(self.records, self.entries, self.returns)


if __name__ == '__main__':
    unittest.main()
