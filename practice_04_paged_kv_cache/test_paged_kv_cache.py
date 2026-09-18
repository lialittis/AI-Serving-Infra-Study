import unittest

from paged_kv_cache import BlockPool, BlockTable, Request


class PagedKVCacheTest(unittest.TestCase):
    def test_six_token_example(self):
        pool = BlockPool(num_blocks=4, block_size=4)
        self.assertEqual(pool.snapshot(), (None, None, None, None))
        request = Request("A", pool)
        request.append(["A0", "A1", "A2", "A3", "A4", "A5"])
        self.assertEqual(request.block_table.physical_blocks, (0, 1))
        self.assertEqual(pool.snapshot(), (
            ("A0", "A1", "A2", "A3"), ("A4", "A5", None, None), None, None,
        ))
        self.assertEqual([request.locate(i) for i in range(6)],
                         [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)])

    def test_fragmented_blocks_preserve_logical_order(self):
        pool = BlockPool(4, 4)
        a, b = Request("A", pool), Request("B", pool)
        a.append(f"A{i}" for i in range(6))
        b.append(f"B{i}" for i in range(4))
        before = pool.snapshot()
        a.append(["A6", "A7"])
        self.assertEqual(pool.free_count, 1)  # Filling a tail does not allocate.
        self.assertEqual(a.block_table.physical_blocks, (0, 1))
        a.append(["A8"])
        self.assertEqual(a.block_table.physical_blocks, (0, 1, 3))
        self.assertEqual(a.locate(7), (1, 3))
        self.assertEqual(a.locate(8), (3, 0))
        self.assertEqual(a.read_all(), [f"A{i}" for i in range(9)])
        self.assertEqual(b.read_all(), [f"B{i}" for i in range(4)])
        self.assertEqual(pool.snapshot()[0], before[0])  # No relocation of B0.
        self.assertEqual(pool.snapshot()[2], before[2])  # No writes into B's block.

    def test_release_and_reuse_clear_old_contents(self):
        pool = BlockPool(4, 4)
        a, b = Request("A", pool), Request("B", pool)
        a.append(f"A{i}" for i in range(6))
        b.append(["B0"])
        a.release()
        a.release()  # Request-level cleanup is idempotent.
        self.assertEqual(a.block_table.physical_blocks, ())
        self.assertEqual(a.num_tokens, 0)
        self.assertEqual(pool.free_count, 3)
        c = Request("C", pool)
        c.append(["C0"])
        self.assertEqual(c.locate(0), (0, 0))
        self.assertEqual(pool.snapshot()[0], ("C0", None, None, None))
        self.assertEqual(b.read_all(), ["B0"])
        with self.assertRaises(IndexError):
            a.read(0)
        with self.assertRaisesRegex(ValueError, "released"):
            a.append(["A6"])

    def test_out_of_capacity_append_is_atomic(self):
        pool = BlockPool(2, 4)
        request = Request("A", pool)
        request.append(["A0", "A1", "A2"])
        before = pool.snapshot()
        # Two additional blocks are needed, but only one is free.
        with self.assertRaises(MemoryError):
            request.append([f"A{i}" for i in range(3, 9)])
        self.assertEqual(pool.snapshot(), before)
        self.assertEqual(request.block_table.physical_blocks, (0,))
        self.assertEqual(request.num_tokens, 3)
        self.assertEqual(request.read_all(), ["A0", "A1", "A2"])
        request.append([f"A{i}" for i in range(3, 8)])  # Exact capacity fits.
        self.assertEqual(pool.free_count, 0)

    def test_logical_bounds_exclude_unused_tail_slots(self):
        pool = BlockPool(2, 4)
        request = Request("A", pool)
        request.append(["A0"])
        self.assertEqual(request.block_table.locate(3), (0, 3))
        self.assertIsNone(pool.read(0, 3))
        for index in (-1, 1, 3):
            with self.subTest(index=index), self.assertRaises(IndexError):
                request.read(index)
        with self.assertRaises(IndexError):
            request.block_table.locate(4)
        with self.assertRaises(IndexError):
            request.block_table.locate(-1)

    def test_empty_request_allocates_nothing(self):
        pool = BlockPool(1, 4)
        request = Request("A", pool)
        request.append([])
        self.assertEqual(request.block_table.physical_blocks, ())
        self.assertEqual(request.read_all(), [])
        self.assertEqual(pool.free_count, 1)
        request.release()
        self.assertEqual(pool.free_count, 1)

    def test_pool_exhaustion_and_invalid_access(self):
        pool = BlockPool(1, 2)
        block = pool.allocate()
        with self.assertRaises(MemoryError):
            pool.allocate()
        for offset in (-1, 2):
            with self.subTest(offset=offset), self.assertRaises(IndexError):
                pool.write(block, offset, "bad")
        for block_id in (-1, 1):
            with self.subTest(block_id=block_id), self.assertRaises(IndexError):
                pool.read(block_id, 0)
        pool.release(block)
        with self.assertRaises(ValueError):
            pool.release(block)
        with self.assertRaises(ValueError):
            pool.read(block, 0)

    def test_invalid_placeholder_does_not_modify_request(self):
        pool = BlockPool(2, 4)
        request = Request("A", pool)
        with self.assertRaises(ValueError):
            request.append(["A0", None])
        self.assertEqual(request.num_tokens, 0)
        self.assertEqual(pool.free_count, 2)

    def test_invalid_sizes(self):
        for num_blocks, block_size in [(0, 4), (4, 0), (-1, 4), (4, -1)]:
            with self.subTest(num_blocks=num_blocks, block_size=block_size):
                with self.assertRaises(ValueError):
                    BlockPool(num_blocks, block_size)
        with self.assertRaises(ValueError):
            BlockTable(BlockPool(1, 4)).ensure_capacity(-1)


if __name__ == "__main__":
    unittest.main()
