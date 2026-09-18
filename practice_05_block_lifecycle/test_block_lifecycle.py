import io
import unittest
from contextlib import redirect_stdout
from dataclasses import FrozenInstanceError

from block_lifecycle import BlockHandle, BlockPool


class BlockLifecycleTest(unittest.TestCase):
    def setUp(self):
        self.output = io.StringIO()
        redirect = redirect_stdout(self.output)
        redirect.__enter__()
        self.addCleanup(redirect.__exit__, None, None, None)

    def test_lifecycle_logs_and_identity(self):
        pool = BlockPool(1, 4)
        a = pool.allocate("A")
        pool.write("A", a, 0, "A0")
        pool.free("A", a)
        b = pool.allocate("B")
        pool.write("B", b, 0, "B0")
        self.assertEqual(a.block_id, b.block_id)
        self.assertNotEqual(a, b)
        self.assertEqual((a.generation, b.generation), (1, 2))
        self.assertEqual(self.output.getvalue().splitlines(), [
            "ALLOC request=A block=0 generation=1",
            "WRITE request=A block=0 generation=1 offset=0 value=A0",
            "FREE  request=A block=0 generation=1",
            "ALLOC request=B block=0 generation=2",
            "WRITE request=B block=0 generation=2 offset=0 value=B0",
        ])

    def test_freed_handle_rejects_all_operations_before_reuse(self):
        pool = BlockPool(1, 4)
        a = pool.allocate("A")
        pool.free("A", a)
        for operation in (
            lambda: pool.read("A", a, 0),
            lambda: pool.write("A", a, 0, "old"),
            lambda: pool.free("A", a),
        ):
            with self.assertRaisesRegex(ValueError, "FREE"):
                operation()

    def test_stale_handle_cannot_read_write_or_free_new_allocation(self):
        # Reusing the SAME request name proves owner checks alone are insufficient.
        pool = BlockPool(1, 4)
        old = pool.allocate("A")
        pool.free("A", old)
        current = pool.allocate("A")
        pool.write("A", current, 0, "new")
        for operation in (
            lambda: pool.read("A", old, 0),
            lambda: pool.write("A", old, 0, "old"),
            lambda: pool.free("A", old),
        ):
            before = self.output.getvalue()
            with self.assertRaisesRegex(ValueError, "stale handle"):
                operation()
            self.assertEqual(pool.read("A", current, 0), "new")
            self.assertEqual(self.output.getvalue(), before)
        with self.assertRaises(MemoryError):
            pool.allocate("B")

    def test_reuse_clears_every_slot(self):
        pool = BlockPool(1, 4)
        a = pool.allocate("A")
        for offset in range(4):
            pool.write("A", a, offset, f"A{offset}")
        pool.free("A", a)
        b = pool.allocate("B")
        self.assertEqual([pool.read("B", b, i) for i in range(4)], [None] * 4)

    def test_generations_are_per_block_and_increase_only_on_allocation(self):
        pool = BlockPool(2, 1)
        a, b = pool.allocate("A"), pool.allocate("B")
        self.assertEqual((a, b), (BlockHandle(0, 1), BlockHandle(1, 1)))
        with self.assertRaises(MemoryError):
            pool.allocate("C")
        pool.free("A", a)
        c = pool.allocate("C")
        self.assertEqual(c, BlockHandle(0, 2))
        pool.write("B", b, 0, "B0")
        pool.free("C", c)
        self.assertEqual(pool.allocate("D"), BlockHandle(0, 3))
        self.assertEqual(pool.read("B", b, 0), "B0")

    def test_wrong_owner_cannot_access_current_handle(self):
        pool = BlockPool(1, 4)
        a = pool.allocate("A")
        for operation in (
            lambda: pool.read("B", a, 0),
            lambda: pool.write("B", a, 0, "wrong"),
            lambda: pool.free("B", a),
        ):
            with self.assertRaisesRegex(ValueError, "belongs to request=A"):
                operation()
        pool.write("A", a, 0, "A0")
        self.assertEqual(pool.read("A", a, 0), "A0")

    def test_invalid_inputs_and_immutable_handle(self):
        for sizes in ((0, 4), (1, 0), (-1, 4)):
            with self.assertRaises(ValueError):
                BlockPool(*sizes)
        pool = BlockPool(1, 4)
        with self.assertRaises(ValueError):
            pool.allocate("")
        a = pool.allocate("A")
        with self.assertRaises(FrozenInstanceError):
            a.generation = 2
        for offset in (-1, 4):
            with self.assertRaises(IndexError):
                pool.write("A", a, offset, "bad")
            with self.assertRaises(IndexError):
                pool.read("A", a, offset)
        for block_id in (-1, 1):
            with self.assertRaises(IndexError):
                pool.free("A", BlockHandle(block_id, 1))
        with self.assertRaises(ValueError):
            pool.write("A", a, 0, None)
        self.assertIsNone(pool.read("A", a, 0))


if __name__ == "__main__":
    unittest.main()
