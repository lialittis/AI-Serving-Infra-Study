import io
import unittest
from contextlib import redirect_stdout

from stale_block_access import Allocator, StaleBlockAccess, main


class StaleBlockAccessTest(unittest.TestCase):
    def setUp(self):
        self.output = io.StringIO()
        redirect = redirect_stdout(self.output)
        redirect.__enter__()
        self.addCleanup(redirect.__exit__, None, None, None)

    def test_demo_reports_the_exact_lifetime_violation(self):
        main()
        self.assertEqual(self.output.getvalue(), (
            "ALLOC A B0:g1\n\n"
            "A remembers B0:g1\n\n"
            "FREE  A B0:g1\n\n"
            "ALLOC B B0:g2\n"
            "WRITE B B0:g2\n\n"
            "A READ B0:g1\n"
            "STALE BLOCK ACCESS\n"
            "request=A\n"
            "block=0\n"
            "expected_generation=1\n"
            "current_generation=2\n"
        ))

    def test_stale_read_reports_original_request_and_preserves_new_data(self):
        allocator = Allocator()
        a = allocator.allocate("A")
        allocator.free(a)
        b = allocator.allocate("B")
        allocator.write(b, "B's KV data")
        with self.assertRaises(StaleBlockAccess) as caught:
            allocator.read(a)
        error = caught.exception
        self.assertEqual((error.request_id, error.block_id,
                          error.expected_generation, error.current_generation),
                         ("A", 0, 1, 2))
        self.assertEqual(a.generation, 1)  # The old reference never becomes g2.
        self.assertEqual(allocator.read(b), "B's KV data")

    def test_stale_write_and_free_cannot_damage_current_allocation(self):
        allocator = Allocator()
        a = allocator.allocate("A")
        allocator.free(a)
        b = allocator.allocate("B")
        allocator.write(b, "new data")
        for operation in (lambda: allocator.write(a, "old data"),
                          lambda: allocator.free(a)):
            with self.assertRaises(StaleBlockAccess):
                operation()
            self.assertEqual(allocator.read(b), "new data")
        with self.assertRaises(MemoryError):
            allocator.allocate("C")

    def test_same_request_name_still_requires_a_new_generation(self):
        allocator = Allocator()
        old = allocator.allocate("A")
        allocator.free(old)
        current = allocator.allocate("A")
        with self.assertRaises(StaleBlockAccess):
            allocator.read(old)
        allocator.write(current, "current A data")
        self.assertEqual(allocator.read(current), "current A data")

    def test_live_access_and_other_blocks_do_not_trigger_false_alarms(self):
        allocator = Allocator(2)
        a, b = allocator.allocate("A"), allocator.allocate("B")
        allocator.write(b, "B data")
        allocator.free(a)
        c = allocator.allocate("C")
        self.assertEqual((c.block_id, c.generation), (0, 2))
        self.assertIsNone(allocator.read(c))
        self.assertEqual(allocator.read(b), "B data")
        allocator.write(c, "C data")
        self.assertEqual(allocator.read(c), "C data")

    def test_free_without_reuse_is_a_state_error_not_a_generation_mismatch(self):
        allocator = Allocator()
        a = allocator.allocate("A")
        allocator.free(a)
        for operation in (lambda: allocator.read(a), lambda: allocator.free(a)):
            with self.assertRaisesRegex(ValueError, "FREE"):
                operation()


if __name__ == "__main__":
    unittest.main()
