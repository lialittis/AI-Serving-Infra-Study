"""Check that characterization never upgrades pending markers to race proof."""
import copy
import unittest
from analyze import summarize_trial


def trial():
    return dict(id="t00", mode="omit", threads=1, address_a="100", tensor_python_owner_released=True,
                copy_all_elements_correct=True,
                progress_after_release=dict(read_start=False, read_end=False, done=False),
                observations=[dict(index=0, pointer="100", reused=True,
                                   before=dict(read_start=False, read_end=False, done=False),
                                   after=dict(read_start=True, read_end=True, done=False))],
                statistics_before=dict(num_alloc_retries=0, num_ooms=0),
                statistics_after_window=dict(num_alloc_retries=0, num_ooms=0),
                trace_entries_for_a=[dict(action="alloc"), dict(action="free_requested"), dict(action="free_completed")],
                block_after_release=[dict(state="inactive")], block_after_synchronize=[dict(state="active_allocated")],
                block_after_poll=[dict(state="active_allocated")], allocations_after_completion=[],
                release=dict(thread_id=1), submitting_main_thread=1)


class InterpretationTests(unittest.TestCase):
    def test_terminal_pending_does_not_establish_pending_copy(self):
        result = summarize_trial(trial())
        self.assertTrue(result["reuse_while_terminal_marker_pending"])
        self.assertFalse(result["reuse_while_copy_end_marker_pending"])
        self.assertIn("no demonstrated race", result["interpretation"])

    def test_pointer_equality_must_match_recorded_boolean(self):
        sample = trial()
        sample["observations"][0]["pointer"] = "200"
        with self.assertRaisesRegex(ValueError, "address equality"):
            summarize_trial(sample)

    def test_storage_release_and_no_retry_are_required(self):
        sample = trial()
        sample["trace_entries_for_a"] = [dict(action="alloc")]
        with self.assertRaisesRegex(ValueError, "storage free"):
            summarize_trial(sample)
        sample = trial()
        sample["statistics_after_window"]["num_alloc_retries"] = 1
        with self.assertRaisesRegex(ValueError, "retry confounds"):
            summarize_trial(sample)

    def test_no_match_is_not_evidence_of_protection(self):
        sample = copy.deepcopy(trial())
        sample["observations"][0].update(pointer="200", reused=False)
        result = summarize_trial(sample)
        self.assertIsNone(result["first_reuse_index"])
        self.assertFalse(result["reuse_while_terminal_marker_pending"])


if __name__ == "__main__":
    unittest.main()
