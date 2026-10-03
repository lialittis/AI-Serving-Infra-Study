"""Check that corruption claims stay tied to element counts and mode controls."""
import copy
import unittest
from analyze import summarize_trial

ELEMENTS = 8


def trial(mode="omit", outcome="intact", reused=True):
    original = ELEMENTS if outcome == "intact" else 0
    sentinel = ELEMENTS if outcome == "fully_overwritten" else 0
    if outcome == "mixed":
        original, sentinel = 3, ELEMENTS - 3
    other = ELEMENTS if outcome == "unexpected_values" else 0
    if outcome == "unexpected_values":
        original = sentinel = 0
    return dict(id="t00", mode=mode, address_a="100", tensor_python_owner_released=True,
                write_target_all_sentinel=True, elements=ELEMENTS,
                observed_original_count=original, observed_sentinel_count=sentinel,
                observed_other_count=other, observed_outcome=outcome,
                progress_after_release=dict(read_start=False, read_end=False, done=False),
                observations=[dict(index=0, pointer="100" if reused else "200", reused=reused,
                                   before=dict(read_start=False, read_end=False, done=False),
                                   after=dict(read_start=False, read_end=False, done=False))],
                statistics_before=dict(num_alloc_retries=0, num_ooms=0),
                statistics_after_window=dict(num_alloc_retries=0, num_ooms=0),
                trace_entries_for_a=[dict(action="alloc"), dict(action="free_requested")],
                block_after_release=[dict(state="inactive")],
                block_after_synchronize=[dict(state="active_allocated")],
                write_index=0, write_address_reused=reused, write_pointer="100" if reused else "200",
                release=dict(thread_id=1), submitting_main_thread=1)


class ClassificationTests(unittest.TestCase):
    def test_omit_overwritten_reports_consumed_candidate_data(self):
        result = summarize_trial(trial("omit", "fully_overwritten"))
        self.assertTrue(result["write_address_reused"])
        self.assertIn("read storage after the candidate write", result["interpretation"])

    def test_omit_intact_is_reported_not_hidden(self):
        result = summarize_trial(trial("omit", "intact"))
        self.assertEqual(result["observed_outcome"], "intact")

    def test_mixed_outcome_is_classified(self):
        result = summarize_trial(trial("omit", "mixed"))
        self.assertEqual(result["observed_outcome"], "mixed")

    def test_unexpected_values_are_not_interpreted_as_overwrite(self):
        result = summarize_trial(trial("omit", "unexpected_values"))
        self.assertIn("neither", result["interpretation"])


class ModeControlTests(unittest.TestCase):
    def test_record_reuse_is_a_failure(self):
        with self.assertRaisesRegex(ValueError, "prevent in-window"):
            summarize_trial(trial("record", "intact", reused=True))

    def test_record_intact_without_reuse_passes(self):
        result = summarize_trial(trial("record", "intact", reused=False))
        self.assertFalse(result["write_address_reused"])

    def test_join_corruption_is_a_failure(self):
        with self.assertRaisesRegex(ValueError, "protection path"):
            summarize_trial(trial("join", "fully_overwritten"))

    def test_join_requires_reuse(self):
        with self.assertRaisesRegex(ValueError, "did not reuse"):
            summarize_trial(trial("join", "intact", reused=False))

    def test_synced_corruption_is_a_failure(self):
        with self.assertRaisesRegex(ValueError, "checker control"):
            summarize_trial(trial("synced", "fully_overwritten"))

    def test_write_target_must_retain_sentinel(self):
        sample = trial()
        sample["write_target_all_sentinel"] = False
        with self.assertRaisesRegex(ValueError, "sentinel"):
            summarize_trial(sample)

    def test_element_counts_must_cover_observed(self):
        sample = trial()
        sample["observed_original_count"] = ELEMENTS - 1
        with self.assertRaisesRegex(ValueError, "cover observed"):
            summarize_trial(sample)

    def test_storage_release_is_required(self):
        sample = trial()
        sample["trace_entries_for_a"] = [dict(action="alloc")]
        with self.assertRaisesRegex(ValueError, "storage free"):
            summarize_trial(sample)


if __name__ == "__main__":
    unittest.main()
