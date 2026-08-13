import unittest

import torch

from observe_kv_cache import TinyCausalLM, cache_bytes, verify_cached_equals_full


class KVCacheTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        torch.set_num_threads(1)
        self.model = TinyCausalLM().eval()

    def test_cache_shape_and_growth(self) -> None:
        with torch.inference_mode():
            prefill = self.model(torch.tensor([[1, 5, 9, 2]]))
            self.assertEqual(len(prefill.past_key_values), 2)
            for key, value in prefill.past_key_values:
                self.assertEqual(tuple(key.shape), (1, 2, 4, 8))
                self.assertEqual(tuple(value.shape), (1, 2, 4, 8))

            decoded = self.model(torch.tensor([[7]]), prefill.past_key_values)
            for key, value in decoded.past_key_values:
                self.assertEqual(tuple(key.shape), (1, 2, 5, 8))
                self.assertEqual(tuple(value.shape), (1, 2, 5, 8))

    def test_old_entries_are_preserved(self) -> None:
        with torch.inference_mode():
            before = self.model(torch.tensor([[1, 5, 9, 2]])).past_key_values
            after = self.model(torch.tensor([[7]]), before).past_key_values
            for (old_key, old_value), (new_key, new_value) in zip(before, after):
                self.assertTrue(torch.equal(old_key, new_key[:, :, :-1, :]))
                self.assertTrue(torch.equal(old_value, new_value[:, :, :-1, :]))

    def test_cached_and_full_execution_match(self) -> None:
        with torch.inference_mode():
            difference = verify_cached_equals_full(self.model, [1, 5, 9, 2, 7])
        self.assertLess(difference, 1e-5)

    def test_cache_memory_formula(self) -> None:
        with torch.inference_mode():
            cache = self.model(torch.tensor([[1, 5, 9, 2]])).past_key_values
        # 2 (K,V) * 2 layers * 1 batch * 2 heads * 4 tokens * 8 dims * 4 bytes
        self.assertEqual(cache_bytes(cache), 1024)


if __name__ == "__main__":
    unittest.main()
