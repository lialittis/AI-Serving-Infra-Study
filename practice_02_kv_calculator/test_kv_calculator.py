import contextlib
import io
import json
import unittest

from kv_calculator import (
    KVCacheConfig,
    derive_head_dim,
    estimate_kv_cache,
    format_bytes,
    main,
)


class KVCalculatorTest(unittest.TestCase):
    def test_practice_01_configuration(self) -> None:
        estimate = estimate_kv_cache(
            KVCacheConfig(
                layers=2,
                batch_size=1,
                sequence_length=4,
                kv_heads=2,
                head_dim=8,
                dtype="fp32",
            )
        )
        self.assertEqual(estimate.total_bytes, 1024)
        self.assertEqual(estimate.bytes_per_token, 256)

    def test_gqa_example_is_one_gibibyte(self) -> None:
        estimate = estimate_kv_cache(
            KVCacheConfig(
                layers=32,
                batch_size=1,
                sequence_length=8192,
                kv_heads=8,
                head_dim=128,
                dtype="bf16",
            )
        )
        self.assertEqual(estimate.total_bytes, 1024**3)
        self.assertEqual(format_bytes(estimate.total_bytes), "1.00 GiB")

    def test_batch_and_sequence_scale_linearly(self) -> None:
        base = estimate_kv_cache(KVCacheConfig(2, 1, 4, 2, 8, "fp16"))
        larger = estimate_kv_cache(KVCacheConfig(2, 3, 8, 2, 8, "fp16"))
        self.assertEqual(larger.total_bytes, base.total_bytes * 6)

    def test_int4_uses_half_byte_per_scalar(self) -> None:
        estimate = estimate_kv_cache(KVCacheConfig(1, 1, 1, 1, 1, "int4"))
        # Two scalars, one each for K and V, occupy one byte in total.
        self.assertEqual(estimate.total_bytes, 1)

    def test_derives_head_dimension(self) -> None:
        self.assertEqual(derive_head_dim(None, 4096, 32), 128)
        self.assertEqual(derive_head_dim(128, None, None), 128)

    def test_rejects_incomplete_or_inconsistent_dimensions(self) -> None:
        with self.assertRaisesRegex(ValueError, "provide --head-dim"):
            derive_head_dim(None, 4096, None)
        with self.assertRaisesRegex(ValueError, "divisible"):
            derive_head_dim(None, 100, 32)
        with self.assertRaisesRegex(ValueError, "either --head-dim"):
            derive_head_dim(128, 4096, 32)

    def test_rejects_non_positive_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "sequence-length"):
            estimate_kv_cache(KVCacheConfig(2, 1, 0, 2, 8, "fp16"))

    def test_json_cli(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            return_code = main(
                [
                    "--layers", "2",
                    "--sequence-length", "4",
                    "--kv-heads", "2",
                    "--head-dim", "8",
                    "--dtype", "fp32",
                    "--json",
                ]
            )
        result = json.loads(output.getvalue())
        self.assertEqual(return_code, 0)
        self.assertEqual(result["total_bytes"], 1024.0)


if __name__ == "__main__":
    unittest.main()
