"""Estimate the raw memory occupied by a Transformer KV cache.

This calculator models a dense cache in which every sequence has the same length.
It excludes allocator, block-table, padding, fragmentation, and runtime overhead.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import Optional, Sequence


DTYPE_BITS = {
    "fp32": 32,
    "float32": 32,
    "fp16": 16,
    "float16": 16,
    "bf16": 16,
    "bfloat16": 16,
    "fp8": 8,
    "int8": 8,
    "int4": 4,
}


@dataclass(frozen=True)
class KVCacheConfig:
    layers: int
    batch_size: int
    sequence_length: int
    kv_heads: int
    head_dim: int
    dtype: str

    def validate(self) -> None:
        for name in ("layers", "batch_size", "sequence_length", "kv_heads", "head_dim"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name.replace('_', '-')} must be greater than zero")
        if self.dtype.lower() not in DTYPE_BITS:
            supported = ", ".join(sorted(DTYPE_BITS))
            raise ValueError(f"unsupported dtype {self.dtype!r}; choose one of: {supported}")


@dataclass(frozen=True)
class KVCacheEstimate:
    config: KVCacheConfig
    dtype_bits: int
    cached_scalars: int
    bytes_per_token_per_layer: Fraction
    bytes_per_token: Fraction
    bytes_per_sequence: Fraction
    total_bytes: Fraction


def estimate_kv_cache(config: KVCacheConfig) -> KVCacheEstimate:
    """Calculate cache size with exact integer/fractional arithmetic."""
    config.validate()
    dtype_bits = DTYPE_BITS[config.dtype.lower()]

    # The factor of 2 represents one key and one value scalar at each location.
    cached_scalars = (
        2
        * config.layers
        * config.batch_size
        * config.sequence_length
        * config.kv_heads
        * config.head_dim
    )
    bytes_per_token_per_layer = Fraction(
        2 * config.kv_heads * config.head_dim * dtype_bits, 8
    )
    bytes_per_token = bytes_per_token_per_layer * config.layers
    bytes_per_sequence = bytes_per_token * config.sequence_length
    total_bytes = bytes_per_sequence * config.batch_size

    return KVCacheEstimate(
        config=config,
        dtype_bits=dtype_bits,
        cached_scalars=cached_scalars,
        bytes_per_token_per_layer=bytes_per_token_per_layer,
        bytes_per_token=bytes_per_token,
        bytes_per_sequence=bytes_per_sequence,
        total_bytes=total_bytes,
    )


def derive_head_dim(
    head_dim: Optional[int], hidden_size: Optional[int], attention_heads: Optional[int]
) -> int:
    """Use an explicit head dimension or derive it from a model config."""
    if head_dim is not None:
        if hidden_size is not None or attention_heads is not None:
            raise ValueError(
                "use either --head-dim or both --hidden-size and --attention-heads"
            )
        return head_dim

    if hidden_size is None or attention_heads is None:
        raise ValueError(
            "provide --head-dim, or provide both --hidden-size and --attention-heads"
        )
    if hidden_size <= 0 or attention_heads <= 0:
        raise ValueError("hidden-size and attention-heads must be greater than zero")
    if hidden_size % attention_heads != 0:
        raise ValueError("hidden-size must be divisible by attention-heads")
    return hidden_size // attention_heads


def format_bytes(value: Fraction) -> str:
    """Format bytes using binary (KiB, MiB, GiB, TiB) units."""
    numeric = float(value)
    units = ("bytes", "KiB", "MiB", "GiB", "TiB", "PiB")
    unit_index = 0
    while numeric >= 1024 and unit_index < len(units) - 1:
        numeric /= 1024
        unit_index += 1
    if unit_index == 0:
        return f"{numeric:,.2f} {units[unit_index]}" if value.denominator != 1 else f"{int(value):,} bytes"
    return f"{numeric:,.2f} {units[unit_index]}"


def estimate_as_dict(estimate: KVCacheEstimate) -> dict:
    """Return a JSON-compatible representation of an estimate."""
    result = asdict(estimate.config)
    result.update(
        {
            "dtype_bits": estimate.dtype_bits,
            "cached_scalars": estimate.cached_scalars,
            "bytes_per_token_per_layer": float(estimate.bytes_per_token_per_layer),
            "bytes_per_token": float(estimate.bytes_per_token),
            "bytes_per_sequence": float(estimate.bytes_per_sequence),
            "total_bytes": float(estimate.total_bytes),
        }
    )
    return result


def print_report(estimate: KVCacheEstimate) -> None:
    config = estimate.config
    print("KV-cache estimate")
    print("=================")
    print(
        "Formula: 2 (K + V) × layers × batch × sequence × "
        "KV heads × head dimension × dtype bytes"
    )
    print()
    print("Inputs")
    print(f"  layers:             {config.layers:,}")
    print(f"  batch size:         {config.batch_size:,}")
    print(f"  sequence length:    {config.sequence_length:,} tokens")
    print(f"  KV heads:           {config.kv_heads:,}")
    print(f"  head dimension:     {config.head_dim:,}")
    print(f"  cache dtype:        {config.dtype} ({estimate.dtype_bits} bits)")
    print()
    print("Results")
    print(
        "  per token, layer:   "
        f"{format_bytes(estimate.bytes_per_token_per_layer)}"
    )
    print(f"  per token, all layers: {format_bytes(estimate.bytes_per_token)}")
    print(f"  per sequence:       {format_bytes(estimate.bytes_per_sequence)}")
    print(f"  total for batch:    {format_bytes(estimate.total_bytes)}")
    print(f"  exact total bytes:  {float(estimate.total_bytes):,.0f}")
    print()
    print(
        "Note: this is raw tensor storage. A serving runtime may use more memory "
        "because of block rounding, fragmentation, metadata, and temporary buffers."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, required=True, help="decoder layer count")
    parser.add_argument("--batch-size", type=int, default=1, help="concurrent sequences")
    parser.add_argument(
        "--sequence-length", type=int, required=True, help="cached tokens per sequence"
    )
    parser.add_argument(
        "--kv-heads",
        type=int,
        required=True,
        help="number of key/value heads (num_key_value_heads in many configs)",
    )
    dimension = parser.add_argument_group("head dimension")
    dimension.add_argument("--head-dim", type=int, help="dimension of each attention head")
    dimension.add_argument("--hidden-size", type=int, help="model hidden size")
    dimension.add_argument("--attention-heads", type=int, help="query attention-head count")
    parser.add_argument(
        "--dtype",
        default="bf16",
        type=str.lower,
        choices=sorted(DTYPE_BITS),
        help="KV-cache scalar type (default: bf16)",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        head_dim = derive_head_dim(args.head_dim, args.hidden_size, args.attention_heads)
        estimate = estimate_kv_cache(
            KVCacheConfig(
                layers=args.layers,
                batch_size=args.batch_size,
                sequence_length=args.sequence_length,
                kv_heads=args.kv_heads,
                head_dim=head_dim,
                dtype=args.dtype,
            )
        )
    except ValueError as error:
        parser.error(str(error))

    if args.json:
        print(json.dumps(estimate_as_dict(estimate), indent=2, sort_keys=True))
    else:
        print_report(estimate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
