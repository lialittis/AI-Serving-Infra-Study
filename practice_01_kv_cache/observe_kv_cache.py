"""Observe a decoder-only Transformer's KV cache using CPU-only PyTorch.

The public cache format intentionally resembles the legacy Hugging Face format:

    past_key_values[layer_index] == (key_tensor, value_tensor)

Both tensors use [batch, heads, cached_tokens, head_dim] layout.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor, nn


LayerCache = Tuple[Tensor, Tensor]
PastKeyValues = Tuple[LayerCache, ...]


@dataclass
class ModelOutput:
    logits: Tensor
    past_key_values: PastKeyValues
    attention_weights: Tuple[Tensor, ...]


class CausalSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def _split_heads(self, tensor: Tensor) -> Tensor:
        batch, tokens, _ = tensor.shape
        return tensor.view(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self, hidden_states: Tensor, past_key_value: Optional[LayerCache] = None
    ) -> Tuple[Tensor, LayerCache, Tensor]:
        query = self._split_heads(self.q_proj(hidden_states))
        new_key = self._split_heads(self.k_proj(hidden_states))
        new_value = self._split_heads(self.v_proj(hidden_states))

        past_tokens = 0
        if past_key_value is not None:
            past_key, past_value = past_key_value
            past_tokens = past_key.size(2)
            key = torch.cat((past_key, new_key), dim=2)
            value = torch.cat((past_value, new_value), dim=2)
        else:
            key, value = new_key, new_value

        # Query position i may see cached positions <= past_tokens + i.
        query_tokens = query.size(2)
        key_tokens = key.size(2)
        query_positions = past_tokens + torch.arange(query_tokens, device=query.device)
        key_positions = torch.arange(key_tokens, device=query.device)
        allowed = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)

        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~allowed.view(1, 1, query_tokens, key_tokens), float("-inf"))
        attention_weights = torch.softmax(scores, dim=-1)
        context = attention_weights @ value
        context = context.transpose(1, 2).contiguous().view(
            hidden_states.size(0), query_tokens, -1
        )
        return self.out_proj(context), (key, value), attention_weights


class DecoderBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_size)
        self.attention = CausalSelfAttention(hidden_size, num_heads)
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    def forward(
        self, hidden_states: Tensor, past_key_value: Optional[LayerCache]
    ) -> Tuple[Tensor, LayerCache, Tensor]:
        attention_output, cache, weights = self.attention(
            self.attention_norm(hidden_states), past_key_value
        )
        hidden_states = hidden_states + attention_output
        hidden_states = hidden_states + self.ffn(self.ffn_norm(hidden_states))
        return hidden_states, cache, weights


class TinyCausalLM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 32,
        hidden_size: int = 16,
        num_heads: int = 2,
        num_layers: int = 2,
        max_positions: int = 128,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_positions, hidden_size)
        self.layers = nn.ModuleList(
            DecoderBlock(hidden_size, num_heads) for _ in range(num_layers)
        )
        self.final_norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(
        self, input_ids: Tensor, past_key_values: Optional[PastKeyValues] = None
    ) -> ModelOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, tokens]")
        if past_key_values is not None and len(past_key_values) != len(self.layers):
            raise ValueError("past_key_values must contain one (key, value) pair per layer")

        past_tokens = 0 if past_key_values is None else past_key_values[0][0].size(2)
        positions = torch.arange(
            past_tokens, past_tokens + input_ids.size(1), device=input_ids.device
        )
        hidden_states = self.token_embedding(input_ids) + self.position_embedding(positions)

        next_cache = []
        all_weights = []
        for index, layer in enumerate(self.layers):
            layer_past = None if past_key_values is None else past_key_values[index]
            hidden_states, layer_cache, weights = layer(hidden_states, layer_past)
            next_cache.append(layer_cache)
            all_weights.append(weights)

        logits = self.lm_head(self.final_norm(hidden_states))
        return ModelOutput(logits, tuple(next_cache), tuple(all_weights))


def cache_bytes(cache: PastKeyValues) -> int:
    return sum(tensor.numel() * tensor.element_size() for pair in cache for tensor in pair)


def describe_cache(cache: PastKeyValues, show_values: int) -> None:
    for layer_index, (key, value) in enumerate(cache):
        print(f"  layer {layer_index}: K {list(key.shape)}, V {list(value.shape)}")
        print(
            f"    K[batch=0, head=0, newest token, :{show_values}] = "
            f"{key[0, 0, -1, :show_values].tolist()}"
        )
        print(
            f"    V[batch=0, head=0, newest token, :{show_values}] = "
            f"{value[0, 0, -1, :show_values].tolist()}"
        )
    print(f"  total cache memory: {cache_bytes(cache)} bytes")


def verify_cached_equals_full(model: TinyCausalLM, token_ids: Sequence[int]) -> float:
    complete = torch.tensor([token_ids], dtype=torch.long)
    full_logits = model(complete).logits

    cache: Optional[PastKeyValues] = None
    step_logits = []
    for token_id in token_ids:
        output = model(torch.tensor([[token_id]], dtype=torch.long), cache)
        cache = output.past_key_values
        step_logits.append(output.logits)
    cached_logits = torch.cat(step_logits, dim=1)
    return (full_logits - cached_logits).abs().max().item()


def parse_prompt(text: str) -> list[int]:
    try:
        tokens = [int(item.strip()) for item in text.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("prompt must be comma-separated integers") from error
    if not tokens:
        raise argparse.ArgumentTypeError("prompt must contain at least one token")
    return tokens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", type=parse_prompt, default=parse_prompt("1,5,9,2"))
    parser.add_argument("--decode-token", type=int, default=7)
    parser.add_argument("--show-values", type=int, default=4)
    args = parser.parse_args()

    torch.manual_seed(7)
    torch.set_num_threads(1)
    model = TinyCausalLM().eval()
    all_tokens = args.prompt + [args.decode_token]
    if min(all_tokens) < 0 or max(all_tokens) >= 32:
        parser.error("token IDs must be between 0 and 31")

    with torch.inference_mode():
        print("=== 1. PREFILL: process the entire prompt ===")
        print(f"input token IDs: {args.prompt}")
        prefill = model(torch.tensor([args.prompt], dtype=torch.long))
        describe_cache(prefill.past_key_values, args.show_values)
        print(
            "  attention shape at layer 0: "
            f"{list(prefill.attention_weights[0].shape)}  [batch, heads, queries, keys]"
        )

        old_cache = prefill.past_key_values
        print("\n=== 2. DECODE: process only one new token and reuse the cache ===")
        print(f"new input token ID: [{args.decode_token}]")
        decoded = model(
            torch.tensor([[args.decode_token]], dtype=torch.long),
            past_key_values=old_cache,
        )
        describe_cache(decoded.past_key_values, args.show_values)
        print(
            "  attention shape at layer 0: "
            f"{list(decoded.attention_weights[0].shape)}  [batch, heads, queries, keys]"
        )

        prefix_unchanged = all(
            torch.equal(old_key, new_key[:, :, :-1, :])
            and torch.equal(old_value, new_value[:, :, :-1, :])
            for (old_key, old_value), (new_key, new_value) in zip(
                old_cache, decoded.past_key_values
            )
        )
        max_difference = verify_cached_equals_full(model, all_tokens)

        print("\n=== 3. CORRECTNESS CHECKS ===")
        print(f"old K/V entries unchanged after append: {prefix_unchanged}")
        print(f"max |full logits - cached logits|: {max_difference:.3e}")
        print(f"numerically equivalent: {max_difference < 1e-5}")


if __name__ == "__main__":
    main()
