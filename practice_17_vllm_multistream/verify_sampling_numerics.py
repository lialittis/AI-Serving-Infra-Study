"""Separate numerical check of installed sampler paths with identical positive q.

Only exponential draws are replaced with a fixed tensor. The installed stream
switch, event synchronization, softmax, division and argmax paths execute on NPU.
This is not a throughput run or a test of top-k/top-p filtering/distribution.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    import torch_npu  # noqa: F401
    import vllm_ascend.sample.sampler as installed
    torch.npu.set_device(0)
    source = Path(inspect.getfile(installed))
    (args.output / "sampler.py").write_bytes(source.read_bytes())
    generator = torch.Generator().manual_seed(710)
    records = []
    for batch, vocab in [(4, 257), (32, 151936), (64, 128256)]:
        logits_cpu = torch.randn((batch, vocab), generator=generator, dtype=torch.float32) * 2
        q_cpu = torch.rand((batch, vocab), generator=generator, dtype=torch.float32) + .25
        probs_cpu = logits_cpu.softmax(-1)
        reference = (probs_cpu / q_cpu).argmax(-1)
        logits = logits_cpu.to("npu")
        fixed_q = q_cpu.to("npu")
        torch.npu.synchronize()
        for enabled in (False, True):
            config = SimpleNamespace(enable_reduce_sample=False, enable_async_exponential=enabled)
            # Avoid engine/distributed configuration: initialize only the state
            # used by the installed methods under test.
            top = installed.AscendTopKTopPSampler.__new__(installed.AscendTopKTopPSampler)
            torch.nn.Module.__init__(top)
            top.logprobs_mode = "raw_logprobs"
            top.apply_top_k_top_p = installed.apply_top_k_top_p
            top.top_k = None
            parent = installed.AscendSampler.__new__(installed.AscendSampler)
            torch.nn.Module.__init__(parent)
            parent.topk_topp_sampler = top
            parent.async_exponential_event = torch.npu.Event()
            def fixed_draw(tensor, *unused, **kwargs):
                return tensor.copy_(fixed_q)
            with patch.object(installed, "get_ascend_config", return_value=config):
                with patch.object(torch.Tensor, "exponential_", fixed_draw):
                    if enabled:
                        parent.do_async_exponential(batch, vocab, {})
                    actual, _ = top.forward_native(logits.clone(), {}, None, None)
                    actual_cpu = actual.cpu()
                    torch.testing.assert_close(actual_cpu, reference, rtol=0, atol=0)
                    identity = bool(torch.equal(top.q.cpu(), q_cpu)) if enabled else None
                    if enabled and not identity:
                        raise AssertionError("async q differs from fixed q")
                # Also exercise real exponential generation and confirm that the
                # published q is finite/positive after its own completion event.
                parent.do_async_exponential(batch, vocab, {})
                parent.async_exponential_event.synchronize()
                real_q = top.q.cpu()
                finite_positive = bool(torch.isfinite(real_q).all() and (real_q > 0).all())
                if not finite_positive:
                    raise AssertionError("non-positive or non-finite exponential q")
            records.append({"batch": batch, "vocab": vocab, "enabled": enabled,
                            "exact_argmax_match": True, "async_fixed_q_equal": identity,
                            "real_exponential_finite_positive": finite_positive,
                            "actual_tokens": actual_cpu.tolist()})
    result = {"status": "passed", "cases": records,
              "source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "scope": "Installed sampling consumer and stream/event paths with fixed q; real q finite/positive check. k/p=None; not a distribution or full-model accuracy test."}
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
