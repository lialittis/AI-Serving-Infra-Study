"""Separate real DMA correctness run; never used for performance timing."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    import torch
    import torch_npu
    import vllm_ascend.vllm_ascend_C
    from vllm_ascend.simple_kv_offload.npu_mem_ops import build_params, copy_blocks
    from vllm_ascend.simple_kv_offload.worker import SimpleCPUOffloadNPUWorker
    torch.npu.set_device(0)
    matrix = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    actual = (matrix.to("npu") @ matrix.T.to("npu")).cpu()
    assert torch.equal(actual, matrix @ matrix.T)
    results = []
    for offset in (0, 128):
        views, expected = {}, {}
        keep_alive = []
        for i, page in enumerate((512, 768)):
            cpu = ((torch.arange(8*page+offset, dtype=torch.int32)+17*i) % 251).to(torch.uint8)
            raw = cpu.to("npu")
            keep_alive.append(raw)
            tensor = raw[offset:].view(8, page)
            name = "K" if i == 0 else "V"
            views.update(SimpleCPUOffloadNPUWorker._build_block_views(name, tensor, 8))
            expected[name] = cpu[offset:].view(8, page).view(torch.int8)
        host = {name: torch.zeros((8, v.shape[1]), dtype=torch.int8, pin_memory=True)
                for name, v in views.items()}
        assert all(x.is_pinned() for x in host.values())
        store_stream, load_stream = torch.npu.Stream(), torch.npu.Stream()
        store_stream.wait_stream(torch.npu.current_stream())
        src, mid, dst = [1, 3, 5], [4, 1, 6], [7, 2, 0]
        with torch.npu.stream(store_stream):
            copy_blocks(src, mid, build_params(views, host, 1))
            done = torch.npu.Event()
            done.record(store_stream)
        done.synchronize()
        for name in host:
            assert torch.equal(host[name][mid], expected[name][src])
        with torch.npu.stream(load_stream):
            load_stream.wait_event(done)
            copy_blocks(mid, dst, build_params(host, views, 0))
            loaded = torch.npu.Event()
            loaded.record(load_stream)
        loaded.synchronize()
        for name, v in views.items():
            restored = v.cpu()
            assert torch.equal(restored[dst], expected[name][src])
            untouched = [i for i in range(8) if i not in dst]
            assert torch.equal(restored[untouched], expected[name][untouched])
        results.append(dict(offset_bytes=offset, source_blocks=src, cpu_blocks=mid,
                            destination_blocks=dst, k_and_v_exact=True, untouched_exact=True))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(dict(status="passed", matrix_exact=True, cases=results), indent=2)+"\n")
    print(a.output.read_text())


if __name__ == "__main__":
    main()
