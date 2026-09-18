"""Small NPU arithmetic check; this is not a hardware health certification."""

import importlib.metadata
import sys


def main():
    import torch
    import torch_npu  # noqa: F401 - explicitly register the Ascend backend

    print("Python:", sys.executable, sys.version)
    for package in ("torch", "torch-npu", "vllm", "vllm-ascend", "triton-ascend"):
        print(package, importlib.metadata.version(package))
    print("NPU available:", torch.npu.is_available())
    print("NPU count:", torch.npu.device_count())
    if not torch.npu.is_available():
        raise RuntimeError("NPU is unavailable; check device mapping and CANN environment")
    print("Logical device 0:", torch.npu.get_device_name(0))
    x = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    actual = (x.to("npu:0") @ x.to("npu:0")).cpu()
    torch.testing.assert_close(actual, x @ x)
    print("NPU_MATMUL_PASS", actual.tolist())


if __name__ == "__main__":
    main()
