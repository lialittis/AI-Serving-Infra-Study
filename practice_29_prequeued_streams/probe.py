"""Tiny real-device qualification, including an intentional watchdog release."""
import argparse
from contextlib import nullcontext
import faulthandler
import os
from pathlib import Path
import time

from gate import Gate
from common import save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    import torch
    import torch_npu
    torch.npu.set_device(0)
    streams = [torch.npu.Stream(), torch.npu.Stream()]
    x = torch.ones(1024, device='npu')
    outputs = [torch.empty_like(x) for _ in streams]
    for stream, output in zip(streams, outputs):
        with torch.npu.stream(stream):
            torch.add(x, 2, out=output)
    torch.npu.synchronize()
    records = []
    for rescue in (False, True):
        gate = Gate(streams, timeout=0.25 if rescue else 2)
        ends = [torch.npu.Event() for _ in streams]
        profiler = (nullcontext() if rescue else torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
            experimental_config=torch_npu.profiler._ExperimentalConfig(
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(a.output / 'profiler'))))
        try:
            with profiler:
                with torch.profiler.record_function('P29/gate_close'):
                    gate.close()
                for role, (stream, output, end) in enumerate(zip(streams, outputs, ends)):
                    with torch.npu.stream(stream), torch.profiler.record_function(f'P29/tiny/{role}'):
                        torch.add(x, 2, out=output)
                        end.record()
                    gate.stamp('python_submit_return', role=role)
                time.sleep(0.05)  # Qualification only: the gate must still be closed.
                assert gate.query() == [1, 1], 'work escaped the closed gate'
                with torch.profiler.record_function('P29/gate_release'):
                    if rescue:
                        # Deliberately wait for the safety mechanism, no compute thread.
                        gate.timer.join(timeout=2)
                    else:
                        gate.open()
                for end in ends:
                    end.synchronize()
            assert gate.rescue == rescue
            assert gate.query() == [0, 0]
            assert all(torch.equal(output.cpu(), torch.full((1024,), 3.0)) for output in outputs)
            records.append(dict(intentional_rescue=rescue, passed=True, log=gate.log))
        finally:
            gate.open()
            torch.npu.synchronize()
            gate.destroy()
            save(a.output / 'gate_checks.json', records)
    save(a.output / 'status.json', dict(status='passed', pid=os.getpid()))


if __name__ == '__main__':
    faulthandler.enable()
    main()
