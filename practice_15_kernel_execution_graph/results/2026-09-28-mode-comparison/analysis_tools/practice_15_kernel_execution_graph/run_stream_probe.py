"""Small real-NPU control experiment for cross-stream event dependencies.

This is a separate calibration workload, not a substitute for the model trace.
Explicitly owns all tensors until the final wait; no allocator lifetime guessing.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    import torch_npu
    torch.npu.set_device(0)
    a, b = torch.npu.Stream(), torch.npu.Stream()
    event = torch.npu.Event()
    # Keep all allocations alive across the complete measured interval.
    x = torch.ones((256, 256), device='npu', dtype=torch.float32)
    y, z, out = torch.empty_like(x), torch.empty_like(x), torch.empty_like(x)
    torch.npu.synchronize()  # Initialization boundary, outside the measured graph.
    def tensor(t):
        return dict(data_ptr=str(t.data_ptr()), storage_ptr=str(t.untyped_storage().data_ptr()),
                    shape=list(t.shape), stride=list(t.stride()), device=str(t.device),
                    dtype=str(t.dtype), bytes=t.numel()*t.element_size())
    records = []
    @contextmanager
    def operation(name, stream, reads=(), writes=(), generation=None):
        with torch.profiler.record_function('P15/' + name):
            record = dict(name=name, label='P15/' + name, sequence=len(records),
                          raw_stream_handle=str(stream.npu_stream),
                          host_stream_id=str(stream.stream_id), pid=os.getpid(),
                          reads=list(reads), writes=list(writes), generation=generation,
                          host_start_ns=time.monotonic_ns())
            records.append(record)
            yield
            record['host_end_ns'] = time.monotonic_ns()
            record['event_handle'] = str(event.npu_event) if generation else None
    def workload():
        with torch.npu.stream(a):
            with operation('produce_y', a, ('x',), ('y',)):
                torch.add(x, 1, out=y)
            with operation('record_1', a, generation=1):
                event.record(a)
        with torch.npu.stream(b):
            with operation('wait_1', b, generation=1):
                b.wait_event(event)
            with operation('consume_y', b, ('y',), ('z',)):
                torch.mul(y, 2, out=z)
            # Reuse the same event: later waits refer to generation 2.
            with operation('record_2', b, generation=2):
                event.record(b)
        with torch.npu.stream(a):
            with operation('wait_2', a, generation=2):
                a.wait_event(event)
            with operation('consume_z', a, ('x', 'z'), ('out',)):
                torch.add(z, x, out=out)
            with operation('record_3', a, generation=3):
                event.record(a)
        with operation('host_wait_3', a, generation=3):
            event.synchronize()
    # Warm the exact kernels and stream/event path before profiling.
    workload()
    records.clear()
    config = torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1)
    with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
            schedule=torch_npu.profiler.schedule(wait=0, warmup=1, active=1, repeat=1),
            record_shapes=True, experimental_config=config,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(args.output / 'profiler'))) as prof:
        prof.step()
        workload()
        prof.step()
    correct = bool(torch.equal(out.cpu(), torch.full((256, 256), 5.0)))
    import inspect
    sources = {}
    for name, path in [('run_stream_probe.py', Path(__file__)),
                       ('streams.py', Path(inspect.getfile(torch_npu.npu.Stream)))]:
        data = path.read_bytes()
        (args.output / name).write_bytes(data)
        sources[name] = hashlib.sha256(data).hexdigest()
    result = dict(workload='controlled two-stream probe; not vLLM',
                  torch_version=torch.__version__, torch_npu_version=torch_npu.__version__,
                  pid=os.getpid(), records=records,
                  resources={k: tensor(v) for k, v in dict(x=x, y=y, z=z, out=out).items()},
                  source_sha256=sources, output_equals_five=correct,
                  lifetime='all four tensors retained until after final event wait',
                  extra_synchronization='initialization before profile; explicit measured event wait at end')
    (args.output / 'probe.json').write_text(json.dumps(result, indent=2) + '\n')
    if not correct:
        raise RuntimeError('probe output mismatch')
    print('Probe captured:', args.output, flush=True)


if __name__ == '__main__':
    main()
