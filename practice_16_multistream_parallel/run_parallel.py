"""Observe independent matmul/vector kernels on one versus two real NPU streams.

No model, graph capture, threads, custom kernels, or core-count tuning. Every
tensor remains alive until both streams finish; only submission streams change.
"""
import argparse
from contextlib import contextmanager
import datetime
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--pairs', type=int, default=6)
    parser.add_argument('--rounds', type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.pairs <= 32 or not 1 <= args.rounds <= 10:
        parser.error('pairs must be 1..32; rounds must be 1..10')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output/'sources').mkdir()
    for phase in ['before']:
        result = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True, timeout=30)
        save(output/('device_'+phase+'.json'), dict(exit_code=result.returncode,
                                                  stdout=result.stdout, stderr=result.stderr))
    import torch
    import torch_npu
    torch.npu.set_device(0)
    # Some torch-npu builds expose this option through a setter only.
    original_format = getattr(torch_npu.npu.config, 'allow_internal_format', None)
    torch_npu.npu.config.allow_internal_format = False
    streams = {'A': torch.npu.Stream(), 'B': torch.npu.Stream()}
    # Distinct live storages. Constant, exactly representable inputs make it easy
    # to check every output element after the measured work has completed.
    tensors = {
        'matrix': torch.full((4096, 4096), 1/64, device='npu', dtype=torch.float16),
        'matrix_out': torch.empty((4096, 4096), device='npu', dtype=torch.float16),
        'vector': torch.full((64*1024*1024,), .25, device='npu', dtype=torch.float32),
        'vector_out': torch.empty((64*1024*1024,), device='npu', dtype=torch.float32),
    }
    torch.npu.synchronize()  # All initialization is complete before either stream reads.
    records, trials, checks = [], [], []
    @contextmanager
    def operation(trial, name, stream_name, **extra):
        stream = streams[stream_name]
        label = 'P16/'+trial+'/'+name
        with torch.profiler.record_function(label):
            record = dict(label=label, trial=trial, name=name, stream_name=stream_name,
                          raw_stream_handle=str(stream.npu_stream),
                          host_stream_id=str(stream.stream_id), pid=os.getpid(),
                          sequence=len(records), host_start_ns=time.monotonic_ns(), **extra)
            records.append(record)
            yield record
            record['host_end_ns'] = time.monotonic_ns()

    def workload(mode, trial):
        # synchronize is a round boundary, never inserted between the branches.
        torch.npu.synchronize()
        begin = time.monotonic_ns()
        for i in range(args.pairs):
            with torch.npu.stream(streams['A']):
                with operation(trial, 'mm-%02d'%i, 'A', kind='compute', branch='mm',
                               reads=['matrix'], writes=['matrix_out']):
                    torch.mm(tensors['matrix'], tensors['matrix'], out=tensors['matrix_out'])
            destination = 'A' if mode == 'serial' else 'B'
            with torch.npu.stream(streams[destination]):
                with operation(trial, 'mul-%02d'%i, destination, kind='compute', branch='mul',
                               reads=['vector'], writes=['vector_out']):
                    torch.mul(tensors['vector'], 1.5, out=tensors['vector_out'])
        # Record BOTH terminal markers before the CPU waits for either one.
        # Distinct events per round avoid ambiguous record generations.
        ends = []
        for name in (['A'] if mode == 'serial' else ['A', 'B']):
            event = torch.npu.Event()
            with operation(trial, 'record-'+name, name, kind='event_record') as rec:
                event.record(streams[name])
                rec['event_handle'] = str(event.npu_event)
            ends.append((name, event))
        for name, event in ends:
            with operation(trial, 'wait-'+name, name, kind='host_wait',
                           event_handle=str(event.npu_event)):
                event.synchronize()
        trials.append(dict(id=trial, mode=mode, host_elapsed_ns=time.monotonic_ns()-begin,
                           pairs=args.pairs))

    def validate(trial):
        # Outside each measured trial. Checks all elements, then copies just two booleans.
        good = dict(trial=trial, matrix_all_one=bool(torch.all(tensors['matrix_out']==1).item()),
                    vector_all_point375=bool(torch.all(tensors['vector_out']==.375).item()))
        checks.append(good)
        if not good['matrix_all_one'] or not good['vector_all_point375']:
            raise RuntimeError('incorrect result: '+str(good))

    for mode in ['serial', 'parallel']:
        workload(mode, 'warmup-'+mode)
        validate('warmup-'+mode)
    records.clear()
    trials.clear()
    checks.clear()
    config = torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1)
    with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
            schedule=torch_npu.profiler.schedule(wait=0, warmup=1, active=1, repeat=1),
            record_shapes=True, experimental_config=config,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(output/'profiler'))) as prof:
        prof.step()
        for i in range(args.rounds):
            # Alternate order to avoid always giving one mode the earlier position.
            for mode in (['serial', 'parallel'] if i%2==0 else ['parallel', 'serial']):
                trial = '%02d-%s'%(i+1, mode)
                workload(mode, trial)
                with torch.profiler.record_function('P16/check/'+trial):
                    validate(trial)
        prof.step()
    def tensor(t):
        return dict(data_ptr=str(t.data_ptr()), storage_ptr=str(t.untyped_storage().data_ptr()),
                    bytes=t.numel()*t.element_size(), shape=list(t.shape), stride=list(t.stride()),
                    dtype=str(t.dtype), device=str(t.device))
    sources = {}
    for name, path in [('run_parallel.py', Path(__file__)),
                       ('streams.py', Path(inspect.getfile(torch_npu.npu.Stream)))]:
        data = path.read_bytes()
        (output/'sources'/name).write_bytes(data)
        sources['sources/'+name] = hashlib.sha256(data).hexdigest()
    env_keys = ['ASCEND_HOME_PATH','ASCEND_RT_VISIBLE_DEVICES','ASCEND_VISIBLE_DEVICES',
                'TASK_QUEUE_ENABLE','ASCEND_LAUNCH_BLOCKING','PYTORCH_NPU_ALLOC_CONF']
    save(output/'run.json', dict(
        captured_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        command=[sys.executable]+sys.argv, torch_version=torch.__version__,
        torch_npu_version=torch_npu.__version__, python=sys.version, machine=platform.machine(),
        device=torch.npu.get_device_name(0), environment={k:os.environ.get(k) for k in env_keys},
        allow_internal_format=dict(original=original_format, effective=False),
        rounds=args.rounds, pairs=args.pairs, pid=os.getpid(), trials=trials,
        records=records, checks=checks, resources={k:tensor(v) for k,v in tensors.items()},
        source_sha256=sources, lifetime='all storages retained through both terminal waits',
        workload='independent matmul and scalar multiply; real torch-npu eager operators'))
    print('Captured and verified:', output, flush=True)


if __name__ == '__main__':
    main()
