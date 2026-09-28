"""Measure actual ATB KV-cache writes: token count, launch cores and pipeline metrics.

One explicit NPU stream, eager, BF16, no model or custom kernel. Cache allocation
and correctness readback stay outside timed trials. Hardware profiling is separate
from unprofiled completed-work timing. No manual blockDim/core-count tuning.
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
import random
import subprocess
import sys
import time


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--tokens', default='1,4,10,16,24,32,48,64,128,256')
    p.add_argument('--rounds', type=int, default=7)
    p.add_argument('--iterations', type=int, default=100)
    p.add_argument('--profile-repeats', type=int, default=3)
    args = p.parse_args()
    tokens = sorted(set(int(x) for x in args.tokens.split(',')))
    if not tokens or min(tokens) < 1 or max(tokens) > 512 or not 1 <= args.rounds <= 30 or not 1 <= args.iterations <= 1000 or not 1 <= args.profile_repeats <= 10:
        p.error('tokens: 1..512, rounds: 1..30, iterations: 1..1000, profile repeats: 1..10')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / 'sources').mkdir()
    def device_info(name):
        r = subprocess.run(['npu-smi','info'], text=True, capture_output=True, check=True, timeout=30)
        (out / name).write_text(r.stdout)
    device_info('device_before.txt')
    import torch
    import torch_npu
    torch.npu.set_device(0)
    torch_npu.npu.config.allow_internal_format = False
    props = torch.npu.get_device_properties(0)
    hardware = {k:getattr(props, k, None) for k in ('name','cube_core_num','vector_core_num','multi_processor_count','total_memory','L2_cache_size')}
    stream = torch.npu.Stream()
    # Same per-layer layout as Qwen2.5-0.5B, with a smaller physical pool.
    block_size, num_blocks, kv_heads, head_size = 128, 8, 2, 64
    sentinel = -7
    key_cache = torch.full((num_blocks, block_size, kv_heads, head_size), sentinel, dtype=torch.bfloat16, device='npu')
    value_cache = torch.full_like(key_cache, sentinel)
    def describe(t):
        return dict(shape=list(t.shape), stride=list(t.stride()), dtype=str(t.dtype), device=str(t.device),
                    data_ptr=str(t.data_ptr()), storage_ptr=str(t.untyped_storage().data_ptr()), bytes=t.numel()*t.element_size())
    cases = {}
    for count in tokens:
        # Row id is exactly representable at BF16 precision for the tested range
        # after conversion; full expected tensors use those same converted bits.
        t = torch.arange(count, dtype=torch.float32).reshape(count,1,1)
        h = torch.arange(kv_heads, dtype=torch.float32).reshape(1,kv_heads,1)
        d = torch.arange(head_size, dtype=torch.float32).reshape(1,1,head_size)
        key_cpu = (t + h/2 + d/128).to(torch.bfloat16).contiguous()
        value_cpu = (-t - h/2 - d/128 - 2).to(torch.bfloat16).contiguous()
        slot_cpu = torch.arange(128,128+count,dtype=torch.int32)
        # Construct expected storage on CPU; compare the ENTIRE pool, including
        # untouched prefix/suffix, not merely selected rows or a checksum.
        expected_k = torch.full((num_blocks*block_size,kv_heads,head_size), sentinel,dtype=torch.bfloat16)
        expected_v = expected_k.clone()
        expected_k[slot_cpu.long()] = key_cpu
        expected_v[slot_cpu.long()] = value_cpu
        cases[count] = dict(key_cpu=key_cpu, value_cpu=value_cpu, slot_cpu=slot_cpu,
                            key=key_cpu.to('npu'), value=value_cpu.to('npu'), slots=slot_cpu.to('npu'),
                            expected_k=expected_k.reshape_as(key_cache), expected_v=expected_v.reshape_as(value_cache))
    torch.npu.synchronize()  # initialization boundary on default stream
    source_paths = dict(run_core_probe=Path(__file__), streams=Path(inspect.getfile(torch_npu.npu.Stream)),
                        device_utils=Path(inspect.getfile(torch.npu.get_device_properties)),
                        profiler_config=Path(inspect.getfile(torch_npu.profiler._ExperimentalConfig)))
    import torch_npu.op_plugin.atb._atb_ops as atb
    source_paths['atb_ops'] = Path(atb.__file__)
    for name, source in source_paths.items():
        (out / 'sources' / (name + '.py')).write_bytes(source.read_bytes())
    def launch(count):
        c = cases[count]
        torch_npu._npu_reshape_and_cache(key=c['key'], value=c['value'], key_cache=key_cache,
                                       value_cache=value_cache, slot_indices=c['slots'])
    def reset():
        key_cache.fill_(sentinel)
        value_cache.fill_(sentinel)
        stream.synchronize()
    checks = []
    def validate(count, label):
        c = cases[count]
        k, v = key_cache.cpu(), value_cache.cpu()
        good = dict(label=label, tokens=count, key_pool_equal=bool(torch.equal(k,c['expected_k'])),
                    value_pool_equal=bool(torch.equal(v,c['expected_v'])),
                    input_key_unchanged=bool(torch.equal(c['key'].cpu(),c['key_cpu'])),
                    input_value_unchanged=bool(torch.equal(c['value'].cpu(),c['value_cpu'])),
                    first_slot=int(c['slot_cpu'][0]), last_slot=int(c['slot_cpu'][-1]),
                    checked_elements_per_pool=key_cache.numel())
        checks.append(good)
        if not all(good[k] for k in ('key_pool_equal','value_pool_equal','input_key_unchanged','input_value_unchanged')):
            raise RuntimeError('incorrect KV write: ' + str(good))
    trials, records, profiles = [], [], []
    with torch.npu.stream(stream):
        for count in tokens:
            reset()
            for _ in range(10):
                launch(count)
            stream.synchronize()
            validate(count, 'warmup/' + str(count))
        # Different deterministic order each round; every trial starts from a
        # completed reset and joins its own terminal event before timing stops.
        rng = random.Random(19)
        for round_id in range(args.rounds):
            order = list(tokens)
            rng.shuffle(order)
            for count in order:
                reset()
                terminal = torch.npu.Event()
                begin = time.monotonic_ns()
                for _ in range(args.iterations):
                    launch(count)
                submitted = time.monotonic_ns()
                terminal.record(stream)
                terminal.synchronize()
                finished = time.monotonic_ns()
                label = 'timing/{}/{}'.format(round_id,count)
                trials.append(dict(label=label, round=round_id, tokens=count, iterations=args.iterations,
                                   begin_ns=begin, submitted_ns=submitted, finished_ns=finished,
                                   submit_ns=submitted-begin, completed_ns=finished-begin,
                                   final_wait=True))
                validate(count, label)
        for name, metric in [('plain',torch_npu.profiler.AiCMetrics.AiCoreNone),
                             ('pipe',torch_npu.profiler.AiCMetrics.PipeUtilization)]:
            config = torch_npu.profiler._ExperimentalConfig(profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                                                          aic_metrics=metric)
            profiles.append(dict(name=name, metric=str(metric), level='Level1'))
            with torch_npu.profiler.profile(
                    activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],
                    schedule=torch_npu.profiler.schedule(wait=0,warmup=1,active=1,repeat=1),
                    record_shapes=True, experimental_config=config,
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(out / ('profiler_' + name)))) as prof:
                prof.step()
                for count in tokens:
                    with torch.profiler.record_function('P19/control/reset/' + str(count)):
                        reset()
                    for rep in range(args.profile_repeats):
                        label = 'P19/{}/tokens={}/rep={}'.format(name,count,rep)
                        record = dict(label=label, profile=name, tokens=count, repeat=rep,
                                      pid=os.getpid(), raw_stream_handle=str(stream.npu_stream),
                                      host_stream_id=str(stream.stream_id), host_begin_ns=time.monotonic_ns())
                        with torch.profiler.record_function(label):
                            launch(count)
                        record['host_end_ns'] = time.monotonic_ns()
                        records.append(record)
                    with torch.profiler.record_function('P19/control/join-check/' + str(count)):
                        stream.synchronize()
                        validate(count, name + '/' + str(count))
                prof.step()
    torch.npu.synchronize()
    sources = {str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (out/'sources').glob('*')}
    save(out / 'run.json', dict(captured_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
         command=[sys.executable]+sys.argv, torch_version=torch.__version__, torch_npu_version=torch_npu.__version__,
         python=sys.version, machine=platform.machine(), hardware=hardware,
         operator_schema=str(torch.ops.atb._npu_reshape_and_cache.default._schema),
         config=dict(tokens=tokens, rounds=args.rounds, iterations=args.iterations, profile_repeats=args.profile_repeats,
                     block_size=block_size, num_blocks=num_blocks, kv_heads=kv_heads, head_size=head_size,
                     dtype='torch.bfloat16', sentinel=sentinel, slot_pattern='128 + arange(tokens)',
                     raw_stream_handle=str(stream.npu_stream), allow_internal_format=False),
         resources=dict(key_cache=describe(key_cache), value_cache=describe(value_cache),
                        cases={str(n):{k:describe(c[k]) for k in ('key','value','slots')} for n,c in cases.items()}),
         source_sha256=sources, trials=trials, checks=checks, records=records, profiles=profiles,
         environment={k:os.environ.get(k) for k in ('ASCEND_HOME_PATH','TASK_QUEUE_ENABLE','ASCEND_LAUNCH_BLOCKING','ASCEND_RT_VISIBLE_DEVICES')},
         limits=['Unprofiled timing is host submit-through-completion time for a batch, not isolated kernel latency.',
                 'Repeated writes target the same warmed buffers; this is not a cold-HBM bandwidth benchmark.',
                 'Profiler block counts and aggregate pipeline metrics do not identify physical per-core timelines.']))
    device_info('device_after.txt')
    print('CAPTURED',out,flush=True)


if __name__ == '__main__':
    main()
