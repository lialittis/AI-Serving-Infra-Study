"""One eager same32 case, three schedules. Stop scaling at the first mechanism."""
import argparse
from contextlib import nullcontext
import faulthandler
import os
from pathlib import Path
import time

from common import Bindings, save
from gate import Gate


def gated_pair(torch, caps, streams, strategy, observer=None):
    chosen = {'A': streams['A'], 'B': streams['B'] if strategy == 'parallel' else streams['A']}
    unique = list({id(s): s for s in chosen.values()}.values())
    # Include native gate allocation/cleanup in the full cost, outside release cost.
    begin = time.perf_counter_ns()
    gate = Gate(unique)
    ends = {r: torch.npu.Event(enable_timing=True) for r in caps}
    outputs, submitted, recorded = {}, [], set()
    def mark(kind, role=None):
        return observer.marker(kind, role) if observer else nullcontext()
    try:
        with mark('gate_close'):
            gate.close()
        faulthandler.dump_traceback_later(0.8, repeat=False)
        for role, stream in chosen.items():
            submitted.append(stream)
            with torch.npu.stream(stream):
                with observer.scope(role) if observer else nullcontext():
                    outputs[role] = caps[role].submit()
                with mark('terminal', role):
                    ends[role].record()
                recorded.add(role)
            gate.stamp('python_submit_return', role=role)
        closed_status = gate.query()
        gate.stamp('both_python_submissions_returned', closed_status=closed_status)
    finally:
        faulthandler.cancel_dump_traceback_later()
        with mark('gate_release'):
            gate.open()
        # On a partial/failed submit, first open the gate, then drain streams.
        for role, stream in zip(chosen, submitted):
            with mark('join', role):
                if role in recorded:
                    ends[role].synchronize()
                else:
                    stream.synchronize()
        complete = time.perf_counter_ns()
        gate.destroy()
    released = next(x['monotonic_ns'] for x in gate.log if x['action'] == 'release_begin')
    return outputs, dict(wall_us=(time.perf_counter_ns()-begin)/1000,
        release_to_join_us=(complete-released)/1000, rescued=gate.rescue, gate_log=gate.log,
        python_prequeue=not gate.rescue and closed_status == [1]*len(unique))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING='0', HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1', TRITON_CACHE_DIR=str(a.output / 'triton_cache'),
        VLLM_CACHE_ROOT=str(a.output / 'vllm_cache'), PYTHONDONTWRITEBYTECODE='1')
    import native
    from native import (make_engine, collect_states, GlobalState, Capsule, make_second,
                        compare, fingerprint, ranges, pair)
    from experiment import Observer
    import torch_npu
    status = dict(status='running')
    save(a.output / 'status.json', status)
    torch, llm, runner = make_engine('eager')
    import vllm, vllm_ascend
    save(a.output / 'imports.json', {m.__name__:dict(path=m.__file__, version=getattr(m,'__version__',None))
                                   for m in (torch,torch_npu,vllm,vllm_ascend)})
    save(a.output / 'settings.json', native.settings('eager'))
    checks, measurements = [], []
    try:
        with torch.no_grad():
            with Bindings([(native, 'STEPS', (32,))]):
                snapshots = collect_states(torch, llm, runner, a.output)
            caps = {'A': Capsule('A', runner, GlobalState(), snapshots['A',32]),
                    'B': make_second('eager', runner, snapshots['B',32])}
            weights = fingerprint(dict(runner.model.named_parameters()))
            readonly = fingerprint({r:c.readonly_buffers() for r,c in caps.items()})
            save(a.output / 'weights_before.json', weights)
            save(a.output / 'readonly_before.json', readonly)
            streams = {r: torch.npu.Stream() for r in caps}
            save(a.output / 'streams.json', {r:dict(id=s.stream_id, handle=s.npu_stream) for r,s in streams.items()})
            save(a.output / 'graphs.json', [])  # Reuse P28's exact flow analyzer.

            def prepare():
                for r, c in caps.items():
                    c.restore(snapshots[r,32])
                torch.npu.synchronize()

            def validate(outputs, label):
                for role in caps:
                    check = compare(outputs[role], snapshots[role,32]['expected'])
                    checks.append(dict(label=label, role=role, **check))
                    save(a.output / 'correctness.json', checks)
                    assert check['passed'], 'native exact numerical mismatch'
                save(a.output / 'storage_ranges.json', ranges(list(caps.values())))

            # Small warm-up, identical model work. Gates are qualified separately.
            for i in range(2):
                prepare()
                outputs, _ = pair(torch, caps, streams, 'parallel')
                validate(outputs, f'warmup-{i}')

            cases = [('normal-dual','parallel',False), ('gated-single','serial',True),
                     ('gated-dual','parallel',True)]
            blocked = []
            for label, strategy, gated in cases:
                root = a.output / 'trials' / label
                root.mkdir(parents=True)
                prepare()
                observer = Observer(torch, [], root)
                # Keep P28 labels to reuse its exact CPU/queue/CANN/device/CSV join.
                with torch_npu.profiler.profile(
                    activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                    record_shapes=True, with_stack=False, profile_memory=False,
                    experimental_config=torch_npu.profiler._ExperimentalConfig(
                        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                        aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone),
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(root / 'profiler'))):
                    outputs, timing = (gated_pair(torch,caps,streams,strategy,observer) if gated
                        else pair(torch,caps,streams,strategy,observer=observer))
                save(root / 'observations.json', observer.records)
                save(root / 'trial.json', dict(case='same32', strategy=strategy, order='AB',
                    profile='plain', label=label, gated=gated, timing=timing))
                validate(outputs, label)
                if gated and not timing['python_prequeue']:
                    blocked.append(label)

            # Small unprofiled controls, also reveal profiler-specific blocking.
            for label, strategy, gated in cases:
                for i in range(3):
                    prepare()
                    outputs, timing = (gated_pair(torch,caps,streams,strategy) if gated
                        else pair(torch,caps,streams,strategy))
                    measurements.append(dict(label=label, repeat=i, **timing))
                    save(a.output / 'measurements.json', measurements)
                    validate(outputs, f'{label}-unprofiled-{i}')
                    if gated and not timing['python_prequeue']:
                        blocked.append(label + '-unprofiled')
                        break  # No repeated timeouts or benchmark expansion.
            weights_after = fingerprint(dict(runner.model.named_parameters()))
            readonly_after = fingerprint({r:c.readonly_buffers() for r,c in caps.items()})
            save(a.output / 'weights_after.json', weights_after)
            save(a.output / 'readonly_after.json', readonly_after)
            assert weights == weights_after and readonly == readonly_after
            status.update(status='passed', blocked_cases=blocked,
                outcome='host_submission_blocked' if blocked else 'await_trace_prequeue_verification',
                numerical_checks=len(checks), readonly_resources_unchanged=True)
    except BaseException as exc:
        status.update(status='failed', error=repr(exc))
        raise
    finally:
        save(a.output / 'status.json', status)


if __name__ == '__main__':
    faulthandler.enable()
    main()
