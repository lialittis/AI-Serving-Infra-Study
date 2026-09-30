"""Disposable native worker: reference, qualification, then optional measurement."""
import argparse
from contextlib import nullcontext
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback

from common import Bindings, save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['eager', 'graph'], required=True)
    p.add_argument('--stage', choices=['reference', 'qualify', 'measure', 'plain', 'pipe'], required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING='0', HF_HUB_OFFLINE='1',
                      TRANSFORMERS_OFFLINE='1', TRITON_CACHE_DIR=str(a.output / 'triton_cache'),
                      VLLM_CACHE_ROOT=str(a.output / 'vllm_cache'), PYTHONDONTWRITEBYTECODE='1')
    result = dict(mode=a.mode, stage=a.stage, status='running', pid=os.getpid())
    inference = None
    save(a.output / 'status.json', result)
    try:
        from native import (make_engine, prompts, generate, collect_states, GlobalState,
                            Capsule, make_second, ranges, pair, compare, tensors, fingerprint)
        torch, llm, runner = make_engine(a.mode)
        # Constructing new torch-npu parameters inside inference_mode creates
        # tensors without the version counter required by its format cache.
        # Native execute_model retains its own inference-mode decorator.
        inference = torch.no_grad()
        inference.__enter__()
        import vllm, vllm_ascend, torch_npu
        save(a.output / 'imports.json', {m.__name__: str(m.__file__) for m in (torch, torch_npu, vllm, vllm_ascend)})
        if a.stage == 'reference':
            ids = prompts(llm)
            response = generate(llm, ids['A'])
            save(a.output / 'reference.json', dict(prompt=ids['A'], tokens=list(response.outputs[0].token_ids)))
            result['status'] = 'passed'
            return

        snapshots = collect_states(torch, llm, runner, a.output)
        state_a = GlobalState()
        weights_before = fingerprint(dict(runner.model.named_parameters()))
        save(a.output / 'weights_before.json', weights_before)
        from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
        caps = {'A': Capsule('A', runner, state_a, snapshots['A', 32]),
                'B': make_second(a.mode, runner, snapshots['B', 32])}
        readonly_before = fingerprint({r:c.readonly_buffers() for r,c in caps.items()})
        save(a.output / 'readonly_buffers_before.json', readonly_before)
        save(a.output / 'readonly_buffer_addresses.json', {
            r:{n:dict(address=t.untyped_storage().data_ptr(), bytes=t.untyped_storage().nbytes())
               for n,t in c.readonly_buffers().items()} for r,c in caps.items()})
        streams = {role: torch.npu.Stream() for role in caps}
        save(a.output / 'streams.json', {r: dict(id=s.stream_id, handle=str(s.npu_stream)) for r, s in streams.items()})
        # Python-only restoration check, before any cross-stream submission.
        originals = [(obj, name, getattr(obj, name)) for obj, name, value in state_a.entries]
        try:
            with caps['B'].state.active():
                raise RuntimeError('injected_after_bindings')
        except RuntimeError as exc:
            assert str(exc) == 'injected_after_bindings'
        assert all(getattr(obj, name) is value for obj, name, value in originals)

        checks = []
        for role, capsule in caps.items():
            source = snapshots[role, 32]
            for _ in range(3):
                capsule.restore(source)
                torch.npu.synchronize()
                with torch.npu.stream(streams[role]):
                    output = capsule.submit()
                streams[role].synchronize()
            check = compare(output, source['expected'])
            checks.append(dict(role=role, kind='isolated_native_reference', **check))
            save(a.output / 'qualification.json', checks)
            if not check['passed']:
                raise AssertionError('isolated runner does not match native reference: ' + role)
        save(a.output / 'storage_ranges.json', ranges(list(caps.values())))

        # Save graph identity and pools after the two independent capture sets exist.
        graphs = []
        for role, capsule in caps.items():
            for wi, wrapper in enumerate(list(ACLGraphWrapper._all_instances)):
                if wrapper.vllm_config is not capsule.runner.vllm_config:
                    continue
                for ei, entry in enumerate(wrapper.concrete_aclgraph_entries.values()):
                    if entry.aclgraph is None:
                        continue
                    path = a.output / 'graph_dumps' / f'{role}-{wi}-{ei}.json'
                    path.parent.mkdir(exist_ok=True)
                    entry.aclgraph.debug_dump(str(path))
                    graphs.append(dict(role=role, python_id=id(entry.aclgraph), pool=list(wrapper.graph_pool),
                                       path=str(path.relative_to(a.output))))
        save(a.output / 'graphs.json', graphs)
        if a.mode == 'graph':
            pa = {tuple(g['pool']) for g in graphs if g['role'] == 'A'}
            pb = {tuple(g['pool']) for g in graphs if g['role'] == 'B'}
            assert pa and pb and not (pa & pb), 'graph pools not independent'

        cases = {'same32': {'A': ('A', 32), 'B': ('B', 32)},
                 'mixed16_47': {'A': ('A', 16), 'B': ('B', 47)}}

        def prepare(keys):
            for role, key in keys.items():
                caps[role].restore(snapshots[key])
            torch.npu.synchronize()

        for case, keys in cases.items():
            for swap in (False, True):
                use = dict(zip(('A', 'B'), reversed(list(keys.values())))) if swap else keys
                for strategy in ('serial', 'parallel'):
                    for order in ('AB', 'BA'):
                        prepare(use)
                        outputs, timing = pair(torch, caps, streams, strategy, order)
                        for role in caps:
                            check = compare(outputs[role], snapshots[use[role]]['expected'])
                            checks.append(dict(case=case, swap=swap, strategy=strategy, order=order, role=role, **check))
                            save(a.output / 'qualification.json', checks)
                            if not check['passed']:
                                raise AssertionError('qualification numerical mismatch')
                        ranges(list(caps.values()))

        prepare(cases['same32'])
        try:
            pair(torch, caps, streams, 'parallel', inject=True)
        except RuntimeError as exc:
            assert str(exc) == 'injected_after_first_submit'
        assert all(s.query() for s in streams.values())
        assert all(getattr(obj, name) is value for obj, name, value in originals)
        save(a.output / 'recovery.json', dict(bindings_restored=True, submitted_streams_drained=True,
                                            unrecorded_terminal_event_not_waited=True))

        if a.stage != 'qualify':
            from experiment import run_measurements
            run_measurements(a, torch, caps, streams, snapshots, cases, prepare, pair, compare)
        weights_after = fingerprint(dict(runner.model.named_parameters()))
        save(a.output / 'weights_after.json', weights_after)
        assert weights_after == weights_before, 'read-only model weights changed'
        readonly_after = fingerprint({r:c.readonly_buffers() for r,c in caps.items()})
        save(a.output / 'readonly_buffers_after.json', readonly_after)
        assert readonly_after == readonly_before, 'read-only rotary table changed'
        result.update(status='passed', qualification_checks=len(checks))
    except BaseException as exc:
        result.update(status='failed', exception=type(exc).__name__, reason=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        save(a.output / 'status.json', result)
        if inference is not None:
            inference.__exit__(None, None, None)


if __name__ == '__main__':
    main()
