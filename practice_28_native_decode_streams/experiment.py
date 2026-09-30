"""Balanced unprofiled pairs and independent diagnostic trials."""
from contextlib import contextmanager
import json
from pathlib import Path
import time

from common import Bindings, save


class Observer:
    def __init__(self, torch, graphs, root):
        self.torch, self.graphs, self.root = torch, {g['python_id']: g for g in graphs}, root
        self.records, self.role, self.serial = [], None, 0

    @contextmanager
    def marker(self, kind, role=None):
        label = 'P28/' + kind + ('/' + role if role else '')
        self.records.append(dict(kind=kind, role=role, label=label))
        with self.torch.profiler.record_function(label):
            yield

    @contextmanager
    def scope(self, role):
        previous = self.role
        self.role = role
        label = 'P28/forward/' + role
        self.records.append(dict(kind='forward', role=role, label=label))
        try:
            with self.torch.profiler.record_function(label):
                yield
        finally:
            self.role = previous

    @contextmanager
    def installed(self):
        cls = self.torch.npu.NPUGraph
        original = cls.replay
        def replay(graph, *args, **kwargs):
            if self.role is None:
                return original(graph, *args, **kwargs)
            record = self.graphs[id(graph)]
            assert record['role'] == self.role
            self.serial += 1
            label = f'P28/replay/{self.role}/{self.serial}'
            self.records.append(dict(kind='replay', role=self.role, label=label,
                                     python_id=id(graph), path=record['path']))
            with self.torch.profiler.record_function(label):
                return original(graph, *args, **kwargs)
        with Bindings([(cls, 'replay', replay)]):
            yield


def run_measurements(args, torch, caps, streams, snapshots, cases, prepare, pair, compare):
    measurements, correctness = [], []
    def validate(outputs, keys, info):
        for role in caps:
            check = compare(outputs[role], snapshots[keys[role]]['expected'])
            correctness.append(dict(**info, role=role, **check))
            save(args.output / 'correctness.json', correctness)
            if not check['passed']:
                raise AssertionError('formal numerical check failed')

    if args.stage == 'measure':
        for case, keys in cases.items():
            for strategy in ('serial', 'parallel'):
                for _ in range(3):
                    prepare(keys)
                    output, _ = pair(torch, caps, streams, strategy)
            for repeat in range(12):
                swap, order = bool((repeat // 2) % 2), ('AB' if repeat % 2 == 0 else 'BA')
                use = dict(zip(('A', 'B'), reversed(list(keys.values())))) if swap else keys
                strategies = ('serial', 'parallel') if repeat % 2 == 0 else ('parallel', 'serial')
                for strategy in strategies:
                    prepare(use)
                    torch.npu.reset_peak_memory_stats()
                    before = torch.npu.memory_allocated()
                    output, timing = pair(torch, caps, streams, strategy, order)
                    info = dict(case=case, repeat=repeat, swap=swap, order=order, strategy=strategy)
                    measurements.append(dict(**info, **timing, allocated_before=before,
                        allocated_peak=torch.npu.max_memory_allocated(), reserved_peak=torch.npu.max_memory_reserved()))
                    save(args.output / 'measurements.json', measurements)
                    validate(output, use, info)
    else:
        import torch_npu
        graphs = json.loads((args.output / 'graphs.json').read_text())
        for case, keys in cases.items():
            for strategy in ('serial', 'parallel'):
                for order in ('AB', 'BA'):
                    info = dict(case=case, strategy=strategy, order=order, profile=args.stage)
                    ident = f'{case}-{strategy}-{order}'
                    root = args.output / 'trials' / ident
                    root.mkdir(parents=True, exist_ok=False)
                    prepare(keys)
                    observer = Observer(torch, graphs, root)
                    metric = (torch_npu.profiler.AiCMetrics.PipeUtilization if args.stage == 'pipe'
                              else torch_npu.profiler.AiCMetrics.AiCoreNone)
                    with observer.installed(), torch_npu.profiler.profile(
                        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                        record_shapes=True, with_stack=False, profile_memory=False,
                        experimental_config=torch_npu.profiler._ExperimentalConfig(
                            profiler_level=torch_npu.profiler.ProfilerLevel.Level1, aic_metrics=metric),
                        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(root / 'profiler'))):
                        output, timing = pair(torch, caps, streams, strategy, order, observer)
                    save(root / 'observations.json', observer.records)
                    save(root / 'trial.json', dict(**info, timing=timing))
                    validate(output, keys, info)
