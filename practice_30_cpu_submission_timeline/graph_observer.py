"""Capture identities before warmup, then time replay without new device waits."""
from contextlib import contextmanager
import functools
import inspect
import time
import weakref

from common import Bindings, save


class GraphObserver:
    def __init__(self, torch, output):
        self.output = output
        self.active = None
        self.graphs = weakref.WeakKeyDictionary()
        self.begins = weakref.WeakKeyDictionary()
        self.records, self.entries, self.sources = [], [], {}
        self.serial = 0
        cls = torch.npu.NPUGraph
        begin, end, replay = cls.capture_begin, cls.capture_end, cls.replay

        @functools.wraps(begin)
        def capture_begin(graph, *args, **kwargs):
            self.begins[graph] = (time.perf_counter_ns(), time.thread_time_ns())
            return begin(graph, *args, **kwargs)

        @functools.wraps(end)
        def capture_end(graph, *args, **kwargs):
            result = end(graph, *args, **kwargs)
            finish, cpu = time.perf_counter_ns(), time.thread_time_ns()
            start, cpu_start = self.begins.pop(graph)
            self.serial += 1
            uid = f'g{self.serial}'
            self.graphs[graph] = uid
            path = self.output / 'graph_dumps' / (uid + '.json')
            path.parent.mkdir(exist_ok=True)
            graph.debug_dump(str(path))
            self.records.append(dict(kind='capture', uid=uid,
                path=str(path.relative_to(self.output)), wall_start_ns=start,
                wall_end_ns=finish, wall_us=(finish-start)/1000,
                thread_cpu_us=(cpu-cpu_start)/1000,
                during_measurement=self.active is not None))
            return result

        @functools.wraps(replay)
        def replayed(graph, *args, **kwargs):
            if self.active is None:
                return replay(graph, *args, **kwargs)
            with self.active.phase('replay', dict(uid=self.graphs[graph])):
                return replay(graph, *args, **kwargs)

        self.entries.extend([(cls, 'capture_begin', capture_begin),
                             (cls, 'capture_end', capture_end), (cls, 'replay', replayed)])
        self.source('replay', replay)
        # Cover both exported Python entrypoints; their captured originals call
        # C directly, so these aliases do not cause nested duplicate observations.
        import torch_npu.npu.graphs as graphs
        for owner in (torch.npu, graphs):
            for name in ('graph_task_update_begin', 'graph_task_update_end'):
                original = getattr(owner, name)
                self.source(name, original)
                self.entries.append((owner, name, self.update_wrapper(original, name)))

    def source(self, kind, fn):
        self.sources[kind] = dict(path=inspect.getsourcefile(fn),
            line=inspect.getsourcelines(fn)[1], qualname=fn.__qualname__)

    def update_wrapper(self, original, kind):
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            if self.active is None:
                return original(*args, **kwargs)
            with self.active.phase(kind):
                return original(*args, **kwargs)
        return wrapped

    @contextmanager
    def installed(self):
        originals = [(o, n, n in vars(o), getattr(o, n)) for o, n, _ in self.entries]
        try:
            with Bindings(self.entries):
                yield self
        finally:
            self.active = None
            restored = all((n in vars(o)) == owned and getattr(o, n) == value
                           for o, n, owned, value in originals)
            save(self.output / 'graph_captures.json', self.records)
            save(self.output / 'graph_binding_recovery.json', dict(restored=restored))
            assert restored, 'graph hooks not restored'
