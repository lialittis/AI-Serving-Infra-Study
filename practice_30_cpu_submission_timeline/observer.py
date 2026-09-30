"""Reversible phase scopes. No device reads, injected waits, or hot-path file IO."""
from contextlib import contextmanager
import functools
import inspect
import os
import threading
import time

from common import Bindings


class Observer:
    def __init__(self, torch, llm, runner):
        self.torch, self.llm, self.runner = torch, llm, runner
        self.records, self.sources, self.stack = [], {}, []
        self.index, self.serial = -1, 0
        self.rid = None
        self.entries = []
        engine = llm.llm_engine
        core = engine.engine_core.engine_core
        assert core.batch_queue is None and not core.async_scheduling
        assert core.step_fn.__func__ is core.step.__func__
        self.wrap(engine, 'step', 'engine_step', step_boundary=True)
        # core.step_fn cached its bound method during initialization.
        self.wrap(core, 'step_fn', 'core_step')
        self.wrap(core.scheduler, 'schedule', 'schedule', schedule=True)
        self.wrap(core.scheduler, 'update_from_output', 'scheduler_update')
        self.wrap(engine.output_processor, 'process_outputs', 'output_processing')
        self.wrap(core.model_executor, 'execute_model', 'executor_submit', future=True)
        for name, kind in [('execute_model','execute'), ('_update_states','state_update'),
                           ('_prepare_inputs','prepare_inputs'), ('_preprocess','preprocess'),
                           ('_model_forward','forward'), ('sample_tokens','sample'),
                           ('_sample','sampling'), ('_bookkeeping_sync','bookkeeping'),
                           ('_to_list','token_to_list')]:
            self.wrap(runner, name, kind, execute=kind=='execute')
        self.wrap(runner.model, 'compute_logits', 'logits')
        from vllm.v1.outputs import LogprobsTensors
        self.wrap(LogprobsTensors, 'tolists', 'logprobs_to_cpu')

    def wrap(self, owner, name, kind, step_boundary=False, schedule=False, execute=False, future=False):
        original = getattr(owner, name)
        fn = inspect.unwrap(original)
        self.sources[kind] = dict(path=inspect.getsourcefile(fn), line=inspect.getsourcelines(fn)[1],
                                  qualname=fn.__qualname__)
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            if step_boundary:
                self.index += 1
            extra = {}
            if execute:
                extra['scheduled'] = dict(args[0].num_scheduled_tokens)
                assert len(extra['scheduled']) == 1
                self.rid = next(iter(extra['scheduled']))
            with self.phase(kind, extra) as record:
                result = original(*args, **kwargs)
                if schedule:
                    record['scheduled'] = dict(result.num_scheduled_tokens)
                    if record['scheduled']:
                        self.rid = next(iter(record['scheduled']))
                if future:
                    record['future_done_on_return'] = result.done()
                return result
        self.entries.append((owner, name, wrapped))

    @contextmanager
    def phase(self, kind, extra=None):
        self.serial += 1
        ident = self.serial
        record = dict(id=ident, parent=self.stack[-1] if self.stack else None,
            kind=kind, index=self.index, step=f'p30-measure/step-{self.index:02d}',
            label=f'P30/{self.index:02d}/{kind}/{ident}', pid=os.getpid(),
            tid=threading.get_native_id(), request_id=self.rid, **(extra or {}))
        self.stack.append(ident)
        with self.torch.profiler.record_function(record['label']):
            wall = time.perf_counter_ns()
            cpu = time.thread_time_ns()
            try:
                yield record
            except BaseException:
                record['failed'] = True
                raise
            finally:
                cpu_end = time.thread_time_ns()
                wall_end = time.perf_counter_ns()
                record.update(wall_start_ns=wall, wall_end_ns=wall_end,
                    wall_ns=wall_end-wall, thread_cpu_ns=cpu_end-cpu, request_id=self.rid)
        assert self.stack.pop() == ident
        self.records.append(record)

    @contextmanager
    def installed(self):
        originals = [(o,n,n in vars(o),getattr(o,n)) for o,n,_ in self.entries]
        try:
            with Bindings(self.entries):
                yield self
        finally:
            self.restored = all((n in vars(o))==owned and getattr(o,n)==value
                                for o,n,owned,value in originals)
            assert self.restored, 'observer did not restore exact bindings'
