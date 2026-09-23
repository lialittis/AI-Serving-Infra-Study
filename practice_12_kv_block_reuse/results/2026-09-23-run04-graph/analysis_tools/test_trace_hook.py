"""Check observation plumbing before a real NPU run, without torch installed."""
from contextlib import nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'practice_07_real_request_trace'))
import lifetime_trace as trace

class HookTests(unittest.TestCase):
    def test_scope_event_and_operation_kind_do_not_collide(self):
        trace._local.open={};trace._local.serial=0;trace._local.role='A';trace._local.rid='practice12-A'
        fake_torch=SimpleNamespace(profiler=SimpleNamespace(record_function=lambda label:nullcontext()))
        frame=SimpleNamespace(f_locals={})
        with patch.dict(sys.modules,{'torch':fake_torch}),patch.object(trace.base,'emit') as emit:
            trace.scope(frame,'pool_allocate',before={'free_queue':[1]})
            args,kwargs=emit.call_args
            self.assertEqual(args[0],'scope_enter')
            self.assertEqual(kwargs['kind'],'pool_allocate')
            self.assertEqual(kwargs['before']['free_queue'],[1])
            self.assertEqual(len(trace._local.open),1)

    def test_free_observer_does_not_consume_native_iterator(self):
        trace._local.open={};trace._local.serial=0;trace._local.role='A';trace._local.rid='practice12-A'
        class ForbiddenIterator:
            def __iter__(self):
                raise AssertionError('observer must not consume native free iterator')
        blocks=[SimpleNamespace(block_id=i,ref_cnt=0 if i==0 else 1,is_null=i==0) for i in range(2)]
        pool=SimpleNamespace(blocks=blocks,num_gpu_blocks=2,enable_caching=False,
                             free_block_queue=SimpleNamespace(get_all_free_blocks=lambda:[]))
        frame=SimpleNamespace(f_locals={'self':pool,'ordered_blocks':ForbiddenIterator()})
        fake_torch=SimpleNamespace(profiler=SimpleNamespace(record_function=lambda label:nullcontext()))
        with patch.dict(sys.modules,{'torch':fake_torch}),patch.object(trace.base,'emit') as emit:
            trace.handle('pool_free',frame,'call',None)
            self.assertEqual(emit.call_args[1]['kind'],'pool_free')
            self.assertEqual(blocks[1].ref_cnt,1)

if __name__=='__main__':unittest.main()
