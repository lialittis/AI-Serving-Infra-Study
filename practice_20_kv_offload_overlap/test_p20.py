import importlib.util
import os
from pathlib import Path
import sys
import threading
import time
import types
import unittest
from unittest.mock import patch

from hb import Graph


class CausalTests(unittest.TestCase):
    def graph(self):
        g=Graph()
        for name in ('write','copy','done','query','reuse'):
            g.node(name)
        return g

    def test_time_order_is_not_synchronization(self):
        g=self.graph()
        self.assertFalse(g.verify([dict(source='write',target='copy')])[0]['satisfied'])

    def test_successful_query_orders_reuse(self):
        g=self.graph()
        for a,b in [('write','copy'),('copy','done'),('done','query'),('query','reuse')]:
            g.edge(a,b,'evidence')
        self.assertTrue(g.verify([dict(source='copy',target='reuse')])[0]['satisfied'])

    def test_required_edges_do_not_prove_themselves(self):
        g=self.graph();g.edge('write','copy','stream_order')
        required=[dict(source='copy',target='reuse')]
        self.assertFalse(g.verify(required)[0]['satisfied'])
        self.assertNotIn('reuse',g.successors['copy'])

    def test_generation_change_does_not_remove_hazard(self):
        g=self.graph()
        edge=dict(source='copy',target='reuse',from_generation=1,to_generation=2)
        self.assertFalse(g.verify([edge])[0]['satisfied'])

    def test_cpu_buffer_reuse_requires_load_completion(self):
        g=self.graph();g.edge('copy','done','stream_order')
        self.assertFalse(g.verify([dict(source='copy',target='reuse',medium='CPU')])[0]['satisfied'])
        g.edge('done','reuse','event_sync')
        self.assertTrue(g.verify([dict(source='copy',target='reuse',medium='CPU')])[0]['satisfied'])

    def test_event_handle_reuse_needs_distinct_record_nodes(self):
        g=self.graph();g.node('done2')
        g.edge('copy','done2','new_event_generation');g.edge('done','reuse','old_event_wait')
        self.assertFalse(g.verify([dict(source='copy',target='reuse')])[0]['satisfied'])

    def test_cycle_rejected(self):
        g=self.graph();g.edge('write','copy','x');g.edge('copy','write','x')
        with self.assertRaisesRegex(ValueError,'cycle'):
            g.topological()


class SerialPublicationTests(unittest.TestCase):
    def test_waits_for_copy_still_in_background_queue(self):
        calls=[]
        class Event:
            npu_event=7
            def synchronize(self):
                calls.append('device_complete')
        class Backend:
            _store_params=types.SimpleNamespace(bpb=types.SimpleNamespace(sum=lambda:8))
            _load_params=_store_params
            def launch_copy(self,src,dst,is_store,index,events):
                def work():
                    time.sleep(.02)
                    calls.append('published')
                    events.append((index,Event()))
                self.thread=threading.Thread(target=work)
                self.thread.start()
        backend=Backend()
        class Base:
            def __init__(self):
                self.scheduler_manager=None
                self.worker_handler=types.SimpleNamespace(_backend=backend)
        module_name='vllm_ascend.distributed.kv_transfer.kv_pool.simple_cpu_offload.simple_cpu_offload_connector'
        fake=types.ModuleType(module_name);fake.AscendSimpleCPUOffloadConnector=Base
        torch=types.ModuleType('torch')
        torch.npu=types.SimpleNamespace(current_stream=lambda:types.SimpleNamespace(
            synchronize=lambda:calls.append('compute_complete')))
        spec=importlib.util.spec_from_file_location('p20_test_connector',Path(__file__).with_name('p20_connector.py'))
        module=importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules,{'torch':torch,module_name:fake}),patch.dict(os.environ,
                {'P20_OUTPUT':'/tmp','P20_MODE':'serialized','P20_PHASE':'benchmark'}):
            spec.loader.exec_module(module)
            module.emit=lambda *a,**k:None
            connector=module.P20Connector()
            backend.launch_copy([1],[2],True,5,connector.worker_handler._store_events)
            calls.append('return')
            backend.thread.join()
        self.assertEqual(calls,['compute_complete','published','device_complete','return'])
        self.assertEqual(connector.published,{})


if __name__=='__main__':
    unittest.main()
