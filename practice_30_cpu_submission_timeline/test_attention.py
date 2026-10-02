"""Attention attribution must preserve thread identity and reversible bindings."""
from contextlib import contextmanager
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

from attention_analysis import selected_parent

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'practice_28_native_decode_streams'))
from attention_observer import AttentionObserver


class AttentionTests(unittest.TestCase):
    def test_parent_requires_thread_and_unique_enclosure(self):
        phase=dict(trace_start_us='15', trace_end_us='25', tid=4)
        event=dict(name='vllm::unified_attention_with_output', ts='10', dur='20', tid=4)
        with self.assertRaisesRegex(ValueError, 'enclosing'):
            selected_parent([dict(index=1,event=dict(event,tid=5))],phase)
        with self.assertRaisesRegex(ValueError, 'enclosing'):
            selected_parent([dict(index=i,event=event) for i in (1,2)],phase)
        self.assertEqual(selected_parent([dict(index=1,event=event)],phase)['index'],1)

    def observer(self):
        @contextmanager
        def phase(kind):
            recorded.append(kind)
            yield
        recorded=[]
        obs=AttentionObserver.__new__(AttentionObserver)
        obs.observer=SimpleNamespace(sources={},phase=phase,index=32)
        obs.active=False;obs.entries=[];obs.metadata={};obs.selected=0;obs.context_calls=0
        return obs,recorded

    def test_inactive_wrapper_preserves_arguments_and_result(self):
        obs,recorded=self.observer()
        value=object()
        def call(arg, *, token):
            self.assertIs(arg,value);self.assertIs(token,value)
            return value
        owner=SimpleNamespace(call=call)
        obs.add(owner,'call','attn_test')
        with obs.installed():
            self.assertIs(owner.call(value,token=value),value)
        self.assertIs(owner.call,call)
        self.assertEqual(recorded,[])

    def test_exception_restores_inherited_binding(self):
        obs,recorded=self.observer()
        class Parent:
            def call(self):
                raise RuntimeError('native failure')
        class Child(Parent):
            pass
        obs.add(Child,'call','attn_test')
        with self.assertRaisesRegex(RuntimeError,'native failure'):
            with obs.installed():
                obs.active=True
                Child().call()
        self.assertNotIn('call',vars(Child))
        self.assertFalse(obs.active)
        self.assertTrue(obs.restored)
        self.assertEqual(recorded,['attn_test'])

    def test_context_records_only_first_target_step(self):
        obs,recorded=self.observer()
        value=object()
        owner=SimpleNamespace(call=lambda layer:value)
        obs.add(owner,'call','attn_context',context=True)
        with obs.installed():
            obs.observer.index=31
            self.assertIs(owner.call('layer'),value)
            obs.observer.index=32
            for _ in range(2):self.assertIs(owner.call('layer'),value)
        self.assertEqual(recorded,['attn_context'])
        self.assertEqual(obs.context_calls,1)


if __name__=='__main__':unittest.main()
