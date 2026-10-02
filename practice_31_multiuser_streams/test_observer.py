"""Exercise the current coroutine wrapper and pre-update scheduler metadata."""
from types import SimpleNamespace as NS
import unittest
import observer as o


def frame(module, name, values):
    return NS(f_globals={'__name__':module}, f_code=NS(co_name=name,co_qualname=name),f_locals=values)


class ObserverTests(unittest.TestCase):
    def setUp(self):
        o._records.clear();o._seen.clear();o._occurrences.clear()

    def test_wrapped_completion_coroutine_records_mapping_once(self):
        body=NS(request_id='p31-measure-identity')
        outer=frame('vllm.entrypoints.openai.completion.serving','create_completion',{'request':body})
        inner=frame('vllm.entrypoints.openai.completion.serving','_create_completion',{'request':body})
        o.handle(outer,'call',None);o.handle(inner,'call',None);o.handle(inner,'return',None)
        inner.f_locals.update(request_id='cmpl-p31-measure-identity',request_id_item='cmpl-p31-measure-identity-0')
        o.handle(inner,'call',None);o.handle(inner,'return',None);o.handle(outer,'return',None)
        self.assertEqual([r['kind'] for r in o._records],['received','frontend_map'])
        self.assertEqual(o._records[-1]['external_id'],'cmpl-p31-measure-identity-0')

    def test_phase_uses_progress_before_schedule_updates(self):
        rid='cmpl-p31-measure-identity-0-random'
        req=NS(num_computed_tokens=0,num_prompt_tokens=128)
        f=frame('vllm.v1.core.sched.scheduler','schedule',{'self':NS(requests={rid:req})})
        o.handle(f,'call',None)
        req.num_computed_tokens=128
        o.handle(f,'return',NS(num_scheduled_tokens={rid:128}))
        record=o._records[-1]
        self.assertEqual(record['requests'][rid]['phase'],'prefill')
        self.assertEqual(record['requests'][rid]['computed_before'],0)
        self.assertEqual(record['key'],o.step_key('worker',{rid:128}))


if __name__=='__main__':unittest.main()
