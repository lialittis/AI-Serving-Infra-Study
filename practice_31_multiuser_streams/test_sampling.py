"""Validate sampling causality, identity and device classification independently."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
from analyze import compute_core, overlap, dump
from decimal import Decimal
import tempfile
from sampling_analysis import analyze_sampling
from run import service_command
from run_sampling import case_plan
import observer as o
from test_observer import frame


def sample(enabled):
    stage='async_exponential' if enabled else 'inline_exponential'
    hosts={};tasks=[]
    def task(ident,name,stream,stage,start,stop,native,core='AI_VECTOR_CORE'):
        hosts[ident]=dict(name=name,start_us=str(native),end_us=str(native+1))
        tasks.append(dict(id=ident,name=name,stream=stream,step='s',stage=stage,
                          start_us=str(start),end_us=str(stop),duration_us=str(stop-start),
                          is_compute=not name.startswith('EVENT_'),host_index=ident,cann_index=ident,core_type=core))
    task('f','aclnnMatMul','46','forward',50,100,40)
    task('q','aclnnInplaceUniform','44',stage,55,65,10 if enabled else 105,'DSA_SQE')
    task('q2','aclnnLog','44',stage,65,70,11 if enabled else 106)
    task('e','EVENT_RECORD','44',stage,70,71,12 if enabled else 107)
    task('c','aclnnInplaceDiv','46','sampling_math' if enabled else stage,140,145,130)
    scopes=[dict(step='s',stage='forward',start_us='35',end_us='101'),
            dict(step='s',stage=stage,start_us='5' if enabled else '102',end_us='20' if enabled else '132')]
    metadata=dict(shape=[2,151936],dtype='torch.float32',device='npu:0',data_ptr='123',storage_ptr='123')
    records=[dict(kind='q_ready',key='s',q=metadata,generator_count=0,event_handle='999' if enabled else None)]
    events=[]
    if enabled:
        scopes.append(dict(step='s',stage='sampling_math',start_us='110',end_us='146'))
        scopes.append(dict(step='s',stage='q_wait',tid=9,start_us='111',end_us='119'))
        # Deliberately offset Python wall clock: never use it to filter the
        # profiler's temperature division against the q consumer boundary.
        records.append(dict(kind='q_consume',key='s',q=copy.deepcopy(metadata),event_handle='999',wall_ns=100000))
        events=[dict(name='AscendCL@aclrtSynchronizeEvent',ph='X',ts=112,dur=6,tid=9)]
        # Earlier temperature division must not be mistaken for probs/q.
        task('temp','aclnnInplaceDiv','46','sampler',105,110,104)
    else:
        # Late host submission must precede its device execution, so move q.
        for t in tasks:
            if t['stream']=='44':
                for field in ('start_us','end_us'):t[field]=str(int(t[field])+60)
        task('w','EVENT_WAIT','46',stage,120,135,108)
    result=dict(tasks=tasks,hosts=hosts,scopes=scopes,
                steps=[dict(key='s',requests=['a','b'],scheduler=dict(requests={'a':dict(phase='decode'),'b':dict(phase='decode')}))])
    return result,records,events


class SamplingTests(unittest.TestCase):
    def test_dsa_is_compute_but_copy_and_unknown_are_not(self):
        self.assertTrue(compute_core('DSA_SQE'))
        self.assertTrue(compute_core('AI_VECTOR_CORE'))
        self.assertFalse(compute_core('SDMA_SQE'))
        self.assertFalse(compute_core('DSA_SQE',True))
        self.assertFalse(compute_core('UNKNOWN'))

    def test_off_device_wait_and_on_host_wait(self):
        for enabled in (False,True):
            with self.subTest(enabled=enabled):
                data=analyze_sampling(*sample(enabled),enabled)
                self.assertEqual(data['summary']['issues'],[])
                self.assertEqual(data['summary']['validated_steps'],1)
                s=data['steps'][0]
                self.assertEqual(s['consumer'],'c')
                self.assertEqual(s['synchronization']['kind'],'host_event_synchronize' if enabled else 'device_event_wait')
                self.assertEqual(s['same_step_random_forward_overlap_us'],'15' if enabled else '0')
                self.assertEqual(s['dsa_forward_overlap_us'],'10' if enabled else '0')
                self.assertEqual(s['other_random_forward_overlap_us'],'5' if enabled else '0')
                self.assertEqual(s['native_submission_delta_us'],'-30' if enabled else '65')
                self.assertTrue(data['examples'][0]['task_evidence'])

    def test_identity_or_completion_failure_never_becomes_zero_overlap(self):
        for failure in ('identity','completion'):
            r,records,events=sample(True)
            if failure=='identity':records[-1]['q']['data_ptr']='456'
            else:events[0]['ts']=60;events[0]['dur']=1;r['scopes'][-1].update(start_us='59',end_us='62')
            data=analyze_sampling(r,records,events,True)
            self.assertEqual(data['summary']['validated_steps'],0)
            self.assertEqual(len(data['summary']['issues']),1)

    def test_sampler_overlap_is_not_model_overlap(self):
        r,records,events=sample(False)
        r['tasks'].append(dict(id='softmax',name='Softmax',stream='46',step='s',stage='sampling_math',
                               start_us='116',end_us='124',is_compute=True,core_type='AI_VECTOR_CORE'))
        r['overlap_evidence']=overlap(r['tasks'])['evidence']
        s=analyze_sampling(r,records,events,False)['summary']
        self.assertEqual(s['random_forward_overlap_us'],'0')
        self.assertEqual(s['overlap_patterns'][0]['overlap_us'],'8')
        self.assertEqual({t['name'] for t in s['overlap_patterns'][0]['tasks']},{'Softmax','aclnnInplaceUniform'})

    def test_raw_native_decimal_evidence_export_preserves_precision(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'evidence.json'
            dump(p,dict(ts=Decimal('1790000000000000.001'),dur=Decimal('1.234')))
            self.assertEqual(json.loads(p.read_text())['ts'],'1790000000000000.001')

    def test_only_treatment_changes_service_command_and_abba(self):
        a=NS(model='model',port=8031,max_num_seqs=32,max_num_batched_tokens=4096,precompute=False)
        off=service_command(a,Path('/tmp/case'),'benchmark')
        a.precompute=True
        on=service_command(a,Path('/tmp/case'),'benchmark')
        i=off.index('--additional-config')+1
        self.assertEqual(off[:i]+off[i+1:],on[:i]+on[i+1:])
        self.assertEqual(json.loads(off[i]),{'enable_async_exponential':False})
        self.assertEqual(json.loads(on[i]),{'enable_async_exponential':True})
        self.assertIn('--enforce-eager',on)
        cases=case_plan([8,32],'both')
        self.assertEqual(len(cases),12)
        for c in (8,32):self.assertEqual([enabled for cc,phase,enabled,_ in cases if cc==c and phase=='benchmark'],[False,True,True,False])

    def test_event_identity_selects_q_wait_through_wrappers(self):
        o._records.clear();o._local.scopes={};o._local.api_calls={};o._local.active=True;o._local.key='s'
        o._local.q_consumer_event=NS(npu_event=999)
        try:
            with patch.object(o,'open_scope') as opened,patch.object(o,'close_scope') as closed,patch.object(o,'describe_stream',return_value=None):
                for handle in (777,999):
                    f=frame('torch_npu.npu.streams','synchronize',{'self':NS(npu_event=handle)})
                    # No f_back: actual identity must work through an arbitrary wrapper.
                    o.handle(f,'call',None);o.handle(f,'return',None)
                self.assertEqual(opened.call_count,1);self.assertEqual(closed.call_count,1)
                returns=[r for r in o._records if r['kind']=='stream_api_return']
                self.assertEqual([r['q_wait'] for r in returns],[False,True])
        finally:
            o._local.active=False;o._local.q_consumer_event=None


if __name__=='__main__':unittest.main()
