"""Counterexamples to premature prequeue claims; no NPU required."""
import copy
import unittest
from analyze import gate_evidence


class EvidenceTests(unittest.TestCase):
    def fixture(self):
        events=[dict(ph='X',name='Node@launch',ts='10',dur=2),
                dict(ph='X',name='Node@launch',ts='20',dur=2),
                dict(ph='X',name='AscendCL@aclrtRecordNotify',ts='50',dur=1)]
        tasks=[dict(is_compute=True,stream='44',role=r,start_us=str(60+i*5),end_us=str(62+i*5),
                    name='kernel',submission_index=i) for i,r in enumerate(('A','B'))]
        tasks += [dict(is_compute=False,name='NOTIFY_WAIT',stream='44',start_us='0',end_us='55'),
                  dict(is_compute=False,name='NOTIFY_RECORD',stream='12',start_us='54',end_us='54')]
        result=dict(tasks=tasks,trial=dict(strategy='serial',timing=dict(rescued=False)),
                    cpu_scopes={f'P28/forward/{r}':dict(ts=str(i*15),dur='10') for i,r in enumerate(('A','B'))})
        return events,result

    def test_all_native_returns_before_release(self):
        self.assertTrue(gate_evidence(*self.fixture())['prequeue_proven'])

    def test_python_return_does_not_prove_native_submission(self):
        e,r=self.fixture()
        e[1]['dur']=35  # Native call is still in flight when gate opens.
        value=gate_evidence(e,r)
        self.assertTrue(value['python_forward_returned_before_release'])
        self.assertFalse(value['prequeue_proven'])
        self.assertEqual(value['pending_by_role'],{'B':1})

    def test_watchdog_release_invalidates_benchmark_premise(self):
        e,r=self.fixture()
        r['trial']['timing']['rescued']=True
        self.assertFalse(gate_evidence(e,r)['prequeue_proven'])

    def test_compute_before_wait_completion_is_rejected(self):
        e,r=self.fixture()
        r['tasks'][0]['start_us']='52'
        with self.assertRaisesRegex(AssertionError,'escaped'):
            gate_evidence(e,r)


if __name__=='__main__':unittest.main()
