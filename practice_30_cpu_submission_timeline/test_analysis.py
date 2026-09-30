"""Counterexamples: do not fabricate coverage or add nested/thread wall times."""
import copy
import unittest
from analyze import phase_metrics,validate_queue
from exact_join import validate_csv,validate_steps,validate_graph_chunk
from compare import compare_outputs


class EvidenceTests(unittest.TestCase):
    def phases(self):
        parent=dict(id=1,parent=None,tid=1,label='parent',kind='engine_step',wall_start_ns=0,
                    wall_end_ns=100000,wall_ns=100000,thread_cpu_ns=60000)
        child=dict(id=2,parent=1,tid=1,label='child',kind='forward',wall_start_ns=10000,
                   wall_end_ns=60000,wall_ns=50000,thread_cpu_ns=40000)
        scopes=dict(parent=dict(ts='0',dur=100),child=dict(ts='10',dur=50))
        return [parent,child],scopes,{}

    def test_nested_time_not_counted_twice(self):
        rs=phase_metrics(*self.phases())
        self.assertEqual(sum(r['self_wall_us'] for r in rs),100)
        self.assertEqual(sum(r['self_thread_cpu_us'] for r in rs),60)

    def test_other_thread_is_not_a_nested_child(self):
        rs,scopes,sources=self.phases();rs[1]['tid']=2
        with self.assertRaisesRegex(ValueError,'thread'):phase_metrics(rs,scopes,sources)

    def test_overlapping_siblings_rejected(self):
        rs,scopes,sources=self.phases();r=dict(rs[1],id=3,label='third');rs.append(r);scopes['third']=scopes['child']
        with self.assertRaisesRegex(ValueError,'sibling'):phase_metrics(rs,scopes,sources)

    def test_negative_cpu_self_time_rejected(self):
        rs,scopes,sources=self.phases();rs[1]['thread_cpu_ns']=70000
        with self.assertRaisesRegex(ValueError,'negative'):phase_metrics(rs,scopes,sources)

    def test_missing_step_and_wrong_step_rejected(self):
        rs=[dict(index=i,scheduled={'r':10 if i==0 else 1}) for i in range(64)]
        validate_steps(rs)
        with self.assertRaisesRegex(ValueError,'scheduling'):validate_steps(rs[:-1])
        rs[32]['index']=31
        with self.assertRaisesRegex(ValueError,'scheduling'):validate_steps(rs)

    def test_queue_must_match_both_flow_and_correlation(self):
        q=dict(flow_id='7',enqueue=dict(args=dict(correlation_id=7)),dequeue=dict(args=dict(correlation_id=7)))
        validate_queue(q);q['dequeue']['args']['correlation_id']=8
        with self.assertRaisesRegex(ValueError,'identity'):validate_queue(q)

    def test_kernel_name_alone_is_not_identity(self):
        task=dict(name='mm',stream='1',task_id='7',start_us='100',duration_us='2')
        row={'Name':'mm','Stream ID':'1','Task ID':'7','Start Time(us)':'100','Duration(us)':'2'}
        validate_csv(task,row)
        for field,value in [('Stream ID','2'),('Task ID','8'),('Start Time(us)','101')]:
            bad=dict(row);bad[field]=value
            with self.assertRaisesRegex(ValueError,'identity'):validate_csv(task,bad)

    def test_graph_task_sequence_and_completion_are_required(self):
        dump=[dict(args={'Task Type':'AI_CORE','Stream Id':9,'Task Id':3})]
        tasks=[dict(stream='9',task_id='3',name='mm',start_us='10',end_us='12'),
               dict(stream='9',task_id='4',name='NOTIFY_RECORD',start_us='12',end_us='13')]
        validate_graph_chunk(tasks,dump,dict(start_us='9'),dict(end_us='14'))
        bad=copy.deepcopy(tasks);bad[0]['stream']='8'
        with self.assertRaisesRegex(ValueError,'sequence'):
            validate_graph_chunk(bad,dump,dict(start_us='9'),dict(end_us='14'))
        with self.assertRaisesRegex(ValueError,'enclosure'):
            validate_graph_chunk(tasks,dump,dict(start_us='9'),dict(end_us='12'))
        with self.assertRaisesRegex(ValueError,'size'):
            validate_graph_chunk(tasks[:-1],dump,dict(start_us='9'),dict(end_us='14'))

    def test_same_tokens_do_not_prove_equal_logprobs(self):
        a=dict(tokens=[5],finish_reason='length',logprobs=[{'5':dict(logprob=-1.,rank=1)}])
        b=copy.deepcopy(a);b['logprobs'][0]['5']['logprob']=-1.125
        diff=compare_outputs(a,b)
        self.assertTrue(diff['tokens_equal'])
        self.assertFalse(diff['exact_output_equal'])
        self.assertEqual(diff['shared_logprob_max_abs'],.125)
        b['logprobs'][0]['6']=b['logprobs'][0].pop('5')
        self.assertFalse(compare_outputs(a,b)['logprob_keys_equal'])


if __name__=='__main__':unittest.main()
