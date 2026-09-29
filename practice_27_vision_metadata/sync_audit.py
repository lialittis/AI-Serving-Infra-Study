"""Associate native sync calls with enclosing CPU ops or exact queue flow."""
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from decimal import Decimal
import json


def num(x):return Decimal(str(x))
def end(e):return num(e['ts'])+num(e.get('dur',0))


class Intervals:
    def __init__(self, rows):
        self.rows=sorted(rows,key=lambda x:num(x[1]['ts']))
        self.starts=[num(e['ts']) for _,e in self.rows]
        self.size=1
        while self.size<len(self.rows):self.size*=2
        self.tree=[Decimal('-Infinity')]*(2*self.size)
        for i,(_,e) in enumerate(self.rows):self.tree[self.size+i]=end(e)
        for i in range(self.size-1,0,-1):self.tree[i]=max(self.tree[2*i],self.tree[2*i+1])

    def enclosing(self, event):
        start=num(event['ts']);stop=end(event);found=[]
        limit=bisect_right(self.starts,start)
        pending=[(1,0,self.size)]
        while pending:
            node,left,right=pending.pop()
            if left>=limit or self.tree[node]<stop:continue
            if right-left==1:
                found.append(self.rows[left]);continue
            middle=(left+right)//2
            pending.extend(((2*node,left,middle),(2*node+1,middle,right)))
        return sorted(found,key=lambda x:(num(x[1].get('dur',0)),x[0]))



def audit(folder, syncs):
    files=list(folder.rglob('trace_view.json'));assert len(files)==1
    es=json.loads(files[0].read_text(),parse_float=Decimal)
    if isinstance(es,dict):es=es['traceEvents']
    cpus=defaultdict(list);dequeues=defaultdict(list);starts=defaultdict(list);finishes=defaultdict(list)
    point=lambda e:(e['pid'],e['tid'],num(e['ts']))
    for i,e in enumerate(es):
        if e.get('ph')=='X' and e.get('cat')=='cpu_op':cpus[e['tid']].append((i,e))
        if e.get('ph')=='X' and e.get('cat')=='dequeue':dequeues[e['tid']].append((i,e))
        if e.get('cat')=='async_task_queue':
            if e.get('ph')=='s':starts[str(e['id'])].append((i,e))
            if e.get('ph')=='f':finishes[point(e)].append(e)
    ci={t:Intervals(xs) for t,xs in cpus.items()};qi={t:Intervals(xs) for t,xs in dequeues.items()}
    scopes={e['name']:e for e in es if e.get('ph')=='X' and e.get('name','').startswith('P25/')}
    out=[]
    for record in syncs:
        e=es[record['trace_index']];parents=ci[e['tid']].enclosing(e) if e['tid'] in ci else []
        evidence='same host thread and interval containment';queue_id=None
        if not parents:
            queues=qi[e['tid']].enclosing(e) if e['tid'] in qi else []
            candidates=[]
            for _,q in queues:
                for finish in finishes[point(q)]:
                    source=starts[str(finish['id'])];assert len(source)==1
                    _,s=source[0]
                    for parent in ci[s['tid']].enclosing(s) if s['tid'] in ci else []:
                        candidates.append((parent,str(finish['id'])))
            if candidates:
                candidates.sort(key=lambda x:num(x[0][1].get('dur',0)))
                parents=[candidates[0][0]];queue_id=candidates[0][1];evidence='exact async_task_queue flow and enclosing host operator'
        parent=parents[0] if parents else None
        out.append(dict(**record,host_operator=parent[1]['name'] if parent else None,
            host_trace_index=parent[0] if parent else None,
            host_ancestors=[x['name'] for _,x in ci[parent[1]['tid']].enclosing(parent[1])] if parent else [],
            start_from_vision_us=str(num(e['ts'])-num(scopes[record['scope']]['ts'])),
            queue_id=queue_id,evidence=evidence if parent else 'unresolved'))
    return dict(calls=out,by_operator=dict(Counter(r['host_operator'] or 'unresolved' for r in out)))
