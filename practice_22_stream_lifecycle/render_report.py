"""Offline stream ownership explorer and an evidence-linked replay sequence SVG."""
import argparse
import html
import json
from pathlib import Path


def sequence_svg(data,path):
    r=next(r for r in data['replays'] if r['partition']=='submod_0' and r['phase']=='request-0/decode-1')
    graph=next(g for g in data['graphs'] if g['id']==r['graph'])
    byid={t['id']:t for t in data['tasks']}
    boundary=[byid[k] for k in r['boundary_tasks']];inner=[byid[k] for k in r['internal_tasks']]
    main=boundary[0]['stream'];internal=inner[0]['stream']
    def txt(x,y,s,size=14):return '<text x="{}" y="{}" font-size="{}">{}</text>'.format(x,y,size,html.escape(str(s)))
    s='<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 620"><defs><marker id="arrow" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto"><path d="M0 0L10 4L0 8" fill="#69d4c2"/></marker></defs><rect width="1200" height="620" fill="#101b2c"/><g font-family="sans-serif" fill="#e1ebf7">'
    s+=txt(30,35,'Practice 22 / actual request-0 decode-1 / submod_0',23)
    positions=[120,400,690,1030]
    for x,label in zip(positions,['Python / vLLM-Ascend','CANN','main stream '+main,'graph stream '+internal]):
        s+=txt(x-85,80,label)+'<path d="M{0} 95V485" stroke="#73839a" stroke-dasharray="5 5"/>'.format(x)
    def arrow(x1,x2,y,label):
        return '<path d="M{} {}H{}" stroke="#69d4c2" marker-end="url(#arrow)"/>'.format(x1,y,x2)+txt(min(x1,x2)+8,y-12,label,12)
    s+=arrow(120,400,135,'NPUGraph.replay → aclmdlRIExecuteAsync')
    s+=arrow(400,690,190,'connection_id '+str(boundary[0]['connection_id']))
    s+=txt(600,220,'MODEL_EXECUTE')
    s+=arrow(690,1030,260,'model '+str(graph['model_ids'])+' / exact captured task membership')
    s+='<rect x="677" y="285" width="26" height="132" fill="#dfb060" opacity=".8"/>'
    s+=txt(525,310,'NOTIFY_WAIT',13)
    s+=txt(760,305,str(len(inner)-1)+' captured compute tasks',14)
    s+=txt(760,334,'dump stream/task IDs checked',13)
    s+=txt(760,368,'NOTIFY_RECORD (same live model)',13)
    s+='<path d="M1030 396H710" stroke="#dfb060" stroke-dasharray="5 5"/>'
    s+=txt(730,420,'Completion interval checked; notify ID pair unknown',12)
    s+=txt(550,460,'next task on main stream',13)
    s+=txt(30,515,'Native model handle: '+graph['native_model']+' / capture occurrence: '+graph['id'],13)
    s+=txt(30,542,'Arrows show calls / runtime connection / graph membership; dashed line is NOT an exact event-pair edge.',13)
    s+=txt(30,568,'Vertical spacing is explanatory, not elapsed time. The HTML task timeline uses measured device timestamps.',13)
    s+=txt(30,594,'Both requests reuse the captured graph; no per-replay compilation or stream creation is inferred.',13)
    path.write_text(s+'</g></svg>')


def render(root):
    out=root/'report';out.mkdir(exist_ok=True)
    data={}
    for name in ('eager','graph','sampling-off','sampling-on'):
        run=root/'results'/('2026-09-28-'+name+'-run01')
        if not (run/'analysis/stream_evidence.json').exists():continue
        d=json.loads((run/'analysis/stream_evidence.json').read_text())
        # Keep all tasks interactive, without duplicating every raw observer span.
        selected_spans={s['creation_scope'] for s in d['lifetimes'] if s['creation_scope']}
        selected_spans.update(o['call'] for o in d['python_objects'] if o['acquisition']=='pool_or_new_resource')
        compact={k:d[k] for k in ('case','summary','streams','lifetimes','tasks','edges','event_dependencies','replays','gaps','handle_checks')}
        compact['python_spans']=[p for p in d['python_spans'] if p['id'] in selected_spans]
        compact['graphs']=[{k:v for k,v in g.items() if k!='dump_tasks'} for g in d['graphs']]
        compact['evidence']='../results/'+run.name+'/analysis/stream_evidence.json'
        compact['source_base']='../results/'+run.name+'/sources/'
        data[name]=compact
        if name=='graph':sequence_svg(d,out/'graph_replay_sequence.svg')
    page=(Path(__file__).parent/'viewer.html').read_text()
    page=page.replace('__DATA__',json.dumps(data,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c'))
    (out/'index.html').write_text(page)
    (out/'summary.json').write_text(json.dumps({k:v['summary'] for k,v in data.items()},indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path,nargs='?',default=Path(__file__).parent);a=p.parse_args();render(a.root)
