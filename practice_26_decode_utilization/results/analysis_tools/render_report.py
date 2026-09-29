"""Standalone offline explorer of steps, kernels, dispatch and raw hardware fields."""
import argparse
import base64
import gzip
import html
import io
import json
from pathlib import Path
from analyze import read,number

def svg(data,index=32):
    step=next(s for s in data['steps'] if s['index']==index)
    tasks=[t for t in data['tasks'] if t['step']==step['step']]
    lanes=sorted({t['stream'] for t in tasks},key=int)
    start=min(number(step['start_us']),number(step['host_execute_start_us']));finish=max(number(step['end_us']),number(step['host_sample_end_us']))
    width=1300;height=210+len(lanes)*35;scale=1100/float(finish-start)
    def x(t):return 170+float(number(t)-start)*scale
    def txt(xx,yy,s,size=13):return '<text x="{}" y="{}" font-size="{}">{}</text>'.format(xx,yy,size,html.escape(s))
    out='<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {} {}"><rect width="100%" height="100%" fill="#f5f8fa"/><g fill="#17313e" font-family="sans-serif">'.format(width,height)
    out+=txt(20,28,'Practice 26 / '+data['config']['mode']+' / '+data['config']['profile']+' / decode '+str(index),20)
    out+=txt(20,52,'Measured timestamps; tiny tasks have a minimum visible width. Block Num is not whole-chip utilization.')
    for j,(a,b) in enumerate([('host_execute_start_us','host_execute_end_us'),('host_sample_start_us','host_sample_end_us')]):
        out+='<rect x="{}" y="70" width="{}" height="14" fill="#8367aa"/>'.format(x(step[a]),max(.2,x(step[b])-x(step[a])))
    out+=txt(20,82,'CPU execute/sample')
    out+=txt(20,112,'CANN submission')
    for i in sorted({t['submission_index'] for t in tasks if t['submission_index'] is not None}):
        e=data['host_events'][str(i)]
        out+='<rect x="{}" y="98" width="{}" height="14" fill="#487eaf"><title>{}</title></rect>'.format(x(e['ts']),max(.2,float(number(e['dur']))*scale),html.escape(e['name']))
    for i,lane in enumerate(lanes):
        yy=140+i*35;out+=txt(20,yy+12,'stream '+lane)
        out+='<path d="M170 {}H1270" stroke="#d5dfe4"/>'.format(yy+17)
        for t in tasks:
            if t['stream']!=lane:continue
            color='#198a83' if t['is_compute'] else '#cb8c38' if 'WAIT' in t['name'] else '#6d8496'
            out+='<rect x="{}" y="{}" width="{}" height="16" fill="{}"><title>{}</title></rect>'.format(x(t['start_us']),yy,max(.15,x(t['end_us'])-x(t['start_us'])),color,html.escape('{} | {} us | {} | block {}'.format(t['name'],t['duration_us'],t['id'],t.get('block_num'))))
    out+=txt(170,height-35,'0 us');out+=txt(1120,height-35,'{:.1f} us'.format(float(finish-start)))
    return out+'</g></svg>'

def render(root,out):
    out.mkdir(parents=True,exist_ok=True);cases={}
    comparison=read(root/'comparison.json')
    for mode in ('eager','graph'):
        for profile in ('plain','pipe'):
            name=mode+'-'+profile;d=read(root/name/'analysis/evidence.json')
            # Preserve each exact task identity, raw counters and associated host/CANN call.
            compact_tasks=[]
            for t in d['tasks']:
                r=d['kernel_rows'][t['csv_row']] if t['csv_row'] is not None else None
                host=d['host_events'].get(str(t['host_index']));cann=d['host_events'].get(str(t['submission_index']))
                compact_tasks.append(dict(t,host=host,cann=cann,raw_csv=r))
            cases[name]=dict(summary=d['summary'],steps=d['steps'],tasks=compact_tasks,cores=d['cores'],limits=d['limits'],provenance=d['provenance'])
            (out/(name+'-decode32.svg')).write_text(svg(d))
    buffer=io.BytesIO()
    with gzip.GzipFile(fileobj=buffer,mode='wb',mtime=0) as f:f.write(json.dumps(dict(comparison=comparison,cases=cases),ensure_ascii=False,separators=(',',':')).encode())
    packed=base64.b64encode(buffer.getvalue()).decode()
    template=Path(__file__).with_name('viewer.html').read_text()
    (out/'index.html').write_text(template.replace('__PAYLOAD__',packed))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,default=Path(__file__).parent/'report');a=p.parse_args();render(a.root,a.output)
