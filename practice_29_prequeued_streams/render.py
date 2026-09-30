"""Small static timeline: actual task intervals, full submission and compute zoom."""
import argparse
from decimal import Decimal as D
from html import escape
import json
from pathlib import Path


def render(data, path):
    colors = {'A':'#2563eb', 'B':'#e8790b'}
    compute = [t for t in data['tasks'] if t['is_compute']]
    starts = [D(t['start_us']) for t in compute]
    stops = [D(t['end_us']) for t in compute]
    scopes = {r:data['cpu_scopes']['P28/forward/'+r] for r in ('A','B')}
    origin = min(D(s['ts']) for s in scopes.values())
    release = D(data['gate_evidence']['release_native_us']) if data.get('gate_evidence') else None
    stream_ids = sorted({t['stream'] for t in compute})
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1300" height="660" viewBox="0 0 1300 660">',
        '<style>text{font:13px sans-serif;fill:#172033}.title{font-size:20px;font-weight:bold}.small{font-size:11px}</style>',
        '<rect width="1300" height="660" fill="#f8fafc"/>',
        f'<text x="24" y="30" class="title">Practice 29 — {escape(data["trial"]["label"])} — measured trace</text>',
        f'<text x="24" y="54">A: blue · B: orange · {len(compute)} exact kernel joins · overlap {float(D(data["overlap_us"]))/1000:.3f} ms</text>']
    full=(origin-100, max(stops)+100)
    zoom=(min(starts)-100,max(stops)+100)
    for panel,(lo,hi) in enumerate((full,zoom)):
        top=100+panel*270
        scale=1040/float(hi-lo)
        def x(t):return 225+float(t-lo)*scale
        svg.append(f'<text x="24" y="{top-12}" class="title">{"CPU submission → completion" if panel==0 else "Device compute zoom"}</text>')
        rows={'CPU':top+30,'CANN':top+70, **{s:top+110+40*i for i,s in enumerate(stream_ids)}}
        for name,y in rows.items():
            label=name if name in ('CPU','CANN') else 'NPU stream '+name
            svg.extend([f'<text x="24" y="{y+15}">{label}</text>',f'<line x1="225" y1="{y+25}" x2="1265" y2="{y+25}" stroke="#cbd5e1"/>'])
        def bar(begin,stop,row,color,title,height=20):
            if stop<lo or begin>hi:return
            a,b=max(begin,lo),min(stop,hi)
            svg.append(f'<rect x="{x(a):.3f}" y="{rows[row]}" width="{max(.25,x(b)-x(a)):.3f}" height="{height}" fill="{color}"><title>{escape(title)}</title></rect>')
        for r,s in scopes.items():
            bar(D(s['ts']),D(s['ts'])+D(s['dur']),'CPU',colors[r],r+' forward → logits → sample')
        for t in data['tasks']:
            if t['name']=='NOTIFY_WAIT':
                bar(D(t['start_us']),D(t['end_us']),t['stream'],'#cbd5e1','Native Notify wait; no AI-core busy-wait')
        for t in compute:
            r=t['role']
            bar(D(t['start_us']),D(t['end_us']),t['stream'],colors[r],r+': '+t['name'])
            e=data['host_events'][str(t['submission_index'])]
            bar(D(e['ts']),D(e['ts'])+D(str(e['dur'])),'CANN',colors[r],r+': '+e['name'])
        if release is not None and lo<=release<=hi:
            xx=x(release)
            svg.append(f'<line x1="{xx}" x2="{xx}" y1="{top+8}" y2="{top+188}" stroke="#dc2626" stroke-dasharray="4 3"/>')
            svg.append(f'<text x="{min(xx+4,1150)}" y="{top+4}" fill="#dc2626">release API begins</text>')
        for i in range(6):
            t=lo+(hi-lo)*i/5
            svg.append(f'<text x="{x(t)-12}" y="{top+210}" class="small">{float(t-origin)/1000:.2f} ms</text>')
    svg.extend(['<text x="24" y="642" class="small">Time relative to first CPU forward. Kernel widths are actual intervals; tooltip names identify tasks. Profiler data, not benchmark speedup.</text>','</svg>'])
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text('\n'.join(svg))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    for root in (a.run/'model/trials').iterdir():
        render(json.loads((root/'analysis.json').read_text()),a.output/(root.name+'.svg'))


if __name__=='__main__':main()
