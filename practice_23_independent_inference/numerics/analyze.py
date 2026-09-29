"""Offline numerical evidence verification, exact rational replay, and fair-control summaries."""
import argparse,csv,hashlib,json,statistics,struct,sys
from collections import defaultdict
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'practice_17_vllm_multistream'))
from analyze_run import number,end,inside,only,require


def read(p):return json.loads(p.read_text())
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def stats(xs):
    q=statistics.quantiles(xs,n=4,method='inclusive') if len(xs)>1 else [xs[0]]*3
    return dict(n=len(xs),median=statistics.median(xs),q1=q[0],q3=q[2],min=min(xs),max=max(xs))

def replay_exact(operands,report):
    manifest=read(operands/'manifest.json');data={}
    for name,m in manifest['tensors'].items():
        p=operands/m['file'];require(sha(p)==m['sha256'] and p.stat().st_size==m['bytes'],'operand identity '+name);data[name]=p.read_bytes()
    def value(name,i):
        bits=struct.unpack_from('<H',data[name],i*2)[0]
        return Fraction(struct.unpack('<f',struct.pack('<I',bits<<16))[0])
    checks=[]
    for r in report:
        n,k=manifest['tensors'][r['operator']]['shape']
        for c in r['exact_spotchecks']:
            row,col=divmod(c['flat_index'],n)
            exact=sum((value('inputs',row*k+i)*value(r['operator'],col*k+i) for i in range(k)),Fraction(0))
            require(exact==Fraction(int(c['exact_dot_numerator']),int(c['exact_dot_denominator'])),'rational dot mismatch')
            require(float(exact)==c['reference_fp64'],'FP64 reference mismatch')
            checks.append(dict(operator=r['operator'],index=c['flat_index'],verified=True))
    return checks


def kernel_evidence(root):
    events=json.loads(only(root.rglob('trace_view.json'),'trace').read_text(),parse_float=Decimal)
    if isinstance(events,dict):events=events['traceEvents']
    with only(root.rglob('kernel_details.csv'),'kernel csv').open() as f:rows=list(csv.DictReader(f))
    records={r['label']:r for r in read(root/'scopes.json')}
    scopes={e['name']:e for e in events if e.get('ph')=='X' and e.get('name') in records}
    require(set(scopes)==set(records),'missing operator scope')
    points=defaultdict(list);starts=defaultdict(list);finishes=defaultdict(list)
    def point(e):return e['pid'],e['tid'],number(e['ts'])
    for i,e in enumerate(events):
        if e.get('ph')=='X':points[point(e)].append((i,e))
        elif e.get('ph')=='s':starts[e.get('cat'),str(e['id'])].append(e)
        elif e.get('ph')=='f':finishes[(e.get('cat'),)+point(e)].append(e)
    def source(e,kind):
        finish=only(finishes[(kind,)+point(e)],kind+' finish');start=only(starts[kind,str(finish['id'])],kind+' start')
        i,event=only(points[point(start)],kind+' host');return i,event,str(finish['id'])
    found=[];used=set()
    for index,e in enumerate(events):
        a=e.get('args',{})
        if e.get('ph')!='X' or a.get('Task Type') not in ('AI_CORE','AI_VECTOR_CORE','MIX_AIC','MIX_AIV'):continue
        hi,host,hflow=source(e,'async_npu');ci,cann,cflow=source(e,'HostToDevice')
        label=only((s for s in scopes if inside(scopes[s],host)),'containing scope')
        require(str(cann['args']['connection_id'])==str(a['connection_id']),'CANN connection')
        j,row=only(((j,r) for j,r in enumerate(rows) if r['Name']==e['name'] and r['Stream ID'].strip()==str(a['Physic Stream Id'])
            and r['Task ID'].strip()==str(a['Task Id']) and number(r['Start Time(us)'])==number(e['ts'])),'CSV identity')
        require(j not in used,'duplicate CSV use');used.add(j)
        require(abs(number(row['Duration(us)'])-number(e['dur']))<=Decimal('.001'),'duration')
        expected_m=1024 if records[label]['mode']=='single' else 2048
        require(row['Input Shapes'].strip('"')==f'{expected_m},896;4864,896','matmul shape')
        found.append(dict(scope=label,kernel=e['name'],input_shapes=row['Input Shapes'],input_dtypes=row['Input Data Types'],
            trace_index=index,host_trace_index=hi,cann_trace_index=ci,torch_flow=hflow,cann_flow=cflow,
            connection_id=str(a['connection_id']),stream=str(a['Physic Stream Id']),task_id=str(a['Task Id']),csv_row=j))
    require(len(used)==len(rows)==12,'isolated kernel coverage')
    for label,r in records.items():require(sum(k['scope']==label for k in found)==(2 if r['mode']=='single' else 1),'scope launch count')
    return found


def analyze(root):
    names=['numerics-locate-r01','numerics-operator-r01','numerics-control-r01','numerics-control-all-r01',
           'numerics-control-model-r01','numerics-rounding-r02','numerics-fp32-formal-r01']
    original_root=Path(__file__).resolve().parents[1]/'results/published'
    original=read(original_root/'environment.json')
    require(read(original_root/'summary.json')['invalid_cells']==['prefill-1024/batch'],'original BF16 failure status changed')
    manifest={}
    for name in names:
        run=root/name
        env=read(run/'environment.json')
        for key in ('torch','torch_npu'):
            require(env[key]==original[key],'runtime version changed '+name)
        if 'checkpoint' in env:require(env['checkpoint']==original['checkpoint'],'checkpoint changed '+name)
        for f,v in read(run/'sources.json').items():
            path=run/'sources'/f;require(sha(path)==v['sha256'],'source changed '+str(path));manifest[name+'/'+f]=v['sha256']
    contracts=Path(__file__).with_name('contracts.json')
    if contracts.exists():require(manifest==read(contracts)['sources'],'source contracts changed')
    locate=root/names[0];operator=root/names[1];rounding=root/'numerics-rounding-r02';formal=root/'numerics-fp32-formal-r01'
    require(read(locate/'completed.json')['first_module']=='model.layers.0.mlp.gate_proj','first divergent module')
    for f in ('layers.json','modules.json'):
        require(all(c['exact'] for cs in read(locate/f)['unperturbed'].values() for c in cs),'observer changes results')
    modules=read(locate/'modules.json')['records']
    first_gate=next(i for i,r in enumerate(modules) if r['module']=='model.layers.0.mlp.gate_proj')
    require(all(r['exact'] for r in modules[:first_gate]),'earlier divergence')
    bounds=read(rounding/'rounding_bounds.json');require(all(r['repeat_exact'] for r in bounds),'native repeat instability')
    require(all(b['violations']==0 and b['max_bound_fraction']<=1 for r in bounds for b in r['bounds']),'rounding-bound violations')
    require(read(operator/'operands/manifest.json')['source_pt_sha256']==read(rounding/'plan.json')['operands_sha256'],'operand provenance')
    exact=replay_exact(operator/'operands',bounds);kernels=kernel_evidence(rounding)
    require(all(not read(root/n/'completed.json')['all_valid'] for n in ('numerics-control-r01','numerics-control-all-r01')),'partial control failures not reproduced')
    require(read(root/'numerics-control-model-r01/completed.json')['all_valid'],'full FP32 qualification')
    checks=read(formal/'checks.json');rows=read(formal/'measurements.json');plan=read(formal/'plan.json')
    require(len(rows)==len(checks)==144 and len({(r['case'],r['round'],r['mode']) for r in rows})==144,'formal sample coverage')
    require(plan['precision']=='full_fp32' and not plan['hf32'],'precision configuration')
    require(all(c['valid'] and c['greedy_equal'] and c['tensors']==49 for r in checks for c in r['checks']),'formal numerics')
    require(all(c['exact'] for r in checks if r['mode']!='batch' for c in r['checks']),'serial/parallel mismatch')
    require(read(formal/'weight_check.json')['unchanged'],'shared model state changed')
    performance=[]
    for phase,length in plan['cases']:
        case=f'{phase}-{length}';rs=[r for r in rows if r['case']==case];modes={};paired={}
        for mode in ('serial','parallel','batch'):
            xs=[r for r in rs if r['mode']==mode]
            modes[mode]=dict(wall_ms=stats([r['wall_us']/1000 for r in xs]),tasks_per_second=stats([2e6/r['wall_us'] for r in xs]),
                first_ready_ms=stats([min(r['ready_us'].values())/1000 for r in xs]),last_ready_ms=stats([max(r['ready_us'].values())/1000 for r in xs]),
                incremental_peak_mib=stats([(r['allocated_peak']-r['allocated_before'])/2**20 for r in xs]))
            if mode!='serial':
                ratios=[]
                for x in xs:
                    baseline=only((r for r in rs if r['mode']=='serial' and r['round']==x['round']),'paired baseline')
                    ratios.append(100*(x['wall_us']/baseline['wall_us']-1))
                paired[mode]=dict(change_percent=stats(ratios),faster_pairs=sum(x<0 for x in ratios))
        performance.append(dict(case=case,modes=modes,paired=paired))
    summary=dict(status='passed',first_layer=0,first_module='mlp.gate_proj',original_bf16_failure_preserved=True,
        native_batch_disagreements={r['operator']:r['disagreements'] for r in bounds},exact_rational_replays=len(exact),
        native_kernels_verified=len(kernels),kernel_family=sorted({k['kernel'] for k in kernels}),
        full_fp32_samples=len(rows),full_fp32_max_abs=max(c['max_abs'] for r in checks for c in r['checks']),
        full_fp32_original_tolerance_pass=True,performance=performance,
        conclusion='Shape-dependent matmul results consistent with accumulation/rounding; specific native tiling/reduction not observed. Full-FP32 is a separate control, not a BF16 fix.')
    out=root/'analysis';out.mkdir(exist_ok=True);save(out/'summary.json',summary);save(out/'kernel_evidence.json',kernels);save(out/'rational_replay.json',exact)
    print(json.dumps({k:v for k,v in summary.items() if k!='performance'},indent=2));return summary,manifest


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);analyze(p.parse_args().root)
