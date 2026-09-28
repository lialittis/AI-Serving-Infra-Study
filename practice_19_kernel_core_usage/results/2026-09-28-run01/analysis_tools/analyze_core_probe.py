"""Validate real KV-write core observations and keep timing domains separate."""
import argparse
from collections import Counter, defaultdict
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import statistics


def require(ok, message):
    if not ok:
        raise ValueError(message)


def only(values, message):
    values = list(values)
    require(len(values) == 1, message + ' (found {})'.format(len(values)))
    return values[0]


def number(x):
    return Decimal(str(x).strip())


def end(e):
    return number(e['ts']) + number(e.get('dur', 0))


def inside(a, b):
    return a['pid'] == b['pid'] and a['tid'] == b['tid'] and number(a['ts']) <= number(b['ts']) and end(b) <= end(a)


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def stats(values):
    values = [float(x) for x in values]
    require(values, 'missing samples')
    return dict(count=len(values), median=statistics.median(values), min=min(values), max=max(values))


def analyze(run, metadata=None, traces=None, csv_rows=None):
    meta = metadata if metadata is not None else json.loads((run/'run.json').read_text())
    for name, sha in meta['source_sha256'].items():
        require(digest(run/name) == sha, 'source fingerprint mismatch')
    config, hw = meta['config'], meta['hardware']
    require(hw['cube_core_num'] > 0 and hw['vector_core_num'] > 0, 'missing device core query')
    require(config['block_size'] == 128 and config['num_blocks'] == 8 and config['kv_heads'] == 2 and config['head_size'] == 64 and config['dtype'] == 'torch.bfloat16', 'unexpected tensor configuration')
    require(config['slot_pattern'] == '128 + arange(tokens)', 'unexpected slot mapping')
    resources = meta['resources']
    spans = sorted((int(x['data_ptr']),int(x['data_ptr'])+x['bytes']) for x in [resources['key_cache'],resources['value_cache']] + [t for c in resources['cases'].values() for t in c.values()])
    require(all(a[1] <= b[0] for a,b in zip(spans,spans[1:])), 'tensor storage alias')
    for key in ('key_cache','value_cache'):
        require(resources[key]['shape'] == [8,128,2,64] and resources[key]['stride'] == [16384,128,64,1], 'KV cache layout mismatch')
    for n in config['tokens']:
        r = resources['cases'][str(n)]
        require(r['key']['shape'] == r['value']['shape'] == [n,2,64] and r['slots']['shape'] == [n], 'case shape mismatch')
    checks = {x['label']:x for x in meta['checks']}
    expected_checks = {'warmup/' + str(n) for n in config['tokens']} | {p + '/' + str(n) for p in ('plain','pipe') for n in config['tokens']}
    expected_checks |= {'timing/{}/{}'.format(r,n) for r in range(config['rounds']) for n in config['tokens']}
    require(len(checks) == len(meta['checks']) and set(checks) == expected_checks, 'correctness coverage mismatch')
    for check in checks.values():
        require(all(check[x] is True for x in ('key_pool_equal','value_pool_equal','input_key_unchanged','input_value_unchanged')), 'KV correctness failure')
        require(check['first_slot'] == 128 and check['last_slot'] == 127 + check['tokens'] and check['checked_elements_per_pool'] == 8*128*2*64, 'full-pool correctness scope mismatch')
    trials = meta['trials']
    require(Counter((t['round'],t['tokens']) for t in trials) == Counter((r,n) for r in range(config['rounds']) for n in config['tokens']), 'timing coverage mismatch')
    for t in trials:
        require(t['final_wait'] is True and t['iterations'] == config['iterations'] and
                t['completed_ns'] == t['finished_ns']-t['begin_ns'] and t['submit_ns'] == t['submitted_ns']-t['begin_ns'] and
                0 < t['submit_ns'] <= t['completed_ns'], 'invalid completed-work timing')
    tasks, inventories, provenance = [], {}, {}
    for profile in ('plain','pipe'):
        trace_path = only((run/('profiler_'+profile)).rglob('trace_view.json'), 'trace')
        csv_path = only((run/('profiler_'+profile)).rglob('kernel_details.csv'), 'kernel CSV')
        if traces is not None:
            events = traces[profile]
        else:
            events = json.loads(trace_path.read_text(),parse_float=Decimal)
            if isinstance(events,dict):
                events=events['traceEvents']
        if csv_rows is not None:
            rows = csv_rows[profile]
        else:
            with csv_path.open() as f:
                rows=list(csv.DictReader(f))
        provenance[profile] = dict(trace_sha256=digest(trace_path), csv_sha256=digest(csv_path))
        records = [r for r in meta['records'] if r['profile'] == profile]
        require(Counter((r['tokens'],r['repeat']) for r in records) == Counter((n,i) for n in config['tokens'] for i in range(config['profile_repeats'])), 'profile record coverage')
        scopes = {r['label']:only((e for e in events if e.get('ph') == 'X' and e.get('name') == r['label']), 'operator scope') for r in records}
        control = [e for e in events if e.get('ph') == 'X' and e.get('name','').startswith('P19/control/')]
        complete, starts, finishes = defaultdict(list),defaultdict(list),defaultdict(list)
        def point(e):
            return e['pid'],e['tid'],number(e['ts'])
        for i,e in enumerate(events):
            if e.get('ph') == 'X':
                complete[point(e)].append((i,e))
            elif e.get('ph') == 's':
                starts[e.get('cat'),str(e['id'])].append((i,e))
            elif e.get('ph') == 'f':
                finishes[(e.get('cat'),)+point(e)].append((i,e))
        def source(task, cat):
            fi,f=only(finishes[(cat,)+point(task)], cat+' endpoint')
            si,s=only(starts[cat,str(f['id'])], cat+' start')
            ei,e=only(complete[point(s)], cat+' source')
            return e,dict(flow_id=str(f['id']),start_index=si,end_index=fi,source_index=ei)
        matched = []
        excluded = []
        for i,task in enumerate(events):
            if task.get('ph') != 'X' or 'Task Type' not in task.get('args',{}):
                continue
            if task['name'] in ('PROFILING_ENABLE','PROFILING_DISABLE'):
                excluded.append(dict(trace_index=i,name=task['name'],reason='profiler control'))
                continue
            host,hproof=source(task,'async_npu')
            matching = [r for r in records if inside(scopes[r['label']],host)]
            if not matching:
                c=only((c for c in control if inside(c,host)), 'unexplained task outside KV scopes')
                excluded.append(dict(trace_index=i,name=task['name'],reason=c['name']))
                continue
            record=only(matching,'task scope association')
            require(task['name']=='ReshapeAndCacheNdKernel' and host['name']=='ReshapeCacheOperation','unexpected kernel/host operator')
            cann,cproof=source(task,'HostToDevice')
            require(cann['args']['connection_id']==task['args']['connection_id'],'CANN connection mismatch')
            a=task['args']
            j,row=only(((j,row) for j,row in enumerate(rows) if row['Name']==task['name'] and
                       str(row['Task ID']).strip()==str(a['Task Id']) and str(row['Stream ID']).strip()==str(a['Physic Stream Id']) and
                       number(row['Start Time(us)'])==number(task['ts'])), 'kernel CSV identity')
            require(j not in matched and abs(number(row['Duration(us)'])-number(task['dur']))<=Decimal('.001'), 'kernel duration/duplicate CSV mismatch')
            matched.append(j)
            n=record['tokens']
            shapes='"{0},2,64;{0},2,64;8,128,2,64;8,128,2,64;{0}"'.format(n)
            require(row['Input Shapes']==shapes,'kernel input shape mismatch')
            require(row['Accelerator Core']=='AI_VECTOR_CORE' and a['Task Type']=='AI_VECTOR_CORE', 'unexpected compute core type')
            cores=int(row['Block Num']);mix=int(row['Mix Block Num'])
            require(0 < cores <= hw['vector_core_num'] and mix == 0,'invalid reported core count')
            metrics={k:row[k] for k in row if k.startswith(('aiv_','aic_','aicore_','cube_utilization'))}
            if profile=='pipe':
                require(all(k in metrics and metrics[k] not in ('','N/A') for k in ('aiv_time(us)','aiv_vec_ratio','aiv_scalar_ratio','aiv_mte2_ratio','aiv_mte3_ratio')), 'pipeline metrics absent')
            tasks.append(dict(label=record['label'],profile=profile,tokens=n,repeat=record['repeat'],
                              name=task['name'],physical_stream=str(a['Physic Stream Id']), task_id=a['Task Id'],
                              trace_index=i,start_us=str(task['ts']),end_us=str(end(task)),duration_us=str(task['dur']),
                              host=host,cann=cann,torch_flow=hproof,cann_flow=cproof,connection_id=a['connection_id'],
                              csv_index=j,raw_csv=row,core_type=row['Accelerator Core'],block_num=cores,mix_block_num=mix,metrics=metrics))
        selected=[t for t in tasks if t['profile']==profile]
        require(Counter(t['label'] for t in selected)==Counter(r['label'] for r in records),'one KV kernel per scope required')
        require(len({t['physical_stream'] for t in selected})==1,'expected one physical stream')
        ordered=sorted(selected,key=lambda t:number(t['start_us']))
        require(all(number(a['end_us'])<=number(b['start_us']) for a,b in zip(ordered,ordered[1:])),'same-stream kernel overlap')
        # Existing native stream joins bound the output readback. Host submit
        # ranges alone would not establish completion.
        for n in config['tokens']:
            scope=only((s for s in control if s['name']=='P19/control/join-check/'+str(n)), 'join scope')
            waits=[e for e in events if e.get('ph')=='X' and e.get('name','').startswith('AscendCL@aclrtSynchronizeStream') and
                   e.get('args',{}).get('Thread Id',e.get('tid'))==scope['tid'] and
                   number(scope['ts'])<=number(e['ts']) and end(e)<=end(scope)]
            require(waits,'missing native stream completion wait')
            require(max(number(t['end_us']) for t in selected if t['tokens']==n)<=min(end(e) for e in waits),'KV write after completion boundary')
        inventories[profile]=dict(selected_tasks=len(selected),excluded_tasks=excluded,physical_stream=selected[0]['physical_stream'],raw_kernel_csv_rows=len(rows))
    summaries=[]
    for n in config['tokens']:
        group=[t for t in tasks if t['tokens']==n]
        core_counts={t['block_num'] for t in group}
        require(len(core_counts)==1,'core count differs across repeats/profile modes')
        core_count=next(iter(core_counts))
        time_trials=[t for t in trials if t['tokens']==n]
        pipe=[t for t in group if t['profile']=='pipe']
        summaries.append(dict(tokens=n,reported_vector_cores=core_count,observed_min_tokens_capacity=core_count==min(n,hw['vector_core_num']),
                              kv_write_bytes=n*2*64*2*2,
                              host_completed_per_call_us=stats(Decimal(t['completed_ns'])/t['iterations']/1000 for t in time_trials),
                              host_submit_per_call_us=stats(Decimal(t['submit_ns'])/t['iterations']/1000 for t in time_trials),
                              plain_kernel_us=stats(number(t['duration_us']) for t in group if t['profile']=='plain'),
                              pipe_kernel_us=stats(number(t['duration_us']) for t in pipe),
                              pipeline_median={k:statistics.median(float(t['metrics'][k]) for t in pipe) for k in ('aiv_time(us)','aiv_vec_ratio','aiv_scalar_ratio','aiv_mte2_ratio','aiv_mte3_ratio')}))
    return dict(hardware=hw,config=config,tasks=tasks,cases=summaries,profiles=inventories,
                correctness_checks=len(checks),unprofiled_trials=len(trials),provenance=provenance,
                summary=dict(all_correct=True,observed_core_rule_all_samples=all(c['observed_min_tokens_capacity'] for c in summaries),
                             exact_operator_tasks=len(tasks),per_core_timeline_available=False,
                             rule_scope='empirical rule for this dtype/layout/slot pattern and tested token counts; not a general kernel tiling proof'),
                limits=meta['limits'] + ['Pipeline ratios may overlap and are not a pie chart or whole-chip utilization.',
                                        'The ATB kernel internal token-to-physical-core assignment was not instrumented.'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    a=p.parse_args();data=analyze(a.run)
    out=a.run/'analysis';out.mkdir(exist_ok=True)
    (out/'core_evidence.json').write_text(json.dumps(data,ensure_ascii=False,indent=2,default=str)+'\n')
    print(json.dumps(dict(hardware=data['hardware'],summary=data['summary'],cases=data['cases']),ensure_ascii=False))
    from render_core_report import render
    render(data,out)


if __name__=='__main__':main()
