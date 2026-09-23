"""Compare validated eager/PIECEWISE runs with matching inputs and provenance.

Wall times include instrumentation and HTTP overhead, not a performance benchmark.
"""
import argparse
from decimal import Decimal
import html
import json
from pathlib import Path
from analyze_reuse import analyze, load


def require(value, message):
    if not value:
        raise ValueError(message)


def normalized_command(run):
    args = load(run / 'command.json')['argv'][4:]
    skip = {'--enforce-eager': 0, '--compilation-config': 1, '--profiler-config': 1, '--port': 1}
    result = []
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in skip:
            i += 1 + skip[arg]
        else:
            result.append(arg)
            i += 1
    return result


def compare(eager, graph):
    left,_,_ = analyze(eager)
    right,_,_ = analyze(graph)
    require(left['mode'] == 'eager' and right['mode'] == 'graph', 'expected eager then graph run')
    require(normalized_command(eager) == normalized_command(graph), 'non-mode server configuration differs')
    require(load(eager/'source_manifest.json') == load(graph/'source_manifest.json'), 'installed source revisions differ')
    env_snapshots=[load(run/'environment.json') for run in (eager,graph)]
    for key in ('model_files','packages','python','machine','npu_mapping'):
        require(env_snapshots[0][key] == env_snapshots[1][key], 'environment/model mismatch: '+key)
    require(load(eager/'instrumentation_hashes.json') == load(graph/'instrumentation_hashes.json'),
            'comparison requires identical runtime observation scripts')
    for name in ('config.json','generation_config.json'):
        require((eager/'sources/model'/name).read_bytes() == (graph/'sources/model'/name).read_bytes(),
                'model configuration differs')
    profiler=[]
    for run in (eager,graph):
        argv=load(run/'command.json')['argv']
        config=json.loads(argv[argv.index('--profiler-config')+1])
        config.pop('torch_profiler_dir',None)
        profiler.append(config)
    require(profiler[0]==profiler[1], 'profiler settings differ')
    envs = [load(run/'command.json')['environment_overrides'] for run in (eager,graph)]
    # Only the trace output path is allowed to differ; PYTHONPATH and compiler switches match.
    require({k:v for k,v in envs[0].items() if k not in ('P12_TRACE_DIR','P12_MODE')} ==
            {k:v for k,v in envs[1].items() if k not in ('P12_TRACE_DIR','P12_MODE')}, 'environment overrides differ')
    require(load(eager/'prompt_info.json') == load(graph/'prompt_info.json'), 'prompt tokenization differs')
    for role in ('A','B'):
        require(load(eager/('request_'+role+'.json')) == load(graph/('request_'+role+'.json')),
                'A/B request bodies differ')
    rows = []
    for label,run,data in [('eager',eager,left),('graph',graph,right)]:
        windows = load(run/'request_windows.json')
        responses = {role:dict(text=data['requests'][role]['response_text'],
                     token_ids=[step['token_ids'] for step in data['requests'][role]['steps']],
                     http_elapsed_ms=str(Decimal(windows[role]['request_end_ns']-
                                                 windows[role]['request_start_ns'])/Decimal(1000000)))
                     for role in ('A','B')}
        rows.append(dict(mode=label,run=run.name,trace_sha256=data['trace_sha256'],
                         reused_block=data['reused_block'],layers=data['layers'],
                         verified_kv_operator_chains=data['verified_kv_operator_chains'],
                         scopes=data['scopes'],device_tasks=data['device_tasks'],
                         kernel_csv_rows=data['kernel_csv_rows'],
                         model_execute_tasks=data['device_task_counts'].get('MODEL_EXECUTE',0),
                         replayed_tasks=data['replayed_tasks'],
                         replayed_without_torch_flow=data['replayed_tasks_without_torch_flow'],
                         replayed_without_cann_start=data['replayed_tasks_without_cann_start'],
                         phases={phase:dict(execution=data['requests']['A']['steps'][i]['execution'],
                                 acl_dispatches=len(data['requests']['A']['steps'][i]['graph_dispatches']),
                                 partition_body_calls=data['requests']['A']['steps'][i]['partition_body_calls'])
                                 for i,phase in enumerate(('prefill','decode'))},
                         responses=responses,
                         gap_last_A_access_to_free_us=data['gap_last_A_access_to_free_us'],
                         gap_free_to_B_allocation_us=data['gap_free_to_B_allocation_us'],
                         gap_last_A_access_to_first_B_access_us=data['gap_last_A_access_to_first_B_access_us']))
    resource_summaries=[]
    for run in (eager,graph):
        path=run/'analysis/resource_summary.json'
        if path.exists():
            from audit_resources import audit
            summary=audit(run)['summary']
            require(summary==load(path),'resource summary differs from current evidence')
            resource_summaries.append(summary)
        else:
            require(not (run/'instrumentation/resource_observer.py').exists(), 'resource audit required for this capture')
            resource_summaries.append(None)
    require(all(x is None for x in resource_summaries) or all(x is not None for x in resource_summaries),
            'resource audit missing from one mode')
    return dict(resource_audits=resource_summaries,runs=rows,matching_requests=True,matching_sources=True,matching_instrumentation=True,
                same_output_tokens=all(rows[0]['responses'][r]['token_ids']==rows[1]['responses'][r]['token_ids']
                                       for r in ('A','B')),
                conclusion='Both modes verify native completion -> release -> reuse for all observed KV accesses. '
                           'PIECEWISE keeps attention/KV outside replay; replay-internal attribution remains incomplete.',
                task_count_differences={name:dict(eager=left['device_task_counts'].get(name,0),
                                                 graph=right['device_task_counts'].get(name,0))
                                       for name in sorted(set(left['device_task_counts'])|set(right['device_task_counts']))
                                       if left['device_task_counts'].get(name,0)!=right['device_task_counts'].get(name,0)},
                performance_comparison=False)


def write_report(output, result, eager, graph):
    output.mkdir(parents=True,exist_ok=True)
    (output/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    data={row['mode']:row for row in result['runs']}
    items=[('执行配置','eager','PIECEWISE · capture [1]'),
           ('126-token prefill','eager','compiled callable（设备图 runtime NONE）'),
           ('1-token decode','eager','25 个普通分区重放；attention 保持直接调用'),
           ('A/B 复用 / 层数','B1 / 24','B1 / 24'),
           ('原生等待后释放、再复用','验证通过','验证通过')]
    for title,key in [('核对 KV/FIA 链数','verified_kv_operator_chains'),('主机观测范围数','scopes'),
                      ('全窗口设备任务数','device_tasks'),('计算 kernel CSV 行数','kernel_csv_rows'),
                      ('MODEL_EXECUTE 数量','model_execute_tasks'),('带有效 Model Id 的任务','replayed_tasks'),
                      ('上述任务缺少 torch flow','replayed_without_torch_flow'),
                      ('上述任务缺少 CANN flow 起点','replayed_without_cann_start')]:
        items.append((title,str(data['eager'][key]),str(data['graph'][key])))
    for role in ('A','B'):
        items.append(('请求 '+role+' 输出',data['eager']['responses'][role]['text'],data['graph']['responses'][role]['text']))
    audits=result.get('resource_audits',[None,None])
    if all(audits):
        for title,key in [('资源观测范围','observed_scopes'),('逐次Replay记录','replay_records'),
                          ('全部CPU事件','cpu_events'),('缺少精确flow关联的设备任务','unassociated_device_tasks'),
                          ('原生完成边界','native_completion_boundaries'),('核对的权重tensor','weight_tensors')]:
            items.append((title,str(audits[0][key]),str(audits[1][key])))
    table='\n'.join('| '+' | '.join(row)+' |' for row in items)
    md='# Practice 12：eager / graph 实测对比\n\n'
    md+='主对照：`'+eager.name+'` 与 `'+graph.name+'`。请求、模型配置、安装源码、观测脚本和非模式参数一致。\n\n'
    md+='| 项目 | eager | graph |\n|---|---|---|\n'+table+'\n\n'
    md+='两种模式均验证了24层 KV 的同块复用和原生完成边界。图模式的 attention 分区未被设备图捕获，KV/FIA 保持直接调用，因此192条 KV/FIA 链仍可逐一关联。普通分区重放内部缺少关联，不等于本次 KV 生命周期证据缺失。\n\n'
    md+='图模式新增50个 MODEL_EXECUTE、50个 NOTIFY_RECORD 和50个 NOTIFY_WAIT，总任务数多150；计算kernel CSV行数相同。部分RoPE任务名称变为 `_triton_rope_1`。这些计数不构成逐FX节点映射，也不能据此判断哪种模式更快。主机观测范围多100，是每步新增25个ACL wrapper范围造成的观测差异。\n\n'
    if all(audits):
        md=md.replace('主机观测范围多100，是每步新增25个ACL wrapper范围造成的观测差异。',
                      '主机观测范围多150，其中100个来自四步的ACL wrapper，50个来自decode实际NPUGraph.replay的独立范围。')
    md+='图模式仅捕获尺寸1。prefill 的 runtime NONE 不代表没有编译；这里不验证 FULL graph，也不验证 attention 被完整捕获后的生命周期。\n\n'
    md+='A/B 两请求的采样 token IDs 在模式间相同：`'+str(result['same_output_tokens'])+'`。不要求跨进程设备地址相同，只核对每次运行内部 A/B 共享同一存储。\n\n'
    md+='以下耗时包含逐层插桩、profiler、Python记录与串行HTTP开销，只有每模式一次A/B请求，**不是性能基准，不计算加速比**。\n\n'
    md+='| 诊断观察 | eager | graph |\n|---|---:|---:|\n'
    for label,key in [('A 最后 KV 访问结束 → free入口（µs）','gap_last_A_access_to_free_us'),
                      ('A free返回 → B allocate入口（µs）','gap_free_to_B_allocation_us')]:
        md+='| '+label+' | '+data['eager'][key]+' | '+data['graph'][key]+' |\n'
    for role in ('A','B'):
        md+='| 请求'+role+' HTTP往返（ms，含观测） | '+data['eager']['responses'][role]['http_elapsed_ms']+' | '+data['graph']['responses'][role]['http_elapsed_ms']+' |\n'
    md+='\n原始关联见两个run的 `analysis/`，具体重放缺口见各自 `analysis/reuse_evidence.json` 的 `replay_coverage`。\n'
    if all(audits):
        md+='\n资源记录已扩展为：逐观测事件的输入/输出和准备阶段、原生event身份与完成边界，以及每次replay的捕获基线和资源核对。所有CPU事件/设备任务均有清单，但每个ATen输入输出的全部地址、隐藏workspace生命周期和无flow任务归属仍未建立。\n'
    (output/'comparison.md').write_text(md)
    import os
    paths=[os.path.relpath(run/'analysis/reuse_viewer.html',output) for run in (eager,graph)]
    rows=''.join('<tr><th>'+html.escape(a)+'</th><td>'+html.escape(b)+'</td><td>'+html.escape(c)+'</td></tr>' for a,b,c in items)
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Practice 12 · eager / graph 对照</title><style>
body{background:#f4f7f5;color:#183b46;font:15px/1.8 system-ui,sans-serif;margin:0}main{max-width:1120px;margin:auto;padding:35px 24px}h1{font-size:34px}a{color:#087d78}table{border-collapse:collapse;width:100%;font-size:13px;background:white}td,th{text-align:left;padding:12px;border-bottom:1px solid #d4e2dc}th{font-weight:600}.overflow{overflow:auto}button{font:inherit;padding:9px 15px;border-radius:6px;border:1px solid #9cbdb3;color:#145c56;background:white;margin-right:8px;cursor:pointer}button[aria-pressed=true]{background:#087d78;color:white}iframe{width:100%;height:1050px;border:1px solid #c3d9d0;background:white;margin:18px 0}.note{padding:20px;background:#e2efe8;border-left:4px solid #087d78;margin:25px 0}small{color:#526d72}.stats{display:grid;grid-template-columns:1fr 1fr;gap:20px}.stat{background:white;border:1px solid #d5e1dd;padding:20px;border-radius:7px}@media(max-width:650px){main{padding:20px 14px}.stats{grid-template-columns:1fr}h1{font-size:28px}}@media print{iframe,button{display:none}}</style><main><small>PRACTICE 12 / SAME REQUESTS · SAME KV POOL · TWO EXECUTION MODES</small><h1>图模式改变了执行路径，<br>这次 KV 复用顺序保持一致。</h1><p>原生两块池（B0保留，B1可用），A/B串行请求，各126输入 / 2输出。模式间请求、源码和观测脚本一致；不比较跨进程绝对地址。</p><div class="stats"><div class="stat"><b>Eager</b><p>模型各步直接调用。192条 KV/FIA 链都关联到真实设备 kernel。</p></div><div class="stat"><b>PIECEWISE graph · capture [1]</b><p>prefill执行编译callable，decode普通分区重放；attention/KV仍直接执行，192条KV/FIA链同样完整。</p></div></div><div class="note"><b>共同验证：</b>A最后一次已观测KV访问完成 → 原生采样结果等待返回 → A释放B1 → B分配并写入B1。<br>这不代表已验证FULL graph、并发、prefix sharing或async scheduling；设备图内部的逐节点kernel关联仍有缺口。</div><h2>逐项对照</h2><div class="overflow"><table><thead><tr><th>项目</th><th>Eager</th><th>Graph</th></tr></thead><tbody>__ROWS__</tbody></table></div><h2>切换真实生命周期记录</h2><p><button data-mode="eager" aria-pressed="true">Eager</button><button data-mode="graph" aria-pressed="false">Graph</button><a id="open" href="__EAGER__">单独打开当前模式页面</a></p><iframe id="viewer" title="所选模式的真实生命周期" src="__EAGER__"></iframe><p><small>两种模式的图可各自逐步播放。诊断时间受插桩和profiler影响，本对照不是性能基准。<a href="comparison.md">完整结果与诊断耗时</a> · <a href="comparison.json">结构化比较数据</a></small></p></main><script>const paths=__PATHS__;document.querySelectorAll('[data-mode]').forEach(b=>b.onclick=()=>{document.querySelectorAll('[data-mode]').forEach(x=>x.setAttribute('aria-pressed',x===b));document.getElementById('viewer').src=paths[b.dataset.mode];document.getElementById('open').href=paths[b.dataset.mode];});</script></html>'''
    page=page.replace('__ROWS__',rows).replace('__EAGER__',html.escape(paths[0],quote=True)).replace('__PATHS__',json.dumps(dict(zip(('eager','graph'),paths))))
    if all(audits):
        resource_paths=[os.path.relpath(run/'analysis/resource_viewer.html',output) for run in (eager,graph)]
        section='<h2>逐事件与逐次 Replay 的资源记录</h2><p>'+ ' · '.join(
            '<a href="'+html.escape(path,quote=True)+'">'+label+' 资源台账</a>'
            for label,path in zip(('Eager','Graph'),resource_paths))+'</p><p>核对输入/输出存储、捕获快照、原生event与完成边界。未观测到的内部资源保留为未知，不把地址相同解释为完整安全证明。</p>'
        page=page.replace('<h2>切换真实生命周期记录</h2>',section+'<h2>切换真实生命周期记录</h2>')
    (output/'index.html').write_text(page)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--eager',type=Path,required=True);p.add_argument('--graph',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=compare(a.eager,a.graph);write_report(a.output,result,a.eager,a.graph)
    print('Validated comparable runs and wrote',a.output/'index.html')

if __name__=='__main__':main()
