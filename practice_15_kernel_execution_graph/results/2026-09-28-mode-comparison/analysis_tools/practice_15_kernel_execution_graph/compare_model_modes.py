"""Compare matched model-mode captures; no timing-based speedup claim."""
import argparse
from collections import Counter
import html
import json
import os
from pathlib import Path

from build_mode_graph import analyze, load, require


def normalized_command(run):
    args = load(run / 'command.json')['argv']
    normalized = []
    i = 0
    while i < len(args):
        key = args[i]
        if key == '--enforce-eager':
            i += 1
        elif key == '--compilation-config':
            i += 2
        elif key == '--profiler-config':
            config = json.loads(args[i + 1])
            config.pop('torch_profiler_dir')
            normalized.extend([key, config])
            i += 2
        else:
            normalized.append(key)
            i += 1
    return normalized


def compare(eager, graph):
    require(normalized_command(eager) == normalized_command(graph), 'non-mode command mismatch')
    for name in ('request.json', 'warmup_request.json', 'prompt_info.json', 'source_manifest.json', 'instrumentation_hashes.json'):
        require(load(eager / name) == load(graph / name), 'comparison mismatch: ' + name)
    envs = [load(p / 'environment.json') for p in (eager, graph)]
    for key in ('python', 'packages', 'model_files', 'model_path', 'machine', 'os_release', 'environment', 'source_repositories', 'npu_mapping'):
        require(envs[0][key] == envs[1][key], 'environment mismatch: ' + key)
    overrides = [load(p / 'command.json')['environment_overrides'] for p in (eager, graph)]
    require([{k:v for k,v in e.items() if k not in ('P13_TRACE_DIR', 'TRITON_CACHE_DIR')} for e in overrides][0] ==
            [{k:v for k,v in e.items() if k not in ('P13_TRACE_DIR', 'TRITON_CACHE_DIR')} for e in overrides][1], 'observer environment mismatch')
    data = [analyze(p) for p in (eager, graph)]
    require([d['summary']['mode'] for d in data] == ['eager', 'graph'], 'wrong comparison order')
    tokens = []
    for d in data:
        returns = [s['exit'] for s in d['parameter_scopes'].values() if s['entry']['kind'] == 'to_list']
        tokens.append([e['token_ids'] for e in sorted(returns, key=lambda e:e['step'])])
    counts = [Counter(n['name'] for n in d['nodes'] if n['kind'] == 'kernel') for d in data]
    return dict(matching_configuration=True, matching_sources=True, matching_requests=True,
                same_output_token_ids=tokens[0] == tokens[1], output_token_ids=tokens,
                runs=[dict(run=p.name, **d['summary']) for p,d in zip((eager, graph), data)],
                task_count_differences={n:dict(eager=counts[0][n], graph=counts[1][n]) for n in sorted(set(counts[0]) | set(counts[1])) if counts[0][n] != counts[1][n]},
                performance_benchmark=False)


def report(out, result, eager, graph):
    out.mkdir(parents=True, exist_ok=True)
    (out / 'comparison.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    left, right = result['runs']
    fields = [('设备任务（不含 profiler 控制）', 'kernel_tasks'), ('计算 kernel CSV 行数', 'compute_csv_rows'),
              ('物理 stream 数', 'physical_streams'), ('具有两条精确 host flow 的任务', 'tasks_with_both_flows'),
              ('关联到异步队列的任务', 'tasks_with_queue'), ('NPUGraph.replay 次数', 'replay_calls'),
              ('MODEL_EXECUTE 任务', 'model_execute_tasks'), ('有 runtime connection 的 replay 边界任务', 'replay_boundary_tasks_with_runtime_connection'),
              ('有 Model Id、缺少逐算子 flow 的任务', 'model_tasks_without_both_flows'),
              ('原生结果完成边界', 'native_completion_boundaries'), ('Attention 调用（24 层 × 4 步）', 'attention_invocations'),
              ('本次观察到的 Triton 编译', 'compiler_calls'), ('正式请求期间编译', 'measured_compiles')]
    rows = [(label, left[key], right[key]) for label, key in fields]
    rows += [('RoPE 的确定 RAW 边', left['edges_by_kind'].get('data_contract', 0), right['edges_by_kind'].get('data_contract', 0)),
             ('KV 存储候选边', left['edges_by_kind'].get('storage_candidate', 0), right['edges_by_kind'].get('storage_candidate', 0))]
    links = [os.path.relpath(p / 'analysis/index.html', out) for p in (eager, graph)]
    intro = ('同一模型、BF16、单卡、10 输入 / 4 输出、同脚本与源码，重新采集 eager 与 PIECEWISE capture [1]。'
             'prefill 执行编译 callable；三个 decode 各重放 25 个普通分区，attention 保持直接调用。')
    limits = ('26 条物理 stream 不等于 26 路计算并行。缺少图内 flow 时不建立逐 kernel 的 Python/FX 归属；'
              'NOTIFY 任务没有配对证据，不补画跨流同步边。无数据边表示未覆盖，不表示独立。'
              '本实验带插桩、只有一个正式请求，不报告性能加速比。')
    text = '# Practice 15：eager / PIECEWISE 执行图对照\n\n' + intro + '\n\n'
    text += '| 项目 | eager | graph |\n|---|---:|---:|\n' + '\n'.join('| {} | {} | {} |'.format(*r) for r in rows)
    text += '\n\n输出 token IDs 一致：`{}`。输出：`{}`。\n\n{}\n'.format(result['same_output_token_ids'], right['response'], limits)
    text += '\n[打开 eager 图]({}) · [打开 graph 图]({})\n'.format(*links)
    (out / 'comparison.md').write_text(text)
    table = ''.join('<tr><td>{}</td><td>{}</td><td>{}</td></tr>'.format(*(html.escape(str(v)) for v in r)) for r in rows)
    diagram = '''<svg xmlns="http://www.w3.org/2000/svg" width="1060" height="410" viewBox="0 0 1060 410" role="img" aria-label="eager 与 PIECEWISE decode 的提交路径">
<style>text{font:16px sans-serif;fill:#193247}.box{fill:#edf4fd;stroke:#668eae} .warn{fill:#fff0d6;stroke:#c39438}path{stroke:#63809f;stroke-width:2;fill:none}</style>
<defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8"/></marker></defs>
<text x="24" y="30">eager：直接算子调用与逐任务 profiler flow</text>
<rect class="box" x="24" y="50" width="220" height="55" rx="8"/><text x="40" y="83">CPU / PyTorch 算子</text>
<rect class="box" x="320" y="50" width="290" height="55" rx="8"/><text x="340" y="83">异步队列 → CANN 下发</text>
<rect class="box" x="695" y="50" width="330" height="55" rx="8"/><text x="715" y="83">NPU kernel（物理 stream 46）</text>
<path d="M244,78 H318 M610,78 H693" marker-end="url(#a)"/>
<text x="24" y="160">PIECEWISE decode：普通分区 replay 与 attention 直接调用交替</text>
<rect class="box" x="24" y="185" width="220" height="70" rx="8"/><text x="40" y="214">NPUGraph.replay</text><text x="40" y="238">捕获基线 / 输入输出地址</text>
<rect class="box" x="320" y="185" width="290" height="70" rx="8"/><text x="336" y="214">aclmdlRIExecuteAsync</text><text x="336" y="238">connection_id</text>
<rect class="box" x="695" y="185" width="330" height="70" rx="8"/><text x="712" y="214">MODEL_EXECUTE / NOTIFY_WAIT</text><text x="712" y="238">主执行 stream 上的下发边界</text>
<path d="M244,219 H318 M610,219 H693" marker-end="url(#a)"/>
<rect class="warn" x="24" y="292" width="1001" height="72" rx="8"/><text x="42" y="321">图内 kernel + NOTIFY_RECORD：保留 Model Id、物理 stream、Task Id、时间和 CSV</text><text x="42" y="347">缺少逐算子 flow：不把它们强行映射到上面的某次 replay，也不虚构 NOTIFY 依赖</text>
<text x="24" y="397">两模式：KV 写入 / FIA 直接调用；原生结果 event 完成 → CPU 等待返回 → 后续步骤</text></svg>'''
    (out / 'execution_paths.svg').write_text(diagram)
    (out / 'index.html').write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Practice 15 · eager / graph 执行图对照</title><style>body{font:16px/1.7 system-ui;margin:0;color:#193247;background:#f4f6fa}main{max-width:1060px;margin:auto;padding:24px}table{border-collapse:collapse;width:100%;background:white}td,th{padding:9px;border-bottom:1px solid #dbe3ea;text-align:left}a{color:#1767a5}.scroll{overflow:auto}nav{display:flex;gap:24px;flex-wrap:wrap}img{width:1060px}h1{font-size:27px}</style><main><h1>Practice 15：eager / PIECEWISE 执行图</h1><p>''' + html.escape(intro) + '</p><nav><a href="' + links[0] + '">打开 eager 执行图</a><a href="' + links[1] + '">打开 graph 执行图与 75 次 replay</a></nav><div class="scroll"><img src="execution_paths.svg" alt="两种执行模式的调用路径与关联缺口"></div><div class="scroll"><table><tr><th>项目</th><th>eager</th><th>graph</th></tr>' + table + '</table></div><p>输出 token IDs 一致：' + str(result['same_output_token_ids']) + '；输出文本：' + html.escape(right['response']) + '</p><p>' + html.escape(limits) + '</p><a href="comparison.json">机器可读对照与各 kernel 数量变化</a></main></html>')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--eager', type=Path, required=True)
    p.add_argument('--graph', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = compare(a.eager, a.graph)
    report(a.output, result, a.eager, a.graph)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
