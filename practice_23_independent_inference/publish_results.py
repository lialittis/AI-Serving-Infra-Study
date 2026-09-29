"""Publish compact P3a artifacts and a reproducible evidence manifest."""
import argparse,gzip,hashlib,json,shutil
from pathlib import Path


def read(p):return json.loads(p.read_text())
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('raw',type=Path);p.add_argument('archive',type=Path);a=p.parse_args()
    root=Path(__file__).resolve().parent;out=root/'results/published';out.mkdir(exist_ok=True)
    run=a.raw/'formal-r02';s=read(run/'analysis/summary.json')
    assert s['status']=='analysis_passed' and s['unsatisfied']==0
    for f in ('plan.json','environment.json','measurements.json','correctness.json','weight_check.json','layouts.json','inputs.json','completed.json','config.json'):
        shutil.copyfile(run/f,out/f)
    shutil.copyfile(run/'analysis/summary.json',out/'summary.json');shutil.copyfile(a.archive,out/'evidence.tgz')
    with gzip.GzipFile(filename=str(out/'execution_graph.json.gz'),mode='wb',mtime=0) as f:f.write((run/'analysis/execution_graph.json').read_bytes())
    manifest={str(p.relative_to(a.raw)):dict(bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(a.raw.rglob('*')) if p.is_file() and 'analysis' not in p.relative_to(a.raw).parts}
    save(out/'evidence_manifest.json',manifest)
    save(out/'archives.json',dict(file='evidence.tgz',sha256=digest(out/'evidence.tgz'),bytes=(out/'evidence.tgz').stat().st_size,
        remote='/data/tianchi/practice_23_independent_inference/results',
        omitted='CANN binary raw buffers, FRAMEWORK ranges and profiler databases remain on remote host; trace+CSV sufficient for this analysis'))
    save(out/'excluded_runs.json',{'qualification-r01':'128-token prefill/decode basic full-model qualification; not performance samples',
        'formal-r01':'Stopped at prefill-1024 batch numerical threshold failure; 128-token samples excluded from formal statistics',
        'qualification-long-r01':'Confirmed serial repeat and parallel exact; only long-prefill batching fails original tolerance',
        'formal-r02':'Final capture; retains 14 failed prefill-1024 batch checks (12 performance + 2 diagnostic); invalid cell excluded from benefit claims'})
    save(out/'validation.json',dict(status='completed_with_one_invalid_comparison',analysis_status=s['status'],
        performance_samples=s['performance_samples'],diagnostic_trials=s['diagnostic_trials'],
        failed_numeric_comparisons=s['failed_numeric_comparisons'],invalid_cells=s['invalid_cells'],
        boundary_requirements=s['boundary_requirements'],unsatisfied=0,cross_task_order_checks=s['cross_task_order_checks'],
        device_tasks=s['device_tasks'],compute_tasks=s['compute_tasks'],weights_and_buffers_unchanged=True,
        complete_exact_kernel_memory_dag=False))
    lines=['# P3a：独立完整模型 forward 的双流并发与 batching','',
        '本轮已完成 Qwen2.5-0.5B-Instruct 完整预训练模型的单卡 eager harness。比较同样两条序列的串行、双 stream 和 batch=2：双流在 prefill 上缩短总完成时间，decode 基本持平；数值通过的三种形状中，batch=2 更快。1024-token prefill 的 batch 未通过数值校验，不能采用其收益结论。','',
        '查看[交互执行图](report/index.html)、[原始性能样本](results/published/measurements.json)、[数值检查](results/published/correctness.json)与[验证记录](results/published/validation.json)。','',
        '## 环境和工作量','',
        '2026-09-29，Ascend 910B2C 单卡，物理 NPU 5 / 逻辑 0；torch 2.10.0、torch-npu 2.10.0、Transformers 5.5.4。复用已有权重，不修改系统安装。24 层、BF16、eager attention、默认 RoPE、无 sliding window。一个 CPU 线程提交；这不是 vLLM 的 native runner、continuous batching、HTTP serving 或 graph replay。','',
        '每任务一次完整 forward，返回最后一个位置 logits 与更新后的全部 KV；prefill 长度 128 / 1024，decode 上下文长度 128 / 1024。decode 的前缀预先用相同基线生成；cache clone、batch 拼接、输入准备、同步就绪均在计时外。因此不包含 cache 合批成本、排队、权重加载或完整文本生成。','',
        '两任务共享一份只读权重，权重及模型 buffer 的 SHA256 前后相同。A/B 持有独立可变 KV，所有活跃缓存字节范围无跨任务重叠；输出和输入一直保留到两个末尾 event 完成。mask 和位置张量可只读共享。原生算子 workspace 由框架管理，未恢复其精确访存。','',
        '## 无 profiler 性能','',
        '每形状／模式预热 3 次，正式 12 轮；六种模式顺序轮换，AB/BA 交替，共 144 个 pair 样本。以共同输入已就绪为起点，以两个任务全部完成为终点，包含主机提交和 event 控制。下表为 ms/pair 中位数；配对变化先逐轮除以 serial 再取中位数。','',
        '| 形状 | serial | 双流 | batch=2 | 双流配对变化 | 双流更快的轮次 |',
        '|---|---:|---:|---:|---:|---:|']
    for r in s['performance']:
        m=r['modes'];b='%.3f'%(m['batch']['wall_us']['median']/1000)
        if not m['batch']['numerically_valid']:b+='（数值未通过）'
        lines.append('| %s | %.3f | %.3f | %s | %+.2f%% | %d/12 |'%(r['case'],m['serial']['wall_us']['median']/1000,m['parallel']['wall_us']['median']/1000,b,r['paired']['parallel']['change_percent']['median'],r['paired']['parallel']['faster_pairs']))
    lines+=['','这是本轮交替测量的描述性结果，完整 IQR、极值、每任务 NPU 就绪时间、主机观察时间和 allocated/reserved 峰值见[汇总](results/published/summary.json)和原始样本。没有做跨进程／跨日统计显著性检验。','',
        '## 吞吐与首任务延迟的取舍','',
        '| 形状 | 模式 | 较早完成 ms | 两任务完成 ms | tasks/s | forward 增量峰值 MiB |',
        '|---|---|---:|---:|---:|---:|']
    for r in s['performance']:
        for mode,m in r['modes'].items():
            throughput='%.1f'%m['tasks_per_second']['median'] if m['numerically_valid'] else '不采纳'
            lines.append('| %s | %s | %.3f | %.3f | %s | %.2f |'%(r['case'],mode,m['first_ready_us']['median']/1000,m['last_ready_us']['median']/1000,throughput,m['incremental_peak']['median']/2**20))
    lines+=['','设备就绪时间用公共 origin event 到各 terminal event 的 elapsed_time，包含提交供给造成的空闲；wall time 还含 host 等待返回。first/last 与 A/B 身份区分：AB/BA 交替后按每次较早／较晚任务统计，A/B 单独分布另存 JSON。batch 两任务同时就绪。','',
        'prefill 双流缩短整体完成时间，但较早任务比串行更晚完成，不能把吞吐收益表述为两个任务都降低延迟。三个数值通过的形状里，batch 优于双流；batch 使用更大的算子临时内存。显存增量从准备完成后计算，decode 初始 KV 不在增量内；所有模式共享同样的已保留基线和权重。','',
        '## 真实 kernel 执行图','',
        f"24 个诊断 trial；完整采集含 {s['device_tasks']:,} 个设备任务、{s['compute_tasks']:,} 个计算任务（包括输入准备／校验等 scope 外任务）。所有 CSV 计算条目已核对精确身份。下表重叠来自各 forward 内实际 kernel 区间的并集求交，不是两个 forward 起止范围的交集。",'',
        '| 形状 | 双流计算重叠 µs（两次） | 实际计算 stream |',
        '|---|---|---|']
    for case in dict.fromkeys(t['case'] for t in s['trials']):
        ts=[t for t in s['trials'] if t['case']==case and t['mode']=='parallel']
        lines.append('| %s | %s | %s |'%(case,' / '.join('%.3f'%t['overlap_us'] for t in ts),' / '.join(ts[0]['streams'])))
    lines+=['',
        '1024-token prefill 两次诊断均观察到约 12.8 ms 的跨任务计算重叠；两条计算 stream 为本轮物理 43 / 44，默认 origin stream 为 46。短 prefill 和两个 decode 形状的诊断均为 0。短 prefill 的无 profiler 耗时改善尚不能直接归因于计算重叠。', '',
        '任务边界关系：`输入／独立初始 KV 就绪 → origin event → A/B 各自 wait → 完整 forward → 各自 terminal event → host join → 校验／释放`。串行额外由同一 stream 的 FIFO 将先提交的完整 forward 排在后提交者之前；双流两条任务链仅共享起点与主机汇合，没有 A→B 数据依赖。batch 只有一条共同执行链，每个 kernel 处理两个 batch 行。','',
        f"独立校验了 {s['boundary_requirements']} 条输入就绪／输出完成要求和 {s['cross_task_order_checks']} 项跨任务顺序检查；未将这些要求添加到同步图中自证。实际设备 event-wait 边 {s['event_wait_edges']} 条，只有 API 证据的跨流等待 {s['api_only_waits']} 次。同流等待由 FIFO 覆盖。",'',
        '该图是完整模型运行的 **kernel execution DAG**，包含实际任务、FIFO、event 和 host completion。它仍不是全模型每个 native kernel 的完整精确内存数据依赖 DAG；没有据此给出自动 stream 分配或理论最大加速比。跨 trial 的所有主机阶段也未建成完整因果图；HB 结论限定于已声明的每个 trial 内边界。','',
        'Profiler 与性能采集分开，诊断只增加外层 forward/event scope，仍可能改变主机提交时序。设备诊断有无重叠都不能直接当作无 profiler 的逐微秒重放；耗时和 trace 是两类互补证据。','',
        '## 数值失败项','',
        '全部 serial/parallel logits 与 24 层 KV 逐元素相同。三个通过形状的 batch 也逐元素相同。1024-token prefill 的 batch 两任务贪心 token 与 serial 相同，但 logits 最大绝对差分别为 0.3125 / 0.25，KV 最大绝对差分别为 2.453125 / 1.736328125；预设 `atol=0.0625, rtol=0.02` 不通过。','',
        f"最终保留 {s['failed_numeric_comparisons']} 个失败对照（12 次性能、2 次诊断），没有调整容差。独立资格复核证明该差异随 long-prefill batching 出现，串行重复与双流完全一致。尚未定位到具体原生算子／tiling 舍入机制，不能直接宣称是正常 BF16 舍入，也不能当作 stream 竞态。进一步量化 batching 收益前需定位首次分歧或建立可靠高精度参考。",'',
        '## 重放与下一步','',
        '[证据包](results/published/evidence.tgz)保留 trace、kernel CSV、源码、参数、全部样本及中断／资格检查；[文件哈希](results/published/evidence_manifest.json)和[归档说明](results/published/archives.json)支持独立重放。初始化硬件历史 Alarm 仍存在，实验前后记录中未见其他 NPU 作业；未修改或重启设备。复现命令见 [README](README.md)。','',
        '建议优先定位 long-prefill batching 的首个数值分歧，使长输入三方比较也具备有效性；随后单独测 graph replay 的提交开销影响。P3b 视觉／语言并发仍需已有适配模型及图像输入，尚未开展。','']
    (root/'RESULTS.md').write_text('\n'.join(lines));print(json.dumps(read(out/'validation.json'),indent=2))


if __name__=='__main__':main()
