"""Publish replayable evidence, compact graph and measured result tables."""
import argparse,gzip,hashlib,json,shutil
from pathlib import Path


def read(p):return json.loads(p.read_text())
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('raw',type=Path);p.add_argument('archive',type=Path);a=p.parse_args()
    root=Path(__file__).resolve().parent;out=root/'results/published';out.mkdir(exist_ok=True)
    run=a.raw/'formal-r01';s=read(run/'analysis/summary.json');assert s['status']=='analysis_passed' and s['unsatisfied']==0
    for f in ('plan.json','environment.json','measurements.json','correctness.json','weight_check.json','captures.json','lifetime_checks.json','completed.json'):
        shutil.copyfile(run/f,out/f)
    shutil.copyfile(run/'analysis/summary.json',out/'summary.json');shutil.copyfile(a.archive,out/'evidence.tgz')
    with gzip.GzipFile(filename=str(out/'execution_graph.json.gz'),mode='wb',mtime=0) as f:f.write((run/'analysis/execution_graph.json').read_bytes())
    manifest={str(p.relative_to(a.raw)):dict(bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(a.raw.rglob('*')) if p.is_file() and 'analysis' not in p.relative_to(a.raw).parts}
    save(out/'evidence_manifest.json',manifest)
    save(out/'archives.json',dict(file='evidence.tgz',sha256=digest(out/'evidence.tgz'),bytes=(out/'evidence.tgz').stat().st_size,
        remote='/data/tianchi/practice_24_graph_replay/results',qualification='qualification-r01: 18 checks, excluded from formal performance statistics',
        omitted='CANN binary buffers, FRAMEWORK and profiler databases remain on remote; trace, CSV, source snapshots and graph dumps retained'))
    save(out/'validation.json',{k:s[k] for k in ('status','performance_samples','diagnostic_trials','numerical_pairs','lifetime_pairs','replay_invocations','internal_graph_tasks','boundary_requirements','unsatisfied','cross_task_order_checks','device_tasks','compute_tasks','complete_exact_kernel_memory_dag','exact_notify_id_pairing')})
    lines=['# Practice 24：全模型 fixed-step graph replay 对照','',
        '完整 FP32 配置的 eager / graph 六种对照均通过数值检查。Graph 双流在四种形状中都快于 graph 串行，但 batch=2 仍然最快。串行 graph 虽有两条内部计算 stream，也没有跨任务并行：调用 stream 上的 completion → launch 顺序将两张图串起来。','',
        '查看[交互执行图](report/index.html)、[原始样本](results/published/measurements.json)、[完整汇总](results/published/summary.json)及[验证记录](results/published/validation.json)。','',
        '## 环境、精度与工作量','',
        '2026-09-29，Ascend 910B2C，物理 NPU 5 / 逻辑 0，torch / torch-npu 2.10.0、Transformers 5.5.4，Qwen2.5-0.5B-Instruct 24 层完整 checkpoint。复用原 BF16 checkpoint 数值后 model.float()，模型参数／计算为 FP32，HF32 关闭；显式 attention mask 仍为既有 BF16 mask，在算子中按需要转换。不是另一个 FP32 原始 checkpoint，也不是 BF16 性能修复。','',
        '一个 CPU 提交线程，两个独立输入与初始 KV，共享只读权重。每任务一次完整 forward，输出最后位置 logits 和全部 24 层 KV。prefill 长度 128 / 1024，decode 初始上下文 128 / 1024。decode graph 固定当前位置与初始 KV 长度；每次回填输入和初始 KV 后重算同一步，不模拟不断增长 KV 的自回归生成。','',
        '三种策略分别为：两个 forward 共用调用 stream、分别进入两条调用 stream、合成 batch=2。每种形状 capture A / B / AB 三张图，12 张图在采集期间全部存活，各自独立 pool；串行和双流复用同一对 A/B 图。没有 superkernel 或自动 capture dispatch。不是 native vLLM runner、请求 scheduler、HTTP serving 或原生多模态路径。','',
        '## 无 profiler 计时','',
        '每格预热 3 次，正式 12 次；六种配置轮换位置、后半轮逆序，AB/BA 与原始／交换 payload 四组合各重复 3 次，共 288 个 pair 样本。计时从共同 origin event 提交前开始，到两个 terminal event 的主机等待返回结束；包括提交／同步控制。capture、输入／KV 准备和回填、全局输入就绪同步、校验与 profiler 均在计时外。','',
        '| 形状 | eager serial ms | eager 双流 ms | eager batch ms | graph serial ms | graph 双流 ms | graph batch ms |',
        '|---|---:|---:|---:|---:|---:|---:|']
    cases=list(dict.fromkeys(r['case'] for r in s['performance']))
    for case in cases:
        ps={r['backend']:r for r in s['performance'] if r['case']==case}
        values=[ps[b]['modes'][m]['wall_us']['median']/1000 for b in ('eager','graph') for m in ('serial','parallel','batch')]
        lines.append('| '+case+' | '+' | '.join('%.3f'%v for v in values)+' |')
    lines+=['','上表为中位数 ms/pair；下面先逐轮配对算比例，再取中位数。负值为耗时下降。','',
        '| 形状 | graph 双流相对 graph serial | graph batch 相对 graph serial | graph 双流相对 eager 双流 | graph 双流更快于 graph serial |',
        '|---|---:|---:|---:|---:|']
    for r in s['performance']:
        if r['backend']!='graph':continue
        c=next(x for x in s['replay_vs_eager'] if x['case']==r['case'] and x['mode']=='parallel')
        lines.append('| %s | %+.2f%% | %+.2f%% | %+.2f%% | %d/12 |'%(r['case'],r['paired']['parallel']['change_percent']['median'],r['paired']['batch']['change_percent']['median'],c['change_percent']['median'],r['paired']['parallel']['faster_pairs']))
    lines+=['','这是单次进程内交替测量的描述性结果，不是跨日统计显著性结论。完整 IQR、极值、A/B 及 first/last 完成时间、tasks/s、host submission 时间、allocated/reserved 峰值均保存在汇总；交互报告列出主要分布。tasks/s 仅指固定 forward 单元，不代表 serving goodput。','',
        '## 设备上是否真正交叠','',
        '| 形状 | eager 双流交叠 µs（两次） | graph 双流交叠 µs（两次） | graph 内部 stream（排序） | replay 调用 stream |',
        '|---|---|---|---|---|']
    for case in cases:
        ts={b:[t for t in s['trials'] if t['case']==case and t['backend']==b and t['mode']=='parallel'] for b in ('eager','graph')}
        lines.append('| %s | %s | %s | %s | %s |'%(case,*(' / '.join('%.3f'%t['overlap_us'] for t in ts[b]) for b in ('eager','graph')),' / '.join(ts['graph'][0]['streams']),' / '.join(ts['graph'][0]['caller_streams'])))
    lines+=['','交叠按 A/B 实际 compute kernel 区间各自求并集后求交，不用 forward 包围跨度。所有串行诊断交叠为 0。Profiler 独立采集 48 个 trial，可能改变 eager 主机供给，因此 trace 不能视为无 profiler 计时的逐微秒重放。','',
        'Graph 降低逐算子主机提交开销；本轮短 prefill/decode 的双流 graph 也出现了真实设备交叠。收益同时受到计算资源竞争、带宽和 batch 算子形状影响；没有恢复 tiling 或逐物理核利用率，不能仅凭重叠量分解各因素的贡献。','',
        '## 最终 execution DAG','',
        'eager：`origin → stream wait → 本流 kernel FIFO → terminal event → host join`。graph：`origin → 调用流 wait → MODEL_EXECUTE → 内部流 kernel FIFO → NOTIFY_RECORD → 调用流 NOTIFY_WAIT → terminal event → host join`。','',
        '串行 A/B 的调用流相同，先提交图的 NOTIFY_WAIT 在后提交图的 MODEL_EXECUTE 前；AB/BA 次序交替，内部流虽不同仍具先提交任务 → 后提交任务 happens-before。双流分别 launch / wait，各图只有自己的内部 FIFO，两任务间没有数据依赖与强制串行边。batch 图是一条处理两行的计算链，不能拆成两条独立 A/B kernel 链。','',
        f"完整 trace 核对 {s['device_tasks']:,} 个设备任务、{s['compute_tasks']:,} 个计算任务（含准备／校验）；40 次诊断 replay 对应 {s['internal_graph_tasks']:,} 个 graph 内部任务。{s['boundary_requirements']} 条输入就绪／输出完成要求和 {s['cross_task_order_checks']} 项跨任务顺序检查全部通过；要求不作为证明边加入 HB 图。",'',
        '普通 kernel 通过精确 PyTorch flow、CANN flow/connection 与 CSV 的 stream/task/start/duration 核对。Graph boundary 没有普通逐 kernel host flow，用同 native thread 的完整 replay scope 包含关系及唯一 CANN connection 关联 MODEL_EXECUTE / NOTIFY_WAIT。内部任务使用存活期间唯一 Model ID、debug dump 完整 stream/task 序列、逐图 replay 顺序与边界包围共同核对。脚本源码哈希锁定“capture 对象 → 同对象 dump → 同对象 replay”的绑定；没有按最近时间戳配对。','',
        'debug dump 的 ts/dur 是示意排布，仅使用 Model ID、stream、task、编译符号和 task type；profiler 的 aclnn 名称可能不同。图内 kernel 没有本次新的 host launch，残留／悬空的 capture flow 不作为逐 kernel 调用证据。','',
        '这里是完整 forward 的执行 DAG，仍未恢复每个 native kernel 的精确内存访问、workspace 数据 DAG、原生 RI 句柄或精确 NOTIFY ID 配对。graph launch / completion 边表达捕获对象的 replay 语义及本次边界证据，不冒充底层 NOTIFY 标识。未进行自动 stream 分配或理论关键路径加速预测。','',
        '## capture、内存和正确性','',
        '| 形状 | 图 | capture ms | 首次 replay ms | capture allocated 增量 MiB |',
        '|---|---|---:|---:|---:|']
    for c in s['captures']:
        lines.append('| %s | %s | %.3f | %.3f | %.2f |'%(c['case'],c['name'],c['capture_ms'],c['first_replay_ms'],(c['capture_allocated_after']-c['capture_allocated_before'])/2**20))
    lines+=['','capture 墙钟包含 torch-npu graph context 的同步／清理及捕获；各图只有一次观测。首次 replay 从调用到 capture stream 完成，独立于正式 pair 计时。memory_allocated 增量包含图相关活跃分配，不等价于全部 private pool reserved；graph 的静态缓冲已在 capture 时保留，所以 replay 增量 0 不表示无内存成本。所有 graph 共存，跨形状绝对峰值含此前图，不能直接作隔离部署显存比较。','',
        '关键生命周期处理：DynamicCache 在 capture 中把 layer.keys/values 改成输出引用，但 replay 仍读取最初输入地址。因此单独保留原始 input KV tensors；每次 replay 前在完成上次 terminal join 后回填原始地址，输出保留到读取结束。input IDs 克隆为图私有缓冲。12 张图的可变输入／KV／输出字节范围无跨图交叠；mask/position 和权重只读共享。','',
        f"336 次正式／诊断 pair 数值对照、24 次输入／初始 KV 交换复用对照全部通过；每次检查两个任务各 49 个 tensor，贪心 token 一致。serial/parallel 与同形状 eager 基线逐元素相同；batch 保持原 atol=0.0625、rtol=0.02，所有检查最大绝对差 {s['max_abs']:.9g}。权重与 buffer 前后 SHA256 相同，图输出地址及固定 KV 长度持续有效。18 次先行资格对照不计入正式性能。",'',
        '原 Practice 23 的 BF16 long-prefill batch 失败保持原状；本轮结论限定完整 FP32 配置。硬件历史 Alarm 仍存在，前后 npu-smi 已归档，未修改系统或设备。','',
        '## 重放与后续','',
        '[证据包](results/published/evidence.tgz)、[manifest](results/published/evidence_manifest.json)保留 trace、CSV、源码、图 dump、资格检查及全部正式结果。命令见 [README](README.md)。','',
        '下一计划项为 P3b：先确认适配的视觉模型、已有权重和图像输入，再比较请求 A 的语言计算与请求 B 的视觉编码。固定步 graph 已通过；若进一步研究持续 decode，需单独设计静态 KV 容量、位置更新、图 bucket 与复用规则，不能直接把本图当作动态生成循环。','']
    (root/'RESULTS.md').write_text('\n'.join(lines));print(json.dumps(read(out/'validation.json'),indent=2))


if __name__=='__main__':main()
