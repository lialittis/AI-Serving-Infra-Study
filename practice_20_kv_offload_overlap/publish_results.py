"""Validate the completed evidence and regenerate the compact result report."""
import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import shutil


def read(path):
    return json.loads(path.read_text())


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--diagnostics',type=Path,required=True)
    p.add_argument('--performance',type=Path,required=True)
    p.add_argument('--archive',type=Path,required=True)
    a=p.parse_args();root=Path(__file__).resolve().parent;out=root/'results/published'
    out.mkdir(parents=True,exist_ok=True)
    performance=read(a.performance)
    assert performance['status']=='passed' and performance['formal_samples']==60
    summaries=[];golden={};response_keys=None
    runs=sorted(a.diagnostics.glob('block-*'))
    assert len(runs)==3, 'expected all three diagnostic runs'
    for run in runs:
        s=read(run/'analysis/summary.json');summaries.append(s)
        assert s['acyclic'] and s['unsatisfied_requirements']==0
        assert set(s['unassociated_task_names'])<= {'PLACE_HOLDER_SQE'}
        if s['mode']=='serialized':
            assert all(Decimal(x['overlap_us'])==0 for x in s['overlap'])
        keys=set()
        cases=sorted(run.glob('measure-*'))
        assert len(cases)==2, 'expected both diagnostic prefix lengths'
        for case in cases:
            for response in case.glob('*.json'):
                if response.stem in ('inputs','summary'):
                    continue
                key=case.name,response.stem
                keys.add(key)
                tokens=read(response)['token_ids']
                if key in golden:
                    assert tokens==golden[key], 'diagnostic output mismatch'
                else:
                    golden[key]=tokens
        assert len(keys)==16, 'expected eight request responses per prefix length'
        if response_keys is not None:
            assert keys==response_keys, 'missing diagnostic response'
        response_keys=keys
        (out/('diagnostic-'+s['mode']+'.json')).write_text(json.dumps(s,indent=2)+'\n')
    assert {s['mode'] for s in summaries}=={'native','serialized','recompute'}
    for name in ('numerics-before.json','numerics-after.json'):
        assert read(out/name)['status']=='passed'
    shutil.copyfile(a.performance,out/'performance.json')
    locations=dict(diagnostic_archive=dict(local=str(a.archive.resolve()),
        remote='/data/tianchi/practice_20_kv_offload_overlap/results/p1-diagnostic-evidence.tgz',
        bytes=a.archive.stat().st_size,sha256=digest(a.archive)),
        local_full_graphs=[str((r/'analysis/execution_graph.json').resolve()) for r in sorted(a.diagnostics.glob('block-*'))],
        remote_full_graphs=['/data/tianchi/practice_20_kv_offload_overlap/results/diagnostic-r03/'+r.name+'/analysis/execution_graph.json' for r in runs],
        full_graph_sha256={r.name:digest(r/'analysis/execution_graph.json') for r in runs},
        benchmark_archive=dict(path='benchmark-evidence.tgz',sha256=digest(out/'benchmark-evidence.tgz')))
    (out/'evidence_locations.json').write_text(json.dumps(locations,indent=2)+'\n')
    validation=dict(status='passed',formal_samples=60,warmup_samples=60,
        exact_cross_mode_tokens=True,diagnostic_tokens_equal=True,matching_transfer_bytes=True,
        diagnostic_runs=len(summaries),device_tasks=sum(s['device_tasks'] for s in summaries),
        compute_tasks=sum(s['compute_tasks'] for s in summaries),
        kv_requirements=sum(s['required_edges'] for s in summaries),
        unsatisfied_requirements=0,graphs_acyclic=3,serialized_dma_compute_overlap_us=0,
        numerical_checks_before_after='passed',unknown_runtime_placeholders=sum(s['unassociated_device_tasks'] for s in summaries))
    (out/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    lines=['# P1 实测结果：KV 回载确有重叠，但本轮重算更快','',
           '在当前单卡 Qwen2.5-0.5B eager 负载中，真实 CPU prefix 命中、D2H 保存、H2D 回载和 block 复用均已验证。原生传输与独立请求 B 的计算发生重叠；强制串行组没有计算／DMA 重叠。回载没有带来端到端优势。','',
           '打开[离线交互图](report/index.html)，或查看[机器可读验证结果](results/published/validation.json)。','',
           '## 无 profiler 性能','',
           '下表为中位数，单位 ms；括号内为首 token 延迟的 Q1–Q3。每模式每长度 10 个正式周期，另有等量预热。','',
           '| 前缀 token | 模式 | A′ 首 token（IQR） | A′ 完成 | B 完成 |',
           '|---|---|---:|---:|---:|']
    for c in performance['cases']:
        for mode in ('native','serialized','recompute'):
            s=c['modes'][mode];t=s['reload_ttft_ms']
            lines.append('| %d | %s | %.2f（%.2f–%.2f） | %.2f | %.2f |'%(
                c['prefix_tokens'],mode,t['median'],t['q1'],t['q3'],s['reload_latency_ms']['median'],s['b_latency_ms']['median']))
    lines+=['','原生组相对串行组的首 token 延迟，两种长度均变慢；A′ 总完成时间和 B 完成时间相对串行组的差异未形成一致结论。相对重算组，原生组在上述指标均更慢。这里的“一致”仅指前后两对服务方向相同且合并 IQR 不重叠，不是统计显著性检验。','',
            '服务顺序为 native → serialized → recompute → recompute → serialized → native。计时在远端 loopback 客户端进行，等待实际流式 token 到达与完整响应。两请求完成窗口由较长的 B 主导，不将其解释为持续到达场景的服务容量。','',
            '性能测量关闭 profiler 和重型 Python 观察器，保留少量命中／传输记账；延迟包含这部分开销。串行组包含主机屏障，不能据此单独量化“换一个 stream ID”的成本。','',
            '## 真实传输与依赖','',
            '| 模式 | 设备任务 | 计算任务 | KV 约束 | 未满足 | 物理 stream |',
            '|---|---:|---:|---:|---:|---|']
    for s in summaries:
        lines.append('| %s | %d | %d | %d | %d | %s |'%(s['mode'],s['device_tasks'],s['compute_tasks'],
            s['required_edges'],s['unsatisfied_requirements'],', '.join(sorted(s['streams']))))
    native=next(s for s in summaries if s['mode']=='native')
    reloads=[x for x in native['overlap'] if x['direction']=='H2D']
    lines+=['', '原生组两次 H2D 分别搬运 12 / 36 MiB，与 B 计算区间实际交叠 **%s / %s µs**。串行组仍使用独立的传输 stream，但全部传输的计算重叠均为 0。stream 数字仅在各自运行内关联。'%(reloads[0]['b_compute_overlap_us'],reloads[1]['b_compute_overlap_us']), '',
            '每周期 D2H 总量分别为 205.5 / 229.5 MiB，包含 A、压力请求和 B 的已完成 block；两个 offload 模式严格一致。A′ 的 NPU prefix hit 为 0，CPU hit 分别为 1024 / 3072 token。重算组没有 KV DMA。','',
            '两种 offload 图各验证 786 条约束：386 条同代数据依赖、382 条存储换代约束、16 条 CPU 缓存发布和 2 条回载释放约束。第二个周期出现 CPU pool 的真实驱逐／复用。全部要求在独立构建的同步图中可达。','',
            '同步链包括：','',
            '- KV producer → 计算流中采样结果的 event → 原生主机等待 → 后续调度与后台提交 → D2H。`confirmed_tokens` 本身不是设备屏障。',
            '- D2H → 完成 event 查询 → worker 完成消息 → scheduler 发布 CPU prefix，并释放额外的 NPU block 引用。',
            '- H2D → 完成查询 → request 恢复调度 → attention 消费；CPU/NPU 的额外引用在完成确认后释放。',
            '- 同一物理 block 换代前，旧传输访问必须先结束。图分别保存逻辑分配与设备覆盖，不把主机分配时刻当作设备写入。','',
            '## 正确性与观测边界','',
            '60 个正式周期及 60 个预热周期在三模式下逐 token 一致；六个独立诊断周期的输出也一致。独立真实 DMA 检查覆盖 K/V 分离、非零 storage offset、块映射和未覆盖区域，实验前后均逐字节通过。','',
            '后台线程不继承 PyTorch CPU scope，原始异步关联会把其 event 错归到主线程。两份 offload 诊断共纠正 36 条这类归属，使用 CANN connection、enqueue/dequeue correlation 与源码约束下的 FIFO 批次顺序。记录数、句柄和每批 memcpy 数均须一致；没有按最近时间猜测，也没有用两套时钟的数值相近补边。','',
            '三份诊断保留了 9 条无 host 关联的 `PLACE_HOLDER_SQE` runtime 任务，仅连接物理 stream FIFO，不虚构数据含义。KV 访问以完整批次和 24 层 attention 调用边界保守建模，原生 workspace 仍未知，`complete_exact_model_data_dag=false`。','',
            '本轮仅覆盖正常 prefix 缓存驱逐，不覆盖活动请求抢占／取消。源码检查发现 Ascend runner 未调用新版 `handle_preemptions`；本实验遇到活动抢占会停止，不把当前结果推广到那条路径。没有升级或覆盖系统安装。','',
            '环境为 Ascend 910B2C、CANN 9.0.0、torch/torch-npu 2.10.0、vLLM 0.21.0、vLLM-Ascend 0.21.0rc1。既有硬件 Alarm 仍在；基础数值和往返拷贝前后通过，但不能据此认证硬件健康。最后已无实验 NPU 进程。','',
            '## 归档与复现','',
            '- [完整性能样本压缩包](results/published/benchmark-evidence.tgz)：包含原始响应、预热、命中／搬运记录、服务命令、源码与仪器快照。',
            '- [性能汇总](results/published/performance.json)、[环境](results/published/environment.json)、[数值检查](results/published/numerics-after.json)。',
            '- [诊断文件哈希清单](results/published/p1-diagnostic-evidence.manifest.json)与[归档位置](results/published/evidence_locations.json)。完整诊断包保留在远端 `/data/tianchi/practice_20_kv_offload_overlap/results/p1-diagnostic-evidence.tgz`，本地为 `/tmp/p1-diagnostic-evidence.tgz`。',
            '- 完整图在本地／远端对应诊断的 `analysis/execution_graph.json`；本地原始诊断位于本 practice 的 `results/diagnostic-r03/`。这些大文件不加入 Git。','',
            '诊断包 SHA256：`%s`。'%locations['diagnostic_archive']['sha256'],'',
            '首轮短前缀资格检查、串行长前缀资格检查及早期诊断仍保留在远端 results；一次端口 TIME_WAIT 导致的诊断启动失败未采集设备数据，修复后使用新目录重试。它们均未混入正式性能样本。','',
            '当前结论支持继续研究更大模型或更长前缀下的收益交叉点；无需假设“有重叠就一定加速”。本次没有完成对回载／调度成本的逐项归因。','']
    (root/'RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps(validation,indent=2))


if __name__=='__main__':
    main()
