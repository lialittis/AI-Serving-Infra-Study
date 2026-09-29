"""Publish P3b measurements, execution graph and split reproducible evidence."""
import argparse,gzip,hashlib,json,shutil
from pathlib import Path


def read(p):return json.loads(p.read_text())
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('raw',type=Path);p.add_argument('archive',type=Path);a=p.parse_args()
    root=Path(__file__).resolve().parent;out=root/'results/published';out.mkdir(exist_ok=True);run=a.raw/'formal-r02';s=read(run/'analysis/summary.json')
    assert s['status']=='analysis_passed' and s['unsatisfied']==0
    for f in ['plan.json','environment.json','cases.json','measurements.json','correctness.json','full_vs_split.json','weight_check.json','completed.json','images.json']:
        shutil.copyfile(run/f,out/f)
    shutil.copyfile(run/'analysis/summary.json',out/'summary.json')
    with gzip.GzipFile(filename=str(out/'execution_graph.json.gz'),mode='wb',mtime=0) as f:f.write((run/'analysis/execution_graph.json').read_bytes())
    parts=[]
    with a.archive.open('rb') as f:
        while block:=f.read(48*1024*1024):
            target=out/f'evidence.tgz.part-{len(parts):02d}';target.write_bytes(block);parts.append(dict(file=target.name,bytes=len(block),sha256=digest(target)))
    save(out/'archives.json',dict(parts=parts,reassembled_sha256=digest(a.archive),bytes=a.archive.stat().st_size,
        remote='/data/tianchi/practice_25_multimodal_overlap/results',
        omitted='CANN binary buffers, FRAMEWORK ranges, profiler databases and formal-r01 interrupted diagnostic traces remain on remote. Final formal-r02 trace+CSV retained.'))
    save(out/'evidence_manifest.json',{str(p.relative_to(a.raw)):dict(bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(a.raw.rglob('*')) if p.is_file() and 'analysis' not in p.relative_to(a.raw).parts})
    save(out/'excluded_runs.json',{'qualification-r01':'Startup failed: physical NPU 5 was incorrectly put into ASCEND_RT_VISIBLE_DEVICES in a container already mapping it to logical 0; no model computation.',
        'qualification-r02':'Initial full/split prefill and two-stream prefill/decode checks passed; no native generation decode comparison yet.',
        'qualification-r03':'Extended native generation decode equivalence passed; excluded from performance.',
        'formal-r01':'96 prefill samples then interrupted at native decode reference: raw top-level forward lacked generation input/position slicing (1025 vs 513). All samples excluded; script fixed to use native generate for the reference.',
        'formal-r02':'Final complete 192-sample run; only this run contributes performance and dependency conclusions.'})
    save(out/'validation.json',{k:s[k] for k in ['status','performance_samples','numerical_pairs','diagnostic_trials','native_equivalence_pairs','all_exact','max_abs','boundary_requirements','unsatisfied','cross_stage_order_checks','device_tasks','compute_tasks','native_vision_sync_calls','complete_exact_kernel_memory_dag']})
    lines=['# P3b：视觉编码与另一请求语言计算的实测结果','',
        'Qwen2.5-VL-3B 的真实视觉／语言路径可以跨 stream 重叠，但收益取决于语言阶段和主机提交顺序：先提交 A prefill，再提交 B vision 时得到改善；先提交 vision 时没有同样的收益。A 为单步 decode 时，本轮基本持平。','',
        '查看[交互执行图](report/index.html)、[原始测量](results/published/measurements.json)、[完整汇总](results/published/summary.json)与[验证记录](results/published/validation.json)。','',
        '## 模型、输入和工作量','',
        '2026-09-29，Ascend 910B2C，容器逻辑 0 / 物理 5；torch / torch-npu 2.10.0，Transformers 5.5.4。完整预训练 BF16 checkpoint，32 层 ViT（含 patch merger）、36 层语言 decoder；eager attention、单 CPU 提交线程。权重与 buffer 前后 SHA256 相同，安装环境未修改。历史硬件 Alarm 仍在，实验前后 npu-smi 记录中无其他 NPU 作业。','',
        'A 的图片为另一张照片，其视觉编码提前完成，图文上下文为 512 tokens。A 阶段分别为完整语言 prefill 或基于该前缀的一步 decode。B 使用 [两张公开示例照片](../datasets/p3b_images/README.md)，像素预算为 64 / 256 个合并后视觉 token；保留宽高比，因此记录实际 grid 和 token 数。','',
        '| B 图片 | 预算 | 实际视觉 tokens | B 图文 tokens | grid (t,h,w) |',
        '|---|---:|---:|---:|---|']
    for c in s['cases']:
        if c['phase']=='prefill':lines.append('| %s | %d | %d | %d | %s |'%(c['B']['image'],c['B']['budget'],c['B']['visual_tokens'],c['B']['language_tokens'],c['B']['grid']))
    lines+=['','每次测量都执行 `A language + B vision + B merge/language`。B 语言消费包含在 pair 完成时间中，不能把“仅完成 B 视觉编码”当作 B 完成。A 视觉、decode 前缀、图像预处理／传输、KV clone 均在计时外。这个工作单元是两个请求的指定阶段，不是完整文本生成、HTTP TTFT 或 serving SLO。','',
        '两模式各预热 3 次，每格各测 12 次；LV/VL 交替，模式顺序每两轮交换，四种组合各 3 次。192 个正式 pair 样本；另有 32 个独立 profiler trial（每格两模式 × 两种顺序）。','',
        '## 按提交顺序比较性能','',
        '**LV**：先提交 A language，再提交 B vision。**VL**：先提交 B vision，再提交 A language。两者都在语言 stream 上排入后续 B merge/language，且先等待 B 的视觉 event。表中耗时为 ms/pair 中位数；百分比先逐轮配对，再取中位数，负值表示双流更快。','',
        '| 形状 | LV serial | LV 双流 | LV 配对变化 | VL serial | VL 双流 | VL 配对变化 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in s['performance']:
        o=r['by_order'];lines.append('| %s | %.3f | %.3f | %+.2f%% | %.3f | %.3f | %+.2f%% |'%(r['case'],o['LV']['serial']['wall_us']['median']/1000,o['LV']['parallel']['wall_us']['median']/1000,r['paired']['LV']['change_percent']['median'],o['VL']['serial']['wall_us']['median']/1000,o['VL']['parallel']['wall_us']['median']/1000,r['paired']['VL']['change_percent']['median']))
    lines+=['','完整 IQR、极值、每顺序更快的轮次、A/V/B 就绪时间、主机提交和显存峰值在汇总中。单进程、同日交替测量不代表跨日统计显著性；百分点很小的变化不能视为稳定加速。先语言提交的 prefill 收益并不意味着 A 本身一定更快，资源竞争也可能延长 A 的完成时间。','',
        '| 形状 | LV A 完成 serial / 双流 ms | LV B 完成 serial / 双流 ms |',
        '|---|---:|---:|']
    for r in s['performance']:
        o=r['by_order']['LV'];lines.append('| %s | %.3f / %.3f | %.3f / %.3f |'%(r['case'],o['serial']['ready_us']['A']['median']/1000,o['parallel']['ready_us']['A']['median']/1000,o['serial']['ready_us']['B']['median']/1000,o['parallel']['ready_us']['B']['median']/1000))
    lines+=['','公共 origin 到 terminal event 的设备就绪时间包含主机供给不足造成的空闲；pair wall time还包含主机等待返回。权重共享，A/B KV 独立；绝对峰值与 forward 增量分别记录，输入和准备好的前缀不在增量内。','',
        '## 实际 kernel 交叠与主机供给','',
        '| 形状 | 双流 LV 计算交叠 µs | 双流 VL 计算交叠 µs | vision 原生同步调用数 LV / VL |',
        '|---|---:|---:|---:|']
    for r in s['performance']:
        ts={t['order']:t for t in s['trials'] if t['case']==r['case'] and t['mode']=='parallel'}
        lines.append('| %s | %.3f | %.3f | %d / %d |'%(r['case'],ts['LV']['overlap_us'],ts['VL']['overlap_us'],ts['LV']['vision_native_syncs'],ts['VL']['vision_native_syncs']))
    lines+=['','交叠使用 A language 与 B vision 的真实 compute kernel 区间并集求交；不是模块包围范围相交。串行诊断全部为 0，B 语言消费始终在 B 视觉完成之后。Profiler 与正式计时分开，诊断改变主机提交时序的可能性仍存在。','',
        '安装的视觉 eager 路径会构造窗口元数据，并对设备上的累计序列长度执行 `.tolist()` 等主机读取。Trace 的 vision scope 中确有 `aclrtSynchronizeStream`／`WithTimeout` 调用；同一 CPU 线程在视觉路径内部等待时，尚不能提交后面的 A language。计数是 API 调用条数，可能含嵌套调用，不等价于互不重叠的等待次数，也没有将其时长直接相加。','',
        '因此，“两个模块放到两条 stream”还不足以保证重叠。LV 能先为设备提供语言计算，再推进视觉路径；VL 的视觉 host 调用在返回前已推进并等待了大量设备工作。这里的因果解释由已安装源码和同步 trace 支持，没有据此声称已重建全部隐式 CPU 同步边，或把单步 decode 的微小变化归因为稳定并发收益。','',
        '## 特征 handoff、KV 与执行 DAG','',
        '`输入就绪 → A language` 与 `输入就绪 → B vision → V_done event → masked_scatter → B language` 构成主体。A/B language 共用 L stream，FIFO 保证 A 完成后 B 才执行；串行模式还通过同一 stream 排序 A language 与 B vision。双流模式没有人为增加两者间的数据依赖。','',
        f"完整正式 trace 共 {s['device_tasks']:,} 个设备任务，其中 {s['compute_tasks']:,} 个计算任务，kernel CSV 全部精确关联。{s['boundary_requirements']} 条输入、视觉特征、embedding、语言顺序及输出完成要求全部有显式同步可达证明；另外检查 {s['cross_stage_order_checks']} 项 A/V 显式顺序。",'',
        '每个 trial 核对 `get_image_features` 输出、`masked_scatter` 输入的真实地址及布局相同，merge 输出与语言输入相同；实际 feature 消费 kernel 通过精确 host flow 对应 `aclnnInplaceMaskedScatter`。来源脚本与安装源码哈希固定。A/B KV、features、merge embeddings 的活跃字节范围无交叠；输入及跨流 features 引用一直保留到 terminal joins 之后。','',
        '所有普通设备任务核对 PyTorch flow、CANN connection 和 CSV 的 stream/task/start/duration。flow ID 重复时由真实队列 enqueue/dequeue 关系消歧，不按最近时间戳配对。原生 event wait 未产生独立 SQE 时保留带 API 证据的逻辑屏障，不虚构设备任务。','',
        '要求与证明边分开构造；删除特征等待、输入等待或 host join 后，相应证明必须失败。该图覆盖完整指定阶段的 kernel 执行与显式同步，仍不是 native workspace 的完整访存 DAG。隐式 host 阻塞有观测记录，尚未构成完整 CPU happens-before 图，不能把显式图中的“不可达”解释成所有底层机制均不存在顺序限制。','',
        '## 正确性与中断记录','',
        '最终 224 次正式／诊断 pair 校验全部逐元素一致：视觉 features，以及 A/B 每个输出的 logits + 36 层 K/V（各 73 个 tensor），最大绝对差为 0。8 个正式 case 另与完整原生图文 forward / generation 参考逐元素一致，贪心 token 一致。没有放宽容差。','',
        '原生模型含共享 `rope_deltas` 缓存。本 harness 按请求预计算 MRoPE，正式执行中显式传位置，模型共享缓存保持 None；权重及全部 buffer 前后哈希相同。原生参考在并发计时外执行，返回前清理共享位置状态。','',
        '资格 r01 在模型计算前因错误的设备环境变量退出；容器已将物理 5 映射为逻辑 0，去掉覆盖后通过。正式 r01 完成 96 个 prefill 样本后，在 native decode 参考处因缺少 generation 的输入／位置切片发生长度不匹配。修正为原生 generate 参考并通过资格 r03 后，正式 r02 重新测量全部 8 格；r01 样本不混入统计。相关脚本、日志、失败和校验记录见[排除说明](results/published/excluded_runs.json)。','',
        '## 证据与下一步','',
        '[归档说明](results/published/archives.json)列出证据分片与 SHA256；合并解压后可独立重放原始 trace/CSV。CANN 原始二进制、数据库及被排除 formal-r01 的诊断 trace 留在远端，最终正式 trace 完整保留。复现命令见 [README](README.md)。','',
        'P3b 已完成真实模型阶段 harness 验收。后续可单独验证静态视觉元数据预计算或固定形状 graph 是否减少 host 同步，再决定是否推进 native vLLM 多模态调度；这些优化尚未实施。原计划 P4 的 tile 级与多卡通信仍为能力调查。','']
    (root/'RESULTS.md').write_text('\n'.join(lines));print(json.dumps(read(out/'validation.json'),indent=2))


if __name__=='__main__':main()
