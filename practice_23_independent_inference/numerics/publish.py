"""Publish the numerical diagnosis without changing the original BF16 evidence."""
import argparse,json,hashlib,shutil
from pathlib import Path
from analyze import analyze,read,save,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('raw',type=Path);p.add_argument('archive',type=Path);a=p.parse_args()
    summary,contracts=analyze(a.raw)
    root=Path(__file__).resolve().parents[1];out=root/'results/published/numerics';out.mkdir(parents=True,exist_ok=True)
    save(Path(__file__).with_name('contracts.json'),dict(sources=contracts))
    copies={
        'summary.json':'analysis/summary.json','kernel_evidence.json':'analysis/kernel_evidence.json','rational_replay.json':'analysis/rational_replay.json',
        'layers.json':'numerics-locate-r01/layers.json','modules.json':'numerics-locate-r01/modules.json',
        'rounding_bounds.json':'numerics-rounding-r02/rounding_bounds.json','rounding_plan.json':'numerics-rounding-r02/plan.json',
        'operator_rounding.json':'numerics-operator-r01/rounding.json',
        'fp32-plan.json':'numerics-fp32-formal-r01/plan.json','fp32-checks.json':'numerics-fp32-formal-r01/checks.json',
        'fp32-measurements.json':'numerics-fp32-formal-r01/measurements.json','fp32-environment.json':'numerics-fp32-formal-r01/environment.json',
        'fp32-weight_check.json':'numerics-fp32-formal-r01/weight_check.json'}
    for dst,src in copies.items():shutil.copyfile(a.raw/src,out/dst)
    shutil.copyfile(a.archive,out/'evidence.tgz')
    manifest={f.relative_to(a.raw).as_posix():dict(sha256=sha(f),bytes=f.stat().st_size) for f in sorted(a.raw.rglob('*')) if f.is_file() and 'analysis' not in f.relative_to(a.raw).parts}
    save(out/'evidence_manifest.json',manifest)
    save(out/'archives.json',dict(sha256=sha(out/'evidence.tgz'),bytes=(out/'evidence.tgz').stat().st_size,
        remote='/data/tianchi/practice_23_independent_inference/results/numerics-*',
        includes='All investigation runs, source snapshots, exact BF16 operands, final raw trace and kernel CSV',
        omitted='Torch .pt operands duplicate the exported .bf16 data. Large full FP64 reference tensors and binary profiler intermediates remain on remote host.'))
    save(out/'excluded_runs.json',dict(numerics_rounding_r01='Preliminary profiler capture lacked explicit schedule; final kernel proof uses numerics-rounding-r02 with explicit closure and bitwise repeat check',
        numerics_control_r01='MLP-only FP32 linears: long-prefill comparison still fails; qualification only',
        numerics_control_all_r01='All FP32 linears with BF16 activations: long-prefill comparison still fails; qualification only',
        numerics_control_model_r01='Full FP32 qualification, not performance samples',
        formal='Only numerics-fp32-formal-r01 contributes the new 144 performance samples; no cross-precision paired speedup claim'))
    s=summary
    text=['# P3a 数值定位：batch 形状改变了 BF16 投影的舍入结果','',
        '**定位已完成；原生 BF16 的失败对照继续保留。** 首次分歧出现在第 0 层 MLP 的 `gate_proj`：进入该层投影的输入完全一致，batch=1 和 batch=2 的独立矩阵乘仍能复现不同结果。差异与累加／舍入误差一致，未发现这组分歧需要双 stream 才能触发。','',
        '另完成全模型 FP32 配置下的 144 个三方性能样本，四形状数值均通过。**这是独立精度控制组，不能替代或“修复”原始 BF16 结果。** 查看[数值汇总](results/published/numerics/summary.json)、[原始 BF16 结果](RESULTS.md)和[重放证据](results/published/numerics/evidence.tgz)。','',
        '## 首次分歧与传播','',
        '复用原始 1024-token A/B 输入、checkpoint、eager attention 和独立 KV。先逐层观察，再细化第 0 层子模块，最后完全脱离模型单独运行投影。钩子采集后的完整 logits 和全部 KV 与无钩子结果逐元素相同；诊断不计入性能。','',
        '| 位置 | 观察结果 |','|---|---|',
        '| embedding、第 0 层 input norm、Q/K/V、attention 输出、post-attention norm | A/B 两行分别完全一致 |',
        '| 第 0 层 `mlp.gate_proj` | 首次分歧；共 9,961,472 个元素中 667 个不同，最大差异 0.0078125 |',
        '| 同层 `mlp.up_proj` | 802 个元素不同，最大差异 0.00390625 |',
        '| 第 0 层完整输出 | A/B 最大差异均为 0.00390625 |',
        '| 第 23 层完整输出 | A/B 最大差异为 2.25 / 1.75 |',
        '| 最后位置 logits | A/B 最大差异为 0.3125 / 0.25，复现原始失败 |','',
        '![逐层差异](report/numerics-layer-errors.svg)','',
        '图表示实际观测到的层输出差异增长，包含上游误差传播及后续新舍入差异；没有证明全部最终误差都只由第一个 gate projection 引起。只替换 MLP 或全部线性层的精度仍失败，也说明局部改动不足以消除完整路径的分歧。','',
        '## 独立算子与高精度参考','',
        '保存进入投影的精确 BF16 输入和权重，在同一默认 stream 上分别执行两个 batch=1 投影和一个 batch=2 投影；batch 重复执行逐元素相同。CPU FP64 参考使用这些 BF16 数值的精确提升，不使用 NPU 输出作为参考。','',
        '| 投影 | 独立 batch=1 相对 FP64 的 L2 误差 | batch=2 相对 FP64 的 L2 误差 |',
        '|---|---:|---:|',
        '| gate_proj | 0.001655710017 | 0.001655710016 |',
        '| up_proj | 0.001659410485 | 0.001659410487 |','',
        '两种形状的总体误差量级几乎一致，不能认定 batch=1 就是高精度真值。个别接近零的消去结果，按 BF16 ULP 计数可能很大，因此没有用“所有差异都只有一个 ULP”概括整个投影。','',
        '对每个结果，计算 FP64 点积到该 BF16 输出舍入区间的距离，再检查它是否小于 `γ_K · Σ|xᵢwᵢ|`，其中 `K=896`、`u=2⁻²⁴`、`γ_K=Ku/(1−Ku)`。两种形状、两个投影全部通过，最大只用了该误差界约 0.08%。这是与 FP32 累加及 BF16 舍入相容的数值证据，不是对未公开 native 归约实现的证明。','',
        '28 个覆盖分歧、最大误差与接近零位置的点积，进一步用 Python `Fraction` 精确有理数运算复核，全部与 FP64 参考相同。仓库的 BF16 二进制操作数可用 Python 标准库独立重放这些点积，不依赖 Ascend。完整 FP64 tensor 保留在远端，原始代码可重新计算所有位置。','',
        '## 原生 kernel 证据','',
        '12 个隔离投影 kernel 均通过 PyTorch flow、CANN connection、CSV 的 stream/task/start/duration 关联。算子族均为 `aclnnMatmul_MatMulCommon_MatMulV2`；batch=1 的输入矩阵为 `[1024,896]`，batch=2 为 `[2048,896]`，权重均为 `[4864,896]`。BF16 与 FP32 输入 dtype 也由 CSV 核对。','',
        '证据支持“输入数值相同、M 维形状变化后原生 matmul 输出不同”。本轮未取得内部 tiling key、归约树或指令级记录，不能进一步断言是哪一种 split-K 或特定 kernel 缺陷。','',
        '## 精度控制组与有效比较','',
        '| 配置 | 1024-token prefill 三方数值状态 |','|---|---|',
        '| 原始 BF16 | batch 未通过；原始失败保留 |',
        '| 仅 MLP Linear 用 FP32，输出转回 BF16 | batch 仍未通过 |',
        '| 全部 Linear 用 FP32，输出转回 BF16 | batch 仍未通过 |',
        '| 全部模型参数及中间激活用 FP32 | 四形状全部通过；完成正式三方测量 |','',
        '全 FP32 组将原 checkpoint 的 BF16 权重值精确提升为 FP32；关闭 HF32，三个策略使用同一模型状态、同一组输入和各自独立 KV。每形状每模式预热 3 次、测 12 轮，六种模式顺序轮换、AB/BA 交替。沿用原始 batch 容差 `atol=0.0625, rtol=0.02`，serial/parallel 仍要求完全一致，另核对贪心 token 和全部 24 层 KV。没有放宽原始 BF16 门限。','',
        f"全部 144 个新样本通过，最大差异 {s['full_fp32_max_abs']:.10g}。下表只在 FP32 配置内部比较，单位 ms/pair、中位数；显存、首末任务就绪时间、IQR 和原始样本另存 JSON。",'',
        '| 形状 | serial | 双流 | batch=2 | 双流配对变化 | batch 配对变化 |',
        '|---|---:|---:|---:|---:|---:|']
    for r in s['performance']:
        m=r['modes'];text.append('| %s | %.3f | %.3f | %.3f | %+.2f%% | %+.2f%% |'%(r['case'],m['serial']['wall_ms']['median'],m['parallel']['wall_ms']['median'],m['batch']['wall_ms']['median'],r['paired']['parallel']['change_percent']['median'],r['paired']['batch']['change_percent']['median']))
    text+=['','在这个独立精度配置中，batch=2 的四种形状均快于双流；长 prefill 双流也比串行快。不能把这里的速度或显存用于修正原 BF16 的性能表；不同精度还改变了初始 decode KV 的数值。此阶段没有采集完整 FP32 execution graph，不能从这些计时反推出其实际计算交叠量。','',
        '## 验证与复现','',
        '最终证据包保存全部定位、失败控制组、合格控制组、正式样本、代码快照、BF16 操作数及最终隔离 trace。初版 profiler 没有显式 schedule，已由完整闭合的 r02 采集替代；没有混入新的性能统计。','',
        '```bash',
        'mkdir -p /tmp/p23-numerics-replay',
        'tar -xzf practice_23_independent_inference/results/published/numerics/evidence.tgz -C /tmp/p23-numerics-replay',
        'python practice_23_independent_inference/numerics/analyze.py /tmp/p23-numerics-replay',
        "python -m unittest discover -s practice_23_independent_inference/numerics -p 'test_*.py'",
        '```','',
        '远端重做入口依次为 `numerics/locate.py`、`operator_probe.py`、`control.py --scope model`、`rounding_evidence.py`、`benchmark_control.py`；每次使用不存在的新输出目录。详细参数及版本在各 run 的 plan、environment、sources 中。','',
        '本项已完成首次数值分歧定位、高精度核验和独立精度配置下的有效比较。原生 BF16 跨 batch 的严格数值等价仍未解决，不因本次定位而改为通过。下一项按计划核验 graph capture/replay 的输入与 KV 条件，再开展同精度的调度对照。','']
    (root/'NUMERICS.md').write_text('\n'.join(text))
    print('published',out)


if __name__=='__main__':main()
