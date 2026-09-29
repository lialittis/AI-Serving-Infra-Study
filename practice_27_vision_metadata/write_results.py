"""Render factual result tables from verified P27 summaries."""
import json
from pathlib import Path


def main():
    root=Path(__file__).resolve().parent;s=json.loads((root/'results/published/summary.json').read_text());variants=s['variants']
    lines=['# 视觉元数据预计算：实测结果','',
        '本轮完成 8 个场景、576 个无 profiler 样本、96 个独立诊断 trial；含预热/资格的 816 次输出比较全部逐元素一致，8 个场景另与原生完整 forward/generate 对照通过。','',
        '视觉阶段同步 API 从 native 的每次 91 条降为 lengths 的 7 条，再降为 cached 的 0 条。仅预计算 attention 分段长度不足以让 VL（先视觉后语言）产生重叠；完整元数据预计算后，8 个场景的 VL 都观察到真实 compute kernel 重叠。','',
        'cached 相对自身单流的 VL pair 耗时下降：prefill 为 7.18%–7.98%，decode 为 5.17%–10.74%。其 LV prefill 下降 6.86%–12.04%，LV decode 基本持平。因此 P25 的 decode 无明显收益只适用于当时的元数据与提交路径，不能推广成 decode 天生无法并发。','',
        '完整元数据准备每场景观测约 0.57–0.66 ms，额外有效设备数据约 136 / 621 KiB，均在正式 pair 计时之外。','',
        '## 方法与工作量','',
        '完整预训练 Qwen2.5-VL-3B-Instruct，BF16 eager，Ascend 910B2C（容器逻辑 0 / 物理 5），单 CPU 提交线程。A 为 512-token 语言 prefill 或一个准备好前缀的 decode 步；B 为两张图片、两档实际视觉 token 数 54 / 247，随后运行自己的语言阶段。','',
        'native 保留原实现；lengths 仅改 attention 分段长度来源，保留其他视觉元数据操作；cached 再预计算窗口索引、逆序索引和 RoPE cos/sin，并让索引提前驻留设备。全部网络层和像素计算保留。三组使用相同 stream/event/handoff，A/B KV 和位置状态独立。方法替换仅在实验进程内有效，原安装文件和权重未修改。','',
        '六个变体/流配置轮换位置，每种 LV/VL 顺序下各覆盖全部六个测量位置。每配置预热 3 次，计时 12 次（每种顺序 6 次）。性能计时不带 profiler，诊断与正式计时分开。','',
        '## 性能：分开看单流优化和双流收益','',
        '下面是 ms/pair 中位数。括号内为同轮双流相对同一变体单流的配对百分比中位数；负值表示双流更快。native 与 cached 的差额包含元数据路径优化，不应全算作并发收益。完整 IQR/极值、变体相对 native 的配对变化、A/V/B 就绪和主机提交记录见 [summary.json](results/published/summary.json)。','',
        '| 场景 | 变体 | LV 单流 / 双流 | LV 双流变化 | VL 单流 / 双流 | VL 双流变化 |','|---|---|---:|---:|---:|---:|']
    for case in [p['case'] for p in variants['native']['performance']]:
        for variant,v in variants.items():
            p=next(p for p in v['performance'] if p['case']==case);o=p['by_order']
            lines.append(f"| {case} | {variant} | {o['LV']['serial']['wall_us']['median']/1000:.3f} / {o['LV']['parallel']['wall_us']['median']/1000:.3f} | {p['paired']['LV']['change_percent']['median']:+.2f}% | {o['VL']['serial']['wall_us']['median']/1000:.3f} / {o['VL']['parallel']['wall_us']['median']/1000:.3f} | {p['paired']['VL']['change_percent']['median']:+.2f}% |")
    lines+=['','## 实际重叠与同步','',
        '重叠来自 A 语言与 B 视觉的实际 compute kernel 区间并集求交，单位 ms；不是两个 Python forward 包围区间相交。API 调用数可能含嵌套，不等于独立等待次数，未将其 duration 相加。这里的零同步只指 vision_B scope，跨流 event 和末尾 host join 仍然存在。','',
        '| 场景 | 变体 | LV 重叠 ms | VL 重叠 ms | 每 trial 视觉同步 API（LV / VL） |','|---|---|---:|---:|---:|']
    for case in [p['case'] for p in variants['native']['performance']]:
        for variant,v in variants.items():
            ts={t['order']:t for t in v['trials'] if t['case']==case and t['mode']=='parallel'}
            lines.append(f"| {case} | {variant} | {ts['LV']['overlap_us']/1000:.3f} | {ts['VL']['overlap_us']/1000:.3f} | {ts['LV']['vision_native_syncs']} / {ts['VL']['vision_native_syncs']} |")
    lines+=['','原实现的分段长度 `.tolist()` 发生于每个 window attention 的 Q/K/V 切分。lengths 去掉这类读取后，仍保留 CPU 索引进入设备时的拷贝和 `unique_consecutive` 相关同步。逐条关联见 [sync_audit.json](results/published/sync_audit.json)：依据同线程区间包含或精确 async_task_queue flow，而非最近时间戳。','',
        '首个场景中，lengths 在视觉末尾仍有 `aten::index → aten::to → aten::_to_copy → aten::copy_` 内的同步；结合固定源码的执行顺序，该末尾索引可对应到 merger 后 CPU reverse_indices 的设备索引消费；这是源码与 trace 的联合定位。cached 将索引提前驻留设备，连同其他静态元数据一起移出执行阶段。该组合对照说明只移除 attention 分段读取仍不足以解除 VL 的阻塞；本轮没有进一步独立区分窗口、逆序和位置编码各自的性能贡献。','',
        '## 准备成本与请求延迟','',
        '下列是每个场景一次实际准备的观测值，不是准备耗时分布。总准备包含元数据设备运算/上传及同步，发生于计时前。持久化字节是新增长期存活元数据 tensor 的有效字节，不等同于 allocator reserved 增量；三组对照期间都保留同一份预计算对象，显存峰值比较不是“原实现完全不持有元数据”的生产部署比较。','',
        '| 场景 | CPU 长度准备 ms | 完整准备 ms | 持久化设备字节 |','|---|---:|---:|---:|']
    for p in s['preparation']:lines.append(f"| {p['case']} | {p['cpu_lengths_us']/1000:.3f} | {p['total_us']/1000:.3f} | {p['persistent_device_bytes']} |")
    lines+=['','pair 时间包含 A 语言、B 视觉、B merge/语言及结束等待；图片预处理、输入传输、A 视觉/前缀、KV clone 和上面的预计算不在计时内。不能把已复用元数据的 pair 时间直接解释为首次请求或 HTTP 端到端时间。','',
        '| cached 场景 | LV A 完成 单流 / 双流 ms | VL A 完成 单流 / 双流 ms |','|---|---:|---:|']
    for p in variants['cached']['performance']:
        o=p['by_order'];lines.append(f"| {p['case']} | {o['LV']['serial']['ready_us']['A']['median']/1000:.3f} / {o['LV']['parallel']['ready_us']['A']['median']/1000:.3f} | {o['VL']['serial']['ready_us']['A']['median']/1000:.3f} / {o['VL']['parallel']['ready_us']['A']['median']/1000:.3f} |")
    lines+=['','A 完成时间以公共起始 event 为原点，包含前序排队与主机供给间隙；不是独立 kernel 服务时间。总耗时下降不保证每个请求的延迟改善，资源竞争仍需独立硬件计数器实验。','',
        '## 执行依赖与验证','',
        '| 变体 | 设备任务 | 计算任务 | 已证明边界要求 | 视觉同步 API 总数 |','|---|---:|---:|---:|---:|']
    for variant,v in variants.items():lines.append(f"| {variant} | {v['device_tasks']} | {v['compute_tasks']} | {v['boundary_requirements']} | {v['native_vision_sync_calls']} |")
    lines+=['','所有 kernel CSV 记录均精确关联。真实视觉 feature 指针与消费端一致，A/B KV 等活跃存储范围隔离，feature event 等待保护 B 语言消费；要求没有作为证明边添加回图。权重/buffer 前后哈希一致，共享 rope_deltas 保持 None。','',
        '图仅覆盖显式 FIFO/event/host join 和真实 kernel 时间线；原生 workspace 的完整访存 DAG、全部隐式 CPU 因果边、算力/带宽竞争程度仍未恢复。诊断会扰动提交时序，单次诊断的重叠时长不能直接相减得到正式性能收益。','',
        '## 归档与边界','',
        '[交互对照与逐 kernel 图](report/index.html) · [归档清单](results/published/archives.json) · [验证记录](results/published/replay_validation.json)。','',
        'qualification-r01 在 case 构造前因缺失 phase 参数失败，未产生有效样本；修正后的 qualification-r02 通过。正式结果仅来自 formal-r01，失败与资格记录一起保留。大型 CANN 二进制/数据库留在远端，正式 trace_view/kernel CSV 全部可本地重放。远端与本地主机时钟有偏移，关联与耗时只使用同一采集时钟。','',
        '这仍是 HF eager 阶段 harness，没有 graph replay 或 vLLM 在线服务调度。下一步可在固定元数据路径上做 graph 对照，或先独立采集计算/带宽计数器，评估并发的资源代价；本轮未实施这些后续工作。']
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
