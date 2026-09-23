#!/usr/bin/env python3
"""Build the offline learning report from curated explanations and archived facts.

Standard library only, Python >= 3.7. Never connects to the inference server.
"""
import hashlib
import html
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
P09 = 'practice_09_operator_trace/results/2026-09-22-run01/'
P10 = 'practice_10_model_graph/results/2026-09-23-run02/'
P11 = 'practice_11_attention_execution/results/2026-09-23-run02/'
SOURCES = {
    'p07': 'practice_07_real_request_trace/RESULTS.md',
    'p08': 'practice_08_real_kv_mapping/RESULTS.md',
    'p09': 'practice_09_operator_trace/OPERATOR_FLOW.md',
    'p10': 'practice_10_model_graph/RESULTS.md',
    'p11': 'practice_11_attention_execution/RESULTS.md',
    'runner': P11 + 'sources/vllm_ascend/vllm_ascend/worker/model_runner_v1.py',
    'model': P09 + 'full_analysis/sources/vllm/vllm/model_executor/models/qwen2.py',
    'attention': P11 + 'sources/vllm_ascend/vllm_ascend/attention/attention_v1.py',
    'routes': P09 + 'full_analysis/coverage.json',
    'async': P09 + 'operator_links.json',
    'graph': P10 + 'analysis/summary.json',
    'execution': P11 + 'analysis/attention_evidence.json',
    'mapping': 'practice_08_real_kv_mapping/results/2026-09-22-run01/token_mapping.csv',
    'graph_viewer': P10 + 'analysis/graph_viewer.html',
    'attention_viewer': P11 + 'analysis/attention_viewer.html',
}


def step(src, dst, title, kind, detail, memory, proof, evidence):
    return dict(src=src, dst=dst, title=title, kind=kind, detail=detail,
                memory=memory, proof=proof, evidence=evidence)


def journeys():
    # The lanes describe responsibilities, NOT six separate OS processes.
    startup = [
        step(0, 2, '启动服务：加载本地模型与配置', 'control',
             '模型目录保存权重、config 和 tokenizer。vLLM 建立 Qwen2 模型，Ascend 插件接入设备实现。这里是启动阶段的结构说明；请求窗口 profiler 没有覆盖权重加载搬运。',
             '磁盘：模型文件；CPU：配置、模块对象、加载暂存。', '源码解释 / 启动日志', 'model'),
        step(2, 5, '权重驻留 NPU；建立 KV 存储池', 'data',
             '权重装入设备存储；框架按预算创建各层 KV cache。箭头概括加载结果，不代表已经追踪到每次权重 DMA 或具体加载缓冲策略。',
             'NPU HBM：持久权重 + 各层 K/V pool。CPU tensor 对象保存 shape、stride、device 和存储引用。', '实测布局 + 机制说明', 'p11'),
        step(2, 2, 'TorchDynamo 捕获模型 FX 图', 'control',
             'PyTorch 追踪真实 Qwen2Model.forward，vLLM 编译后端接收 FX GraphModule。852 个节点描述模型主体，输出 hidden states；LM head、采样和调度不在这张图里。',
             'CPU：FX 图、节点元数据、编译 callable。图中的 tensor 元数据不等于保存了全部数值。', '实测 · P10 / P11', 'graph'),
        step(2, 3, '49 个分区；为尺寸 1 捕获设备图', 'command',
             '25 个普通计算分区与 24 个 attention 边界交替。当前只 capture [1]，为后续单 token decode 做准备。FX 编译图与 ACL 设备图是不同表示。',
             '主机 / 运行时：分区 callable、设备图执行对象与关联缓冲；内部图对象的具体存储未解析。', '实测配置 / 源码', 'p11'),
        step(3, 4, '预热并执行捕获所需的设备工作', 'command',
             '启动和预热会执行真实计算。算子可能来自已安装库，也可能来自 Triton JIT 的编译缓存；本轮正式 profiler 在预热之后开始，未捕获完整编译或代码加载过程。',
             'NPU：执行代码及工作区；HBM：预热和图捕获需要的缓冲。不能把它们计成正式请求的 token。', '机制说明；编译过程未观测', 'p09'),
        step(2, 0, '服务就绪，开始正式请求', 'control',
             '下面 prefill / decode 以 P11 的 126 输入、2 输出请求为主线。完整系统时序是教学整理，证据来自不同粒度的观测；不是一份逐行原始 trace。',
             '权重和 KV pool 已存在；请求尚未占用自己的逻辑块。', '实测 · P11', 'p11'),
    ]
    prefill = [
        step(0, 1, '请求：126 个 token IDs，最多生成 2 个', 'data',
             'P11 直接发送 token ID 14990 × 126；没有在这次请求中执行文本分词。文本 → tokenizer → IDs 的入口由 P07 单独验证。API 与 EngineCore 是不同进程。',
             'CPU：HTTP 请求、token ID 列表、采样参数与 request 状态。', '实测 · P07 / P11', 'p11'),
        step(1, 1, '调度 126 个位置；分配逻辑块到 B2', 'control',
             'CPU allocator 从已存在的池中分配块 ID。本次请求 block table=[2]；这不是新建一块 NPU tensor，也不代表 B2 对所有运行固定。',
             'CPU：request → block table [2]；NPU KV pool 保持原有容量。', '实测 · P11', 'execution'),
        step(1, 2, '传入调度结果与 block IDs', 'control',
             '本次 UniProc 配置中 scheduler、worker、runner 是同一进程的不同职责。Runner 准备 input_ids、positions、长度和 attention metadata。',
             'CPU：调度输出与主机输入缓冲。调度器管理 ID，不处理完整 K/V 数值。', '实测 P07；P11 元数据', 'p07'),
        step(2, 5, 'H2D：输入和索引缓冲', 'data',
             'Runner 将需要的输入复制到 NPU；归档源码含 copy_to_gpu / non_blocking 路径。H2D 是 Host to Device。此箭头概括数据去向，不是绕过 CANN 的物理线路；未逐份关联全部拷贝的地址和字节数。',
             'NPU HBM：input_ids、positions、设备 block table；CPU 仍保留调度元数据。', '源码解释；拷贝任务 P09 实测', 'runner'),
        step(2, 3, '调用算子 / 入队；使用编译 callable', 'command',
             '126-token prefill 不匹配设备图 capture [1]，使用已编译 callable。Python / torch-npu / Triton 提交工作；不是逐条等待设备完成。',
             'CPU：算子参数、tensor 元数据、主机任务队列。传入设备地址并不等于复制 tensor 数值。', '实测 · P09 / P11', 'p11'),
        step(3, 4, 'CANN 出队、下发到 NPU stream', 'command',
             'P09 用 correlation ID 核对 Enqueue / Dequeue，再用 flow ID 连接 CANN launch 与设备任务。下面概括一组算子，每个模型层都会继续产生设备工作。',
             '运行时：stream 上排队的计算、复制与 event 任务。', '实测 · P09', 'p09'),
        step(4, 5, 'Embedding → Norm → QKV → RoPE', 'compute',
             '模型用已加载权重计算 hidden states 和 Q/K/V。Q、K 做 RoPE，V 不做。本轮首层 Q=[126,14,64]，K/V=[126,2,64]；QKV 位于 NPU，不是由 CPU 按 token ID 查出 KV。',
             'HBM：当前层中间张量。芯片内部还使用本地存储与计算单元，本实验没有追踪内部每次读写。', '实测 shape + 模型源码', 'model'),
        step(4, 5, '首层：将新 K/V 写入持久 cache', 'compute',
             'ReshapeAndCacheNdKernel 写入 cache；按 CPU block table 推得预期 slot 256～381。P11 未读回 slot 数值；真正读回并核对数值的是独立的 P08。其余层分别写入自己的 cache。',
             'HBM：第一层 cache[B2, offset 0..125]。各层保存不同的 K/V 数值。', '实测 kernel / 元数据 · P11', 'p11'),
        step(5, 4, '首层 FIA：读取本轮 Q、K、V', 'data',
             '本次状态 PrefillNoCache，FIA 直接使用当前 K/V，block_table=None，KV 长度 126。虽然已写 cache，不能因此断言本轮 FIA 一定从 cache 读取。',
             'HBM：本轮 Q/K/V → attention 计算；持久 KV 留给后续 decode。', '实测 · P11', 'execution'),
        step(4, 5, 'Attention output → view → O projection', 'compute',
             'Attention 将结果写进传入的 output buffer，Python 返回 None。下一分区 view 成 [126,896] 后由 O projection 读取；已验证共享地址和设备先后关系。',
             'HBM：output buffer 被原地更新。view 改变元数据，不会因此复制整份数据。', '实测 · P11', 'p11'),
        step(4, 5, '残差 / MLP；继续其余层与 final norm', 'compute',
             '每层执行归一化、attention、输出投影、MLP，并传递 hidden states / residual，共 24 层。许多残差加法与 norm 融合，没有独立 Add kernel。',
             'HBM：临时 hidden states、residual 与工作区；各层 KV 持续保存。临时缓冲的精确回收时刻未追踪。', '源码 + P09 算子计数', 'p09'),
        step(4, 5, 'LM head → penalty → argmax → token t₀', 'compute',
             'Runner 选择需要预测的位置计算 logits。本请求 temperature=0，但模型默认 repetition_penalty=1.1 仍有效：先调整 logits，再 argmax。t₀ 是位置 126 的新 ID，不是它的 KV。',
             'NPU：logits 和新 token ID。模型主体 FX 图在 hidden states 结束，LM head 与采样位于其外。', 'P09 实测；P11 runner 源码', 'runner'),
        step(5, 2, 'D2H：取回采样 ID，等待所需结果就绪', 'data',
             '当前关闭 async scheduling，_bookkeeping_sync 将采样结果转换为主机列表。CPU 使用结果前必须满足完成依赖；不把这解释为每个算子都全设备同步。',
             'CPU：新 token ID；NPU：权重、KV 继续驻留。没有把整个 KV cache 搬回 CPU。', '源码解释 + P09 同步事件', 'runner'),
        step(2, 1, '更新：computed=126，output=1', 'control',
             '请求还没有结束，t₀ 将作为下一轮的输入。此时逻辑序列长 127，但 KV 只覆盖位置 0～125；位置 126 的 t₀ 还没有自己的 KV。',
             'CPU：请求进度；NPU：126 个已计算位置的每层 KV。', '实测 · P11 / P07', 'p11'),
    ]
    decode = [
        step(1, 2, '调度 t₀：本轮只处理位置 126', 'control',
             '上轮新生成的 token 现在才进入 forward。Runner 已缓存采样 ID；不意味着调度器必须重新传回完整 token 列表。请求仍使用 B2。',
             'CPU：新增 1 个待计算位置；历史 prompt 无需重新做完整 forward。', '源码 / 实测长度 · P11', 'runner'),
        step(2, 5, '准备单 token 输入、位置和 metadata', 'data',
             '输入长度为 1，可见 KV 长度将变为 127。block table 是 [1,16] 的 NPU tensor；shape 中的 16 是表容量，不是这次请求用了 16 块。',
             'HBM：新输入和设备索引；持久 K/V 仍在原来的 cache。', '实测 · P11', 'execution'),
        step(2, 3, '普通分区：ACL graph replay', 'command',
             '尺寸 1 匹配已捕获设备图。ACLGraphWrapper 重放普通分区，跳过分区内部逐算子的 Python 调用；attention 边界仍通过其可观测路径执行。',
             'CPU / 运行时：已有图执行对象；HBM：图所引用的输入、输出缓冲。', '实测 · P11', 'p11'),
        step(3, 4, '提交图执行与 attention 设备任务', 'command',
             'P11 有 25 个 MODEL_EXECUTE 任务。244 个带有效 Model Id 的设备任务缺少完整框架关联，不能按名字或时间强行归属于某个 FX 节点。',
             'NPU：真实运行重放任务；未知的是精确 FX → 内部 kernel 对应关系。', '实测及缺口 · P11', 'p11'),
        step(4, 5, '对 t₀ 计算 Q/K/V 与 RoPE', 'compute',
             '首层 Q=[1,14,64]，当前 K/V=[1,2,64]。本轮只产生 1 个新位置的 K/V，而不是重新产生全部历史 K/V。',
             'NPU：新 Q/K/V；后续每层会计算该层的新 Q/K/V。', '实测布局 · P11', 'execution'),
        step(4, 5, '将 t₀ 的新 K/V 写入 cache', 'compute',
             'CPU 映射推导的预期 slot 为 2×128+126=382。第一层 KV kernel 为 stream 46 / task 1907，执行结束后才开始本层 FIA。',
             'HBM：B2 的 offset 126 新增 KV，之前 126 个位置仍可供 attention 使用。', '实测 kernel；slot 是推导值', 'p11'),
        step(5, 4, 'FIA：新 Q 读取历史 + 当前 KV', 'data',
             '状态 DecodeOnly；FIA 读取 KV cache 的 [11761,128,128] view，存储地址与原 [11761,128,2,64] cache 一致，可见长度为 127。Q 仍来自本轮。',
             'HBM：持久 cache → attention。view 的末维 128=2 KV heads×64，不是新复制了一份 cache。', '实测 · P11', 'execution'),
        step(4, 5, 'Attention 写 output；下一分区消费', 'compute',
             'output 与下一重放分区入口的输入共享存储已核对。分区代码包含 view → O projection；decode 内部没有再次进入该 linear 的 Python 函数，具体 kernel 映射尚未补齐。',
             'HBM：原地更新的 output buffer → 下一分区。', '实测入口 / 源码解释内部', 'p11'),
        step(4, 5, '24 层结束；LM head 与采样生成 t₁', 'compute',
             't₁ 位于逻辑位置 127。当前 forward 计算的是 t₀ 的 KV，同时预测 t₁；这是两个不同 token 的事件。',
             'NPU：新 token ID；KV 已覆盖位置 0～126。', '实测 token 计数 / 模型机制', 'p11'),
        step(5, 2, 'D2H：取得 t₁ 的 ID', 'data',
             '取回生成结果，满足 CPU 读取的完成依赖。关闭异步调度仍会有异步算子入队、CANN 下发和设备图重放。',
             'CPU：第二个输出 ID；KV 数据保持设备驻留。', '源码解释 / P09 异步证据', 'runner'),
        step(2, 1, '达到输出长度：computed=127，output=2', 'control',
             '逻辑序列 126+2=128，但只计算了 127 个位置。t₁ 不再进入下一轮，因此没有为它生成 KV。下一页展示请求结束与释放。',
             'CPU：完成状态；设备 cache 中本请求的有效位置截至 126。', '实测 · P11', 'p11'),
    ]
    finish = [
        step(2, 1, '返回 token IDs，判断停止条件', 'data',
             '调度器处理输出与结束状态。这是请求生命周期的一部分，不属于模型主体 FX 计算图。P07 已观察到 update_from_output 与请求清理入口。',
             'CPU：结果 IDs、computed 数、finish reason。', '实测 · P07', 'p07'),
        step(1, 1, '释放 request 对 block 的占用', 'control',
             '释放意味着 allocator 可以在规则允许时重新分配物理块；不等于销毁整个 KV pool、擦除旧数据或仅凭这个事件就证明设备上没有在途访问。',
             'CPU：块所有权 / 池状态变化。HBM：KV pool 继续存在，后续写入可覆盖旧值。', 'P07 清理入口；机制说明', 'p07'),
        step(1, 0, 'Engine 输出 → 反分词 → HTTP 响应', 'data',
             'API 路径将结果 IDs 转为文本。本次 P11 返回“ syntax,”。非流式响应在完成后返回；流式请求的逐段发送行为没有在这次实验中测量。',
             'CPU：输出文本和 HTTP JSON。客户端不会收到模型权重或 KV tensor。', '响应实测 · P11', 'p11'),
        step(1, 1, '后续请求可复用块；真实 lifetime 待研究', 'control',
             'P05/P06 的 B0:g1 与 B0:g2 演示了旧引用风险，但 generation 检查是教学实现，不能当作 vLLM 已部署的 detector。真实 free/reuse 与在途设备任务的关系是下一阶段问题。',
             '逻辑生命周期改变，物理地址可以不变。当前尚未构成真实 KV lifetime violation detector。', '概念练习 / 尚未观测', 'p08'),
    ]
    return dict(startup=startup, prefill=prefill, decode=decode, finish=finish)


def static_sequence(phase, steps):
    """Standalone SVG: all steps visible, same rows as the interactive viewer."""
    xs = [75, 223, 371, 519, 667, 815]
    names = ['客户端 / API', 'Scheduler', 'Runner / PyTorch', 'CANN / 队列', 'NPU stream', 'HBM 存储']
    colors = dict(control='#687d88', data='#3267b1', command='#ab6019', compute='#087d78')
    height = 145 + 59 * len(steps)
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="{}" viewBox="0 0 900 {}" role="img">'.format(height, height),
             '<title>{} 请求时序 / 教学摘要</title>'.format(phase),
             '<desc>来自独立实验和归档源码的教学整理，不按真实时间缩放。泳道按职责划分；HBM 是存储。</desc>',
             '<style>text{font-family:system-ui,"Noto Sans CJK SC",sans-serif}</style>',
             '<rect width="900" height="{}" fill="white"/><rect width="900" height="75" fill="#f1f5f2"/>'.format(height), '<defs>']
    for kind, color in colors.items():
        parts.append('<marker id="{}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10{}" fill="{}" stroke="{}"/></marker>'.format(
            kind, '' if kind == 'command' else 'Z', 'none' if kind == 'command' else color, color))
    parts.append('</defs>')
    for x, name in zip(xs, names):
        parts.append('<text x="{}" y="30" text-anchor="middle" font-size="12" fill="#284650">{}</text><line x1="{}" x2="{}" y1="60" y2="{}" stroke="#bacbc7" stroke-dasharray="4 5"/>'.format(x, name, x, x, height-65))
    for i, s in enumerate(steps):
        y, src, dst = 105 + 59*i, xs[s['src']], xs[s['dst']]
        same = src == dst
        title_x, anchor = (src+14, 'start') if same else ((src+dst)/2, 'middle')
        path = 'M{},{}h55v18h-55'.format(src, y) if same else 'M{},{}H{}'.format(src, y+10, dst)
        parts.append('<text x="17" y="{}" font-size="11" fill="#547078">{:02d}</text><text x="{}" y="{}" text-anchor="{}" font-size="12" fill="#294551">{}</text><path d="{}" fill="none" stroke="{}" stroke-width="1.8" {} marker-end="url(#{})"/>'.format(
            y+9, i+1, title_x, y-6, anchor, html.escape(s['title']), path, colors[s['kind']],
            'stroke-dasharray="6 4"' if s['kind'] == 'command' else '', s['kind']))
    parts.append('<text x="20" y="{}" font-size="11" fill="#547078">{} · 教学时序，不按时间比例。灰：控制；蓝：数据；橙色虚线：异步命令；绿：设备工作。</text></svg>'.format(height-25, phase))
    return '\n'.join(parts) + '\n'


def main():
    paths = list(dict.fromkeys(SOURCES.values()))
    for path in paths:
        if not (ROOT / path).is_file():
            raise FileNotFoundError(path)
    load = lambda key: json.loads((ROOT / SOURCES[key]).read_text())
    p09, p10, p11, async_data = [load(k) for k in ('routes', 'graph', 'execution', 'async')]
    tasks = []
    for item in p11['attention_tasks']:
        k = item['kernel']
        tasks.append(dict(phase=item['phase'], name=k['name'], task=k['args']['Task Id'],
                          stream=k['args']['Physic Stream Id'], duration=k['dur']))
    data = dict(sources={k: '../../' + v for k, v in SOURCES.items()},
                journeys=journeys(), p09=p09,
                graph={k: p10[k] for k in ('nodes', 'edges', 'node_kinds', 'partitions', 'layers')},
                p11=dict(tasks=tasks, device_tasks=p11['device_tasks'],
                         replayed_tasks=p11['replayed_tasks'], phases=p11['phases']),
                async_examples=async_data['examples'])
    payload = json.dumps(data, ensure_ascii=False, separators=(',', ':')).replace('<', '\\u003c')
    template = (HERE / 'report.template.html').read_text()
    assert template.count('__REPORT_DATA__') == 1
    (HERE / 'index.html').write_text(template.replace('__REPORT_DATA__', payload))
    # Editable text counterpart to the interactive SVG, generated from the same steps.
    participants = ['API', 'Scheduler', 'Runner', 'CANN', 'NPU', 'HBM']
    labels = ['客户端与 API · CPU', 'Scheduler · CPU', 'Runner 与 PyTorch · CPU',
              'CANN 与主机队列', 'NPU stream', 'NPU HBM 存储']
    lines = ['sequenceDiagram', '    autonumber']
    lines += ['    participant {} as {}'.format(a, b) for a, b in zip(participants, labels)]
    for phase, steps in journeys().items():
        lines.append('    Note over API,HBM: {} / 教学时序，非原始 trace 时间轴'.format(phase))
        for s in steps:
            arrow = '-)' if s['kind'] == 'command' else '->>'
            title = s['title'].replace(';', '；').replace('\n', ' ')
            lines.append('    {}{}{}: {}'.format(participants[s['src']], arrow,
                                                participants[s['dst']], title))
    (HERE / 'sequence.mmd').write_text('\n'.join(lines) + '\n')
    for phase, steps in journeys().items():
        (HERE / ('sequence-' + phase + '.svg')).write_text(static_sequence(phase, steps))
    manifest = dict(inputs=[dict(path=p, sha256=hashlib.sha256((ROOT / p).read_bytes()).hexdigest())
                            for p in paths],
                    note='Summarizes existing independent runs; no new inference run. Relative paths from repo root.')
    (HERE / 'evidence_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print('Built index.html, sequence.mmd and evidence_manifest.json from {} archived inputs.'.format(len(paths)))


if __name__ == '__main__':
    main()
