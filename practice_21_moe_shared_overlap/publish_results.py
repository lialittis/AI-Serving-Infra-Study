"""Publish replayable raw evidence, compact results and the P2 findings."""
import argparse,hashlib,json,shutil
from pathlib import Path


def read(p):return json.loads(p.read_text())
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--formal',type=Path,required=True);p.add_argument('--light',type=Path,required=True)
    p.add_argument('--formal-archive',type=Path,required=True);p.add_argument('--light-archive',type=Path,required=True)
    p.add_argument('--excluded-archive',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parent;out=root/'results/published';out.mkdir(exist_ok=True)
    s=read(a.formal/'analysis/summary.json');light=read(a.light/'summary.json');plan=read(a.formal/'plan.json')
    assert s['status']==light['status']=='passed' and s['diagnostic_trials']==24 and len(light['trials'])==24
    assert s['unsatisfied_data_edges']==0 and all(float(t['overlap_us'])==0 for t in light['trials'])
    assert read(a.formal/'weight_check.json')['unchanged']
    for f in ('plan.json','measurements.json','correctness.json','lifetime_checks.json','environment.json','weight_check.json'):
        shutil.copyfile(a.formal/f,out/f)
    for f in ('execution_graph.json','summary.json'):shutil.copyfile(a.formal/'analysis'/f,out/f)
    shutil.copyfile(a.light/'summary.json',out/'light-summary.json')
    archives={}
    for name,src in [('formal-evidence.tgz',a.formal_archive),('light-evidence.tgz',a.light_archive),('excluded-evidence.tgz',a.excluded_archive)]:
        shutil.copyfile(src,out/name);archives[name]=dict(bytes=src.stat().st_size,sha256=digest(src))
    manifest={}
    for tag,run in [('formal-r01',a.formal),('light-r01',a.light)]:
        for f in sorted(run.rglob('*')):
            if f.is_file() and 'analysis' not in f.relative_to(run).parts and f.name!='summary.json':
                manifest[tag+'/'+f.relative_to(run).as_posix()]=dict(bytes=f.stat().st_size,sha256=digest(f))
    save(out/'evidence_manifest.json',manifest)
    save(out/'archives.json',dict(archives=archives,remote='/data/tianchi/practice_21_moe_shared_overlap/results',
        replay='Extract formal-evidence.tgz and light-evidence.tgz; run analyze.py and analyze_light.py; source digests are pinned in contracts.json'))
    excluded={
        'qualification-r01':'单层 harness 尚未初始化原生 weight-prefetch 对象；初始化阶段失败。',
        'qualification-r02':'成功的基本资格检查；不属于正式性能样本。',
        'pilot-r01':'FP32 CPU top-k 在 BF16 并列分数处选择不同专家；参考算法尚未处理合法并列。',
        'pilot-r02':'对 BF16 路由权重错误使用 FP32 精度检查；正式门限尚未确定。',
        'routing-check-r01':'用于记录原生路由权重与数学参考差异。',
        'pilot-r03':'验证 BF16 中间舍入不能简单替代融合路由的舍入语义。',
        'pilot-r04':'1024 token 下跨设备 router matmul 不能要求逐位相同；专家输出容差未改。',
        'pilot-r05':'重复使用同一个 forward context，触发原生 MoE layer index 计数检查。',
        'pilot-r06':'成功的完整试采；次数与预热量较少，未混入正式性能数据。'}
    save(out/'excluded_runs.json',excluded)
    validation=dict(status='passed',formal_samples=len(read(a.formal/'measurements.json')),
        full_layer_pairs=48,calls_per_sample=20,cross_mode_exact_checks=12,cpu_reference_checks=12,
        queued_input_reuse_calls=72,diagnostic_trials=24,light_diagnostic_trials=24,
        device_tasks=s['device_tasks'],compute_tasks=s['compute_tasks'],light_compute_tasks=light['compute_tasks'],
        required_data_edges=s['required_data_edges'],unsatisfied_data_edges=0,
        device_event_wait_edges=s['event_wait_edges'],api_only_cross_stream_waits=s['cross_stream_wait_apis_without_device_task'],
        weights_unchanged=True,complete_exact_kernel_memory_dag=False)
    save(out/'validation.json',validation)
    lines=['# P2 结果：同一次 MoE forward 确有双 stream，本轮未获得计算重叠或加速','',
        '本轮完成单卡 eager 的真实层级对照：直接实例化安装版本的 `Qwen2MoeSparseMoeBlock`，实际执行 `AscendFusedMoE`。开启 shared-expert overlap 后，shared 计算进入第二条物理 stream；同步与输出均通过核验，但当前四种形状没有观察到 shared/routed 计算区间交集，无 profiler 的完整层耗时反而增加。','',
        '查看[离线交互图](report/index.html)、[完整执行图 JSON](results/published/execution_graph.json)和[验证记录](results/published/validation.json)。','',
        '## 负载与比较边界','',
        'BF16，TP=DP=EP=1，hidden=1024，8 个 routed expert，top-2，routed intermediate=512，shared intermediate=1024。使用确定性构造的权重与三个随机输入种子，没有下载预训练 MoE 权重；这是缩小尺寸的真实模型层，不是完整模型 serving 验证。','',
        '只切换 `layer.experts.multistream_overlap_shared_expert`，权重、输入、路由和计算 backend 相同。初始化时原生 shared split 一致性检查通过；gate 多流、量化、EPLB 和 graph 模式均未启用。TP=1 的 ALLGATHER 路径不产生跨卡通信。','',
        '## 独立性能测量','',
        '每种形状、每种模式先预热 20 次，再测 12 组，每组连续 20 次，交替反转模式顺序。总计 192 组样本，其中完整层 serial/parallel 共 48 个配对，另外两模式测量独立 shared/routed 分支。计时包含每次 forward context、原生提交／等待、路由与最终合并，结束时等待末尾 event；初始化、校验、profiler 和 Python 方法观察器不在计时内。','',
        '下表单位为 µs/call，中位数；完整 IQR、极值与每组原始记录见[汇总](results/published/summary.json)及[样本](results/published/measurements.json)。','',
        '| tokens | serial | parallel | shared only | routed only | 配对耗时变化中位数 | parallel 更快 |',
        '|---:|---:|---:|---:|---:|---:|---:|']
    for r in s['performance']:
        m=r['modes'];lines.append('| %d | %.2f | %.2f | %.2f | %.2f | +%.2f%% | %d/12 |'%(r['tokens'],m['serial']['median'],m['parallel']['median'],m['shared_only']['median'],m['routed_only']['median'],r['paired_change_percent']['median'],r['parallel_faster_pairs']))
    lines += ['','四种形状的配对中位数均变慢，合并 IQR 不重叠；32-token 形状有一对反向样本，其余方向一致。这是一个进程内交替测量的描述性结果，没有据此声称统计显著性或生产服务吞吐。独立分支耗时不能相加当作完整层耗时。','',
        '## 执行图与原生等待','',
        '24 次详细诊断共 538 个设备任务、384 个计算 kernel，每次包含 16 个计算任务（含一个显式输入 producer）。同流组的计算均在本轮物理 stream 46；双流组 shared 使用 44，其余使用 46。编号只在各次运行内有效。','',
        '分支图为 `输入 → router → routed experts → 合并`，以及 `输入 → shared gate/up → activation/down/shared gate → 合并`。每次 forward 的 8 条边界数据要求单独生成，总计 192 条，全部在实际 happens-before 图中可达。它们不是用于自证的同步边。','',
        '双流原生同步包含：','',
        '1. main 在 routed 工作前记录 `before_routed_experts`；shared 等待输入就绪。',
        '2. shared gate/up 还等待 `before_dispatch`，然后才提交第一段矩阵乘。',
        '3. shared activation/down/gate 等待 `before_combine`；该事件在 routed FFN 后、token combine 前记录。这是当前实现的阶段调度约束，不应误读成 shared 的数据来自 routed 输出。',
        '4. main 通过 `wait_stream(shared)` 等待 shared 末尾 event，再将两个输出相加。','',
        '详细图记录 46 条有物理 `EVENT_WAIT` 的跨流边；另有 2 次跨流等待在 CANN 中有 `aclrtStreamWaitEvent` 调用，但没有对应设备 SQE。后者通过精确 enqueue/dequeue correlation 关联原生 API，并以逻辑等待节点连接后续同流提交，未虚构设备任务或等待时长。36 次同流等待未生成设备 wait，已有 FIFO 提供顺序。','',
        '所有计算节点均通过 PyTorch flow、CANN connection 和 `kernel_details.csv` 的 stream/task/start/duration 身份核对。没有未关联设备任务，没有根据最近时间戳猜边。','',
        '## 重叠与观察器对照','',
        '详细诊断的 12 次双流 forward，shared/routed 实际 kernel 区间交集均为 0 µs。另运行不安装方法包装器和 TorchDispatchMode 的[轻量诊断](results/published/light-summary.json)：相同权重、四种形状、每模式三次，也得到 0 µs，共独立核对 384 个计算 kernel。','',
        '轻量对照仍启用了 profiler，因此不能宣称逐微秒还原无 profiler 执行。本轮无 profiler 性能也没有收益。源码先提交 routed 再提交 shared，只有设备尚有未执行的独立工作时才可能重叠；当前结果与提交／同步开销较大、单层计算较短的情形一致，但本轮没有对每项开销进行因果归因。','',
        '## 数值、存储与证据边界','',
        '12 组输入的 serial/parallel 输出逐元素完全相同。独立 CPU 专家计算使用 `atol=0.003, rtol=0.02`，最大绝对误差为 0.00390625，全部通过组合容差。CPU/NPU router matmul 使用 `atol=1e-4, rtol=1/128`，路由权重使用 `atol=0, rtol=1/128`，依据 BF16 精度制定并在正式测量前固定。','',
        'BF16 softmax 有并列 top-k；23 行的合法专家选择与 CPU 的默认 tie-break 不同。校验先核对被选分数属于 CPU 的 BF16 top-k，再在 CPU 独立计算所选专家和数学归一化权重，没有放宽最终输出门限来接受不同专家的输出。','',
        '权重全程存活，实验前后处理后权重哈希相同；每种形状连续 18 次覆盖同一个输入 buffer 并调用双流层，共 72 次，无中途主机同步，末尾统一校验全部输出。原生 shared→main join 也保证下一次 main 写输入之前旧 shared 读取已结束。shared 临时张量在 shared stream 上生成和消费，合并通过原生 join 衔接。','',
        '图覆盖真实设备任务、event 和调用边界张量地址／布局。CANN 内部 workspace 的精确分配与访存没有恢复，不能将这些边界要求宣传为完整原生 kernel 内存 DAG。未测试 graph、量化、多卡、完整预训练模型、服务请求调度或自动 stream 分配。','',
        '安装版本与 P1 相同，未升级或修改系统源码。设备原有 Alarm 仍在；数值通过不等同于硬件健康认证。正式采集前无其他 NPU 进程，结束记录只含本实验进程；收尾期间另见其他 VLLM 作业，未停止或修改它。','',
        '## 复现与归档','',
        '- [正式证据包](results/published/formal-evidence.tgz)：原始 profiler、测量、数值、源码和仪器快照。',
        '- [轻量证据包](results/published/light-evidence.tgz)：独立无方法观察器的 profiler 采集。',
        '- [排除项证据](results/published/excluded-evidence.tgz)及[排除原因](results/published/excluded_runs.json)：初始化与参考算法试采，不混入正式性能结果。',
        '- [原始文件哈希清单](results/published/evidence_manifest.json)、[压缩包哈希与远端位置](results/published/archives.json)。','',
        '复现命令见 [README](README.md)。下一步若继续研究该功能的收益，应扩大真实层规模或单独验证 graph 模式；本轮的负结果不能推广为所有 MoE、多卡通信或量化路径均无收益。','']
    (root/'RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps(validation,indent=2))


if __name__=='__main__':main()
