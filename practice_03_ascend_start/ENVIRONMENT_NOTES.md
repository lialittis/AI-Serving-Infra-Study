# Ascend 环境调查与下载资料（历史记录）

本文保留 2026-09-17 至 09-18 的环境调查过程。当前复现入口为 [README](README.md)，
压测结果与限制见 [RESULTS](RESULTS.md)。下文早期下载路径是备用路径，当前已上传模型位于
`/data/huggingface_home/hub/Qwen2.5-0.5B-Instruct`。历史文中“尚未启动 API”等描述属于当时状态。

采集日期：2026-09-17。目标实例：`ssh ascend910`，工作目录：`/data/tianchi`。
本文依据实际环境和对应版本官方文档整理。密码不保存在代码或文档中。

2026-09-18 更新：用户已将模型上传到
`/data/huggingface_home/hub/Qwen2.5-0.5B-Instruct`，Transformers 和 vLLM 离线生成均已通过。
直接使用现有模型的命令、`spawn` 修正和日志位置见 [本次推理测试记录](RUN_2026-09-18.md)。
下面下载章节保留为参考；使用已上传模型时可跳过下载，运行 Python 脚本时传入 `--model`，
运行 `serve.sh` 时将该路径作为第一个参数。

## 1. 本次实测结果

| 项目 | 结果 |
| --- | --- |
| SSH | 密码登录成功；现有密钥未能完成认证 |
| 登录身份 / 实例 | root / 容器式算力实例 |
| OS / CPU 架构 | openEuler 24.03 LTS-SP3 / x86_64 |
| 可见 NPU | 1 × Ascend 910B2C，HBM 65536 MiB（64 GiB） |
| 设备编号 | `npu-smi` 物理卡号 5；PyTorch 逻辑设备 `npu:0` |
| 初始 HBM 用量 | 3426 MiB；容器内未显示运行中的 NPU 进程 |
| 数据盘 | `/data` 约 59 GiB，检查时可用约 56 GiB |
| Python | `/usr/local/python3.12.13/bin/python`，3.12.13 |
| CANN | 环境路径 `/usr/local/Ascend/cann-9.0.0` |
| 驱动 / 固件 | 26.0.rc1 / 9.0.0.0.205；工具报告兼容性 OK |
| PyTorch / torch-npu | 2.10.0+cpu / 2.10.0 |
| vLLM / vLLM Ascend | 包元数据 0.21.0+empty / 0.21.0rc1；运行时 vLLM 0.21.0 |
| Triton | triton 3.2.0 + triton-ascend 3.2.1；没有名为 triton-npu 的分发包 |
| Transformers / ModelScope | 5.5.4 / 1.37.1 |
| NPU 运算 | `is_available() == True`，4×4 FP32 矩阵乘法与 CPU 结果一致 |
| vLLM 平台 | `ascend` 插件加载成功，`NPUPlatform` |
| 模型下载网络 | ModelScope 的 Qwen 配置文件 HTTP 200；HF 官网直连失败；后续复查确认 HF 镜像已配置，但镜像 HEAD/GET 连接超时 |

主要依赖与 [v0.21.0rc1 发布说明](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.21.0rc1)
列出的 CANN 9.0.0、torch/torch-npu 2.10.0、triton-ascend 3.2.1 相符。
这不等于所有算子、模型或性能均已验证。`torch` 的 `+cpu` 后缀本身不是问题：
这里通过 `torch_npu` 提供 NPU 后端，已有实际 NPU 运算通过。

### 当前需要处理的硬件告警

以下只读命令可复查：

```bash
npu-smi info
npu-smi info -t health -i 5 -c 0
npu-smi info -t ecc -i 5
```

实际故障输出：

```text
Health Status     : Alarm
Error Code        : 80C98001
Error Information : node type=AIC, sensor type=RAS State, event state=module error can not be fixed
```

不指定芯片时看到的 `Health: OK` 属于 MCU，不代表 AI 芯片正常。HBM ECC 计数为 0
也不能排除这个 AIC 告警。[Ascend 官方故障管理配置](https://github.com/Ascend/mind-cluster/blob/master/docs/en/scheduling/usage/appliance/01_npu_hardware_fault_detection_and_rectification.md)
把 `80C98001` 列入 `RestartNPUCodes`。这说明需要平台侧排查与恢复，不能仅凭该错误文字
断言硬件永久损坏；也不能据此保证重启虚拟实例即可恢复。请将以上输出交给算力平台，
由平台判断复位、更换卡或进一步诊断。

2026-09-17 的验证止于基础算术、包导入与 CLI 检查。
2026-09-18 已按用户要求继续测试，使用用户上传的权重完成 Transformers 与 vLLM 离线生成；
尚未启动 API 服务或运行 Triton 自定义内核。本次推理通过不等于硬件健康认证。

## 2. 连接和环境准备

本地终端：

```bash
ssh ascend910
```

如果本地遇到本次检查发现的
`Bad owner or permissions on /etc/crypto-policies/back-ends/openssh.config`，可使用：

```bash
ssh -F ~/.ssh/config ascend910
```

`-F` 显式使用个人配置并跳过系统 SSH 配置；本次通过已有主机密钥验证后登录成功。
登录时还出现了 RSA 附加主机密钥证明的签名警告；没有关闭主机密钥检查，也没有改写系统配置。
该警告可交由 SSH 网关维护方检查。

远程终端：

```bash
cd /data/tianchi
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export PATH=/usr/local/python3.12.13/bin:$PATH
export HF_HOME=/data/tianchi/cache/huggingface
export MODELSCOPE_CACHE=/data/tianchi/cache/modelscope
export VLLM_CACHE_ROOT=/data/tianchi/cache/vllm
mkdir -p models logs "$HF_HOME" "$MODELSCOPE_CACHE" "$VLLM_CACHE_ROOT"
python -c 'import torch, torch_npu; print(torch.npu.is_available(), torch.npu.device_count())'
```

本次登录 shell 已配置 CANN，可直接 import。不要覆盖已有 `LD_LIBRARY_PATH`；如果新会话报
`libatb.so` 缺失，再定位并 source `/usr/local/Ascend/nnal` 下实际安装版本的 `set_env.sh`。
本例沿用未设置 `ASCEND_RT_VISIBLE_DEVICES` 的实测配置。只有一张可见卡，直接使用 `npu:0`；
不要把物理卡号 5 直接填成 `npu:5`，也不要照搬多卡教程中的 `0,1,2,3`。

学习脚本目录为 `/data/tianchi/practice_03_ascend_start`。如需从本地仓库重新上传：

```bash
# 在本地仓库根目录执行；网关若不支持 SFTP，可使用 scp -O
scp -F ~/.ssh/config -r practice_03_ascend_start ascend910:/data/tianchi/
```

## 3. 学习顺序与组件关系

建议按“基础算术 → Transformers 生成 → vLLM 离线批处理 → HTTP 服务 → 性能实验”推进。
硬件告警处理后，先执行 `python practice_03_ascend_start/check_npu.py` 复查基础算术。

| 组件 | 在这次学习中的作用 |
| --- | --- |
| 驱动 / 固件 / CANN | 设备访问、运行时和底层算子 |
| PyTorch + torch_npu | 张量运算、模型执行，使用 `npu` 设备 |
| Transformers | 易读的 tokenizer、模型加载和 `generate()`，适合理解推理流程 |
| vLLM + vllm-ascend | 推理调度、KV cache 管理、批处理与 API 服务；Ascend 插件提供 NPU 适配 |
| Triton Ascend | 编写与优化设备算子；运行现有小模型时不必先自己写内核 |

已有环境足够开始，不需要先 `pip install -U`。新建普通 venv 会看不到现有系统包，
重装 torch/vllm 也可能打破版本组合；先使用上面的现有解释器。

首个练习选择 **Qwen2.5-0.5B-Instruct**：体积小，方便观察输入、输出和缓存。
其官方模型卡记载 0.49B 参数，BF16 纯权重估算约 0.98 GB，运行时还需要 KV cache 和工作区，
不能把权重大小当作总显存需求。[模型卡及 Transformers 示例](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)。

后续可试 Qwen3-0.6B，再扩展到 Qwen3-4B/8B。当前版本的官方快速入门使用 Qwen3-0.6B；
Qwen3 的 thinking 模式和 chat template 应单独阅读，不能只改路径就假设输出行为相同。
见 [对应版本快速入门](https://docs.vllm.ai/projects/ascend/en/v0.21.0rc/quick_start.html)
和 [Qwen3 部署教程](https://docs.vllm.ai/projects/ascend/en/v0.21.0rc/tutorials/models/Qwen3-Dense.html)。

## 4. 下载一个模型，分别用于两种推理方式

镜像复查：`/root/.bashrc:51` 已设置 `export HF_ENDPOINT=https://hf-mirror.com`，
登录 shell 中的变量及 `huggingface_hub.constants.ENDPOINT` 均为该镜像地址。
最初使用写死官网 URL 的 HTTP 探测没有经过镜像，不能据此断言 `hf download` 不可用。
后续实际执行以下命令时出现重试；独立 HEAD/GET 请求镜像也在 12 秒后连接超时，
因此本次失败不是命令语法或镜像变量未加载导致的。网络情况可能变化，可重新尝试：

```bash
hf download Qwen/Qwen2.5-0.5B-Instruct \
  --local-dir /data/tianchi/models/Qwen2.5-0.5B-Instruct
```

只验证小文件时，在模型 ID 后加 `config.json`，并将目录改为
`/data/tianchi/hf-mirror-check`。本次小文件下载未成功；大模型权重路径的可达性尚未验证。
通过非交互 SSH 或脚本运行时，也可在命令前显式加
`HF_ENDPOINT=https://hf-mirror.com`，避免依赖 `.bashrc` 的加载方式。

已验证 ModelScope 配置文件可达；下面会真正下载约 1 GB 量级的模型文件。
使用已安装的 ModelScope，将文件放在工作目录，避免重复下载：

```bash
cd /data/tianchi
python - <<'PY'
from modelscope import snapshot_download
path = snapshot_download(
    'Qwen/Qwen2.5-0.5B-Instruct',
    local_dir='/data/tianchi/models/Qwen2.5-0.5B-Instruct',
    allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja'],
)
print(path)
PY
```

下载 API 参数已对远程安装版本检查；参考 [ModelScope snapshot_download 实现](https://github.com/modelscope/modelscope/blob/master/modelscope/hub/snapshot_download.py)。
本例使用默认仓库修订；需要严格复现实验时，另行固定并记录模型 revision。

## 5. 用 Transformers 观察单次生成

在远程执行：

```bash
cd /data/tianchi
python practice_03_ascend_start/transformers_generate.py
```

脚本用 BF16、`npu:0` 和 eager attention，从本地加载权重。先经 chat template 构造输入，
再调用 `generate()`，只解码新生成 token。这里的 eager 实现便于排查基础问题，不是性能优化配置。

结合前两个练习，观察三个阶段：tokenizer 把文本变成 token；prefill 处理整段 prompt 并建立
KV cache；decode 逐 token 生成并复用缓存。`max_new_tokens` 只限制新增长度。

## 6. 用 vLLM 跑离线批处理

结束 Transformers 进程后，再运行：

```bash
cd /data/tianchi
python practice_03_ascend_start/vllm_generate.py
```

脚本提交两个对话，让 vLLM 调度它们；无需手工导入 `vllm_ascend`。
应看到 `Platform plugin ascend is activated`。Python 入口使用 `if __name__ == '__main__'`
保护，适配 worker 进程启动。脚本在导入 vLLM 前设置
`VLLM_WORKER_MULTIPROC_METHOD=spawn`；本环境默认 `fork` 实测会报 NPU 无法在子进程重新初始化。

| 参数 | 本例取值及目的 |
| --- | --- |
| `tensor_parallel_size` | 1，使用一张可见卡 |
| `dtype` | bfloat16，明确推理精度 |
| `max_model_len` | 2048，限制输入加输出的总长度 |
| `max_num_seqs` | 4，限制同时调度的序列数 |
| `max_num_batched_tokens` | 2048，限制一次调度的 token 预算 |
| `gpu_memory_utilization` | 0.3，约以整卡 30% 为实例内存预算；名字含 GPU，在 Ascend 仍沿用 |
| `enforce_eager` | True，先排除图编译/捕获带来的额外复杂度 |

这些是本练习选定的保守起点，已在 2026-09-18 的 Qwen2.5-0.5B-Instruct 离线测试中通过，
不是性能最优配置。
内存预算不是实际精确占用，也不是跨进程的配额隔离。必要时根据启动日志调整。

## 7. 启动 vLLM API 服务并通过 SSH 访问

离线推理结束后，在远程前台启动：

```bash
cd /data/tianchi
set -o pipefail
bash practice_03_ascend_start/serve.sh 2>&1 | tee logs/qwen-small.log
```

服务绑定 `127.0.0.1:8000`。看到 `Application startup complete` 后，在本地另一终端建立隧道：

```bash
ssh -F ~/.ssh/config -o ExitOnForwardFailure=yes \
    -N -L 127.0.0.1:8000:127.0.0.1:8000 ascend910
```

在本地第三个终端请求（也可在远程直接执行）：

```bash
curl --fail-with-body http://127.0.0.1:8000/v1/models

curl --fail-with-body http://127.0.0.1:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{
      "model": "qwen-small",
      "messages": [{"role": "user", "content": "请用三句话解释 KV cache。"}],
      "temperature": 0,
      "max_completion_tokens": 128
    }'
```

`qwen-small` 是 `--served-model-name` 定义的服务别名，必须与请求对应。
在运行服务的终端按 Ctrl-C 停止，再关闭隧道。网关的端口转发能力本轮未实测；
若报 `administratively prohibited`，需检查算力平台的隧道入口要求。
本地 8000 被占用时，把隧道左侧端口改为 18000，并访问本地 18000。

## 8. 建议的后续练习与排错

先比较单请求输出，再比较两个请求的批处理。之后每次只调整一个变量：输入长度、输出长度、
并发数或 eager/graph 模式。记录 TTFT（首 token 延迟）、TPOT（后续 token 平均耗时）、
tokens/s 和峰值 HBM。将首次加载/编译与预热后的推理分开计时；小模型能用于理解机制，
不能直接代表大模型吞吐量。

| 现象 | 优先检查 |
| --- | --- |
| `Alarm`、AIC 错误或算子超时 | 先复查硬件健康和故障码；当前 `80C98001` 应交给平台处理 |
| `No module named torch_npu` | `which python` 是否为已安装环境 |
| `libascendcl.so` / `libatb.so` 缺失 | CANN / NNAL 的环境脚本是否加载 |
| NPU 数量 0 或编号错误 | 容器设备映射、可见性变量；本实例用逻辑 0 |
| 访问 Hugging Face 失败 | 检查 SDK 的 `HF_ENDPOINT` 和实际镜像网络；本次镜像连接也超时，可改用 ModelScope |
| `config.json` 存在但加载失败 | 检查权重、tokenizer 是否完整下载，检查首次异常 |
| OOM / KV cache 空间不足 | 查其他进程；降低长度、并发；结合日志调整实例内存预算 |
| 图编译失败 | 先保持 `--enforce-eager`；功能通过后再研究图模式 |
| API 无法访问 | 先远程 curl，再查 SSH 隧道；确认服务启动完成 |

继续阅读：[TorchNPU 官方项目](https://github.com/Ascend/pytorch)、
[Triton Ascend 入门](https://github.com/triton-lang/triton-ascend/blob/main/docs/en/quick_start.md)。
这些项目的 main 文档会变化；尤其 Triton 新文档可能针对 CANN 9.1，不能直接照抄升级当前 9.0 环境。
