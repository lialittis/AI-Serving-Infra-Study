# Practice 03：Ascend 910B2C 推理与并发实验

本实验已完成：NPU 基础算术、Transformers 单次生成、vLLM 离线批处理、HTTP API 请求，
以及固定输入/输出长度下的并发 1 与 4 对照。实测输出吞吐为 94.06 → 337.66 token/s，
约 3.59 倍；具体结论及适用范围见 [RESULTS.md](RESULTS.md)。

本地目录：`AI-Serving-Infra-Study/practice_03_ascend_start/`。
远端副本：`/data/tianchi/practice_03_ascend_start/`（工作目录副本，不是独立 Git 仓库）。
代码和资料在两边保持一致；模型权重仅存于远端，未复制进仓库。

## 1. 环境与模型

| 项目 | 本实验环境 |
| --- | --- |
| 登录 | `ssh ascend910`；如本地系统 SSH 配置权限报错，用 `ssh -F ~/.ssh/config ascend910` |
| 远端工作目录 | `/data/tianchi` |
| 模型 | `/data/huggingface_home/hub/Qwen2.5-0.5B-Instruct` |
| NPU | 1 × Ascend 910B2C，64 GiB；物理卡 5，程序中逻辑设备 `npu:0` |
| 系统 | openEuler 24.03 LTS-SP3，x86_64 |
| Python | `/usr/local/python3.12.13/bin/python`，3.12.13 |
| CANN | 9.0.0，`/usr/local/Ascend/cann-9.0.0` |
| torch / torch-npu | 2.10.0+cpu / 2.10.0 |
| vLLM / vLLM Ascend | 0.21.0+empty / 0.21.0rc1 |
| Triton / Triton Ascend | 3.2.0 / 3.2.1 |
| Transformers | 5.5.4 |

完整包版本、源码 commit、关键源码 SHA256，以及模型文件 SHA256 在
[environment.json](results/2026-09-18/environment.json)。该快照在压测后采集，
不是压测进程启动时的完整环境转储。9 月 17 日记录的 ModelScope 为 1.37.1，
9 月 18 日采集时为 1.40.1；本次推理和压测使用本地模型，不依赖 ModelScope 下载。
整理时运行采集脚本得到的附加快照见
[environment-after-packaging.json](results/2026-09-18/environment-after-packaging.json)。

沿用已有环境，无需升级依赖。设备告警及网络调查保留在 [ENVIRONMENT_NOTES.md](ENVIRONMENT_NOTES.md)，
本轮按用户要求继续功能测试，没有更改驱动或执行设备复位。

## 2. 准备远端终端

以下运行命令均在远端执行。每个新终端先运行：

```bash
cd /data/tianchi
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export PATH=/usr/local/python3.12.13/bin:$PATH
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p logs
```

已上传模型不必再次下载。默认路径已写入脚本；也可以通过 Python 的 `--model` 或 shell
脚本第一个位置参数改用其他路径。不要同时启动离线推理与服务端，以免两者争用 NPU。

## 3. 依次复现功能测试

先做基础 NPU 运算检查：

```bash
python practice_03_ascend_start/check_npu.py
```

预期出现 `NPU_MATMUL_PASS`。然后运行 Transformers：

```bash
set -o pipefail
python -u practice_03_ascend_start/transformers_generate.py \
  --prompt '请用一句话介绍你自己。' --max-new-tokens 64 \
  2>&1 | tee logs/transformers-replay.log
```

预期打印 `Model device: npu:0`、新 token 数和生成文本。原始输出在
[Transformers 日志](results/2026-09-18/logs/transformers-smoke-2026-09-18.log)。
进程退出后运行 vLLM 离线批处理：

```bash
python -u practice_03_ascend_start/vllm_generate.py \
  2>&1 | tee logs/vllm-replay.log
```

预期两个 Question/Answer 和正常退出。`vllm_generate.py` 已在导入 vLLM 前强制使用
`VLLM_WORKER_MULTIPROC_METHOD=spawn`。详细运行记录见 [RUN_2026-09-18.md](RUN_2026-09-18.md)。

## 4. 启动 API 服务

终端 A：

```bash
set -o pipefail
bash practice_03_ascend_start/serve.sh 2>&1 | tee logs/server-replay.log
```

保持前台进程运行，直到出现 `Application startup complete`。服务配置为：

| 参数 | 值 |
| --- | --- |
| 监听地址 / 模型别名 | `127.0.0.1:8000` / `qwen-small` |
| TP / dtype | 1 / BF16 |
| 最大上下文 / 调度序列数 | 2048 / 4 |
| 每步 token 预算 | 2048 |
| 实例内存使用比例 | 0.3 |
| 执行方式 | eager，worker 使用 spawn |

终端 B（同一远端实例）：

```bash
curl --fail-with-body http://127.0.0.1:8000/v1/models
curl --fail-with-body http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen-small","messages":[{"role":"user","content":"请用一句话介绍你自己。"}],"max_completion_tokens":64,"temperature":0}'
```

预期 HTTP 200 且 `choices[0].message.content` 有文本。用户已确认本步骤通过；
原始聊天 HTTP 响应未单独保存。若从本地访问，可另外建立 SSH 隧道：

```bash
# 本地终端，保持运行；性能实验本身仍在远端执行
ssh -F ~/.ssh/config -N -L 127.0.0.1:8000:127.0.0.1:8000 ascend910
```

## 5. 复现并发对照实验

保持终端 A 的服务运行，在终端 B 执行：

```bash
bash practice_03_ascend_start/benchmark_concurrency.sh
```

脚本先检查 `qwen-small` 服务，再采集环境和模型指纹，随后顺序运行两组实验：

| 配置 | 并发 1 | 并发 4 |
| --- | ---: | ---: |
| 输入 / 输出 token 数 | 128 / 64 | 128 / 64 |
| 正式请求数 / 预热请求数 | 20 / 2 | 20 / 2 |
| 随机种子 | 1 | 4 |
| 客户端最大并发 | 1 | 4 |
| 发起速率 / temperature | inf / 0 | inf / 0 |
| random range ratio | 0 | 0 |
| EOS | 忽略 | 忽略 |

随机输入由客户端生成，无需网络下载数据集。`--backend openai` 表示 API 协议，
请求发往本机 `/v1/completions`，不是请求外部服务。
`request-rate=inf` 表示在并发限制内尽快发起请求，不代表 NPU 同时处理 20 个请求。

每次生成独立目录 `/data/tianchi/benchmark_results/concurrency-XXXXXXXX/`，保存：

- `commands.sh`：实际执行命令，可审阅完整参数。
- `environment.json`：环境、设备信息、模型指纹。
- `concurrency-1.json`、`concurrency-4.json`：原始统计及逐请求明细。
- 同名 `.log`：命令日志。
- `summary.md`：两组比较表。

也可指定模型和**尚不存在**的输出目录（父目录须存在）：

```bash
bash practice_03_ascend_start/benchmark_concurrency.sh \
  /data/huggingface_home/hub/Qwen2.5-0.5B-Instruct \
  /data/tianchi/benchmark_results/my-new-run
```

原始实验没有开启 `--save-detailed`；新脚本额外保存逐请求明细，不改变请求参数。
脚本不重启或关闭服务；完成全部实验后，在终端 A 按 Ctrl-C 停止。

## 6. 本地查看结果，无需 NPU

在本地仓库根目录执行：

```bash
python3 practice_03_ascend_start/summarize_results.py
```

默认读取归档的原始两份 JSON，检查两组均成功、总输入 2560、总输出 1280，再生成比较表。
也可传入新的结果目录。结论及统计解释见 [RESULTS.md](RESULTS.md)。
复现意味着重现步骤与趋势，不承诺延迟数值逐位一致。

本地和远端均可在实验目录内执行 `sha256sum -c SHA256SUMS` 检查归档完整性。
清单不包含 Python 缓存或用户自己添加的文件（例如远端 `example.py`）。

## 7. 已解决的问题

| 问题 | 实际原因与修正 |
| --- | --- |
| NPU 在子进程初始化失败 | 默认 fork 不适用；脚本使用 spawn 并保留 main 入口保护 |
| 输入长度最小值为 0 | `--random-range-ratio 1` 错误；固定长度应设为 0 |
| `Connection refused` | API 服务已退出；保持终端 A 运行再执行客户端 |
| Peak concurrent requests 高于上限 | 该字段按秒桶统计活跃过的请求，不等于瞬时并发 |

输入长度 128 / 512 / 1536 的对照实验仅被提出，**尚未执行**，不属于本轮结论。
