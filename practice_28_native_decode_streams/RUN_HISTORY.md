# 运行记录

所有目录均位于远端 `/data/tianchi/practice_28_native_decode_streams/results/`。本地同名 `*-archive` 保存便携证据；各次失败均未覆盖或混入正式统计。

| 运行 | 用途及结果 | 恢复 |
|---|---|---|
| `qualification-01` | KV 为 byte storage 的 BF16 view，直接 deepcopy 失败；改为显式值克隆并保留重复引用 | 通过 |
| `qualification-02` | eager 34 项通过；用真实 Attention metadata 重捕获误入 FULL graph task-group 路径，报 107029；改为原生 dummy capture 与固定输入地址 | 通过 |
| `qualification-03` | 在 inference_mode 中创建第二模型参数，torch-npu format cache 需要的版本计数不可用；模型构建改用 no_grad | 通过 |
| `qualification-04` | eager 34 项通过；graph 输出需在 inference_mode 中原地写入；仅执行阶段对齐原生推理上下文 | 通过 |
| `suite-01` | 完整成功套件：288 个正式 pair，32 条诊断 trace；每个执行子进程先通过 34 项资格检查 | 通过 |
| `qualification-final` | 扩展 buffer 地址检查发现原生共享的 RoPE 固定查找表；因尚未分类为只读而停止 | 通过 |
| `qualification-final-02` | 直接查阅远端读取／写入源码，单独审计固定 RoPE 表；两后端各 34 项通过，前后表哈希一致 | 通过 |

`qualification-02` 的失败发生后，native 析构未退出；核对 launcher 记录及命令行后，仅向本实验自有进程组发送 TERM。之后控制器增加了“失败状态已写出但析构挂起”的自动清理路径。没有设备 reset，也没有操作其他服务。

`qualification-01` 仅保存当时的脚本哈希与日志；从 `qualification-02` 起保存完整冻结的 collector。正式 `suite-01` 另有最终 `analysis_tools` 及辅助函数快照，32 条 trace 在本地离线重分析后结果逐文件哈希完全一致。

最终增强审计修改了检查范围与控制器的错误记录，不改变 `settings`、状态采集、capsule 输入恢复／提交、第二 runner 构建、pair 计时流程或测量驱动。相应 AST／文件一致性结果见 `results/final_audit.json`。正式采集仍按其冻结版本复现，不将后来的审计代码冒充采集时版本。
