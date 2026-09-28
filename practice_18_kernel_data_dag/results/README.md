# 本地原始证据

`2026-09-28-run01`：dispatcher / storage generation 基础采集。

`2026-09-28-run02`：增加 CPU 整数 staging values。

`2026-09-28-run03`：增加同步路径 positions 源码等价契约。

`2026-09-28-run04`：进入 Python 自定义算子内部重启 dispatcher 观测，补齐 FIA 临时
输出到最终 output 的拷贝来源。这是正式报告的数据来源；前三轮仅作采集排错证据，
其并行度数字不能作为最终结论。

这些目录包含 profiler、tensor 地址、服务器日志、环境和安装源码，仅留在本地工作区
与授权实验服务器，不纳入 git。可分享结果位于上一层 `report/`。
