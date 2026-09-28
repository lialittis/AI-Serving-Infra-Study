发现了一个会影响方案的兼容性问题：Ascend 的 NPUOffloadingSpec 仍引用旧版 vLLM 接口，
但当前 vLLM 已删除这些模块，并修改了缓存注册接口。因此，文件虽在，原配置不能直接运行。

继续检查后发现了更合适的入口：当前安装版本还注册了 AscendSimpleCPUOffloadConnector，
已经处理了 Ascend 独立的 K/V 存储，并使用后台拷贝线程和两条传输 stream。
旧 NPUOffloadingSpec 的不兼容，不代表整个环境没有可用的 offload 路径
