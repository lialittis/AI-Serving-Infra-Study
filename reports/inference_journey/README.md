# 大模型推理流程：阶段性 HTML 报告

在本地浏览器打开 [index.html](index.html)。GitHub 的文件页面不会直接执行 HTML；
请先在本地打开仓库中的文件，或从仓库根目录启动静态服务：

```bash
python3 -m http.server 8080 --bind 127.0.0.1
# 浏览器打开 http://127.0.0.1:8080/reports/inference_journey/
```

报告包含：

- 服务启动、prefill、decode、结束释放四个阶段的交互时序图，可逐步播放、拖动和点击。
- 磁盘 / CPU / NPU 的数据分布，以及 token → block → slot 的映射演示。
- 模型计算、ATen / torch-npu / ATB / 自定义 C++ / Triton 实现、CANN 下发的关系。
- 主机队列、设备 stream、异步执行和结果同步的实测示例。
- FX 图、原地修改与存储别名、设备图捕获 / 重放的区别。
- 独立实验的证据入口、结论和未覆盖的问题。

HTML 内嵌 CSS、JavaScript 与数据，不依赖网络、CDN 或前端服务；单独复制 HTML 也能使用交互功能。
本地证据链接需要保留仓库目录结构，静态图下载需要保留旁边的 SVG / Mermaid 文件。
浏览器打印可保存正文和当前所选时序图为 PDF。完整静态图另见：
[启动](sequence-startup.svg)、[prefill](sequence-prefill.svg)、
[decode](sequence-decode.svg)、[结束](sequence-finish.svg)；
[sequence.mmd](sequence.mmd) 是四个阶段的 Mermaid 源码。

## 来源与边界

以 Practice 11 的 126 输入 / 2 输出请求为主线；完整请求路径、真实 KV 数值核对、
主机队列和全算子路径分别引用 Practice 07、08、09；模型图引用 Practice 10 / 11。
这些是不同运行，时序图是依据证据整理的教学图，不是拼接后的原始 trace，也不按时间比例绘制。
主线未启用 prefix caching、chunked prefill、async scheduling 或多请求并发。

每步附有存储说明、证据类别与来源。明确区分已观测数据、源码解释、计算推导与未观测行为；
特别保留 P11 中重放内部 kernel 无法逐一关联 FX 节点的缺口。
本报告没有重新启动推理、连接远端或改变实验状态。

## 重建

在仓库根目录执行（Python ≥ 3.7，仅标准库）：

```bash
python3 reports/inference_journey/build_report.py
```

编辑 `report.template.html` 修改正文 / 样式 / 交互；编辑 `build_report.py` 修改步骤说明和来源。
构建器从现有结构化结果提取统计、设备任务，内嵌到 `index.html`，并生成四张 SVG、Mermaid 源码
和 [evidence_manifest.json](evidence_manifest.json)。后者记录输入文件的路径与 SHA256。
所有链接均相对仓库，生成过程不访问服务器，不依赖 NPU。
