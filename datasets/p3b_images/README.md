# P3b 图像输入

两张 Qwen 官方公开示例照片，保留下载原始字节，用于视觉编码与另一请求语言计算的并发实验。

| 文件 | 内容 | 原始尺寸 | 来源 |
|---|---|---|---|
| `beach.jpeg` | 海滩上的人物与狗 | 2048 × 1365 | [Qwen-VL 示例](https://github.com/QwenLM/Qwen-VL/blob/master/README.md) |
| `beijing.jpeg` | 北京城市建筑 | 614 × 410 | [Qwen-VL 图像目录](https://github.com/QwenLM/Qwen-VL/tree/master/assets/mm_tutorial) |

远端目录：`/data/tianchi/datasets/p3b_images/`。下载地址、字节数与 SHA256 见 [manifest.json](manifest.json)。图片经过完整 JPEG 解码检查；后续由模型 processor 控制分辨率并记录实际视觉 token 数，保留原图用于复现。

模型目录：`/data/huggingface_home/hub/Qwen2.5-VL-3B-Instruct`。已确认目录包含权重分片、索引、tokenizer、processor 配置及聊天模板；模型运行资格仍需另行测试。
