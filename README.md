# 📚 KnowledgeHub 知识库

[![GitHub stars](https://img.shields.io/github/stars/Le0v1n/KnowledgeHub)](https://github.com/Le0v1n/KnowledgeHub/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/Le0v1n/KnowledgeHub)](https://github.com/Le0v1n/KnowledgeHub/network)

> 这个仓库存放了日常的学习笔记，欢迎大家来访！如果你有疑问请 [联系我](#5-联系我) 😊 如果对你有帮助，请 ⭐ 一下

## 📋 目录导航

<details>
<summary><strong>点击展开目录</strong></summary>

- [1. 计划与完成情况](#1-计划与完成情况)
- [2. 简介](#2-简介)
- [3. 仓库结构](#3-仓库结构)
- [4. 其他说明](#4-其他说明)
- [5. 联系我](#5-联系我)

</details>

## 1. 计划与完成情况

### 📅 2025

```mermaid
gantt
	dateFormat YY-MM-DD
	axisFormat %y.%m.%d
	title 2025计划与记录

	✅Ultralytics源码学习: done, 25-01-01, 4w
	✅学习mermaid语法: done, 2025-01-07, 1w
	✅学习vim语法: done, 2025-01-13, 1d
	✅学习VSCode注释高亮语法: done, 2025-01-17, 1d
	✅BLIP2学习: done, after a1, 2w
	✅GLIP学习: done, after a2, 1w
	✅GroundDINO学习: done, after a3, 1w
	✅Lora学习: done, after a4, 1w
	✅QLora学习: done, after a5, 1w
	✅Bubo-GPT学习: done, after a6, 1w
	✅LLM多卡推理学习: done, after a7, 1w
	✅FlashAttention学习: done, after a8, 1w
	✅HuggingFace库学习: done, after a9, 1w
```

### 📅 2024

```mermaid
gantt
	dateFormat YY-MM-DD
	axisFormat %y.%m.%d
	title 2024计划与记录

	section 文档编写
	✅更新rich相关文档: done, 24-01-31, 1w
	✅编写tqdm.rich相关文档: done, 24-01-31, 1w
	✅编写知识蒸馏文档: done, 24-02-20, 1w
	✅学习glob: done, 2024/03/06, 1w

	section 脚本编写
	✅xml2yolo.py: done, 2024-06-05, 2024-06-07
	✅yolo2xml.py: done, 2024-06-05, 2024-06-08
	✅yolo2json.py: done, 2024-06-05, 2024-06-09
	✅json2yolo.py: done, 2024-06-05, 2024-06-09
	✅xml2json.py: done, 2024-06-05, 2024-06-09
	✅json2xml.py: done, 2024-06-05, 2024-06-09
	✅create_empty_labels.py: done, 2024-06-05, 2024-06-09
	✅create_dataset.py: done, 2024-06-05, 2024-06-09
```

## 2. 简介

这个仓库存放了日常的学习笔记，内容涵盖：

| 领域 | 内容 |
|------|------|
| 🤖 大模型 | LangChain、RAG、Prompt工程、Transformer |
| 👁️ 计算机视觉 | 目标检测、语义分割、图像分类、人脸识别、人体姿态估计 |
| 🧠 深度学习 | PyTorch、模型部署、模型量化 |
| 🐍 Python 编程 | 编程技巧、常用库 |
| 🐧 Linux 系统 | Shell脚本、Git |
| 📦 模型部署 | ONNX、模型转换 |

更多内容请见：
- 📖 [CSDN 博客-Le0v1n](https://blog.csdn.net/weixin_44878336)：这里有很多有趣的内容
- 🎬 [Bilibili 视频-L0o0v1N](https://space.bilibili.com/13187602)：这里有视频版内容

## 3. 仓库结构

### 3.1. LargeModel → 大模型相关

<details>
<summary><strong>点击展开详细内容</strong></summary>

- [📂nanobot](./LargeModel/nanobot/nanobot)：nanobot 相关内容
- [📂RAG](./LargeModel/RAG)：RAG 学习笔记
- [📂Transformer](./LargeModel/Transformer)：Transformer 相关内容
- [📂code](./LargeModel/code)：代码实现
- [📂CLI](./LargeModel/CLI)：命令行工具
- [📂CLIP](./LargeModel/CLIP)：CLIP 模型相关

</details>

### 3.2. ObjectDetection → 目标检测相关

<details>
<summary><strong>点击展开详细内容</strong></summary>

- [📂YOLOv5](./ObjectDetection/YOLOv5/)：YOLOv5 相关内容
- [📂YOLO-World](./ObjectDetection/YOLO-World)：YOLO-World
- [📂DETR](./ObjectDetection/DETR)：DETR 相关
- [📂Ultralytics](./ObjectDetection/Ultralytics)：Ultralytics 源码学习
- [📂Metrics](./ObjectDetection/Metrics)：评价指标

</details>

### 3.3. SemanticSegmentation → 语义分割相关

<details>
<summary><strong>点击展开详细内容</strong></summary>

- [📂Fast-SCNN](./SemanticSegmentation/Fast-SCNN)：Fast-SCNN 相关内容

</details>

### 3.4. Classification → 图像分类相关

<details>
<summary><strong>点击展开详细内容</strong></summary>

- [📂ViT](./Classification/ViT)：Vision Transformer 相关

</details>

### 3.5. CLIP → CLIP模型

<details>
<summary><strong>点击展开详细内容</strong></summary>

- [📂clip](./CLIP/clip)：CLIP 模型代码
- [📂notebooks](./CLIP/notebooks)：Jupyter notebooks

</details>

### 3.6. FaceRecognition → 人脸识别

<details>
<summary><strong>点击展开详细内容</strong></summary>

- 人脸识别相关学习笔记

</details>

### 3.7. HumanPoseEstimation → 人体姿态估计

<details>
<summary><strong>点击展开详细内容</strong></summary>

- 人体姿态估计相关学习笔记

</details>

### 3.8. Papers → 论文阅读

<details>
<summary><strong>点击展开详细内容</strong></summary>

- 论文阅读笔记

</details>

### 3.9. Quantization → 模型量化

<details>
<summary><strong>点击展开详细内容</strong></summary>

- 模型量化相关学习内容

</details>

### 3.10. PyTorch → PyTorch相关

<details>
<summary><strong>点击展开详细内容</strong></summary>

- PyTorch 相关学习笔记
- [如何精确统计模型推理时间](./PyTorch/如何精确统计模型推理时间)

</details>

### 3.11. Python → Python相关

<details>
<summary><strong>点击展开详细内容</strong></summary>

- [📂Registry](./Python/Registry)：Python注册机制
- [📂Rich-美化](./Python/Rich-美化)：Rich库相关内容
- [📂resolve_import_methods](./Python/resolve_import_methods)：import 问题解决
- [📂多线程与多进程](./Python/多线程与多进程)
- [📂正则表达式](./Python/正则表达式)
- [📂code](./Python/code)：代码实现

</details>

### 3.12. ONNX → ONNX相关

<details>
<summary><strong>点击展开详细内容</strong></summary>

- [📂code](./ONNX/code)：ONNX 相关代码

</details>

### 3.13. Linux → Linux相关

<details>
<summary><strong>点击展开详细内容</strong></summary>

- [📂shell](./Linux/shell)：Shell 脚本
- [📂Git](./Linux/Git)：Git 教程

</details>

### 3.14. Windows → Windows相关

<details>
<summary><strong>点击展开详细内容</strong></summary>

- Windows 使用技巧

</details>

### 3.15. Writing → 与写作相关的

<details>
<summary><strong>点击展开详细内容</strong></summary>

- [📂Office](./Writing/Office)：Office 技巧
- [📂code](./Writing/code)：写作相关代码

</details>

### 3.16. Datasets → 数据集

<details>
<summary><strong>点击展开详细内容</strong></summary>

- [📂VOCdevkit](./Datasets/VOCdevkit)：VOC 数据集
- [📂coco128](./Datasets/coco128)：COCO 128 数据集
- [📂Web](./Datasets/Web)：Web 数据集

</details>

### 3.17. Configs → 配置文件

<details>
<summary><strong>点击展开详细内容</strong></summary>

- [📂Typora Themes](./Configs/Typora Themes)：Typora 主题
- 各类配置文件

</details>

### 3.18. utils → 工具函数

<details>
<summary><strong>点击展开详细内容</strong></summary>

- [📂dataset](./utils/dataset)：数据集处理工具
- [📂data_processing](./utils/data_processing)：数据处理工具
- [📂onnx](./utils/onnx)：ONNX 工具
- [📂pdf](./utils/pdf)：PDF 工具

</details>

### 3.19. scirpts → 脚本工具

<details>
<summary><strong>点击展开详细内容</strong></summary>

- [📂data_process](./scirpts/data_process)：数据处理脚本

</details>

---

## 4. 其他说明

1. 因为 Github 仓库有最大容量限制，所以部分文章的图片引用来自 [我的 CSDN 博客](https://blog.csdn.net/weixin_44878336)。
2. 如果文章有问题（语法、链接错误、文字、版权等），请 [联系我](#5-联系我)。

---

## 5. 联系我

| 联系方式 | 链接 |
|---------|------|
| 📧 发邮件 | [zjkljd@163.com](mailto:zjkljd@163.com) |
| 💬 CSDN私信 | [Le0v1n](https://blog.csdn.net/weixin_44878336) |
| ❓ 新建Issue | [GitHub Issues](https://github.com/Le0v1n/KnowledgeHub/issues/new/choose) |

---

<div align="center">

⭐ Star me if you find this helpful!

</div>
