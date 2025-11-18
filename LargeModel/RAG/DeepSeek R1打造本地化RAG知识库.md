如何使用大模型高效地管理和利用知识，同时解决大模型专业应用方向的能力，成为了迫切需要解决的问题。RAG（Retrieval-Augmented Generation）增强信息检索和生成模型，能够从大规模知识库中检索相关信息并生成高质量的反馈。

本文将详细介绍如何使用ollama、Deepseek R1大语言模型、Nomic-Embed-Text向量模型和CherryStudio共同搭建一个本地的私有RAG知识库。

# 1. 准备工作

## 1.1. 什么是RAG？

RAG（Retrieval-Augmented Generation）是一种结合了信息检索和大模型（LLM）的技术，在对抗大模型幻觉、高效管理用户本地文件以及数据安全保护等方面具有独到的优势。

> - Retrieval，英`[rɪˈtriːvl]` 美`[rɪˈtriːvl]`：检索

<div align=center>
    <img src=./imgs_markdown/2025-11-17-20-10-30.png
    width=50%></br><center></center>
</div>

主要包括：

- **索引**：将文档库分割成较短的Chunk，并通过编码器构建向量索引。
- **检索**：根据问题和chunks的相似度检索相关文档片段。
- **生成**：以检索到的上下文为条件，生成问题的回答。在开始之前，确保我们需要使用的工具和库：

在开始之前，确保我们需要使用的工具和库：
- Ollama
- DeepseekR1 LLM模型
- Nomic-Embed-Text 向量模型
- Cherry Studio

## 1.2. 安装ollama

[ollama官网](https://ollama.com/download)

安装完成之后，打开终端或命令提示符，输入`ollama--version`，确保安装成功。

---

可以通过`ollama --help`获取ollama常用的命令：

| 命令    | 功能描述                                                                          |
| ------- | --------------------------------------------------------------------------------- |
| serve   | 启动 Ollama 服务，为模型运行、管理等操作提供基础环境支持                          |
| create  | 创建自定义模型，支持基于现有模型微调或通过配置文件构建新模型                      |
| show    | 查看指定模型的详细信息，包括模型版本、大小、描述等元数据                          |
| run     | 运行已下载或创建的模型，启动交互式对话或完成指定任务                              |
| stop    | 停止当前正在运行的模型进程，释放占用的系统资源                                    |
| pull    | 从模型仓库（registry）下载模型到本地，支持指定模型版本                            |
| push    | 将本地模型上传至模型仓库，方便共享或跨设备使用                                    |
| signin  | 登录 ollama.com 账号，关联云端服务（如模型同步、云端运行等）                      |
| signout | 退出当前登录的 ollama.com 账号，解除本地与云端的关联                              |
| list    | 列出本地已下载或创建的所有模型，显示模型名称、版本等关键信息                      |
| ps      | 列出当前正在运行的模型进程，包含进程 ID、模型名称、占用资源等状态                 |
| cp      | 复制现有模型，可用于创建模型备份或衍生新的模型实例（支持重命名）                  |
| rm      | 删除本地模型，释放存储空间（删除后不可恢复，需谨慎操作）                          |
| help    | 查看指定命令的详细用法，使用格式为 `ollama help [命令名]`（如 `ollama help run`） |


## 1.3. 配置DeepSeek R1模型

- 【下载模型】从Ollama的官方网站下载DeepSeekR1模型文件：

```bash
ollama pull deepseek-r1:7b
```

- 【启动模型】启动和下载模型是同一个命令，如果没有下载过的新模型会直接下载，以及下载过的则直接启动。

```bash
ollama run deepseek-r1:7b
```

## 1.4. 配置Nomic-Embed-Text模型

- 【下载模型】还是使用ollama进行下载：

```bash
ollama pull nomic-embed-text
```

## 1.5. 安装CherryStudio

CherryStudio目前支持市面上绝大多数服务商的集成，并且支持多服务商的模型统一调度。

> 下载地址：[CherryStudio](https://www.cherry-ai.com/download)


现在已经安装并配置好了ollama、Deepseek R1、Nomic-Embed-Text和cherry studio，接下来我们将它们结合起来搭建一个本地的私有RAG知识库。

## 1.6. 【方式1】使用Cherry Studio搭建RAG本地知识库

### 1.6.1. 数据准备

首先，你需要准备一个知识库数据集。这个数据集可以是一个包含大量文档的目录，也可以是一个预处理的JSON文件。确保每个文档都有一个唯一的ID和文本内容。
我们准备一个凡人修仙传的小说文档。

> 《凡人8万字大纲.pdF》可以在网上搜索到（免费下载的）

### 1.6.2. 构建索引

在CherryStudio中使用Nomic-Embed-Text将知识库中的文档转换为向量表示，并构建一个索引。

<div align=center>
    <img src=./imgs_markdown/2025-11-18-09-03-28.png
    width=100%></br><center></center>
</div>

### 1.6.3. 检索相关信息

使用DeepSeek R1和检索本地向量数据库。

如果我们直接去问DS，那么回答是这样的，与我们的预期不符，这是因为我们还没有给DS添加知识库。

<div align=center>
    <img src=./imgs_markdown/2025-11-18-09-40-42.png
    width=100%></br><center></center>
</div>

那么我们添加完知识库后，需要开启一个新的话题，不然DS会有上下文从而影响效果。

<div align=center>
    <img src=./imgs_markdown/2025-11-18-09-46-16.png
    width=100%></br><center></center>
</div>

<div align=center>
    <img src=./imgs_markdown/2025-11-18-09-46-30.png
    width=100%></br><center></center>
</div>

<div align=center>
    <img src=./imgs_markdown/2025-11-18-09-47-30.png
    width=100%></br><center></center>
</div>

我们可以看到，DS的回答其实并不好。这里只是展示了一个流程，具体效果我们后续再进行优化。

## 1.6. 【方式2】使用Dify搭建RAG本地知识库

前面我们刚刚讲了CherryStudio实现的本地知识库大模型应用，对于很多朋友来说都是零代码的，很容易上手，但是功能相对少和不够灵活。今天我们给大家讲另外一个方式实现本地知识库的AI助手，Dify开源平台。

https://www.bilibili.com/video/BV1gtCMBEE5M?spm_id_from=333.788.player.switch&vd_source=c96423a5c3cff0232795a1e42972c3d4&p=2




























# 2. 知识来源

- [【全网首推】Deepseek R1打造本地RAG知识库，B站最细本地知识库搭建教程，全程干货，内容通俗易懂，小白也能轻松学会！！！](https://www.bilibili.com/video/BV1gtCMBEE5M)