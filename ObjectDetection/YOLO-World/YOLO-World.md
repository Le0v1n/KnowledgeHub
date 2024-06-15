# 1. 前言

YOLO-World 是一种基于 YOLO系列目标检测器的创新方法，它通过视觉语言建模和大规模数据集的预训练，增强了 YOLO 的开放词汇检测能力。具体来说，YOLO-World 引入了一种新的可重新参数化的视觉-语言路径聚合网络（RepVL-PAN）和区域-文本对比损失，以促进视觉和语言信息之间的交互。这种方法在零样本（zero-shot）情况下高效地检测各种物体，并且在具有挑战性的 LVIS 数据集上表现出色，实现了高准确率和高速度的检测。

YOLO-World 的核心创新点包括：
- 实时解决方案：利用 CNN 的计算速度，提供快速的开放词汇检测解决方案。
- 效率和性能：在不牺牲性能的前提下降低计算和资源需求，支持实时应用。
- 利用离线词汇进行推理：引入了 "先提示后检测" 的策略，使用预先计算的自定义提示来提高效率。
- 卓越的基准测试：在标准基准测试中，YOLO-World 的速度和效率超过了现有的开放词汇检测器。
- 应用广泛：YOLO-World 的创新方法为众多视觉任务带来了新的可能性。

## 1.1 零样本（zero-shot）

零样本学习（zero-shot learning）是机器学习领域中的一种技术，它允许模型在没有接受过特定类别训练数据的情况下，识别或预测这些类别。这通常通过利用模型对其他类别的已有知识来实现，或者通过某种形式的语义或属性描述来辅助模型理解新的类别。

## 1.2 LVIS数据集

LVIS（Large Vocabulary Instance Segmentation）数据集是由 Facebook AI Research (FAIR) 开发并发布的一个大规模细粒度词汇级标记数据集。这个数据集专门用于对象检测和实例分割的研究基准，它包含了超过1000类物体的约200万个高质量的实例分割标注，涵盖了164k大小的图像。

LVIS 数据集的特点包括：
1. **大规模和细粒度**：数据集覆盖了广泛的物体类别，提供了详尽的标注，包括小的、部分被遮挡的或难以辨认的对象实例。
2. **高质量标注**：与 COCO 和 ADE20K 数据集相比，LVIS 数据集的标注质量更高，具有更大的重叠面积和更好的边界连续性。
3. **长尾分布**：LVIS 数据集反映了自然图像中类别的Zipfian分布，即少数常见类别和大量罕见类别的长尾分布。
4. **评估优先的设计原则**：数据集的构建采用了评估优先的设计原则，即首先确定如何执行定量评估，然后设计和构建数据集收集流程以满足评估所需数据的需求。
5. **联合数据集**：LVIS 由大量较小的组成数据集联合形成，每个小数据集为单个类别提供详尽标注的基本保证，即该类别的所有实例都被标注。这种设计减少了整体的标注工作量，同时保持了评估的公平性。

LVIS 数据集的构建过程包括六个阶段：目标定位、穷尽标记、实例分割、验证、穷尽标注验证以及负例集标注。数据集的词汇表 V 是通过迭代过程构建的，从大型超级词汇表开始，并使用目标定位过程逐步缩小，最终确定包含 1723 个同义词的词汇表，这也是可以出现在 LVIS 中的类别数量的上限。

LVIS 数据集旨在推动计算机视觉领域中实例分割算法的研究，特别是在低样本学习（low-shot learning）的挑战下，如何有效地从少量示例中学习并识别和分割新的对象类别。数据集的发布为研究者提供了一个丰富的资源，以开发和评估能够处理大规模词汇量和长尾分布的先进算法。

# 1. 安装

目前YOLO-World有两个仓库：

1. [官方基于MMOpenLab实现](https://github.com/AILab-CVC/YOLO-World?tab=readme-ov-file)
2. [Ultralytics实现](https://github.com/ultralytics/ultralytics)

官方实现相比Ultralytics实现有更多的细节，因此这里我们使用官方基于MMOpenLab实现，具体安装请见[installation.md](https://github.com/AILab-CVC/YOLO-World/blob/master/docs/installation.md)。下面是我自己的安装过程：

```bash
# 安装虚拟环境
conda create -n yolo-world python=3.9
conda activate yolo-world

# 根据cuda版本安装PyTorch
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121

# 安装mmcv
pip install -U openmim
mim install mmcv

# 安装mmdet
# 下载mmdet项目压缩包
# 解压
7z x mmdetection-main.zip -o./
# 安装
pip install -e mmdetection-main

# 安装mmyolo
# 下载mmyolo项目压缩包
# 解压
7z x mmyolo-main.zip -o.
# 安装
 pip install -e mmyolo-main

# 安装其他依赖包
pip install opencv-python --upgrade
pip install opencv-python-headless --upgrade
pip install timm==0.6.13 transformers==4.36.2 albumentations
pip install gradio==4.16.0 supervision
pip install onnx onnxruntime onnxsim

# 安装yolo-world项目
pip install -e .
```

# 2. 数据准备

## 2.1 概况

YOLO-World的预训练模型采用了下表列出的几个数据集：

| Data         | Samples |    Type    | Boxes  | Description                                                         |
| :----------- | :-----: | :--------: | :----: | :------------------------------------------------------------------ |
| Objects365v1 |  609k   | detection  | 9,621k | 一个大规模的对象检测数据集，包含超过60万张图像和近1千万个边界框     |
| GQA          |  621k   | grounding  | 3,681k | 包含超过62万张图像和超过368万对问答对的数据集，用于视觉问答任务     |
| Flickr       |  149k   | grounding  |  641k  | 一个包含约14万张图像和641k问答对的数据集，用于视觉问答任务          |
| CC3M-Lite    |  245k   | image-text |  821k  | 一个包含24.5万图像-标题对的数据集，专注于跨模态匹配，共有821k个实例 |

其中：
- detection：指的是对象检测任务，算法需要识别图像中的对象并为它们绘制边界框。
- grounding：将自然语言描述与图像中的具体物体建立联系的过程。
- image-text：涉及将图像内容与相应的文本描述进行匹配的任务，可能包括图像标注、图像描述生成等。

<kbd><b>Question</b></kbd>：grounding和image-text有什么区别？

<kbd><b>Answer</b></kbd>：在视觉领域，"grounding"和"image-text"是两个相关但有所区别的概念：

1. **Grounding**：
   - 在视觉接地（Visual Grounding）任务中，"grounding"指的是将文本描述中的词汇或短语与图像中的具体物体或场景相匹配的过程。这通常涉及到理解和关联语言描述与视觉信息，以识别图像中与文本描述相对应的物体或区域。
   - Grounding任务可以视为一种跨模态的映射，<font color='blue'><b>它要求模型不仅要理解文本的含义，还要将这些文本与图像中的具体视觉实体关联起来</b></font>。

2. **Image-Text**：
   - "Image-Text"通常指的是图像和文本对，这种数据对可以用于多种任务，例如图像描述生成、视觉问答、图像检索等。在这些任务中，图像和文本并不是直接相互映射，而是作为一种多模态数据存在，用于训练和评估模型对视觉和语言信息的联合理解。
   - Image-Text任务更侧重于图像和文本之间的语义关联，可能<font color='blue'><b>不要求模型在图像中精确地定位与文本描述直接对应的物体或区域，而是更关注整体的语义一致性</b></font>。

总的来说，"grounding"更侧重于文本描述与图像中具体物体的精确匹配和定位，而"image-text"则是更广泛的概念，涵盖了图像和文本之间的各种语义关联任务，不一定要求精确的物体定位。

> 在YOLO-World这样的模型中，"grounding"能力使得模型能够根据文本描述检测图像中的物体，而"image-text"数据则可能用于模型的预训练，以提高对视觉和语言信息的联合理解能力。

## 2.2 数据集目录结构

YOLO-World项目的数据集都放入 `data` 目录中，如：

```
├── coco
│   ├── annotations
│   ├── lvis
│   ├── train2017
│   ├── val2017
├── flickr
│   ├── annotations
│   └── images
├── mixed_grounding
│   ├── annotations
│   ├── images
├── mixed_grounding
│   ├── annotations
│   ├── images
├── objects365v1
│   ├── annotations
│   ├── train
│   ├── val
```

## 2.3 数据集下载介绍与下载

| 数据集名称   | 领域      |    类别数    | 图片数量 | 说明                                                    |
| :----------- | :-------- | :----------: | :------: | :------------------------------------------------------ |
| Objects365v1 | detection |     365      |    2M    | 和传统的目标检测数据集一样                              |
| GQA          | grounding | 无明确的类别 |   148k   | 是一个用于问答的数据集                                  |
| Flickr30k    | grounding | 无明确的类别 |   31k    | 每张图片都有5个captions和一系列的bbox（实体版才有bbox） |
| LVIS         | grounding |     1203     |   160k   | 每个类别都会有一个描述语句                              |

### 2.3.1 Objects365v1

- 论文链接：[Objects365: A Large-scale, High-quality Dataset for Object Detection](http://openaccess.thecvf.com/content_ICCV_2019/papers/Shao_Objects365_A_Large-Scale_High-Quality_Dataset_for_Object_Detection_ICCV_2019_paper.pdf)

- 说明：Objects365就是一个传统的目标检测数据集，图片/类别没有对应的描述性文本，但单独的类别可以有近义词。

<a></a>
<div align=center>
    <img src=./imgs_markdown/2024-06-14-16-49-56.png
    width=100%>
    <center>Objects365v1数据集示例</center>
</div></br>

### 2.3.2 GQA

- 论文链接：[GQA: A New Dataset for Real-World Visual Reasoning and Compositional Question Answering](https://arxiv.org/abs/1902.09506v3)
- 说明：GQA数据集其实不是用于目标检测的，它本质上是一个问答数据集，就是说它的每一张图片都有一些列问题以及对应的答案。

<a></a>
<div align=center>
    <img src=./imgs_markdown/2024-06-14-16-56-16.png
    width=100%>
    <center>GQA数据集示例</center>
</div></br>

### 2.3.3 Flickr30k

- 论文链接：[Flickr30k Entities: Collecting Region-to-Phrase Correspondences for Richer Image-to-Sentence Models](https://arxiv.org/abs/1505.04870)

- 说明：原始的Flickr30k数据集包含了31,783张图片，每张图片配有5个由人类标注者提供的参考句子（captions），共计158,915个句子。随后，有研究者为了进一步提升数据集的多模态研究价值，在Flickr30k数据集的基础上进行了扩展，创建了Flickr30k Entities数据集。Flickr30k Entities数据集在原有的图片和句子的基础上增加了244,000个共指链（coreference chains）和276,000个手动标注的边界框（bounding boxes）。这些边界框与图片中提及的实体相对应，极大地丰富了数据集的语义信息，为图像描述、视觉问答等任务提供了更丰富的标注资源。

<a></a>
<div align=center>
    <img src=./imgs_markdown/2024-06-14-16-32-48.png
    width=100%>
    <center>Flickr30k entities数据集示例</center>
</div></br>

### 2.3.4 LVIS

- 论文链接：[LVIS: A Dataset for Large Vocabulary Instance Segmentation](https://arxiv.org/abs/1908.03195)
- 说明：LVIS和传统的目标检测数据集不同的是：
  - LVIS每个object的坐标是多边形
  - LVIS为每个类别都定义了一段文字用来描述该类别
  - 单独的类别可以有近义词

<a></a>
<div align=center>
    <img src=./imgs_markdown/2024-06-14-17-05-06.png
    width=100%>
    <center></center>
</div></br>

<a></a>
<div align=center>
    <img src=./imgs_markdown/2024-06-14-17-02-30.png
    width=100%>
    <center></center>
</div></br>

### 2.3.5 数据集下载地址

| 数据集名称 | 图片下载地址 | 标签下载地址 |
| :--- | :------| :-------------- |
| Objects365v1 | [Objects365 train](https://opendatalab.com/OpenDataLab/Objects365_v1) | [objects365_train.json](https://opendatalab.com/OpenDataLab/Objects365_v1) |
| MixedGrounding | [GQA](https://nlp.stanford.edu/data/gqa/images.zip) | [final_mixed_train_no_coco.json](https://huggingface.co/GLIPModel/GLIP/blob/main/mdetr_annotations/final_mixed_train_no_coco.json) |
| Flickr30k | [Flickr30k](https://shannon.cs.illinois.edu/DenotationGraph/) |[final_flickr_separateGT_train.json](https://huggingface.co/GLIPModel/GLIP/blob/main/mdetr_annotations/final_flickr_separateGT_train.json) |
| LVIS-minival | [COCO val2017](https://cocodataset.org/) | [lvis_v1_minival_inserted_image_name.json](https://huggingface.co/GLIPModel/GLIP/blob/main/lvis_v1_minival_inserted_image_name.json) |


## 2.4 数据集类别

> 对于在Close-set（传统的目标检测数据集）目标检测上微调YOLO-World，建议使用多模态数据集。

### 2.4.1 设置类别

如果您使用 `COCO-format` 自定义数据集，则“不需要”为自定义词汇表/类别定义数据集类。通过 `metainfo=dict(classes=your_classes)`, 在配置文件中显式设置 `CLASSES` 很简单：

```python
coco_train_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(
        type='YOLOv5CocoDataset',
        metainfo=dict(classes=your_classes),  # 这里需要填写具体的类别数
        data_root='data/your_data',  # 数据集ROOT
        ann_file='annotations/your_annotation.json',  # 标签文件（相对ROOT的路径）
        data_prefix=dict(img='images/'),  # 图片所在文件名称（相对ROOT的路径）
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path='data/texts/your_class_texts.json',  # 类别对应的文本文件路径（绝对路径）
    pipeline=train_pipeline)
```

为了训练YOLO-World，我们主要采用两种数据集类：

### 2.4.2 多模态数据集（MultiModalDataset）

`MultiModalDataset` 是预定义数据集类的简单包装器，例如 Objects365 或 COCO ，它将文本（类别文本）添加到数据集实例中以进行格式化输入文本。

`.json` 文件格式如下：

```json
[
    ['A_1','A_2'],
    ['B'],
    ['C_1', 'C_2', 'C_3'],
    ...
]
```

其中：
- `"A_1"`和`"A_2"`是一个类别，二者为近义词，可以表示这一个类别。
- `"B"`：是一个类别，它没有近义词。
- `"C_1", "C_2", "C_3"`是一个类别，三者为近义词，均表示这个类别。

#### LVIS的json文件格式示意（有近义词）：

```json
[
    [
        "aerosol can",
        "spray can"
    ],
    [
        "air conditioner"
    ],
    [
        "airplane",
        "aeroplane"
    ],
    [
        "alarm clock"
    ],
    [
        "alcohol",
        "alcoholic beverage"
    ],
    [
        "...",
        "有近义词！！！！！！"
    ],
    [
        "yogurt",
        "yoghurt",
        "yoghourt"
    ],
    [
        "yoke",
        "yoke animal equipment"
    ],
    [
        "zebra"
    ],
    [
        "zucchini",
        "courgette"
    ]
]
```

#### COCO的json文件格式示意（没有近义词）：

```json
[
    [
        "person"
    ],
    [
        "bicycle"
    ],
    [
        "car"
    ],
    [
        "motorcycle"
    ],
    [
        "airplane"
    ],
    [
        "没有近义词！！！！！"
    ],
    [
        "teddy bear"
    ],
    [
        "hair drier"
    ],
    [
        "toothbrush"
    ]
]
```

#### Object365V1的json文件格式示意（有近义词）：

```json
[
    [
        "person"
    ],
    [
        "sneakers"
    ],
    [
        "chair"
    ],
    [
        "hat"
    ],
    [
        "lamp"
    ],
    [
        "bottle"
    ],
    [
        "cabinet",
        "shelf"
    ],
    [
        "...",
        "有近义词！！！！！"
    ],
    [
        "iron"
    ],
    [
        "flashlight"
    ]
]
```

### 2.4.3 混合文本对齐数据集（YOLOv5MixedGroundingDataset）

`YOLOv5MixedGroundingDataset` 通过支持从 `json` 文件加载文本/标题来扩展 `COCO` 数据集。它是为 `MixedGrounding` 或 `Flickr30K` 设计的，每个对象都有文本标记。

### 2.4.4 自定义数据集

对于自定义数据集，我们建议用户根据用途转换注释文件。

> 💡 请注意，基本上需要将注释转换为标准 COCO 格式。

1. 大词汇量、基础、参考：请遵循 `MixedGrounding` 数据集的注释格式，其中添加 `caption` 和 `tokens_positive` 来为每个对象分配文本。<font color='blue'><b>文本可以是类别或名词短语</b></font>。
2. 自定义词汇表（已修复）：可以采用 `MultiModalDataset` 包装器（wrapper）作为 `Objects365` 并为自定义类别创建`文本json`。

### 2.4.5 CC3M 伪注释（Pseudo Annotations）

以下注释是根据论文中的自动标记过程生成的，然后根据这些注释报告结果。要使用CC3M注释，需要先准备 CC3M 图像。

| Data | Images | Boxes | File |
| :--: | :----: | :---: | :---: |
| CC3M-246K | 246,363 | 820,629 | [Download 🤗](https://huggingface.co/wondervictor/YOLO-World/blob/main/cc3m_pseudo_annotations.json) |
| CC3M-500K | 536,405 | 1,784,405| [Download 🤗](https://huggingface.co/wondervictor/YOLO-World/blob/main/cc3m_pseudo_500k_annotations.json) |
| CC3M-750K | 750,000 | 4,504,805 | [Download 🤗](https://huggingface.co/wondervictor/YOLO-World/blob/main/cc3m_pseudo_750k_annotations.json) |

CC3M的标注示例如下：

```json
{
    "categories": {
        {"name": "appropriate attire", "id": 1},
        {"name": "golfers", "id": 2},
        {"name": "the exhibit", "id": 3},
        {"name": "art museum", "id": 4},
        {"name": "homes", "id": 5},
        {"name": "the finish line", "id": 6},
        {"name": "a runner", "id": 7},
        {"name": "actor", "id": 8},
        {"name": "psych folk artist performs", "id": 9},
        {"name": "the opening night", "id": 10},
    },
    "images": {
        {"id": 0, "file_name": "2916026_474308128", "height": 811, "width": 1024, "caption": "actor , who recently played his first round of golf , feels rules on appropriate attire for golfers are a good idea ."},
        {"id": 1, "file_name": "426852_4023546009", "height": 400, "width": 600, "caption": "the exhibit displays homes commissioned by art museum ."},
        {"id": 2, "file_name": "429649_2882962956", "height": 706, "width": 1024, "caption": "a runner crosses the finish line during recurring competition ."},
        {"id": 3, "file_name": "2906408_1262177123", "height": 447, "width": 640, "caption": "pop artist , film director and actor ."},
        {"id": 4, "file_name": "2115477_1264179008", "height": 612, "width": 408, "caption": "psych folk artist performs at festival"},
        {"id": 5, "file_name": "1665090_1070062237", "height": 681, "width": 1024, "caption": "musical artist performs on stage supporting artist on the opening night of their tour ."},
        {"id": 6, "file_name": "440715_513854996", "height": 900, "width": 600, "caption": "this dress just reminds me so much of one of my bridesmaids !"},
        {"id": 7, "file_name": "443433_3084953150", "height": 438, "width": 640, "caption": "rhythm and blues artist performs as part of the event"},
        {"id": 8, "file_name": "2078134_958326625", "height": 447, "width": 640, "caption": "the village of person in the island"},
        {"id": 9, "file_name": "1263157_4047346620", "height": 408, "width": 612, "caption": "actor arrives at the premiere"},
        {"id": 10, "file_name": "2929378_677160606", "height": 488, "width": 640, "caption": "politician on a state visit visited country and met with religious leader"},
    }, 
    "annotations": {
        {"image_id": 0, "id": 0, "category_id": 1, "area": 422653.1869565472, "bbox": [0.0, 379.05767822265625, 976.2655029296875, 432.92852783203125], "iscrowd": 0, "score": 0.3347477614879608, "tokens": "appropriate attire"},
        {"image_id": 0, "id": 1, "category_id": 2, "area": 790882.0195608903, "bbox": [0.0, 11.648781776428223, 989.0121459960938, 799.668662071228], "iscrowd": 0, "score": 0.3464314341545105, "tokens": "golfers"},
        {"image_id": 1, "id": 2, "category_id": 3, "area": 130493.92645893525, "bbox": [129.6766357421875, 4.1426849365234375, 332.00677490234375, 393.0459747314453], "iscrowd": 0, "score": 0.3408966362476349, "tokens": "the exhibit"},
        {"image_id": 1, "id": 3, "category_id": 3, "area": 18324.43645722326, "bbox": [0.7921218872070312, 125.02227783203125, 109.76447296142578, 166.9432373046875], "iscrowd": 0, "score": 0.3054649829864502, "tokens": "the exhibit"},
        {"image_id": 1, "id": 4, "category_id": 4, "area": 238566.77633925015, "bbox": [0.240753173828125, 0.5902252197265625, 599.0253601074219, 398.2582244873047], "iscrowd": 0, "score": 0.3036207854747772, "tokens": "art museum"},
        {"image_id": 1, "id": 5, "category_id": 5, "area": 238566.77633925015, "bbox": [0.240753173828125, 0.5902252197265625, 599.0253601074219, 398.2582244873047], "iscrowd": 0, "score": 0.3074909448623657, "tokens": "homes"},
        {"image_id": 2, "id": 6, "category_id": 6, "area": 328616.3230983538, "bbox": [255.7866668701172, 18.506240844726562, 502.07691955566406, 654.5139007568359], "iscrowd": 0, "score": 0.3392699360847473, "tokens": "the finish line"},
        {"image_id": 2, "id": 7, "category_id": 7, "area": 65720.81522059813, "bbox": [247.374755859375, 270.8210754394531, 162.9920654296875, 403.2148132324219], "iscrowd": 0, "score": 0.34287354350090027, "tokens": "a runner"},
        {"image_id": 2, "id": 8, "category_id": 7, "area": 313720.4859452478, "bbox": [281.114990234375, 16.223989486694336, 478.8563232421875, 655.1453342437744], "iscrowd": 0, "score": 0.30990859866142273, "tokens": "a runner"},
        {"image_id": 2, "id": 9, "category_id": 7, "area": 96920.24000985548, "bbox": [429.16864013671875, 87.87799072265625, 176.37725830078125, 549.5053100585938], "iscrowd": 0, "score": 0.3093494474887848, "tokens": "a runner"},
        {"image_id": 3, "id": 10, "category_id": 8, "area": 47870.292117924895, "bbox": [226.57835388183594, 104.74246215820312, 147.6943817138672, 324.1172180175781], "iscrowd": 0, "score": 0.3774438202381134, "tokens": "actor"},
        {"image_id": 3, "id": 11, "category_id": 8, "area": 49782.2714155633, "bbox": [342.3399353027344, 93.310546875, 148.91860961914062, 334.29180908203125], "iscrowd": 0, "score": 0.3582227826118469, "tokens": "actor"},
        {"image_id": 3, "id": 12, "category_id": 8, "area": 42798.63825566019, "bbox": [113.33319091796875, 110.71199035644531, 134.9569549560547, 317.12806701660156], "iscrowd": 0, "score": 0.34423449635505676, "tokens": "actor"},
        {"image_id": 4, "id": 13, "category_id": 9, "area": 158787.37313683218, "bbox": [28.66830825805664, 80.38202667236328, 298.2539939880371, 532.389762878418], "iscrowd": 0, "score": 0.370086133480072, "tokens": "psych folk artist performs"},
        {"image_id": 5, "id": 14, "category_id": 10, "area": 690727.2885092197, "bbox": [3.4731175899505615, 0.566997766494751, 1020.6946680545807, 676.722736120224], "iscrowd": 0, "score": 0.30705296993255615, "tokens": "the opening night"},
        {"image_id": 5, "id": 15, "category_id": 11, "area": 377094.0525847074, "bbox": [287.8883972167969, 16.25448989868164, 569.5247497558594, 662.1205711364746], "iscrowd": 0, "score": 0.33824190497398376, "tokens": "musical artist performs"},
        {"image_id": 5, "id": 16, "category_id": 12, "area": 377094.0525847074, "bbox": [287.8883972167969, 16.25448989868164, 569.5247497558594, 662.1205711364746], "iscrowd": 0, "score": 0.31286439299583435, "tokens": "artist"},
        {"image_id": 6, "id": 17, "category_id": 13, "area": 361252.4822749605, "bbox": [56.020660400390625, 67.7717514038086, 473.9666442871094, 762.1896743774414], "iscrowd": 0, "score": 0.39791786670684814, "tokens": "bridesmaids"},
        {"image_id": 6, "id": 18, "category_id": 14, "area": 111691.20079200249, "bbox": [190.32395935058594, 241.3621826171875, 186.20994567871094, 599.8132934570312], "iscrowd": 0, "score": 0.3924933671951294, "tokens": "this dress"},
        {"image_id": 6, "id": 19, "category_id": 14, "area": 262045.76735822484, "bbox": [87.693359375, 79.24378967285156, 344.794189453125, 760.0063323974609], "iscrowd": 0, "score": 0.31617414951324463, "tokens": "this dress"},
        {"image_id": 7, "id": 20, "category_id": 15, "area": 176702.29679879732, "bbox": [159.9798583984375, 53.910037994384766, 481.47216796875, 367.00417709350586], "iscrowd": 0, "score": 0.34347665309906006, "tokens": "performs"},
        {"image_id": 7, "id": 21, "category_id": 16, "area": 26940.760472631548, "bbox": [90.44544982910156, 253.2623291015625, 231.88111877441406, 116.18350219726562], "iscrowd": 0, "score": 0.30747899413108826, "tokens": "part"},
        {"image_id": 7, "id": 22, "category_id": 16, "area": 4047.9430108852684, "bbox": [572.8274536132812, 314.28662109375, 68.55877685546875, 59.04339599609375], "iscrowd": 0, "score": 0.3061348795890808, "tokens": "part"},
        {"image_id": 8, "id": 23, "category_id": 17, "area": 498.5000101849437, "bbox": [618.4949340820312, 280.6375732421875, 19.456787109375, 25.620880126953125], "iscrowd": 0, "score": 0.31343746185302734, "tokens": "person"},
        {"image_id": 8, "id": 24, "category_id": 18, "area": 84702.77219054895, "bbox": [88.44400024414062, 160.23072814941406, 549.9750061035156, 154.0120391845703], "iscrowd": 0, "score": 0.3755825161933899, "tokens": "the village"},
        {"image_id": 8, "id": 25, "category_id": 19, "area": 136346.31888543777, "bbox": [159.03176879882812, 1.7530081272125244, 477.5589904785156, 285.5067574977875], "iscrowd": 0, "score": 0.31633585691452026, "tokens": "the island"},
        {"image_id": 9, "id": 26, "category_id": 8, "area": 190156.06273768027, "bbox": [96.15538787841797, 6.901155471801758, 474.92542266845703, 400.3914165496826], "iscrowd": 0, "score": 0.34347039461135864, "tokens": "actor"},
        {"image_id": 9, "id": 27, "category_id": 20, "area": 231256.54143596182, "bbox": [20.39750862121582, 6.0372772216796875, 577.8034801483154, 400.23390197753906], "iscrowd": 0, "score": 0.35184529423713684, "tokens": "the premiere"},
        {"image_id": 10, "id": 28, "category_id": 21, "area": 275776.2488502312, "bbox": [21.93243980407715, 12.839357376098633, 610.1771793365479, 451.96093559265137], "iscrowd": 0, "score": 0.3917006850242615, "tokens": "a state visit"},
        {"image_id": 10, "id": 29, "category_id": 22, "area": 95793.32920437756, "bbox": [85.48125457763672, 24.463964462280273, 218.02680206298828, 439.36492347717285], "iscrowd": 0, "score": 0.3641899526119232, "tokens": "politician"},
        {"image_id": 10, "id": 30, "category_id": 23, "area": 103885.73604834673, "bbox": [367.5455017089844, 50.02199172973633, 258.4236145019531, 401.9978446960449], "iscrowd": 0, "score": 0.4383631646633148, "tokens": "religious leader"},
    }
}
```

