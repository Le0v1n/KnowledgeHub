<center><b><font size=12>Comprehensive Guide to Ultralytics YOLOv5</font></b></center>

# 1. 自定义数据集

## 1.1 环境安装

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**注意**：
1. 安装 `lxml`
2. Pillow 版本要低于 10.0.0，解释链接: [module 'PIL.Image' has no attribute 'ANTIALIAS' 问题处理](https://baijiahao.baidu.com/s?id=1775432196700665405)

## 1.2 创建数据集

我们自己下载 PASCAL VOC 也行，按照 PASCAL VOC 自建一个也行，具体过程见 [PASCAL VOC 2012数据集讲解与制作自己的数据集](https://blog.csdn.net/weixin_44878336/article/details/124540069)。

> 文章不长

## 1.3 PASCAL VOC 数据集结构

PASCAL VOC 数据集结构如下所示。

```txt
PASCAL VOC 2012 数据集
|
├── VOC2012
|   ├── JPEGImages    # 包含所有图像文件
|   |   ├── 2007_000027.jpg
|   |   ├── 2007_000032.jpg
|   |   ├── ...
|   |
|   ├── Annotations    # 包含所有标注文件（XML格式）
|   |   ├── 2007_000027.xml
|   |   ├── 2007_000032.xml
|   |   ├── ...
|   |
|   ├── ImageSets
|   |   ├── Main
|   |   |   ├── train.txt  # 训练集的图像文件列表
|   |   |   ├── val.txt    # 验证集的图像文件列表
|   |   |   ├── test.txt   # 测试集的图像文件列表
|   |
|   ├── SegmentationClass  # 语义分割的标注
|   |   ├── 2007_000032.png
|   |   ├── ...
|   |
|   ├── SegmentationObject  # 物体分割的标注
|   |   ├── 2007_000032.png
|   |   ├── ...
|   |
|   ├── ...               # 其他可能的子文件夹
|
├── VOCdevkit
|   ├── VOCcode          # 包含用于处理数据集的工具代码
|
├── README
```

我们可以看到，对于我们来说，我们只需要两个文件夹就可以了。

1. JPEGImages: 存放所有的图片
2. Annotations: 存放所有的标注信息

这里我们从 PASCAL VOC 中提取出几张图片，组成 VOC2012-Lite：

<div align=center>
    <img src=./imgs_markdown/2023-10-18-17-19-28.png
    width=30%>
</div>

需要注意的是，YOLOv5 的要求标注文件后缀为 `.txt`，但 Annotations 中的文件后缀是 `.xml`，所以我们需要进行转换。

<details>
<summary>YOLO 标注文件说明 </summary>

标注文件举例：

```txt
0 0.481719 0.634028 0.690625 0.713278
1 0.741094 0.524306 0.314750 0.933389
2 0.254162 0.247742 0.574520 0.687422
```

其中，每行代表一个物体的标注，每个标注包括五个值，分别是：

1. `<class_id>`：物体的类别标识符。在这里，有三个不同的类别，分别用 0、1 和 2 表示。
2. `<center_x>`：物体边界框的中心点 x 坐标，归一化到图像宽度。这些值的范围应在 0 到 1 之间。
3. `<center_y>`：物体边界框的中心点 y 坐标，归一化到图像高度。同样，这些值的范围应在 0 到 1 之间。
4. `<width>`：物体边界框的宽度，归一化到图像宽度。
5. `<height>`：物体边界框的高度，归一化到图像高度。

以第一行为例：

- `<class_id>` 是 0，表示这个物体属于类别 0。
- `<center_x>` 是 0.481719，这意味着物体边界框的中心点 x 坐标位于图像宽度的 48.17% 处。
- `<center_y>` 是 0.634028，中心点 y 坐标位于图像高度的 63.40% 处。
- `<width>` 是 0.690625，边界框宽度占图像宽度的 69.06%。
- `<height>` 是 0.713278，边界框高度占图像高度的 71.33%。

</details>

## 1.4 YOLO 想要的数据集结构

### 1.4.1 YOLOv3

一般而言，YOLOv3 想要的数据结构如下所示：

```txt
YOLOv3 数据集
|
├── images         # 包含所有图像文件
|   ├── image1.jpg
|   ├── image2.jpg
|   ├── ...
|
├── labels         # 包含所有标注文件（每个图像对应一个标注文件）
|   ├── image1.txt
|   ├── image2.txt
|   ├── ...
|
├── classes.names  # 类别文件，包含所有类别的名称
|
├── train.txt      # 训练集的图像文件列表
├── valid.txt      # 验证集的图像文件列表
```

### 1.4.2 YOLOv5

与 YOLOv3 不同，YOLOv5 所需要的数据集结构如下所示：

```txt
|-- test
|   |-- images
|   |   |-- 000000000036.jpg
|   |   `-- 000000000042.jpg
|   `-- labels
|       |-- 000000000036.txt
|       `-- 000000000042.txt
|-- train
|   |-- images
|   |   |-- 000000000009.jpg
|   |   `-- 000000000025.jpg
|   `-- labels
|       |-- 000000000009.txt
|       `-- 000000000025.txt
`-- val
    |-- images
    |   |-- 000000000030.jpg
    |   `-- 000000000034.jpg
    `-- labels
        |-- 000000000030.txt
        `-- 000000000034.txt
```

根据 `.yaml` 配置文件变动而变动的，这里我们以 `coco128.yaml` 为例:

```yaml
# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
# COCO128 dataset https://www.kaggle.com/ultralytics/coco128 (first 128 images from COCO train2017) by Ultralytics
# Example usage: python train.py --data coco128.yaml
# parent
# ├── yolov5
# └── datasets
#     └── coco128  ← downloads here (7 MB)


# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: ../datasets/coco128  # dataset root dir
train: images/train2017  # train images (relative to 'path') 128 images
val: images/train2017  # val images (relative to 'path') 128 images
test:  # test images (optional)

# Classes
names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  4: airplane
  5: bus
  6: train
  7: truck
  8: boat
  9: traffic light
  10: fire hydrant
  ...


# Download script/URL (optional)
download: https://ultralytics.com/assets/coco128.zip
```

coco128 的数据结构如下：

```
.
|-- LICENSE
|-- README.txt
|-- images
|   `-- train2017
|       |-- 000000000009.jpg
|       |-- 000000000025.jpg
|       |-- 000000000030.jpg
|       |-- 000000000034.jpg
|       |-- 000000000036.jpg
|       |-- 000000000042.jpg
        ...
`-- labels
    `-- train2017
        |-- 000000000009.txt
        |-- 000000000025.txt
        |-- 000000000030.txt
        |-- 000000000034.txt
        |-- 000000000036.txt
        |-- 000000000042.txt
        ...
```














## 2.3 选择模型

选择一个预训练模型来开始训练。在这里，我们选择 YOLOv5s，这是第二小和速度最快的可用模型。

<div align=center>
    <img src=https://github.com/ultralytics/yolov5/releases/download/v1.0/model_comparison.png
    width=100%>
</div>

## 2.4 训练

通过指定数据集、批次大小、图像大小以及预训练权重 `--weights yolov5s.pt`（推荐）或随机初始化权重 `--weights` '' `--cfg yolov5s.yaml`（不推荐），来在 COCO128 数据集上训练 YOLOv5s 模型。预训练权重将会自动从最新的 YOLOv5 发布中下载。

```bash
python train.py --img 640 --epochs 3 --data coco128.yaml --weights yolov5s.pt
```

> 为了加快训练速度，可以添加 `--cache ram` 或 `--cache disk` 选项（需要大量的内存/磁盘资源）
> 始终从本地数据集进行训练。挂载或网络驱动器，如 Google Drive，会导致训练速度非常慢。

所有训练结果都会保存在 `runs/train/` 目录下，每次训练都会创建一个递增的运行目录，例如 `runs/train/exp2`、`runs/train/exp3` 等等。

## 2.5 可视化

训练结果会自动记录在 Tensorboard 和 CSV 日志记录器中，保存在 `runs/train` 目录下，每次新的训练都会创建一个新的实验目录，例如 `runs/train/exp2`、`runs/train/exp3` 等。

该目录包含了训练和验证的统计数据、马赛克图像、标签、预测结果、以及经过增强的马赛克图像，还包括 Precision-Recall（PR）曲线和混淆矩阵等度量和图表。

<div align=center>
    <img src=./imgs_markdown/2023-10-18-14-01-15.png
    width=100%>
</div>

结果文件 `results.csv` 在每个 Epoch 后更新，然后在训练完成后绘制为 `results.png`（如下所示）。我们也可以手动绘制任何 `results.csv` 文件：

```python
from utils.plots import plot_results

plot_results('path/to/results.csv')  # plot 'results.csv' as 'results.png'
```

<div align=center>
    <img src=./imgs_markdown/2023-10-18-14-02-01.png
    width=100%>
</div>


















# 知识来源

1. [Ultralytics YOLOv5 Docs](https://docs.ultralytics.com/yolov5/tutorials)