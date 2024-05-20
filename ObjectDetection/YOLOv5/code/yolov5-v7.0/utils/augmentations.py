# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""Image augmentation functions."""

import math
import random

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from utils.general import LOGGER, check_version, colorstr, resample_segments, segment2box, xywhn2xyxy
from utils.metrics import bbox_ioa

IMAGENET_MEAN = 0.485, 0.456, 0.406  # RGB mean
IMAGENET_STD = 0.229, 0.224, 0.225  # RGB standard deviation


class Albumentations:
    # YOLOv5 Albumentations class (optional, only used if package is installed)
    def __init__(self, size=640):
        self.transform = None
        prefix = colorstr("albumentations: ")
        try:
            import albumentations as A

            check_version(A.__version__, "1.0.3", hard=True)  # version requirement

            T = [
                A.RandomResizedCrop(height=size, width=size, scale=(0.8, 1.0), ratio=(0.9, 1.11), p=0.0),
                A.Blur(p=0.01),
                A.MedianBlur(p=0.01),
                A.ToGray(p=0.01),
                A.CLAHE(p=0.01),
                A.RandomBrightnessContrast(p=0.0),
                A.RandomGamma(p=0.0),
                A.ImageCompression(quality_lower=75, p=0.0),
            ]  # transforms
            self.transform = A.Compose(T, bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))

            LOGGER.info(prefix + ", ".join(f"{x}".replace("always_apply=False, ", "") for x in T if x.p))
        except ImportError:  # package not installed, skip
            pass
        except Exception as e:
            LOGGER.info(f"{prefix}{e}")

    def __call__(self, im, labels, p=1.0):
        if self.transform and random.random() < p:
            new = self.transform(image=im, bboxes=labels[:, 1:], class_labels=labels[:, 0])  # transformed
            im, labels = new["image"], np.array([[c, *b] for c, b in zip(new["class_labels"], new["bboxes"])])
        return im, labels


def normalize(x, mean=IMAGENET_MEAN, std=IMAGENET_STD, inplace=False):
    # Denormalize RGB images x per ImageNet stats in BCHW format, i.e. = (x - mean) / std
    return TF.normalize(x, mean, std, inplace=inplace)


def denormalize(x, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    # Denormalize RGB images x per ImageNet stats in BCHW format, i.e. = x * std + mean
    for i in range(3):
        x[:, i] = x[:, i] * std[i] + mean[i]
    return x


def augment_hsv(im, hgain=0.5, sgain=0.5, vgain=0.5):
    # HSV color-space augmentation
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
        dtype = im.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im)  # no return needed


def hist_equalize(im, clahe=True, bgr=False):
    # Equalize histogram on BGR image 'im' with im.shape(n,m,3) and range 0-255
    yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV if bgr else cv2.COLOR_RGB2YUV)
    if clahe:
        c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yuv[:, :, 0] = c.apply(yuv[:, :, 0])
    else:
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])  # equalize Y channel histogram
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR if bgr else cv2.COLOR_YUV2RGB)  # convert YUV image to RGB


def replicate(im, labels):
    # Replicate labels
    h, w = im.shape[:2]
    boxes = labels[:, 1:].astype(int)
    x1, y1, x2, y2 = boxes.T
    s = ((x2 - x1) + (y2 - y1)) / 2  # side length (pixels)
    for i in s.argsort()[: round(s.size * 0.5)]:  # smallest indices
        x1b, y1b, x2b, y2b = boxes[i]
        bh, bw = y2b - y1b, x2b - x1b
        yc, xc = int(random.uniform(0, h - bh)), int(random.uniform(0, w - bw))  # offset x, y
        x1a, y1a, x2a, y2a = [xc, yc, xc + bw, yc + bh]
        im[y1a:y2a, x1a:x2a] = im[y1b:y2b, x1b:x2b]  # im4[ymin:ymax, xmin:xmax]
        labels = np.append(labels, [[labels[i, 0], x1a, y1a, x2a, y2a]], axis=0)

    return im, labels


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """调整和填充图片，使其符合新的形状，同时保持宽高比例。

    该函数将输入图片缩放到指定的高度和宽度，并保持图片的原始宽高比例。
    如果目标形状和原图宽高比例不一致，则在图片周围填充指定的颜色。
    此函数通常用于预处理图片，以便输入到神经网络模型中。

    Args:
        im (cv2): 输入的图片，格式为HWC（高度、宽度、通道）。
        new_shape (tuple, optional): 目标形状，格式为(高度, 宽度)。Defaults to (640, 640).
        color (tuple, optional): 填充颜色，格式为(蓝, 绿, 红)。Defaults to (114, 114, 114)，即灰色。
        auto (bool, optional): 是否自动调整填充，以适应步长。Defaults to True.
        scaleFill (bool, optional): 是否拉伸图片以填满目标形状。Defaults to False.
        scaleup (bool, optional): 是否放大图片。如果为False，则只缩小，不放大。Defaults to True.
        stride (int, optional): 步长，用于自动调整填充。Defaults to 32.

    Returns:
        tuple: 返回一个元组，包含以下三个元素：
            - im: 调整和填充后的图片。
            - ratio: 宽高缩放比率。
            - (dw, dh): 宽度和高度的填充量。
    """
    # 调整和填充图片，同时满足步长倍数的约束
    shape = im.shape[:2]  # 当前图片的形状 [高度, 宽度]
    
    # 如果new_shape是整数，则将其变为宽高相同的元组
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # 尺度比率 (新尺寸 / 旧尺寸)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])  # 宽度或高度中最小的比例
    # 哪条边差距最大就resize哪条边，也可以理解为哪条边差距最大，就不Padding它，因为Padding它代价更高
    
    # 缩放(resize)到输入大小img_size的时候，如果没有设置上采样的话，则只进行下采样，
    # 因为上采样会让图片模糊，可能会影响模型训练效果。
    if not scaleup:  # scaleup=False时，图片只缩小，不放大，默认scaleup=True
        r = min(r, 1.0)  # 限制在[0,1]之间，防止图片出现上采样情况

    # 计算填充
    ratio = r, r  # 宽高比率
    
    # 计算缩放后的新的宽高（无填充）-> 长边直接缩放到新尺寸
    new_shape_unpadding = int(round(shape[1] * r)), int(round(shape[0] * r))  
    
    # 计算宽度和高度分别需要填充的像素个数（其中d表示delta，即差距）-> 长边为0
    dw, dh = new_shape[1] - new_shape_unpadding[0], new_shape[0] - new_shape_unpadding[1]  
    
    """
        如果auto=True， 则为rectangle（宽度或高度一个边进行填充，填充到stride的最小倍数即可 -> 得到的是一个矩形）
        如果auto=False，则为squared  （宽度或高度一个边进行填充，填充到目标尺寸             -> 得到的是一个正方形）
        
        简单来说，auto=True: 获取一个最小的填充，即矩形填充。
    """
    if auto:
        # 宽高填充，保证新的尺寸是步长的整数倍
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # np.mod 用于计算除法的余数，等价于 %
    # 如果scaleFill=True，则不进行填充，直接resize到目标尺寸，任由图片进行拉伸和压缩（等价于直接resize）
    elif scaleFill:  # 拉伸
        dw, dh = 0.0, 0.0
        new_shape_unpadding = (new_shape[1], new_shape[0])  # 宽度x高度
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # 宽高比率
    
    # 将填充均分到两边
    dw /= 2  
    dh /= 2

    # 如果当前尺寸不等于缩放后的尺寸，则进行缩放
    if shape[::-1] != new_shape_unpadding:  # shape原本为[高度, 宽度]，则shape[::-1]表示[宽度, 高度]
        im = cv2.resize(im, new_shape_unpadding, interpolation=cv2.INTER_LINEAR)  # 长边和短边同时缩放/放大，且长边直接为目标尺寸

    # 计算宽度和高度方向需要填充的像素个数
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))  # 高度
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))  # 宽度
    
    # print(f"{r = }")
    # print(f"原图宽x高: {shape[1]}x{shape[0]}")
    # print(f"缩放后的新的宽x高: {new_shape_unpadding[0]}x{new_shape_unpadding[1]}")
    # print(f"宽度需要填充的像素个数: {dw}")
    # print(f"高度需要填充的像素个数: {dh}")       

    # 添加边框，使用常数颜色填充
    im = cv2.copyMakeBorder(
        src=im, 
        top=top, 
        bottom=bottom, 
        left=left, 
        right=right, 
        borderType=cv2.BORDER_CONSTANT,  # 填充方式：常数填充
        value=color)

    # 返回调整后的图片，宽高比率，以及宽高填充
    return im, ratio, (dw, dh)


def random_perspective(
    im, targets=(), segments=(), degrees=10, translate=0.1, scale=0.1, shear=10, perspective=0.0, border=(0, 0)
):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]

    height = im.shape[0] + border[0] * 2  # shape(h,w,c)
    width = im.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -im.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -im.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            im = cv2.warpPerspective(im, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            im = cv2.warpAffine(im, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Visualize
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(im[:, :, ::-1])  # base
    # ax[1].imshow(im2[:, :, ::-1])  # warped

    # Transform label coordinates
    n = len(targets)
    if n:
        use_segments = any(x.any() for x in segments) and len(segments) == n
        new = np.zeros((n, 4))
        if use_segments:  # warp segments
            segments = resample_segments(segments)  # upsample
            for i, segment in enumerate(segments):
                xy = np.ones((len(segment), 3))
                xy[:, :2] = segment
                xy = xy @ M.T  # transform
                xy = xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]  # perspective rescale or affine

                # clip
                new[i] = segment2box(xy, width, height)

        else:  # warp boxes
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform
            xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # clip
            new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
            new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=new.T, area_thr=0.01 if use_segments else 0.10)
        targets = targets[i]
        targets[:, 1:5] = new[i]

    return im, targets


def copy_paste(im, labels, segments, p=0.5):
    # Implement Copy-Paste augmentation https://arxiv.org/abs/2012.07177, labels as nx5 np.array(cls, xyxy)
    n = len(segments)
    if p and n:
        h, w, c = im.shape  # height, width, channels
        im_new = np.zeros(im.shape, np.uint8)
        for j in random.sample(range(n), k=round(p * n)):
            l, s = labels[j], segments[j]
            box = w - l[3], l[2], w - l[1], l[4]
            ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
            if (ioa < 0.30).all():  # allow 30% obscuration of existing labels
                labels = np.concatenate((labels, [[l[0], *box]]), 0)
                segments.append(np.concatenate((w - s[:, 0:1], s[:, 1:2]), 1))
                cv2.drawContours(im_new, [segments[j].astype(np.int32)], -1, (1, 1, 1), cv2.FILLED)

        result = cv2.flip(im, 1)  # augment segments (flip left-right)
        i = cv2.flip(im_new, 1).astype(bool)
        im[i] = result[i]  # cv2.imwrite('debug.jpg', im)  # debug

    return im, labels, segments


def cutout(im, labels, p=0.5):
    # Applies image cutout augmentation https://arxiv.org/abs/1708.04552
    if random.random() < p:
        h, w = im.shape[:2]
        scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16  # image size fraction
        for s in scales:
            mask_h = random.randint(1, int(h * s))  # create random masks
            mask_w = random.randint(1, int(w * s))

            # box
            xmin = max(0, random.randint(0, w) - mask_w // 2)
            ymin = max(0, random.randint(0, h) - mask_h // 2)
            xmax = min(w, xmin + mask_w)
            ymax = min(h, ymin + mask_h)

            # apply random color mask
            im[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]

            # return unobscured labels
            if len(labels) and s > 0.03:
                box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
                ioa = bbox_ioa(box, xywhn2xyxy(labels[:, 1:5], w, h))  # intersection over area
                labels = labels[ioa < 0.60]  # remove >60% obscured labels

    return labels


def mixup(im, labels, im2, labels2):
    """MixUp数据增强的实现。

    MixUp是一种在训练过程中通过线性插值合并两个随机选择的图像和它们对应的标签来增加数据多样性的方法。
    这种方法可以提高模型的泛化能力，并减少过拟合的风险。

    论文链接:
        https://arxiv.org/pdf/1710.09412.pdf

    Args:
        im (numpy.ndarray): 第一张输入图像，形状为 (height, width, channels)，例子：
        labels (numpy.ndarray): 第一张输入图像对应的标签，形状为 (Number of Object, 5)，其中5为
        im2 (numpy.ndarray): 第二张输入图像，形状与 `im` 相同。
        labels2 (numpy.ndarray): 第二张输入图像对应的标签，形状与 `labels` 相同。

    Returns:
        tuple: 包含两个元素的元组，第一个元素是混合后的图像，第二个元素是混合后的标签。
        
        
    Example：
        im.shape = (640, 640, 3)
        labels.shape = (cls, cx, cy, w, h)
        labels = array([[          2,       23.46,      149.84,      56.603,      166.17],
                        [          2,      110.12,       160.6,      147.08,      180.08],
                        [          2,           0,      150.58,      20.573,      165.75],
                        [          2,      131.22,      140.63,      153.98,      148.01],
                        [          2,      152.74,      168.93,      183.13,      186.35],
                        [          2,           0,      144.09,      18.519,      152.54],
                        [         20,      450.44,       282.3,      552.94,      346.79],
                        [         20,      366.06,      298.44,       447.1,      345.52],
                        [         20,      304.99,       308.8,      352.43,      344.38],
                        [         20,      185.41,      298.92,       260.5,      344.44],
                        [          9,      528.41,      625.18,      530.87,      627.81]], dtype=float32)
    """
    # 使用beta分布生成一个混合系数r，这里alpha和beta参数都设为32.0，控制了分布的形状
    r = np.random.beta(32.0, 32.0)  # mixup比例，alpha=beta=32.0

    # 将两张图像按照混合系数r进行线性组合，生成新的合成图像。结果转换为uint8以匹配图像的常规数据类型。
    im = (im * r + im2 * (1 - r)).astype(np.uint8)  # 混合的结果直接覆盖im了
    
    # 将两个标签数组沿第0轴（垂直方向）连接起来，生成新的标签数组
    labels = np.concatenate((labels, labels2), 0)
    
    # 返回混合后的图像和标签数组
    return im, labels


def box_candidates(box1, box2, wh_thr=2, ar_thr=100, area_thr=0.1, eps=1e-16):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates


def classify_albumentations(
    augment=True,
    size=224,
    scale=(0.08, 1.0),
    ratio=(0.75, 1.0 / 0.75),  # 0.75, 1.33
    hflip=0.5,
    vflip=0.0,
    jitter=0.4,
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
    auto_aug=False,
):
    # YOLOv5 classification Albumentations (optional, only used if package is installed)
    prefix = colorstr("albumentations: ")
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        check_version(A.__version__, "1.0.3", hard=True)  # version requirement
        if augment:  # Resize and crop
            T = [A.RandomResizedCrop(height=size, width=size, scale=scale, ratio=ratio)]
            if auto_aug:
                # TODO: implement AugMix, AutoAug & RandAug in albumentation
                LOGGER.info(f"{prefix}auto augmentations are currently not supported")
            else:
                if hflip > 0:
                    T += [A.HorizontalFlip(p=hflip)]
                if vflip > 0:
                    T += [A.VerticalFlip(p=vflip)]
                if jitter > 0:
                    color_jitter = (float(jitter),) * 3  # repeat value for brightness, contrast, satuaration, 0 hue
                    T += [A.ColorJitter(*color_jitter, 0)]
        else:  # Use fixed crop for eval set (reproducibility)
            T = [A.SmallestMaxSize(max_size=size), A.CenterCrop(height=size, width=size)]
        T += [A.Normalize(mean=mean, std=std), ToTensorV2()]  # Normalize and convert to Tensor
        LOGGER.info(prefix + ", ".join(f"{x}".replace("always_apply=False, ", "") for x in T if x.p))
        return A.Compose(T)

    except ImportError:  # package not installed, skip
        LOGGER.warning(f"{prefix}⚠️ not found, install with `pip install albumentations` (recommended)")
    except Exception as e:
        LOGGER.info(f"{prefix}{e}")


def classify_transforms(size=224):
    # Transforms to apply if albumentations not installed
    assert isinstance(size, int), f"ERROR: classify_transforms size {size} must be integer, not (list, tuple)"
    # T.Compose([T.ToTensor(), T.Resize(size), T.CenterCrop(size), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return T.Compose([CenterCrop(size), ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


class LetterBox:
    # YOLOv5 LetterBox class for image preprocessing, i.e. T.Compose([LetterBox(size), ToTensor()])
    def __init__(self, size=(640, 640), auto=False, stride=32):
        super().__init__()
        self.h, self.w = (size, size) if isinstance(size, int) else size
        self.auto = auto  # pass max size integer, automatically solve for short side using stride
        self.stride = stride  # used with auto

    def __call__(self, im):  # im = np.array HWC
        imh, imw = im.shape[:2]
        r = min(self.h / imh, self.w / imw)  # ratio of new/old
        h, w = round(imh * r), round(imw * r)  # resized image
        hs, ws = (math.ceil(x / self.stride) * self.stride for x in (h, w)) if self.auto else self.h, self.w
        top, left = round((hs - h) / 2 - 0.1), round((ws - w) / 2 - 0.1)
        im_out = np.full((self.h, self.w, 3), 114, dtype=im.dtype)
        im_out[top : top + h, left : left + w] = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
        return im_out


class CenterCrop:
    # YOLOv5 CenterCrop class for image preprocessing, i.e. T.Compose([CenterCrop(size), ToTensor()])
    def __init__(self, size=640):
        super().__init__()
        self.h, self.w = (size, size) if isinstance(size, int) else size

    def __call__(self, im):  # im = np.array HWC
        imh, imw = im.shape[:2]
        m = min(imh, imw)  # min dimension
        top, left = (imh - m) // 2, (imw - m) // 2
        return cv2.resize(im[top : top + m, left : left + m], (self.w, self.h), interpolation=cv2.INTER_LINEAR)


class ToTensor:
    # YOLOv5 ToTensor class for image preprocessing, i.e. T.Compose([LetterBox(size), ToTensor()])
    def __init__(self, half=False):
        super().__init__()
        self.half = half

    def __call__(self, im):  # im = np.array HWC in BGR order
        im = np.ascontiguousarray(im.transpose((2, 0, 1))[::-1])  # HWC to CHW -> BGR to RGB -> contiguous
        im = torch.from_numpy(im)  # to torch
        im = im.half() if self.half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0-255 to 0.0-1.0
        return im
