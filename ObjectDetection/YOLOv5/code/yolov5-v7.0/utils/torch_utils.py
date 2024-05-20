# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""PyTorch utils."""

import math
import os
import platform
import subprocess
import time
import warnings
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.general import LOGGER, check_version, colorstr, file_date, git_describe

LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None

# Suppress PyTorch warnings
warnings.filterwarnings("ignore", message="User provided device_type of 'cuda', but CUDA is not available. Disabling")
warnings.filterwarnings("ignore", category=UserWarning)


def smart_inference_mode(torch_1_9=check_version(torch.__version__, "1.9.0")):
    # Applies torch.inference_mode() decorator if torch>=1.9.0 else torch.no_grad() decorator
    def decorate(fn):
        return (torch.inference_mode if torch_1_9 else torch.no_grad)()(fn)

    return decorate


def smartCrossEntropyLoss(label_smoothing=0.0):
    # Returns nn.CrossEntropyLoss with label smoothing enabled for torch>=1.10.0
    if check_version(torch.__version__, "1.10.0"):
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    if label_smoothing > 0:
        LOGGER.warning(f"WARNING ⚠️ label smoothing {label_smoothing} requires torch>=1.10.0")
    return nn.CrossEntropyLoss()


def smart_DDP(model):
    # Model DDP creation with checks
    assert not check_version(torch.__version__, "1.12.0", pinned=True), (
        "torch==1.12.0 torchvision==0.13.0 DDP training is not supported due to a known issue. "
        "Please upgrade or downgrade torch to use DDP. See https://github.com/ultralytics/yolov5/issues/8395"
    )
    if check_version(torch.__version__, "1.11.0"):
        return DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK, static_graph=True)
    else:
        return DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)


def reshape_classifier_output(model, n=1000):
    # Update a TorchVision classification model to class count 'n' if required
    from models.common import Classify

    name, m = list((model.model if hasattr(model, "model") else model).named_children())[-1]  # last module
    if isinstance(m, Classify):  # YOLOv5 Classify() head
        if m.linear.out_features != n:
            m.linear = nn.Linear(m.linear.in_features, n)
    elif isinstance(m, nn.Linear):  # ResNet, EfficientNet
        if m.out_features != n:
            setattr(model, name, nn.Linear(m.in_features, n))
    elif isinstance(m, nn.Sequential):
        types = [type(x) for x in m]
        if nn.Linear in types:
            i = types.index(nn.Linear)  # nn.Linear index
            if m[i].out_features != n:
                m[i] = nn.Linear(m[i].in_features, n)
        elif nn.Conv2d in types:
            i = types.index(nn.Conv2d)  # nn.Conv2d index
            if m[i].out_channels != n:
                m[i] = nn.Conv2d(m[i].in_channels, n, m[i].kernel_size, m[i].stride, bias=m[i].bias is not None)


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """使用 @contextmanager 装饰器定义一个上下文管理器
    这个上下文管理器接受一个参数 local_rank，它表示当前进程的本地进程号

    如果当前进程的 local_rank 不是 -1 或 0，则执行 dist.barrier()
    这意味着除了 local_rank 为 -1 或 0 的进程外，其他进程都会在这里等待
    直到所有进程都到达这个 barrier，它们才会继续执行

    Args:
        local_rank (int): 当前进程的索引
    """
    if local_rank not in [-1, 0]:
        dist.barrier(device_ids=[local_rank])
        
    # yield 关键字用于将控制权返回给调用者。在这里，调用者可以执行一些操作，而其他进程则在上面设置的 barrier 处等待
    yield
    
    # 当调用者完成操作后，如果当前进程的 local_rank 是 0，则再次执行 dist.barrier()。
    # 这确保了 local_rank 为 0 的进程完成其任务后，其他进程才能继续执行
    if local_rank == 0:
        dist.barrier(device_ids=[0])


def device_count():
    # Returns number of CUDA devices available. Safe version of torch.cuda.device_count(). Supports Linux and Windows
    assert platform.system() in ("Linux", "Windows"), "device_count() only supported on Linux or Windows"
    try:
        cmd = "nvidia-smi -L | wc -l" if platform.system() == "Linux" else 'nvidia-smi -L | find /c /v ""'  # Windows
        return int(subprocess.run(cmd, shell=True, capture_output=True, check=True).stdout.decode().split()[-1])
    except Exception:
        return 0


def select_device(device="", batch_size=0, newline=True):
    # device = None or 'cpu' or 0 or '0' or '0,1,2,3'
    
    # 获取输出描述，例子：'YOLOv5 🚀 2024-1-29 Python-3.8.18 torch-2.1.0+cpu '
    s = f"YOLOv5 🚀 {git_describe() or file_date()} Python-{platform.python_version()} torch-{torch.__version__} "  

    # 将原本的 cuda:0 转换为 0（去掉原本的cuda:）
    device = str(device).strip().lower().replace("cuda:", "").replace("none", "")  # to string, 'cuda:0' to '0'

    # 判断指定的设备是否CPU或MPS
    cpu = device == "cpu"
    mps = device == "mps"  # Apple Metal Performance Shaders (MPS)
    
    # 如果指定的设备是的CPU或MPS，则不使用CUDA
    if cpu or mps:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # force torch.cuda.is_available() = False
    # 如果指定的设备不是CPU或MPS（CUDA）
    elif device:  # non-cpu device requested
        # 设置环境变量中可用的CUDA设备索引
        os.environ["CUDA_VISIBLE_DEVICES"] = device  # set environment variable - must be before assert is_available()
        assert torch.cuda.is_available() and torch.cuda.device_count() >= len(
            device.replace(",", "")
        ), f"Invalid CUDA '--device {device}' requested, use '--device cpu' or pass valid CUDA device(s)"
    # 设备优先级：CUDA > MPS > CPU
    if not cpu and not mps and torch.cuda.is_available():  # prefer GPU if available
        devices = device.split(",") if device else "0"  # range(torch.cuda.device_count())  # i.e. 0,1,6,7
        n = len(devices)  # device count
        if n > 1 and batch_size > 0:  # 检查CUDA设备数是否可以被batch_size整除
            assert batch_size % n == 0, f"batch-size {batch_size} not multiple of GPU count {n}"
        space = " " * (len(s) + 1)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)\n"  # bytes to MB（1 << 20 在这个上下文中用作一个快速的常数，用于将字节数转换为兆字节）
        arg = "cuda:0"
    elif mps and getattr(torch, "has_mps", False) and torch.backends.mps.is_available():  # prefer MPS if available
        s += "MPS\n"
        arg = "mps"
    else:  # revert to CPU
        s += "CPU\n"
        arg = "cpu"

    if not newline:
        s = s.rstrip()
    LOGGER.info(s)
    return torch.device(arg)


def time_sync():
    """torch.cuda.synchronize() 是PyTorch中用于GPU操作的一个函数，它的作用是等待当前设备上的所有CUDA操作都完成后才继续执行后续的代码。
    这个函数是同步操作，它会阻塞程序的执行直到所有先前的CUDA操作都执行完毕。
    在测量GPU操作的性能时，这个函数非常有用。因为在GPU上执行的操作是异步的，如果你在发起一个操作之后立即测量时间，
    你可能会得到一个不准确的时间，因为操作可能还没有完成。使用 torch.cuda.synchronize() 可以确保在测量时间之前，
    所有的CUDA操作都已经完成，从而得到一个准确的性能评估。
    """
    # PyTorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def profile(input, ops, n=10, device=None):
    """YOLOv5 speed/memory/FLOPs profiler
    Usage:
        input = torch.randn(16, 3, 640, 640)
        m1 = lambda x: x * torch.sigmoid(x)
        m2 = nn.SiLU()
        profile(input, [m1, m2], n=100)  # profile over 100 iterations
    """
    results = []
    if not isinstance(device, torch.device):
        device = select_device(device)
    print(
        f"{'Params':>12s}{'GFLOPs':>12s}{'GPU_mem (GB)':>14s}{'forward (ms)':>14s}{'backward (ms)':>14s}"
        f"{'input':>24s}{'output':>24s}"
    )

    for x in input if isinstance(input, list) else [input]:
        x = x.to(device)
        x.requires_grad = True
        for m in ops if isinstance(ops, list) else [ops]:
            m = m.to(device) if hasattr(m, "to") else m  # device
            m = m.half() if hasattr(m, "half") and isinstance(x, torch.Tensor) and x.dtype is torch.float16 else m
            tf, tb, t = 0, 0, [0, 0, 0]  # dt forward, backward
            try:
                flops = thop.profile(m, inputs=(x,), verbose=False)[0] / 1e9 * 2  # GFLOPs
            except Exception:
                flops = 0

            try:
                for _ in range(n):
                    t[0] = time_sync()
                    y = m(x)
                    t[1] = time_sync()
                    try:
                        _ = (sum(yi.sum() for yi in y) if isinstance(y, list) else y).sum().backward()
                        t[2] = time_sync()
                    except Exception:  # no backward method
                        # print(e)  # for debug
                        t[2] = float("nan")
                    tf += (t[1] - t[0]) * 1000 / n  # ms per op forward
                    tb += (t[2] - t[1]) * 1000 / n  # ms per op backward
                mem = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0  # (GB)
                s_in, s_out = (tuple(x.shape) if isinstance(x, torch.Tensor) else "list" for x in (x, y))  # shapes
                p = sum(x.numel() for x in m.parameters()) if isinstance(m, nn.Module) else 0  # parameters
                print(f"{p:12}{flops:12.4g}{mem:>14.3f}{tf:14.4g}{tb:14.4g}{str(s_in):>24s}{str(s_out):>24s}")
                results.append([p, flops, mem, tf, tb, s_in, s_out])
            except Exception as e:
                print(e)
                results.append(None)
            torch.cuda.empty_cache()
    return results


def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def de_parallel(model):
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model


def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True


def find_modules(model, mclass=nn.Conv2d):
    # Finds layer indices matching module class 'mclass'
    return [i for i, m in enumerate(model.module_list) if isinstance(m, mclass)]


def sparsity(model):
    # Return global model sparsity
    a, b = 0, 0
    for p in model.parameters():
        a += p.numel()
        b += (p == 0).sum()
    return b / a


def prune(model, amount=0.3):
    # Prune model to requested global sparsity
    import torch.nn.utils.prune as prune

    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            prune.l1_unstructured(m, name="weight", amount=amount)  # prune
            prune.remove(m, "weight")  # make permanent
    LOGGER.info(f"Model pruned to {sparsity(model):.3g} global sparsity")


def fuse_conv_and_bn(conv, bn):
    """将卷积层（Conv2d）和批量归一化层（BatchNorm2d）融合成一个单一的卷积层。
    这样做可以提高模型的性能，因为融合后的层可以减少一些不必要的计算
    
    💡  OBS：该技巧只适用于模型推理，不适用于模型训练！

    Args:
        conv (_type_): 原来的卷积模块
        bn (_type_): 原来的BN模块

    Returns:
        _type_: 融合后的卷积模块
    """
    # Fuse Conv2d() and BatchNorm2d() layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = (
        nn.Conv2d(
            conv.in_channels,              # 输入通道数
            conv.out_channels,             # 输出通道数
            kernel_size=conv.kernel_size,  # 卷积核大小
            stride=conv.stride,            # 步长
            padding=conv.padding,          # 填充
            dilation=conv.dilation,        # 膨胀卷积的膨胀率 
            groups=conv.groups,            # 分组卷积的组数
            bias=True,                     # 是否需要偏置
        ).requires_grad_(False).to(conv.weight.device)  # 不需要计算梯度，并将其移动到与原始卷积层相同的设备上
    )

    # 准备卷积层的权重
    w_conv = conv.weight.clone().view(conv.out_channels, -1)  # 将卷积核权重展平为二维矩阵
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))  # 计算BN的权重缩放因子
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))  # 融合权重

    # 准备空间偏置项
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias  # 如果原始卷积层没有偏置项，则创建全零偏置项
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))  # 计算BN的偏置项
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)  # 融合偏置项

    return fusedconv


def model_info(model, verbose=False, imgsz=640):
    # Model information. img_size may be int or list, i.e. img_size=640 or img_size=[640, 320]
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if verbose:
        print(f"{'layer':>5} {'name':>40} {'gradient':>9} {'parameters':>12} {'shape':>20} {'mu':>10} {'sigma':>10}")
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace("module_list.", "")
            print(
                "%5g %40s %9s %12g %20s %10.3g %10.3g"
                % (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std())
            )

    try:  # FLOPs
        p = next(model.parameters())
        stride = max(int(model.stride.max()), 32) if hasattr(model, "stride") else 32  # max stride
        im = torch.empty((1, p.shape[1], stride, stride), device=p.device)  # input image in BCHW format
        flops = thop.profile(deepcopy(model), inputs=(im,), verbose=False)[0] / 1e9 * 2  # stride GFLOPs
        imgsz = imgsz if isinstance(imgsz, list) else [imgsz, imgsz]  # expand if int/float
        fs = f", {flops * imgsz[0] / stride * imgsz[1] / stride:.1f} GFLOPs"  # 640x640 GFLOPs
    except Exception:
        fs = ""

    name = Path(model.yaml_file).stem.replace("yolov5", "YOLOv5") if hasattr(model, "yaml_file") else "Model"
    LOGGER.info(f"{name} summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients{fs}")


def scale_img(img, ratio=1.0, same_shape=False, gs=32):  # img(16,3,256,416)
    """将图片img(bs,3,y,x)按照比例进行缩放，约束为gs的倍数。

    Args:
        img (_type_): 输入图像张量，具有形状 (batch_size, channels, height, width)
        ratio (float, optional): 图像放缩的比例因子. Defaults to 1.0.
        same_shape (bool, optional): 决定是否保持原始图像的宽高比例. Defaults to False.
        gs (int, optional): 网格大小，用于确保放缩后的图像尺寸是gs的倍数. Defaults to 32.

    Returns:
        _type_: 缩放后的图片
    """
    
    # 如果比例因子为1，直接返回原图，无需放缩
    if ratio == 1.0:
        return img
    
    # 获取图像的高度和宽度
    h, w = img.shape[2:]
    
    # 计算放缩后的新尺寸
    s = (int(h * ratio), int(w * ratio))
    
    # 使用双线性插值放缩图像
    img = F.interpolate(img, size=s, mode="bilinear", align_corners=False)  # resize
    
    # 如果不需要保持原始宽高比例
    if not same_shape:  # pad/crop img
        # 计算新的高度和宽度，确保它们是gs的倍数
        h, w = (math.ceil(x * ratio / gs) * gs for x in (h, w))
        
    return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)  # value=ImageNet的平均像素值


def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():  # b.__dict_.keys(): dict_keys(['training', '_parameters', '_buffers', '_non_persistent_buffers_set', '_backward_hooks', '_is_full_backward_hook', '_forward_hooks', '_forward_pre_hooks', '_state_dict_hooks', '_load_state_dict_pre_hooks', '_load_state_dict_post_hooks', '_modules', 'yaml_file', 'yaml', 'save', 'names', 'inplace', 'stride'])
        if (len(include) and k not in include) or k.startswith("_") or k in exclude:
            continue
        else:
            setattr(a, k, v)


def smart_optimizer(model, name="Adam", lr=0.001, momentum=0.9, decay=1e-5):
    # YOLOv5 3-param group optimizer: 0) weights with decay, 1) weights no decay, 2) biases no decay
    g = [], [], []  # optimizer parameter groups
    bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
    for v in model.modules():
        for p_name, p in v.named_parameters(recurse=0):
            if p_name == "bias":  # bias (no decay)
                g[2].append(p)
            elif p_name == "weight" and isinstance(v, bn):  # weight (no decay)
                g[1].append(p)
            else:
                g[0].append(p)  # weight (with decay)

    if name == "Adam":
        optimizer = torch.optim.Adam(g[2], lr=lr, betas=(momentum, 0.999))  # adjust beta1 to momentum
    elif name == "AdamW":
        optimizer = torch.optim.AdamW(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
    elif name == "RMSProp":
        optimizer = torch.optim.RMSprop(g[2], lr=lr, momentum=momentum)
    elif name == "SGD":
        optimizer = torch.optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
    else:
        raise NotImplementedError(f"Optimizer {name} not implemented.")

    optimizer.add_param_group({"params": g[0], "weight_decay": decay})  # add g0 with weight_decay
    optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})  # add g1 (BatchNorm2d weights)
    LOGGER.info(
        f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}) with parameter groups "
        f'{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias'
    )
    return optimizer


def smart_hub_load(repo="ultralytics/yolov5", model="yolov5s", **kwargs):
    # YOLOv5 torch.hub.load() wrapper with smart error/issue handling
    if check_version(torch.__version__, "1.9.1"):
        kwargs["skip_validation"] = True  # validation causes GitHub API rate limit errors
    if check_version(torch.__version__, "1.12.0"):
        kwargs["trust_repo"] = True  # argument required starting in torch 0.12
    try:
        return torch.hub.load(repo, model, **kwargs)
    except Exception:
        return torch.hub.load(repo, model, force_reload=True, **kwargs)


def smart_resume(ckpt, optimizer, ema=None, weights="yolov5s.pt", epochs=300, resume=True):
    # Resume training from a partially trained checkpoint
    best_fitness = 0.0
    start_epoch = ckpt["epoch"] + 1
    if ckpt["optimizer"] is not None:
        optimizer.load_state_dict(ckpt["optimizer"])  # optimizer
        best_fitness = ckpt["best_fitness"]
    if ema and ckpt.get("ema"):
        ema.ema.load_state_dict(ckpt["ema"].float().state_dict())  # EMA
        ema.updates = ckpt["updates"]
    if resume:
        assert start_epoch > 0, (
            f"{weights} training to {epochs} epochs is finished, nothing to resume.\n"
            f"Start a new training without --resume, i.e. 'python train.py --weights {weights}'"
        )
        LOGGER.info(f"Resuming training from {weights} from epoch {start_epoch} to {epochs} total epochs")
    if epochs < start_epoch:
        LOGGER.info(f"{weights} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {epochs} more epochs.")
        epochs += ckpt["epoch"]  # finetune additional epochs
    return best_fitness, start_epoch, epochs


class EarlyStopping:
    # YOLOv5 simple early stopper
    def __init__(self, patience=30):
        self.best_fitness = 0.0  # i.e. mAP
        self.best_epoch = 0
        self.patience = patience or float("inf")  # epochs to wait after fitness stops improving to stop
        self.possible_stop = False  # possible stop may occur next epoch

    def __call__(self, epoch, fitness):
        if fitness >= self.best_fitness:  # >= 0 to allow for early zero-fitness stage of training
            self.best_epoch = epoch
            self.best_fitness = fitness
        delta = epoch - self.best_epoch  # epochs without improvement
        self.possible_stop = delta >= (self.patience - 1)  # possible stop may occur next epoch
        stop = delta >= self.patience  # stop training if patience exceeded
        if stop:
            LOGGER.info(
                f"Stopping training early as no improvement observed in last {self.patience} epochs. "
                f"Best results observed at epoch {self.best_epoch}, best model saved as best.pt.\n"
                f"To update EarlyStopping(patience={self.patience}) pass a new patience value, "
                f"i.e. `python train.py --patience 300` or use `--patience 0` to disable EarlyStopping."
            )
        return stop


class ModelEMA:
    """Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        # Create EMA
        self.ema = deepcopy(de_parallel(model)).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        self.updates += 1
        d = self.decay(self.updates)

        msd = de_parallel(model).state_dict()  # model state_dict
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:  # true for FP16 and FP32
                v *= d
                v += (1 - d) * msd[k].detach()
        # assert v.dtype == msd[k].dtype == torch.float32, f'{k}: EMA {v.dtype} and model {msd[k].dtype} must be FP32'

    def update_attr(self, model, include=(), exclude=("process_group", "reducer")):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)
