"""INT8 post-training quantization, QAT and latency measurement (x86 / fbgemm, CPU)."""
from __future__ import annotations

import copy
import io
import math
import statistics
import time

import torch
import torch.nn.functional as F
from torch.ao import quantization as tq
from torch.utils.data import DataLoader

from . import data as D

off = (D.STORE - D.CROP) // 2
torch.backends.quantized.engine = "x86"


def norm_cpu(x: torch.Tensor) -> torch.Tensor:
    m = torch.tensor(D.MEAN).view(1, 3, 1, 1) * 255
    s = torch.tensor(D.STD).view(1, 3, 1, 1) * 255
    return (x.float() - m) / s


@torch.no_grad()
def acc_cpu(model, x, y, bs=128) -> float:
    model.eval()
    c = 0
    for i in range(0, len(y), bs):
        xb = norm_cpu(x[i:i + bs, :, off:off + D.CROP, off:off + D.CROP])
        c += (model(xb).argmax(1) == y[i:i + bs]).sum().item()
    return c / len(y)


def size_mb(model) -> float:
    b = io.BytesIO()
    torch.save(model.state_dict(), b)
    return b.tell() / 1e6


@torch.no_grad()
def latency_ms(model, threads: int, bs: int = 1, iters: int = 300) -> float:
    torch.set_num_threads(threads)
    model.eval()
    x = torch.randn(bs, 3, D.CROP, D.CROP)
    for _ in range(30):
        model(x)
    ts = []
    for _ in range(iters):
        t = time.perf_counter(); model(x); ts.append(time.perf_counter() - t)
    return statistics.median(ts) * 1e3


@torch.no_grad()
def latency_gpu_ms(model, bs: int = 1, iters: int = 300) -> float:
    m = copy.deepcopy(model).cuda().eval().to(memory_format=torch.channels_last)
    x = torch.randn(bs, 3, D.CROP, D.CROP, device="cuda").contiguous(memory_format=torch.channels_last)
    with torch.autocast("cuda", dtype=torch.float16):
        for _ in range(30):
            m(x)
        torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            t = time.perf_counter(); m(x); torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
    return statistics.median(ts) * 1e3


def ptq(float_model, x_calib: torch.Tensor, n_calib: int = 512):
    m = copy.deepcopy(float_model).cpu().eval()
    m.fuse_model()
    m.qconfig = tq.get_default_qconfig("x86")
    tq.prepare(m, inplace=True)
    with torch.no_grad():
        for i in range(0, n_calib, 64):
            m(norm_cpu(x_calib[i:i + 64, :, off:off + D.CROP, off:off + D.CROP]))
    tq.convert(m, inplace=True)
    return m


def qat(float_model, xtr, ytr, aug: str, epochs: int = 10, lr: float = 1e-3, seed: int = 0):
    torch.manual_seed(seed)
    m = copy.deepcopy(float_model).cpu().train()
    m.fuse_model(is_qat=True)
    m.qconfig = tq.get_default_qat_qconfig("x86")
    tq.prepare_qat(m, inplace=True)
    m = m.cuda()
    dl = DataLoader(D.CachedDS(xtr, ytr, D.train_transform(aug)), batch_size=64, shuffle=True,
                    drop_last=True, num_workers=8, pin_memory=True)
    norm = D.GpuNormalize().cuda()
    opt = torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9, weight_decay=5e-5)
    total = epochs * len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / total)))
    for ep in range(epochs):
        m.train()
        if ep >= max(1, epochs // 2):
            m.apply(torch.ao.nn.intrinsic.qat.freeze_bn_stats)
        if ep >= max(1, epochs - 3):
            m.apply(tq.disable_observer)
        for xb, yb in dl:
            xb, yb = norm(xb.cuda(non_blocking=True)), yb.cuda(non_blocking=True)
            loss = F.cross_entropy(m(xb), yb)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
    m = m.cpu().eval()
    return tq.convert(m)
