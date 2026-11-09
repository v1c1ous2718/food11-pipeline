"""Single training run. Everything a run does is decided by a config dict."""
from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from . import data as D
from .models import build, n_params

RUNS = D.ROOT / "runs"


@dataclass
class Cfg:
    name: str
    arch: str = "resnet18"
    epochs: int = 80
    bs: int = 64
    opt: str = "sgd"            # sgd | adamw
    lr: float = 0.1
    wd: float = 5e-4
    sched: str = "cosine"       # const | step | cosine | onecycle
    warmup: int = 0             # epochs of linear warmup (cosine/const/step)
    aug: str = "basic"          # none | basic | randaug
    mix: float = 0.0            # prob of applying MixUp/CutMix to a batch
    smooth: float = 0.0         # label smoothing
    tta: bool = False           # flip TTA at eval
    amp: bool = True
    channels_last: bool = True
    seed: int = 0
    # knowledge distillation
    teacher: str = ""           # run name of teacher
    kd_alpha: float = 0.0       # weight on KD term
    kd_T: float = 4.0
    eval_test: bool = True
    # hyperparameter study (defaults = original behaviour)
    momentum: float = 0.9       # SGD only
    nesterov: bool = True       # SGD only
    dropout: float = 0.0        # before the classifier
    hidden: int = 0             # >0: 512 -> hidden -> 11 MLP head (ReLU)
    l1: float = 0.0             # L1 penalty on weight matrices / kernels, added to the loss
    cos_T: float = 1.0          # cosine period as a multiple of the run (2.0 = stops at half amplitude)
    act: str = "relu"          # relu | silu | gelu (ResNet-18 activations)
    restart: int = 0            # >0: cosine with warm restarts every `restart` epochs (SGDR, T_mult 1)
    extra: dict = field(default_factory=dict)


def seed_all(s: int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def lr_at(cfg: Cfg, step: int, total: int, spe: int) -> float:
    """Multiplier on base lr for a given optimizer step."""
    e = step / spe
    if cfg.sched == "onecycle":
        return 1.0  # handled by torch OneCycleLR
    if cfg.warmup and e < cfg.warmup:
        return (step + 1) / (cfg.warmup * spe)
    if cfg.sched == "const":
        return 1.0
    if cfg.sched == "step":
        return 0.1 ** int(e >= 0.5 * cfg.epochs) * 0.1 ** int(e >= 0.75 * cfg.epochs)
    if cfg.sched == "cosine":
        if cfg.restart:
            t = (step % (cfg.restart * spe)) / (cfg.restart * spe)
        else:
            t = (step - cfg.warmup * spe) / max(1, total - cfg.warmup * spe) / cfg.cos_T
        return 0.5 * (1 + math.cos(math.pi * t))
    raise ValueError(cfg.sched)


@torch.no_grad()
def evaluate(model, x: torch.Tensor, y: torch.Tensor, norm, tta=False, bs=256, return_logits=False):
    model.eval()
    dev = next(model.parameters()).device
    off = (D.STORE - D.CROP) // 2
    out = []
    for i in range(0, len(y), bs):
        xb = x[i:i + bs, :, off:off + D.CROP, off:off + D.CROP].to(dev, non_blocking=True)
        xb = norm(xb).contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.float16):
            lo = model(xb).float()
            if tta:
                lo = (lo.softmax(1) + model(xb.flip(3)).float().softmax(1)).log()
        out.append(lo)
    lo = torch.cat(out).cpu()
    acc = (lo.argmax(1) == y).float().mean().item()
    return (acc, lo) if return_logits else acc


def load_model(run: str, device="cuda"):
    meta = json.loads((RUNS / run / "result.json").read_text())
    c = meta["cfg"]
    m = build(c["arch"], c.get("dropout", 0.0), c.get("hidden", 0), c.get("act", "relu"))
    m.load_state_dict(torch.load(RUNS / run / "model.pt", map_location="cpu"))
    return m.to(device)


def run(cfg: Cfg | dict, quiet=False, callback=None, save=True) -> dict:
    cfg = Cfg(**cfg) if isinstance(cfg, dict) else cfg
    out = RUNS / cfg.name
    if (out / "result.json").exists():
        return json.loads((out / "result.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    seed_all(cfg.seed)
    torch.backends.cudnn.benchmark = True
    dev = "cuda"

    xtr, ytr = D.load_cache("train")
    xva, yva = D.load_cache("val")
    xte, yte = D.load_cache("test")
    g = torch.Generator().manual_seed(cfg.seed)
    dl = DataLoader(D.CachedDS(xtr, ytr, D.train_transform(cfg.aug)), batch_size=cfg.bs, shuffle=True,
                    drop_last=True, num_workers=8, pin_memory=True, persistent_workers=True, generator=g)

    model = build(cfg.arch, cfg.dropout, cfg.hidden, cfg.act).to(dev)
    if cfg.channels_last:
        model = model.to(memory_format=torch.channels_last)
    norm = D.GpuNormalize().to(dev)
    teacher = None
    if cfg.teacher:
        teacher = load_model(cfg.teacher, dev).eval().to(memory_format=torch.channels_last)
        for p in teacher.parameters():
            p.requires_grad_(False)

    decay = [p for n, p in model.named_parameters() if p.ndim > 1]
    no_decay = [p for n, p in model.named_parameters() if p.ndim <= 1]
    groups = [{"params": decay, "weight_decay": cfg.wd}, {"params": no_decay, "weight_decay": 0.0}]
    if cfg.opt == "sgd":
        opt = torch.optim.SGD(groups, lr=cfg.lr, momentum=cfg.momentum, nesterov=cfg.nesterov and cfg.momentum > 0)
    elif cfg.opt == "adamw":
        opt = torch.optim.AdamW(groups, lr=cfg.lr)
    else:
        raise ValueError(cfg.opt)
    spe = len(dl)
    total = spe * cfg.epochs
    if cfg.sched == "onecycle":
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=total, pct_start=0.25)
    else:
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_at(cfg, s, total, spe))
    scaler = torch.amp.GradScaler(enabled=cfg.amp)
    mixer = v2.RandomChoice([v2.MixUp(alpha=0.2, num_classes=D.NUM_CLASSES),
                             v2.CutMix(alpha=1.0, num_classes=D.NUM_CLASSES)]) if cfg.mix > 0 else None

    hist = []
    t0 = time.time()
    for ep in range(cfg.epochs):
        model.train()
        tl, tc, tn = 0.0, 0, 0
        for xb, yb in dl:
            xb = norm(xb.to(dev, non_blocking=True))
            yb = yb.to(dev, non_blocking=True)
            target = F.one_hot(yb, D.NUM_CLASSES).float()
            if mixer is not None and random.random() < cfg.mix:
                xb, target = mixer(xb, yb)
            if cfg.smooth:
                target = target * (1 - cfg.smooth) + cfg.smooth / D.NUM_CLASSES
            if cfg.channels_last:
                xb = xb.contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.float16, enabled=cfg.amp):
                logits = model(xb)
                loss = torch.sum(-target * F.log_softmax(logits.float(), 1), 1).mean()
                if cfg.l1:
                    loss = loss + cfg.l1 * sum(w.abs().sum() for w in decay)
                if teacher is not None:
                    with torch.no_grad():
                        tlog = teacher(xb).float()
                    kd = F.kl_div(F.log_softmax(logits.float() / cfg.kd_T, 1), F.softmax(tlog / cfg.kd_T, 1),
                                  reduction="batchmean") * cfg.kd_T ** 2
                    loss = (1 - cfg.kd_alpha) * loss + cfg.kd_alpha * kd
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tl += loss.item() * len(yb)
            tc += (logits.argmax(1) == yb).sum().item()
            tn += len(yb)
        va = evaluate(model, xva, yva, norm, cfg.tta)
        hist.append({"epoch": ep + 1, "loss": tl / tn, "train_acc": tc / tn, "val_acc": va,
                     "lr": opt.param_groups[0]["lr"], "time": time.time() - t0})
        if callback is not None:
            callback(ep, va)
        if not quiet and ((ep + 1) % 20 == 0 or ep == 0):
            print(f"[{cfg.name}] ep {ep+1:3d} loss {tl/tn:.3f} train {tc/tn:.3f} val {va:.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    res = {"cfg": asdict(cfg), "params": n_params(model), "history": hist,
           "val_acc": hist[-1]["val_acc"], "train_acc": hist[-1]["train_acc"],
           "train_time_s": time.time() - t0}
    if cfg.eval_test:
        res["test_acc"] = evaluate(model, xte, yte, norm, cfg.tta)
    if save:
        torch.save(model.state_dict(), out / "model.pt")
    (out / "result.json").write_text(json.dumps(res, indent=1))
    return res
