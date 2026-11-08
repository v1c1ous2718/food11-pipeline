"""Pretrained backbones on Food-11: how much of the gap to a usable classifier is pretraining?

Same splits and selection rule as the from-scratch study (val chooses, test is only reported).
  r18_in1k   torchvision ResNet-18, ImageNet-1k weights, full fine-tune
  clip_b16   CLIP ViT-B/16 image tower; head initialized from the zero-shot text embeddings, full fine-tune
Plus CLIP zero-shot, a CLIP linear probe, and pseudo-labeling with the 6,786 unlabeled images.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from . import data as D
from .train import seed_all

RUNS = D.ROOT / "runs" / "transfer"
CLIP_ID = "openai/clip-vit-base-patch16"
CLASSES = ["bread", "dairy product", "dessert", "egg", "fried food", "meat", "noodles or pasta", "rice",
           "seafood", "soup", "vegetables or fruit"]
NORMS = {"imagenet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
         "clip": ((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))}


# ----------------------------------------------------------------- data
def cache(name: str, side: int) -> tuple[torch.Tensor, torch.Tensor]:
    """uint8 CHW cache at a given side length (144 is the from-scratch cache)."""
    if side == D.STORE:
        return D.load_cache(name)
    p = D.CACHE / f"{name}_{side}.npz"
    if not p.exists():
        from concurrent.futures import ProcessPoolExecutor
        from functools import partial
        items = D.list_split()[name]
        with ProcessPoolExecutor() as ex:
            x = np.stack(list(ex.map(partial(D.load_resized, side=side), [i for i, _ in items], chunksize=64)))
        np.savez(p, x=x, y=np.array([c for _, c in items], dtype=np.int64))
    d = np.load(p)
    return torch.from_numpy(d["x"]).permute(0, 3, 1, 2).contiguous(), torch.from_numpy(d["y"])


class DS(Dataset):
    def __init__(self, x, y, tf):
        self.x, self.y, self.tf = x, y, tf

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.tf(self.x[i]), self.y[i]


class Norm(torch.nn.Module):
    def __init__(self, kind):
        super().__init__()
        m, s = NORMS[kind]
        self.register_buffer("m", torch.tensor(m).view(1, 3, 1, 1) * 255)
        self.register_buffer("s", torch.tensor(s).view(1, 3, 1, 1) * 255)

    def forward(self, x):
        return (x.float() - self.m) / self.s


# ----------------------------------------------------------------- models
class ClipClassifier(torch.nn.Module):
    """CLIP image tower -> L2-normalized embedding -> linear head (init = zero-shot text weights)."""

    def __init__(self):
        super().__init__()
        from transformers import CLIPModel, CLIPTokenizer
        clip = CLIPModel.from_pretrained(CLIP_ID)
        tok = CLIPTokenizer.from_pretrained(CLIP_ID)
        with torch.no_grad():
            t = tok([f"a photo of {c}, a type of food." for c in CLASSES], padding=True, return_tensors="pt")
            w = F.normalize(clip.get_text_features(**t), dim=-1)
        self.vision, self.proj = clip.vision_model, clip.visual_projection
        self.head = torch.nn.Linear(w.shape[1], len(CLASSES), bias=True)
        self.head.weight.data.copy_(w * clip.logit_scale.exp().item())
        self.head.bias.data.zero_()

    def embed(self, x):
        return F.normalize(self.proj(self.vision(pixel_values=x).pooler_output), dim=-1)

    def forward(self, x):
        return self.head(self.embed(x))


def build(arch: str) -> tuple[torch.nn.Module, str]:
    if arch == "r18_in1k":
        from torchvision.models import ResNet18_Weights, resnet18
        m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        m.fc = torch.nn.Linear(m.fc.in_features, D.NUM_CLASSES)
        return m, "imagenet"
    if arch == "clip_b16":
        return ClipClassifier(), "clip"
    raise ValueError(arch)


# ----------------------------------------------------------------- fine-tuning
@dataclass
class Cfg:
    name: str
    arch: str = "r18_in1k"
    side: int = 256             # cache side; eval = center crop
    crop: int = 224
    epochs: int = 30
    bs: int = 64
    lr: float = 1e-3
    wd: float = 0.05
    warmup: float = 1.0         # epochs
    smooth: float = 0.1
    aug: str = "randaug"        # basic | randaug
    pseudo: str = ""            # path to pseudo-label file (npz with idx, y) or ""
    seed: int = 0
    save: bool = False


@torch.no_grad()
def predict(model, x, norm, side, crop, bs=128, tta=False):
    model.eval()
    off = (side - crop) // 2
    out = []
    for i in range(0, len(x), bs):
        xb = norm(x[i:i + bs, :, off:off + crop, off:off + crop].cuda(non_blocking=True))
        with torch.autocast("cuda", dtype=torch.float16):
            p = model(xb).float().softmax(1)
            if tta:
                p = (p + model(xb.flip(3)).float().softmax(1)) / 2
        out.append(p.cpu())
    return torch.cat(out)


def train(cfg: Cfg | dict, quiet=False) -> dict:
    cfg = Cfg(**cfg) if isinstance(cfg, dict) else cfg
    out = RUNS / cfg.name
    if (out / "result.json").exists():
        return json.loads((out / "result.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    seed_all(cfg.seed)
    torch.backends.cudnn.benchmark = True
    xtr, ytr = cache("train", cfg.side)
    xva, yva = cache("val", cfg.side)
    xte, yte = cache("test", cfg.side)
    n_pseudo = 0
    if cfg.pseudo:
        pl = np.load(cfg.pseudo)
        xun, _ = cache("unlabeled", cfg.side)
        idx = torch.from_numpy(pl["idx"])
        xtr = torch.cat([xtr, xun[idx]]); ytr = torch.cat([ytr, torch.from_numpy(pl["y"])])
        n_pseudo = len(idx)
        del xun
    ops = [v2.RandomResizedCrop(cfg.crop, scale=(0.35, 1.0), antialias=True), v2.RandomHorizontalFlip()]
    if cfg.aug == "randaug":
        ops.append(v2.RandAugment(num_ops=2, magnitude=9))
    g = torch.Generator().manual_seed(cfg.seed)
    dl = DataLoader(DS(xtr, ytr, v2.Compose(ops)), batch_size=cfg.bs, shuffle=True, drop_last=True,
                    num_workers=8, pin_memory=True, persistent_workers=True, generator=g)
    model, nk = build(cfg.arch)
    model = model.cuda()
    norm = Norm(nk).cuda()
    decay = [p for p in model.parameters() if p.ndim > 1]
    no_decay = [p for p in model.parameters() if p.ndim <= 1]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg.wd},
                             {"params": no_decay, "weight_decay": 0.0}], lr=cfg.lr)
    spe, total = len(dl), len(dl) * cfg.epochs
    wu = int(cfg.warmup * spe)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / wu if s < wu else 0.5 * (1 + math.cos(math.pi * (s - wu) / max(1, total - wu))))
    scaler = torch.amp.GradScaler()
    hist, t0 = [], time.time()
    for ep in range(cfg.epochs):
        model.train()
        tl = tn = 0
        for xb, yb in dl:
            xb, yb = norm(xb.cuda(non_blocking=True)), yb.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = F.cross_entropy(model(xb).float(), yb, label_smoothing=cfg.smooth)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            tl += loss.item() * len(yb); tn += len(yb)
        va = (predict(model, xva, norm, cfg.side, cfg.crop).argmax(1) == yva).float().mean().item()
        hist.append({"epoch": ep + 1, "loss": tl / tn, "val_acc": va, "time": time.time() - t0})
        if not quiet:
            print(f"[{cfg.name}] ep {ep+1:2d} loss {tl/tn:.3f} val {va:.3f} ({time.time()-t0:.0f}s)", flush=True)
    te = (predict(model, xte, norm, cfg.side, cfg.crop).argmax(1) == yte).float().mean().item()
    res = {"cfg": asdict(cfg), "history": hist, "val_acc": hist[-1]["val_acc"], "test_acc": te,
           "n_train": len(ytr), "n_pseudo": n_pseudo, "train_time_s": time.time() - t0,
           "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30}
    if cfg.save:
        torch.save(model.state_dict(), out / "model.pt")
    (out / "result.json").write_text(json.dumps(res, indent=1))
    print(f"[{cfg.name}] val {res['val_acc']:.4f} test {te:.4f}", flush=True)
    del model, opt, dl
    torch.cuda.empty_cache()
    return res


def load(name: str) -> torch.nn.Module:
    cfg = Cfg(**json.loads((RUNS / name / "result.json").read_text())["cfg"])
    m, _ = build(cfg.arch)
    m.load_state_dict(torch.load(RUNS / name / "model.pt", map_location="cpu"))
    return m.cuda().eval()


# ----------------------------------------------------------------- CLIP without fine-tuning
@torch.no_grad()
def clip_features(side=256, crop=224) -> dict:
    p = RUNS / "clip_b16_features.pt"
    if p.exists():
        return torch.load(p)
    m = ClipClassifier().cuda().eval()
    norm = Norm("clip").cuda()
    off = (side - crop) // 2
    out = {}
    for split in ["train", "val", "test"]:
        x, y = cache(split, side)
        fs = []
        for i in range(0, len(x), 128):
            xb = norm(x[i:i + 128, :, off:off + crop, off:off + crop].cuda())
            with torch.autocast("cuda", dtype=torch.float16):
                fs.append(m.embed(xb).float().cpu())
        out[split] = (torch.cat(fs), y)
    out["zs_w"], out["zs_b"] = m.head.weight.detach().cpu(), m.head.bias.detach().cpu()
    torch.save(out, p)
    return out


def linear_probe(feats: dict, wds=(0.0, 1e-4, 1e-3, 1e-2, 1e-1)) -> dict:
    """Logistic regression on frozen CLIP embeddings; weight decay picked on val."""
    xtr, ytr = feats["train"]
    rows = []
    for wd in wds:
        torch.manual_seed(0)
        lin = torch.nn.Linear(xtr.shape[1], D.NUM_CLASSES)
        opt = torch.optim.LBFGS(lin.parameters(), lr=1, max_iter=500, history_size=20,
                                line_search_fn="strong_wolfe")
        s = 100.0  # CLIP logit scale; makes the problem well conditioned for unit-norm features

        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(lin(xtr * s), ytr) + wd * lin.weight.pow(2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        acc = {k: (lin(feats[k][0] * s).argmax(1) == feats[k][1]).float().mean().item() for k in ["val", "test"]}
        rows.append({"wd": wd, **acc})
    best = max(rows, key=lambda r: r["val"])
    return {"rows": rows, "best": best}
