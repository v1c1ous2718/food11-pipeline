"""Food-11 data: one-time JPEG decode into a uint8 cache, plus loaders.

Splits (fixed, stratified, seed 0):
  train  2640  (240/class) from training/labeled   -> fitting
  val     440  ( 40/class) from training/labeled   -> all model selection
  test    660  ( 60/class) = official validation/  -> reported only, never used to choose
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "food-11"
CACHE = ROOT / "data" / "cache"
STORE = 144   # cached side length
CROP = 128    # training / eval resolution
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
NUM_CLASSES = 11


def load_resized(path: str | Path, side: int = STORE) -> np.ndarray:
    """Resize shorter side to `side`, center crop to side x side, HWC uint8."""
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = side / min(w, h)
    im = im.resize((max(side, round(w * s)), max(side, round(h * s))), Image.BILINEAR)
    w, h = im.size
    l, t = (w - side) // 2, (h - side) // 2
    return np.array(im.crop((l, t, l + side, t + side)), dtype=np.uint8)


def list_split() -> dict[str, list[tuple[str, int]]]:
    rng = np.random.default_rng(0)
    train, val, test = [], [], []
    for c in range(NUM_CLASSES):
        files = sorted((RAW / "training" / "labeled" / f"{c:02d}").glob("*.jpg"))
        idx = rng.permutation(len(files))
        val += [(str(files[i]), c) for i in idx[:40]]
        train += [(str(files[i]), c) for i in idx[40:]]
        test += [(str(f), c) for f in sorted((RAW / "validation" / f"{c:02d}").glob("*.jpg"))]
    unl = [(str(f), -1) for f in sorted((RAW / "training" / "unlabeled" / "00").glob("*.jpg"))]
    return {"train": train, "val": val, "test": test, "unlabeled": unl}


def build_cache() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    from concurrent.futures import ProcessPoolExecutor
    for name, items in list_split().items():
        out = CACHE / f"{name}.npz"
        if out.exists():
            continue
        with ProcessPoolExecutor(os.cpu_count()) as ex:
            x = np.stack(list(ex.map(load_resized, [p for p, _ in items], chunksize=64)))
        y = np.array([c for _, c in items], dtype=np.int64)
        np.savez(out, x=x, y=y, paths=np.array([p for p, _ in items]))
        print(name, x.shape)


def load_cache(name: str) -> tuple[torch.Tensor, torch.Tensor]:
    d = np.load(CACHE / f"{name}.npz")
    return torch.from_numpy(d["x"]).permute(0, 3, 1, 2).contiguous(), torch.from_numpy(d["y"])


def train_transform(aug: str):
    """CPU per-sample transforms on uint8 CHW tensors. Normalisation happens on GPU."""
    if aug == "none":
        return v2.CenterCrop(CROP)
    ops = [v2.RandomResizedCrop(CROP, scale=(0.35, 1.0), antialias=True), v2.RandomHorizontalFlip()]
    if aug == "randaug":
        ops.append(v2.RandAugment(num_ops=2, magnitude=9))
    return v2.Compose(ops)


EVAL_TF = v2.CenterCrop(CROP)


class CachedDS(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor, tf):
        self.x, self.y, self.tf = x, y, tf

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.tf(self.x[i]), self.y[i]


class JpegDS(Dataset):
    """Decode-from-disk dataset; only used to profile the loader."""

    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, c = self.items[i]
        x = torch.from_numpy(load_resized(p)).permute(2, 0, 1)
        return self.tf(x), c


class GpuNormalize(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("m", torch.tensor(MEAN).view(1, 3, 1, 1) * 255)
        self.register_buffer("s", torch.tensor(STD).view(1, 3, 1, 1) * 255)

    def forward(self, x):
        return (x.float() - self.m) / self.s


if __name__ == "__main__":
    build_cache()
