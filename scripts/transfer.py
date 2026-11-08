"""Pretrained-backbone stage. Design fixed before running:
  1. CLIP ViT-B/16 zero-shot and linear probe (no fine-tuning).
  2. LR sweep on seed 0, chosen by val:
       ResNet-18 ImageNet @128 px (same input as the from-scratch runs) and @224 px: lr {3e-4, 1e-3, 3e-3}, 30 epochs
       CLIP ViT-B/16 @224 px: lr {5e-6, 1e-5, 3e-5}, 10 epochs, wd 0.1, batch 32
  3. Best lr of each model x seeds 0-2.
  4. Pseudo-labeling: the best model (val mean) labels the 6,786 unlabeled images with flip TTA;
     keep confidence >= 0.9; retrain the same config on train + pseudo, seeds 0-2.

uv run python scripts/transfer.py [stage ...]   (stages: clip sweep seeds pseudo)
"""
import json
import sys
from statistics import mean, stdev

import numpy as np
import torch

from f11 import transfer as T

SUM = T.RUNS / "summary.json"
MODELS = {
    "r18_in1k_128": dict(arch="r18_in1k", side=144, crop=128, epochs=30, bs=64, wd=0.05, lrs=[3e-4, 1e-3, 3e-3]),
    "r18_in1k_224": dict(arch="r18_in1k", side=256, crop=224, epochs=30, bs=64, wd=0.05, lrs=[3e-4, 1e-3, 3e-3]),
    "clip_b16_224": dict(arch="clip_b16", side=256, crop=224, epochs=10, bs=32, wd=0.1, lrs=[5e-6, 1e-5, 3e-5]),
}


def summary() -> dict:
    return json.loads(SUM.read_text()) if SUM.exists() else {}


def put(k, v):
    s = summary(); s[k] = v
    SUM.parent.mkdir(parents=True, exist_ok=True)
    SUM.write_text(json.dumps(s, indent=1))


def cfg(model, lr, seed, **kw):
    m = {k: v for k, v in MODELS[model].items() if k != "lrs"}
    tag = f"_pseudo" if kw.get("pseudo") else ""
    return {**m, "lr": lr, "seed": seed, "name": f"{model}_lr{lr:g}{tag}_s{seed}", **kw}


def stage_clip():
    f = T.clip_features()
    zs = {k: ((f[k][0] @ f["zs_w"].T + f["zs_b"]).argmax(1) == f[k][1]).float().mean().item() for k in ["val", "test"]}
    lp = T.linear_probe(f)
    print(f"zero-shot val {zs['val']:.4f} test {zs['test']:.4f}; linear probe {lp['best']}", flush=True)
    put("clip_zeroshot", zs); put("clip_linear_probe", lp)


def stage_sweep():
    out = {}
    for m, spec in MODELS.items():
        rows = [T.train(cfg(m, lr, 0, save=True)) for lr in spec["lrs"]]
        out[m] = [{"lr": r["cfg"]["lr"], "val": r["val_acc"], "test": r["test_acc"]} for r in rows]
        print(m, out[m], flush=True)
    put("sweep", out)


def agg(rs, label):
    v, t = [r["val_acc"] for r in rs], [r["test_acc"] for r in rs]
    row = {"label": label, "val": v, "test": t, "val_mean": mean(v), "val_std": stdev(v), "test_mean": mean(t),
           "test_std": stdev(t), "train_time_s": mean(r["train_time_s"] for r in rs),
           "peak_mem_gb": max(r["peak_mem_gb"] for r in rs), "runs": [r["cfg"]["name"] for r in rs]}
    print(f"  {label:34s} val {row['val_mean']*100:.1f}±{row['val_std']*100:.1f}  "
          f"test {row['test_mean']*100:.1f}±{row['test_std']*100:.1f}", flush=True)
    return row


def stage_seeds():
    sw, rows = summary()["sweep"], {}
    for m in MODELS:
        lr = max(sw[m], key=lambda r: r["val"])["lr"]
        rows[m] = {**agg([T.train(cfg(m, lr, s, save=(s == 0))) for s in range(3)], m), "lr": lr}
    put("seeds", rows)


def stage_pseudo(thr=0.9):
    rows = summary()["seeds"]
    best = max(rows, key=lambda m: rows[m]["val_mean"])
    lr, spec = rows[best]["lr"], MODELS[best]
    teacher = T.load(cfg(best, lr, 0)["name"])
    xun, _ = T.cache("unlabeled", spec["side"])
    norm = T.Norm("clip" if spec["arch"] == "clip_b16" else "imagenet").cuda()
    with torch.no_grad():
        p = T.predict(teacher, xun, norm, spec["side"], spec["crop"], tta=True)
    conf, y = p.max(1)
    keep = (conf >= thr).nonzero().squeeze(1)
    path = T.RUNS / f"pseudo_{best}.npz"
    np.savez(path, idx=keep.numpy(), y=y[keep].numpy())
    hist = np.bincount(y[keep].numpy(), minlength=11).tolist()
    print(f"teacher {best} lr {lr}: kept {len(keep)}/{len(y)} at conf >= {thr}; per class {hist}", flush=True)
    del teacher, xun; torch.cuda.empty_cache()
    student = agg([T.train(cfg(best, lr, s, pseudo=str(path))) for s in range(3)], f"{best} + pseudo-labels")
    put("pseudo", {"teacher": best, "lr": lr, "threshold": thr, "kept": len(keep), "total": len(y),
                   "per_class": hist, "student": student, "baseline": rows[best]})


STAGES = {"clip": stage_clip, "sweep": stage_sweep, "seeds": stage_seeds, "pseudo": stage_pseudo}

if __name__ == "__main__":
    for st in sys.argv[1:] or list(STAGES):
        STAGES[st]()
