"""Figures for the report from runs/*.json. Missing stages are skipped."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path(__file__).resolve().parents[1] / "runs"
F = Path(__file__).resolve().parents[1] / "report" / "figs"
F.mkdir(parents=True, exist_ok=True)
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
                     "lines.linewidth": 2, "savefig.dpi": 200, "savefig.bbox": "tight", "figure.facecolor": "white"})


def j(n):
    p = R / n
    return json.loads(p.read_text()) if p.exists() else None


def save(fig, n):
    fig.savefig(F / n); plt.close(fig); print("wrote", n)


def fig_profile():
    p = j("profile.json")
    if not p: return
    fig, ax = plt.subplots(figsize=(6.6, 3.3))
    rows = [r for r in p["loader"] if r["workers"] in (0, 4, 8)]
    labs = [f"{'JPEG decode' if r['source']=='jpeg' else 'uint8 cache'}, {r['aug']}, {r['workers']} workers" for r in rows]
    cols = [C2 if r["source"] == "jpeg" else C1 for r in rows]
    ax.barh(range(len(rows)), [r["img_per_s"] for r in rows], color=cols, height=0.65)
    g = next(r for r in p["gpu"] if r["arch"] == "resnet18" and r["amp"] and r["channels_last"] and r["bs"] == 64)
    ax.axvline(g["img_per_s"], color=INK, lw=1.2, ls="--")
    ax.text(g["img_per_s"] * 1.05, 0.5, f"GPU consumes\n{g['img_per_s']:.0f} img/s", color=INK, fontsize=8.5, va="center")
    for i, r in enumerate(rows):
        ax.text(r["img_per_s"] * 1.04, i, f"{r['img_per_s']:.0f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(range(len(rows)), labs, fontsize=8.5); ax.invert_yaxis()
    ax.set_xscale("log"); ax.set_xlim(200, 6e4); ax.set_xlabel("images / s (log)"); ax.grid(axis="y", visible=False)
    ax.set_title("Data loader throughput vs GPU consumption", loc="left", fontsize=11)
    save(fig, "profile_loader.png")
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    rs = [r for r in p["gpu"] if r["arch"] == "resnet18" and r["bs"] == 64]
    lab = ["FP32", "AMP FP16", "AMP + channels_last"]
    ax.barh(lab, [r["img_per_s"] for r in rs], color=C1, height=0.55)
    for i, r in enumerate(rs):
        ax.text(r["img_per_s"] + 40, i, f"{r['img_per_s']:.0f}  ({r['img_per_s']/rs[0]['img_per_s']:.2f}x)",
                va="center", fontsize=9, color=INK)
    ax.set_xlabel("training images / s (ResNet-18, bs 64, 128 px, RTX 3060)")
    ax.set_xlim(0, max(r["img_per_s"] for r in rs) * 1.35); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    save(fig, "profile_gpu.png")


def fig_lr():
    s = j("summary_lr.json")
    if not s: return
    fig, ax = plt.subplots(figsize=(5.2, 2.9))
    for o, c, lab in [("sgd", C1, "SGD + Nesterov (wd 5e-4)"), ("adamw", C2, "AdamW (wd 0.05)")]:
        rs = [r for r in s["rows"] if r["opt"] == o]
        ax.plot([r["lr"] for r in rs], [r["val"] * 100 for r in rs], "o-", color=c, label=lab, ms=6)
    ax.set_xscale("log"); ax.set_xlabel("peak learning rate"); ax.set_ylabel("val accuracy (%)")
    ax.legend(frameon=False, fontsize=9); ax.set_title("LR sweep (cosine, 80 epochs, seed 0)", loc="left", fontsize=11)
    save(fig, "lr_sweep.png")


def dotplot(rows, title, fname, show_train=False):
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(rows) + 1.0))
    y = np.arange(len(rows))
    vm = [r["val_mean"] * 100 for r in rows]; vs = [r["val_std"] * 100 for r in rows]
    tm = [r["test_mean"] * 100 for r in rows]; ts = [r["test_std"] * 100 for r in rows]
    ax.errorbar(vm, y - 0.13, xerr=vs, fmt="o", color=C1, ms=6, capsize=3, label="val (selection)")
    ax.errorbar(tm, y + 0.13, xerr=ts, fmt="s", color=C2, ms=6, capsize=3, label="test (held out)")
    if show_train:
        ax.plot([r["train_acc"] * 100 for r in rows], y, "^", color=INK2, ms=6, label="train (last epoch, augmented batches)")
    ax.set_yticks(y, [r["label"] for r in rows]); ax.invert_yaxis()
    ax.set_xlabel("accuracy (%), mean ± std over 3 seeds"); ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, fontsize=8, loc="lower right", bbox_to_anchor=(1, 1.0), ncol=3, handletextpad=0.3)
    ax.set_title(title, loc="left", fontsize=11, pad=22)
    save(fig, fname)


def fig_curves():
    s = j("summary_sched.json")
    if not s: return
    fig, axes = plt.subplots(1, 2, figsize=(9, 3), sharey=True)
    pal = [C1, C2, C3, "#eda100", "#6250d6"]
    for ax, opt in zip(axes, ["SGD", "AdamW"]):
        for i, r in enumerate([r for r in s["rows"] if r["label"].startswith(opt + " ")]):
            hs = [j(f"{rn}/result.json")["history"] for rn in r["runs"]]
            v = np.mean([[h["val_acc"] for h in H] for H in hs], 0) * 100
            ax.plot(np.arange(1, len(v) + 1), v, color=pal[i], lw=1.6, label=r["label"][len(opt) + 1:])
        ax.set_title(opt, loc="left", fontsize=11); ax.set_xlabel("epoch")
    axes[0].set_ylabel("val accuracy (%), seed mean"); axes[1].legend(frameon=False, fontsize=8.5)
    save(fig, "sched_curves.png")


def fig_hpo():
    s = j("summary_hpo.json")
    if not s: return
    t = [x for x in s["trials"] if x["value"] is not None or x["state"] == "PRUNED"]
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    done = [x for x in t if x["state"] == "COMPLETE"]; pr = [x for x in t if x["state"] == "PRUNED"]
    ax.scatter([x["lr"] for x in pr], [x["wd"] for x in pr], marker="x", color=INK2, s=30, label="pruned")
    sc = ax.scatter([x["lr"] for x in done], [x["wd"] for x in done], c=[x["value"] * 100 for x in done],
                    cmap="Blues", vmin=min(x["value"] for x in done) * 100 - 5, s=60, edgecolor=INK2, lw=0.6,
                    label="completed")
    b = s["best_params"]; ax.scatter([b["lr"]], [b["wd"]], marker="*", s=220, color=C2, edgecolor=INK, lw=0.6,
                                     label="best", zorder=5)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("learning rate"); ax.set_ylabel("weight decay")
    fig.colorbar(sc, ax=ax, label="val accuracy (%) at 40 ep")
    ax.legend(frameon=False, fontsize=8, loc="best"); ax.set_title("Optuna TPE trials", loc="left", fontsize=11)
    save(fig, "hpo_scatter.png")


def fig_quant():
    s = j("summary_quant.json")
    if not s: return
    from matplotlib.ticker import FixedLocator, NullLocator
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    mk = {"FP32": "o", "INT8 PTQ": "s", "INT8 QAT": "D"}
    for model, c in [("ResNet-18 (teacher)", C1), ("MobileNetV2-0.5 + KD", C2)]:
        rs = {r["variant"]: r for r in s["rows"] if r["model"] == model}
        for v, r in rs.items():
            ax.scatter(r["cpu1_ms"], r["test"] * 100, marker=mk[v], color=c, s=60, edgecolor="white", zorder=3)
        p, q = rs["INT8 PTQ"], rs["INT8 QAT"]
        same = abs(p["test"] - q["test"]) < 0.004 and abs(p["cpu1_ms"] / q["cpu1_ms"] - 1) < 0.05
        labels = [("FP32", rs["FP32"], (6, -14))]
        labels += [("INT8 PTQ = QAT", p, (-8, 8))] if same else [("INT8 PTQ", p, (8, 4)), ("INT8 QAT", q, (8, -14))]
        for t, r, off in labels:
            ax.annotate(f"{t}, {r['size_mb']:.1f} MB", (r["cpu1_ms"], r["test"] * 100), textcoords="offset points",
                        xytext=off, fontsize=8, color=INK2, ha="right" if off[0] < 0 else "left")
        ax.plot([], [], marker="o", ls="", color=c, label=model)
    ax.set_xscale("log"); ax.xaxis.set_major_locator(FixedLocator([2, 3, 5, 8, 12, 20]))
    ax.xaxis.set_minor_locator(NullLocator()); ax.set_xticklabels(["2", "3", "5", "8", "12", "20"])
    ax.set_xlim(1.8, 30); ax.set_ylim(56, 76)
    ax.set_xlabel(f"CPU latency (ms), batch 1, 1 thread, {s['cpu'].split(':')[-1].strip().replace('12th Gen ', '').replace('(R)', '').replace('(TM)', '')}")
    ax.set_ylabel("test accuracy (%)"); ax.legend(frameon=False, fontsize=8.5, loc="center right")
    ax.set_title("Accuracy vs latency after compression", loc="left", fontsize=11, pad=10)
    save(fig, "quant_pareto.png")


def fig_transfer():
    t, a = j("transfer/summary.json"), j("summary_aug.json")
    if not t or "seeds" not in t or not a: return
    sd = t["seeds"]
    rows = [("ResNet-18, from scratch (128 px)", a["best"]["test_mean"], a["best"]["test_std"], INK2),
            ("ResNet-18, ImageNet (128 px)", sd["r18_in1k_128"]["test_mean"], sd["r18_in1k_128"]["test_std"], C1),
            ("ResNet-18, ImageNet (224 px)", sd["r18_in1k_224"]["test_mean"], sd["r18_in1k_224"]["test_std"], C1),
            ("CLIP ViT-B/16, zero-shot", t["clip_zeroshot"]["test"], None, C3),
            ("CLIP ViT-B/16, linear probe", t["clip_linear_probe"]["best"]["test"], None, C3),
            ("CLIP ViT-B/16, fine-tune", sd["clip_b16_224"]["test_mean"], sd["clip_b16_224"]["test_std"], C2)]
    if "pseudo" in t:
        st = t["pseudo"]["student"]
        rows.append(("CLIP ViT-B/16, fine-tune + pseudo-labels", st["test_mean"], st["test_std"], C2))
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(rows) + 0.9))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, m, sdv, c) in zip(y, rows):
        ax.barh(yi, m * 100, color=c, height=0.62)
        if sdv is not None: ax.errorbar(m * 100, yi, xerr=sdv * 100, color=INK, capsize=3, lw=1)
        ax.text(m * 100 + (sdv or 0) * 100 + 0.8, yi, f"{m*100:.1f}" + (f" ± {sdv*100:.1f}" if sdv is not None else ""), va="center", fontsize=9)
    ax.set_yticks(y, [r[0] for r in rows]); ax.set_xlim(50, 106); ax.set_xlabel("Food-11 test accuracy (%)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Pretrained backbones (mean ± std over 3 seeds where shown)", fontsize=10)
    save(fig, "transfer.png")


if __name__ == "__main__":
    fig_profile(); fig_lr(); fig_curves()
    for st, title, tr in [("sched", "Optimizer x LR schedule", False), ("aug", "Augmentation ladder", True),
                          ("hpo", "Hand-picked vs Optuna", False), ("distill", "Distillation (200 epochs)", True),
                          ("distill_light", "Student recipe follow-up (200 epochs)", True)]:
        s = j(f"summary_{st}.json")
        if s: dotplot(s["rows"], title, f"{st}_dots.png", tr)
    fig_hpo(); fig_quant(); fig_transfer()
