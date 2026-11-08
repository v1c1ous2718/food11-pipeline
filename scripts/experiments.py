"""All experiment stages. Model selection uses mean val accuracy only; test is reported, never chosen on.

uv run python scripts/experiments.py [stage ...]   (stages: lr sched aug hpo distill quant)
"""
import json, sys
from statistics import mean, stdev

from f11 import data as D
from f11.train import RUNS, run

SEEDS = [0, 1, 2]
BASE = dict(arch="resnet18", epochs=80, bs=64, aug="basic", sched="cosine")


def name(c: dict) -> str:
    keys = ["arch", "opt", "lr", "wd", "sched", "warmup", "aug", "mix", "smooth", "epochs", "teacher", "kd_alpha", "seed"]
    d = {"opt": "sgd", "lr": 0.1, "wd": 5e-4, "sched": "cosine", "warmup": 0, "aug": "basic", "mix": 0.0,
         "smooth": 0.0, "epochs": 80, "teacher": "", "kd_alpha": 0.0, "seed": 0, **c}
    return "_".join(f"{d[k]}" for k in keys if d[k] != "" ).replace("/", "-")


def go(c: dict) -> dict:
    c = {**BASE, **c}
    return run({**c, "name": name(c)})


def seeds(c: dict, label: str) -> dict:
    rs = [go({**c, "seed": s}) for s in SEEDS]
    v, t = [r["val_acc"] for r in rs], [r["test_acc"] for r in rs]
    out = {"label": label, "cfg": c, "val": v, "test": t, "val_mean": mean(v), "val_std": stdev(v),
           "test_mean": mean(t), "test_std": stdev(t), "train_acc": mean(r["train_acc"] for r in rs),
           "runs": [name({**BASE, **c, "seed": s}) for s in SEEDS]}
    print(f"  {label:28s} val {out['val_mean']*100:.1f}±{out['val_std']*100:.1f}  "
          f"test {out['test_mean']*100:.1f}±{out['test_std']*100:.1f}  train {out['train_acc']*100:.1f}", flush=True)
    return out


def save(stage, obj):
    (RUNS / f"summary_{stage}.json").write_text(json.dumps(obj, indent=1))


def load(stage):
    return json.loads((RUNS / f"summary_{stage}.json").read_text())


def stage_lr():
    print("== LR sweep (cosine, basic aug, seed 0)")
    rows = []
    for opt, wd, lrs in [("sgd", 5e-4, [0.01, 0.03, 0.1, 0.3]), ("adamw", 0.05, [3e-4, 1e-3, 3e-3])]:
        for lr in lrs:
            r = go(dict(opt=opt, lr=lr, wd=wd))
            rows.append({"opt": opt, "lr": lr, "wd": wd, "val": r["val_acc"], "test": r["test_acc"],
                         "history": [h["val_acc"] for h in r["history"]]})
            print(f"  {opt} lr={lr}: val {r['val_acc']:.3f}", flush=True)
    best = {o: max((r for r in rows if r["opt"] == o), key=lambda r: r["val"]) for o in ["sgd", "adamw"]}
    save("lr", {"rows": rows, "best": best})


def stage_sched():
    b = load("lr")["best"]
    sgd = dict(opt="sgd", lr=b["sgd"]["lr"], wd=b["sgd"]["wd"])
    adw = dict(opt="adamw", lr=b["adamw"]["lr"], wd=b["adamw"]["wd"])
    print(f"== Optimizer x schedule (3 seeds) sgd lr={sgd['lr']} adamw lr={adw['lr']}")
    res = []
    for o, olab in [(sgd, "SGD"), (adw, "AdamW")]:
        for s, w, slab in [("const", 0, "constant"), ("step", 0, "step"), ("cosine", 0, "cosine"),
                           ("cosine", 5, "cosine + 5ep warmup"), ("onecycle", 0, "one-cycle")]:
            res.append(seeds({**o, "sched": s, "warmup": w}, f"{olab} {slab}"))
    save("sched", {"rows": res, "best": max(res, key=lambda r: r["val_mean"])})


def stage_aug():
    best = load("sched")["best"]["cfg"]
    base = {k: best[k] for k in ["opt", "lr", "wd", "sched", "warmup"]}
    print(f"== Augmentation ladder (3 seeds) on {base}")
    ladder = [(dict(aug="none"), "no augmentation"),
              (dict(aug="basic"), "+ crop/flip"),
              (dict(aug="randaug"), "+ RandAugment"),
              (dict(aug="randaug", mix=0.5), "+ MixUp/CutMix (p=0.5)"),
              (dict(aug="randaug", mix=0.5, smooth=0.1), "+ label smoothing 0.1")]
    res = [seeds({**base, **a}, lab) for a, lab in ladder]
    # flip TTA, evaluated on the saved models of the last rung (no retraining)
    import torch
    from f11.train import evaluate, load_model
    xva, yva = D.load_cache("val"); xte, yte = D.load_cache("test")
    norm = D.GpuNormalize().cuda()
    v, t = [], []
    for rn in res[-1]["runs"]:
        m = load_model(rn).to(memory_format=torch.channels_last)
        v.append(evaluate(m, xva, yva, norm, tta=True)); t.append(evaluate(m, xte, yte, norm, tta=True))
    tta = {"label": "+ flip TTA (eval only)", "val": v, "test": t, "val_mean": mean(v), "val_std": stdev(v),
           "test_mean": mean(t), "test_std": stdev(t), "cfg": res[-1]["cfg"], "train_acc": res[-1]["train_acc"]}
    print(f"  {tta['label']:28s} val {mean(v)*100:.1f}±{stdev(v)*100:.1f}  test {mean(t)*100:.1f}±{stdev(t)*100:.1f}")
    res.append(tta)
    save("aug", {"rows": res, "best": max(res[:-1], key=lambda r: r["val_mean"])})


def stage_hpo(n_trials=30, epochs=40):
    import optuna
    best = load("aug")["best"]["cfg"]
    print(f"== Optuna TPE, {n_trials} trials x {epochs} epochs, median pruning, val only")

    def objective(trial):
        c = {**BASE, **best, "epochs": epochs, "eval_test": False,
             "lr": trial.suggest_float("lr", 0.01, 0.5, log=True) if best["opt"] == "sgd"
             else trial.suggest_float("lr", 1e-4, 1e-2, log=True),
             "wd": trial.suggest_float("wd", 1e-5, 2e-3, log=True) if best["opt"] == "sgd"
             else trial.suggest_float("wd", 1e-3, 0.3, log=True),
             "mix": trial.suggest_float("mix", 0.0, 1.0),
             "smooth": trial.suggest_float("smooth", 0.0, 0.2)}
        c["name"] = f"hpo/t{trial.number:02d}"

        def cb(ep, va):
            trial.report(va, ep)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return run(c, quiet=True, callback=cb, save=False)["val_acc"]

    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0),
                             pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=15))
    st.optimize(objective, n_trials=n_trials)
    trials = [{"n": t.number, "state": t.state.name, "value": t.value, **t.params,
               "last_step": max(t.intermediate_values) if t.intermediate_values else None} for t in st.trials]
    print("  best", st.best_params, st.best_value)
    tuned = {**best, **st.best_params}
    rows = [seeds(best, "hand-picked (aug stage best)"), seeds(tuned, "Optuna best")]
    save("hpo", {"trials": trials, "best_params": st.best_params, "rows": rows,
                 "best": max(rows, key=lambda r: r["val_mean"])})


def stage_distill(epochs=200):
    best = load("hpo")["best"]["cfg"]
    rec = {k: best[k] for k in ["opt", "lr", "wd", "sched", "warmup", "aug", "mix", "smooth"]}
    print(f"== Distillation, {epochs} epochs, recipe {rec}")
    teacher = seeds({**rec, "epochs": epochs}, "teacher ResNet-18")
    tname = teacher["runs"][max(range(3), key=lambda i: teacher["val"][i])]
    student = seeds({**rec, "arch": "mbv2_050", "epochs": epochs}, "MobileNetV2-0.5 scratch")
    kd = seeds({**rec, "arch": "mbv2_050", "epochs": epochs, "teacher": tname, "kd_alpha": 0.7},
               "MobileNetV2-0.5 + KD")
    save("distill", {"teacher_run": tname, "rows": [teacher, student, kd], "recipe": rec})


def stage_quant():
    import torch
    from f11 import quant as Q
    from f11.models import build
    from f11.train import load_model, evaluate
    s = load("distill")
    xtr, ytr = D.load_cache("train"); xva, yva = D.load_cache("val"); xte, yte = D.load_cache("test")
    student_run = s["rows"][2]["runs"][max(range(3), key=lambda i: s["rows"][2]["val"][i])]
    rows = []
    for label, rn in [("ResNet-18 (teacher)", s["teacher_run"]), ("MobileNetV2-0.5 + KD", student_run)]:
        fm = load_model(rn, "cpu").eval()
        gpu_val = evaluate(fm.cuda(), xva, yva, D.GpuNormalize().cuda())
        fm = fm.cpu()
        variants = [("FP32", fm), ("INT8 PTQ", Q.ptq(fm, xtr)),
                    ("INT8 QAT", Q.qat(fm, xtr, ytr, aug=s["recipe"]["aug"]))]
        for vlab, m in variants:
            r = {"model": label, "variant": vlab, "run": rn, "size_mb": Q.size_mb(m),
                 "val": Q.acc_cpu(m, xva, yva), "test": Q.acc_cpu(m, xte, yte),
                 "cpu1_ms": Q.latency_ms(m, 1), "cpu4_ms": Q.latency_ms(m, 4)}
            if vlab == "FP32":
                r["gpu_fp16_ms"] = Q.latency_gpu_ms(m)
                r["gpu_val_check"] = gpu_val
            rows.append(r)
            print(f"  {label:22s} {vlab:9s} {r['size_mb']:6.2f} MB  val {r['val']:.3f} test {r['test']:.3f}  "
                  f"cpu1 {r['cpu1_ms']:.2f} ms  cpu4 {r['cpu4_ms']:.2f} ms", flush=True)
    save("quant", {"rows": rows, "cpu": open("/proc/cpuinfo").read().split("model name")[1].split("\n")[0].split(":")[-1].strip()})


STAGES = {"lr": stage_lr, "sched": stage_sched, "aug": stage_aug, "hpo": stage_hpo,
          "distill": stage_distill, "quant": stage_quant}  # distill_light registered below



def stage_distill_light(epochs=200):
    """Follow-up: is the student over-regularized by the teacher's recipe?"""
    s = load("distill")
    rec = {**s["recipe"], "aug": "basic", "mix": 0.0, "smooth": 0.0}
    print(f"== Distillation follow-up, student with light regularization {rec}")
    scratch = seeds({**rec, "arch": "mbv2_050", "epochs": epochs}, "MobileNetV2-0.5 scratch, light aug")
    kd = seeds({**rec, "arch": "mbv2_050", "epochs": epochs, "teacher": s["teacher_run"], "kd_alpha": 0.7},
               "MobileNetV2-0.5 + KD, light aug")
    save("distill_light", {"rows": [scratch, kd], "recipe": rec})


STAGES["distill_light"] = stage_distill_light

if __name__ == "__main__":
    for st in sys.argv[1:] or [k for k in STAGES if k != "distill_light"]:
        STAGES[st]()
