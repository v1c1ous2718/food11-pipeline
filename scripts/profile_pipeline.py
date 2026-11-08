"""Where does training time go? Loader throughput vs GPU step throughput."""
import json, time
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from f11 import data as D
from f11.models import build

res = {"loader": [], "gpu": []}

def loader_rate(ds, workers, bs=64, epochs=2):
    """Steady-state: time the last full epoch (persistent workers, prefetch included)."""
    dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=workers, drop_last=True,
                    persistent_workers=workers > 0)
    for _ in range(epochs):
        t = time.perf_counter(); n = 0
        for xb, _ in dl:
            n += len(xb)
        dt = time.perf_counter() - t
    return n / dt

items = D.list_split()["train"]
xtr, ytr = D.load_cache("train")
for src, aug, w in [("jpeg", "basic", 0), ("jpeg", "basic", 4), ("jpeg", "basic", 8), ("jpeg", "basic", 12),
                    ("cache", "basic", 0), ("cache", "basic", 4), ("cache", "basic", 8),
                    ("cache", "randaug", 0), ("cache", "randaug", 4), ("cache", "randaug", 8)]:
    ds = D.JpegDS(items, D.train_transform(aug)) if src == "jpeg" else D.CachedDS(xtr, ytr, D.train_transform(aug))
    r = loader_rate(ds, w)
    res["loader"].append({"source": src, "aug": aug, "workers": w, "img_per_s": r})
    print(res["loader"][-1], flush=True)

torch.backends.cudnn.benchmark = True
x = torch.randint(0, 255, (128, 3, D.CROP, D.CROP), dtype=torch.uint8, device="cuda")
y = torch.randint(0, 11, (128,), device="cuda")
norm = D.GpuNormalize().cuda()
for arch in ["resnet18", "mbv2_050"]:
    for amp, cl in [(False, False), (True, False), (True, True)]:
        m = build(arch).cuda()
        if cl: m = m.to(memory_format=torch.channels_last)
        opt = torch.optim.SGD(m.parameters(), lr=0.01, momentum=0.9)
        scaler = torch.amp.GradScaler(enabled=amp)
        for bs in [64, 128]:
            xb = norm(x[:bs]); yb = y[:bs]
            if cl: xb = xb.contiguous(memory_format=torch.channels_last)
            def step():
                with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                    loss = F.cross_entropy(m(xb), yb)
                opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            for _ in range(15): step()
            torch.cuda.synchronize(); t = time.perf_counter()
            for _ in range(60): step()
            torch.cuda.synchronize()
            r = 60 * bs / (time.perf_counter() - t)
            res["gpu"].append({"arch": arch, "amp": amp, "channels_last": cl, "bs": bs, "img_per_s": r})
            print(res["gpu"][-1], flush=True)

(D.ROOT / "runs").mkdir(exist_ok=True)
(D.ROOT / "runs" / "profile.json").write_text(json.dumps(res, indent=1))
