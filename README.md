# Food-11 CNN training pipeline

An ablation study of the parts of a CNN training pipeline, run on the Food-11 dataset from Hung-yi Lee's ML 2021 HW3. Every model is trained **from scratch**, as the homework requires, on a single RTX 3060.

Full report (in Chinese): [`report/report.md`](report/report.md). Raw results are in `runs/`.

| Topic | Result (test set) |
|---|---|
| Data loader | Decoding JPEGs every batch saturates at 1,550 img/s, while the GPU consumes 3,068 img/s. A uint8 decode cache removes the bottleneck. |
| AMP + channels-last | 1.89× training throughput over FP32. |
| Augmentation ladder (3 seeds) | 37.7% → 69.2% (crop/flip +24.7, RandAugment +4.8; MixUp/CutMix and label smoothing within noise). |
| INT8 PTQ, ResNet-18 | 44.8 → 11.3 MB; 1-thread CPU latency 17.0 → 7.6 ms; accuracy unchanged (73.2 → 73.3). |
| Optuna (TPE + median pruning) | Did **not** beat the hand-picked setting (67.8 vs 69.2). The 440-image validation set is too small to rank close configurations. |
| Knowledge distillation | Small, consistent gain (+1.4 / +2.0 pt). |

Also covered: LR sweep, schedulers (constant, step, cosine, warmup, one-cycle), QAT and TTA.

```bash
uv sync
uv run python scripts/profile_pipeline.py
uv run python scripts/experiments.py            # all stages; or name stages, e.g. aug quant
uv run python scripts/make_figs.py
```
