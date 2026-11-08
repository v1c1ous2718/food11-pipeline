# Food-11：從訓練到 INT8 部署的對照實驗

資料：李宏毅 ML2021 HW3 的 Food-11（11 類）。ResNet-18 與 MobileNetV2-0.5 都從零開始訓練（不使用 pretrained weights，與作業規定一致）。
硬體：RTX 3060 12 GB、Intel Core i5-12400、WSL2。軟體：PyTorch 2.7.1+cu128、torchvision 0.22.1，以 uv 管理。

## 0. 實驗規則

| split | 張數 | 來源 | 用途 |
|---|---|---|---|
| train | 2,640（每類 240） | `training/labeled` | 訓練 |
| val | 440（每類 40） | `training/labeled`，固定 seed 0 分層抽出 | **所有選擇**（LR、schedule、augmentation、HPO、teacher） |
| test | 660（每類 60） | 官方 `validation/` | 只回報，從不參與選擇 |

- 除了 LR sweep（單一 seed），每個設定都跑 3 個 seed（0/1/2），報 mean ± std。
- 全部報最後一個 epoch 的結果，不挑最佳 epoch。
- 影像預先 resize 成 144 px；訓練時 random-resized-crop 到 128 px，評估時 center crop 128 px。
- `cudnn.benchmark` 有開，所以同一個 seed 不保證 bit-exact。
- val 一直比 test 高約 2–6 pt。val 和 train 來自同一批資料，test 是另一個資料夾，兩邊的分佈可能有差異。

## 1. Pipeline profiling：時間花在哪裡

![loader](figs/profile_loader.png)
![gpu](figs/profile_gpu.png)

- 每個 batch 都從 JPEG 解碼的話，8 個 workers 最多 **1,550 img/s**，開到 12 個 workers 還是 1,561 img/s，已經飽和。這只有 GPU 處理速度（ResNet-18、AMP + channels_last、bs 64 時 **3,068 img/s**）的一半，所以 data loader 會是瓶頸。
- 改成啟動時一次解碼成 uint8 cache（train + val + test 總共 225 MB）之後，basic augmentation 有 14,879 img/s，開 RandAugment 也有 6,859 img/s（8 workers），都比 GPU 快，loader 不再是瓶頸。normalize 移到 GPU 上做。
- AMP FP16 讓吞吐量變成 FP32 的 **1.56×**，再加 channels_last 變成 **1.89×**（1,627 → 3,068 img/s）。
- 結果：80 epochs 的 ResNet-18 一次約 85 秒，整套實驗（約 100 次訓練）在單張 3060 上幾小時就跑完。

## 2. Learning rate sweep（cosine，basic aug，80 epochs，seed 0）

![lr](figs/lr_sweep.png)

| optimizer | 0.01 | 0.03 | 0.1 | 0.3 |
|---|---|---|---|---|
| SGD + Nesterov（wd 5e-4） | 57.5 | **61.6** | 56.6 | 52.3 |

| optimizer | 3e-4 | 1e-3 | 3e-3 |
|---|---|---|---|
| AdamW（wd 0.05） | 60.0 | **63.4** | 62.0 |

（val accuracy %）下一階段用 SGD 0.03、AdamW 1e-3。這一階段只跑一個 seed，所以只用來找數量級。

## 3. Optimizer × LR schedule（3 seeds）

![sched](figs/sched_dots.png)
![curves](figs/sched_curves.png)

| | val | test |
|---|---|---|
| SGD constant | 56.7 ± 1.2 | 53.6 ± 3.1 |
| SGD step | 60.9 ± 0.9 | 58.5 ± 0.5 |
| SGD cosine | 59.7 ± 2.7 | 59.3 ± 2.4 |
| SGD cosine + 5ep warmup | 61.5 ± 2.3 | 60.2 ± 0.7 |
| SGD one-cycle | 64.2 ± 0.5 | 58.5 ± 0.5 |
| AdamW constant | 59.8 ± 2.3 | 58.7 ± 1.2 |
| AdamW step | 63.0 ± 1.4 | 60.7 ± 1.4 |
| AdamW cosine | 63.0 ± 0.9 | 60.5 ± 1.6 |
| AdamW cosine + 5ep warmup | 66.3 ± 2.0 | **63.3 ± 0.4** |
| **AdamW one-cycle**（val 最高，被選中） | **67.0 ± 2.2** | 62.4 ± 0.5 |

觀察：
- **不衰減 learning rate 一定最差。** 兩種 optimizer 用 constant LR 都排最後。從 curve 看得出來，step schedule 在第 40 epoch 降 LR 時，val accuracy 會往上跳約 8 pt。
- 在這個設定下，五種 schedule 的 AdamW test 都比 SGD 高 1.2–5.1 pt。不過 SGD 的 LR 只掃了 4 個點，這個比較對 SGD 可能不公平。
- 有衰減的幾種 schedule 之間，差距大多在 1–2 個 std 以內。**val 和 test 的排序不一致**：SGD one-cycle 在 val 排 SGD 第一，test 卻和 step 並列最低；AdamW 這邊 val 選了 one-cycle，test 最好的卻是 cosine + warmup。用 440 張 val 選設定，解析度大約只到 ±2 pt。

## 4. Augmentation 逐項加入（AdamW one-cycle，3 seeds）

![aug](figs/aug_dots.png)

| | val | test | train（最後 epoch） |
|---|---|---|---|
| 不做 augmentation | 37.0 ± 0.6 | 37.7 ± 1.9 | 100.0 |
| + random-resized-crop / flip | 67.0 ± 2.2 | 62.4 ± 0.5 | 97.9 |
| + RandAugment (2, 9) | 69.9 ± 1.3 | 67.2 ± 0.9 | 86.8 |
| + MixUp/CutMix (p = 0.5) | 70.8 ± 0.3 | 68.1 ± 1.5 | 62.5* |
| + label smoothing 0.1 | **71.1 ± 0.8** | **69.2 ± 1.6** | 63.0* |
| + flip TTA（只在評估時） | 72.7 ± 0.6 | 68.9 ± 0.9 | — |

\* train accuracy 是在混合過的 batch 上算的，和前幾列不能直接比。

- 只有 2,640 張圖時，不做 augmentation 會完全記住訓練集（train 100%、test 37.7%）。光加 crop/flip 就讓 test 提高 **24.7 pt**；再加 RandAugment 又提高 **4.8 pt**，而且三個 seed 的 std 都小於這個差距。
- MixUp/CutMix（+0.9）和 label smoothing（+1.1）的方向是正的，但都在 1 個 std 左右，**不能說有顯著效果**。
- Flip TTA 讓 val 提高 1.6 pt，test 卻降了 0.3 pt，所以沒有採用。這又是一個 val 和 test 不一致的例子。
- 整體來看，test 從 37.7% 提高到 69.2%。

## 5. Optuna hyperparameter search

![hpo](figs/hpo_scatter.png)
![hpo2](figs/hpo_dots.png)

- 搜尋方式：TPE sampler，30 個 trials，每個 trial 跑 40 epochs，用 MedianPruner 提早停掉差的 trial（前 8 個 trial 不 prune，前 15 epoch 不 prune）。只看 val accuracy，test 完全不計算。
- 搜尋空間：lr、weight decay（log scale）、MixUp/CutMix 機率、label smoothing。
- **30 個 trial 有 17 個（57%）被提前停掉**，節省了大約一半的 GPU 時間。
- 最佳參數：lr 1.0e-3、wd 0.026、mix 0.014（等於沒開）、smoothing 0.12。lr 和手選的一樣。

| 80 epochs × 3 seeds | val | test | train |
|---|---|---|---|
| 手選（第 4 節最後一列） | 71.1 ± 0.8 | **69.2 ± 1.6** | 63.0 |
| Optuna 最佳參數 | **71.2 ± 0.1** | 67.8 ± 0.6 | 88.2 |

- 兩者 val 幾乎相同；但 Optuna 的參數 test 低 1.4 pt，train 高 25 pt，也就是 overfit 得更多。
- 可能的解釋是 **short-horizon bias**：只跑 40 epochs 的代理任務，會偏好正則化較弱的設定（MixUp 幾乎關掉），但正式訓練是 80 epochs，這時更強的正則化才有回報。這只是推測，這次沒有另外驗證。
- 依照「只看 val」的規則，後面的實驗用的是 Optuna 參數（只贏 0.1 pt）。事後看，用手選的可能一樣好，甚至更好。

## 6. Knowledge distillation（200 epochs，3 seeds）

![distill](figs/distill_dots.png)
![distill2](figs/distill_light_dots.png)

| | params | val | test | train |
|---|---|---|---|---|
| Teacher ResNet-18 | 11.2 M | 73.6 ± 1.3 | **72.4 ± 0.8** | 97.9 |
| MobileNetV2-0.5，從零訓練 | 0.7 M | 59.7 ± 1.8 | 57.4 ± 1.7 | 77.4 |
| MobileNetV2-0.5 + KD（T = 4，α = 0.7） | 0.7 M | 59.7 ± 1.8 | 58.8 ± 1.2 | 77.5 |
| *追加實驗：student 改用較弱的 augmentation（只有 crop/flip，不做 mix 與 smoothing）* | | | | |
| MobileNetV2-0.5，從零訓練，弱 aug | 0.7 M | 53.2 ± 1.8 | 50.6 ± 0.8 | 95.1 |
| MobileNetV2-0.5 + KD，弱 aug | 0.7 M | 55.3 ± 1.3 | 52.6 ± 1.2 | 94.7 |

- 同一套參數，teacher 從 80 epochs 拉長到 200 epochs，test 從 67.8 提高到 72.4。
- student 比 teacher 低約 14 pt。原本懷疑是 teacher 的強正則化讓 student 學不起來（train 只有 77%），所以加跑了追加實驗。**結果推翻了這個假設**：改用弱 augmentation 後 student 反而 overfit（train 95%），test 掉了 6.8 pt。0.7 M 參數的模型在 2,640 張圖上，一樣需要強 augmentation。
- KD 在兩種設定下 test 都比較好（+1.4、+2.0 pt），同一個 seed 互相比較時，6 組中有 5 組 KD 勝出。方向一致，但差距只有 1–2 個 std，算是**小但一致的效果**。
- 尚未測試：teacher 訓練時用了 label smoothing，文獻指出這可能讓 teacher 比較不適合拿來蒸餾（Müller et al., 2019）。這次沒有驗證。

## 7. INT8 quantization（x86 / fbgemm，CPU）

![quant](figs/quant_pareto.png)

每個模型取 val 最高的那個 seed。PTQ 用 512 張訓練圖做 calibration。QAT 從 FP32 權重開始微調 10 epochs（lr 1e-3、cosine），在一半 epoch 時凍結 BN 統計量，最後 3 epochs 凍結 observer。

| 模型 | 版本 | 大小 | val | test | CPU 1 thread | CPU 4 threads |
|---|---|---|---|---|---|---|
| ResNet-18 | FP32 | 44.8 MB | 74.8 | 73.2 | 16.99 ms | 8.42 ms |
| ResNet-18 | INT8 PTQ | 11.3 MB | 74.8 | 73.3 | **7.60 ms** | 3.96 ms |
| ResNet-18 | INT8 QAT | 11.3 MB | 75.0 | 73.3 | 7.55 ms | 3.65 ms |
| MobileNetV2-0.5 + KD | FP32 | 3.0 MB | 61.8 | 60.2 | 4.04 ms | 4.55 ms |
| MobileNetV2-0.5 + KD | INT8 PTQ | 0.98 MB | 62.5 | 60.2 | **2.55 ms** | 2.21 ms |
| MobileNetV2-0.5 + KD | INT8 QAT | 0.98 MB | 60.5 | 58.5 | 2.55 ms | 2.45 ms |

（batch 1、128 px，取 300 次中位數，測試 CPU 為 i5-12400）

- **ResNet-18 做 PTQ 後，模型小 3.96 倍，單執行緒 latency 快 2.24 倍，accuracy 沒有掉**（test 73.2 → 73.3）。PTQ 已經沒有損失，QAT 就沒有可以補回的空間。
- MobileNetV2-0.5 做 PTQ 後小 3.05 倍、快 1.58 倍，accuracy 一樣沒掉。QAT 反而讓 test 低了 1.7 pt。我的 QAT 微調只用 plain cross-entropy，沒有加 KD 和 label smoothing，等於換掉了原本的訓練目標，這是最可能的原因（未驗證）。
- FP32 的 MobileNetV2 用 4 threads 比 1 thread 慢（4.55 vs 4.04 ms）：batch 1 的小模型，平行化的開銷比它省下的計算時間還多。
- 用 GPU 跑 FP16、batch 1 時，ResNet-18 是 1.70 ms，MobileNetV2-0.5 卻是 3.87 ms，反而比較慢。小 batch 時 GPU 的瓶頸是 kernel launch 次數；MobileNetV2 層數多、depthwise conv 運算量小，所以參數少也沒有比較快。

## 8. 結論

1. **最有效的是 augmentation**：test 從 37.7% 提高到 69.2%。其中 crop/flip 加 RandAugment 貢獻 29.5 pt；MixUp/CutMix 和 label smoothing 方向正確，但在雜訊範圍內。
2. **LR 一定要衰減**；在 decayed schedule 之間，差距小於 440 張 val 能分辨的程度。
3. **自動化 HPO 不一定贏過人工設定**：40-epoch 代理任務選出的設定，在 80 epochs 時 test 低了 1.4 pt。
4. **ResNet-18 INT8 PTQ 沒有掉 accuracy**：模型小 4 倍、CPU 快 2.2 倍。QAT 只有在 PTQ 有損失時才值得做。
5. **工程面**：先解碼成 uint8 cache，加上 AMP 和 channels_last，讓 GPU 不再等資料，訓練吞吐量是 FP32 的 1.89 倍。

## 重現

```bash
cd ~/food11-pipeline
uv sync
uv run python -m f11.data                        # 解碼並建立 uint8 cache
uv run python scripts/profile_pipeline.py
uv run python scripts/experiments.py             # lr sched aug hpo distill quant
uv run python scripts/experiments.py distill_light
uv run python scripts/make_figs.py
```

所有數字都來自 `runs/summary_*.json` 和 `runs/profile.json`；每次訓練的逐 epoch 紀錄存在 `runs/<name>/result.json`。
