# Research models (2B)

Six spatio-temporal models trained on the same data, split and metrics, so they can be compared fairly.
Run commands from `Assignment_2B/`. Separate from `core/tune.py` (which tunes the basic LSTM/GRU/GNN).

| Step | Command |
|---|---|
| Build dataset (from `../data`) | `python research/prepare_data.py` → `research/cache/traffic.npz` (not committed, ~20 s to rebuild) |
| Train a model | `python research/train.py --model {gwn,stgcn,agcrn,mtgnn,stid,staeformer} --epochs 30` |

Outputs: `outputs/weights/research/<model>.pt` (best validation epoch), `outputs/logs/research/<model>_metrics.json` and `<model>_train_log.txt`.

- **Task:** last 12 steps (3 h) at all 40 sites → next 4 steps (15/30/45/60 min)
- **Inputs:** normalised flow + time of day
- **Split by time:** train 2014–2023, validate 2024, test 2025-01-01 → 2026-11-16
- **Gaps:** ≤1 h interpolated, longer gaps use the site's weekday/time-of-day average (training years only).
  Loss and metrics ignore filled values; MAPE skips flows under 10 vehicles.
- **Training:** 30 epochs × 60,000 random training windows, Adam lr 1e-3, batch 64,
  early stopping (patience 6), Apple M2 (MPS)

## Training on another PC (Windows / CUDA)

```
cd Assignment_2B
pip install torch numpy pandas holidays          # torch with CUDA, see pytorch.org
python research/prepare_data.py                   # builds research/cache/traffic.npz (~20 s)
python research/train.py --model staeformer --features ext --epochs 30
```
The device is picked automatically (Apple MPS → CUDA → CPU). On a CUDA GPU you can raise
`--batch-size` (e.g. 128) and `--samples-per-epoch` if it trains fast.
The weather data is already downloaded (`research/external/`), so no internet is needed.

## External features (`--features ext`)

`external_features.py` adds 6 channels per 15-min step, the same for all sites and all known in advance:

| Feature | Source |
|---|---|
| Day of week | from the date |
| Victorian public holiday | `holidays` package (incl. Melbourne Cup, AFL Grand Final Friday) |
| School holiday | official Vic term dates 2014–2026 (vic.gov.au/school-term-dates-and-holidays-victoria) |
| COVID-19 lockdown | Melbourne's six lockdowns, 2020–21 (263 days) |
| Rain (mm/h), temperature (°C) | Open-Meteo historical API (ERA5), hourly, centre of Boroondara, cached in `external/` |

Rain is log-transformed and both weather values are standardised on the training years.
STAEformer uses the paper's day-of-week embedding plus a projection of the other 5;
the graph models take them as extra input channels; STID ignores them.
Results are saved with an `_ext` suffix, so the base results above stay as they are.
First 2 epochs on the Mac: val MAE 32.45 vs 32.64 for base STAEformer (run stopped, to be finished on a PC).

## Best STAEformer: feature engineering (23–24 Nov, RTX 5090)

Recipe: `--features ext+lag+wk4+diff+roll+cyc+event`, 80 epochs, cosine LR schedule, validation on every
2024 window, 3 seeds averaged. Each seed trains in about 19 min (14 s per epoch).

```
python research/prepare_data.py
python research/train.py --model staeformer --features ext+lag+wk4+diff+roll+cyc+event --epochs 80 \
    --schedule cosine --val-stride 1 --patience 20 --seed 0 --tag final_s0      # repeat for seeds 1 and 2
python research/ensemble.py staeformer_ext-lag-wk4-diff-roll-cyc-event_final_s0 \
    staeformer_ext-lag-wk4-diff-roll-cyc-event_final_s1 staeformer_ext-lag-wk4-diff-roll-cyc-event_final_s2 \
    --name staeformer_ext-lag-wk4-diff-roll-cyc-event_final_ens3
```

| Test 2025–2026 | MAE | 15 / 30 / 45 / 60 min | RMSE | MAPE |
|---|---|---|---|---|
| STAEformer, original (table below) | 28.30 | | 45.76 | 10.61% |
| + ext, cosine schedule, 3 seeds | 26.41 | | | |
| **+ best features, 3 seeds (weights in `outputs/weights/research/*_final_s{0,1,2}.pt`)** | **23.49** | 22.50 / 23.27 / 23.82 / 24.38 | **38.37** | **8.86%** |
| same model, sensor faults excluded from the test labels | 23.36 | | | |

Single seeds: 23.63, 23.67 and 23.60.

Feature groups (`train.py --features`). Each row is a 40-epoch run, validation MAE on 2024 (seed 1;
seed-to-seed noise about ±0.15):

| Features | Val MAE | Kept |
|---|---|---|
| ext (calendar, lockdown, weather) | 25.64 | yes |
| + lag: same slot last week and yesterday | 24.70 | yes |
| + wk4: same slot, mean of the last 4 weeks | 24.51 | yes |
| + diff, roll, cyc: changes, rolling 24 h / 7 d mean and std, sin/cos of time | 24.19 | yes |
| + event: AFL/cricket at the MCG, Australian Open (`events.py`, `external/mcg_events.csv`) | **23.87** | yes |
| fill flag, holiday-adjacent days, uni semesters, roadworks (each on ext) | 26.0–26.1 | no |
| + det: top-8 detector counts per site | 24.19 | no |
| + nbr: 72 neighbouring intersections as extra nodes (3.4× slower) | 24.17 | no |
| + llm: features proposed by an LLM feedback loop (`fe_loop.py`) | 23.79 / 24.17 (2 seeds) | no |
| + tsf: 8 tsfresh features of the last 24 h (`tsfresh_select.py`, `tsfresh_features.py`) | 24.23 / 23.86 (2 seeds; base 23.87 / 23.78) | no |
| 6 h input window instead of 3 h | worse | no |

- **events:** they barely change the overall MAE, but on 2024 AFL nights the error at the western sites
  2.5–4 h after the start falls from 72 to 50.
- **Where the remaining error comes from:** counting noise accounts for about 80% of the 23.49 (estimated
  floor of about 19 MAE from the data). A perfect area-wide feature could remove at most 3.7%; a perfect
  per-day or per-site-day feature at most 0.4%.
- **Sensor faults** (`clean_faults.py`, `train.py --clean`): 0.56% of values (dead detectors, spikes, stuck
  values, zero runs). Training on cleaned data didn't help (23.38 vs 23.36 on cleaned labels), so it isn't used.
- **tsfresh:** ~780 features on each site's last 24 h, tested against the 5-fold out-of-fold errors of
  2014–2023 (20,000 samples). 181 of 556 pass tsfresh's relevance test (FDR 5%), but the strongest correlation
  with the error is 0.06, and the test looks at one feature at a time. The best feature of each family (20%
  quantile, change quantiles, maximum, mean absolute change, cid_ce, c3, mean crossings, reoccurring values)
  made STAEformer 0.2 worse on average: they mostly repeat `roll`/`diff`, and every extra channel shares the
  same 16-number projection inside STAEformer. Added one at a time (`--tsf-cols`, 40 epochs, seed 1, base
  23.87 val / 23.78 test): 20% quantile 23.94 / 23.87, change quantiles 23.93 / 23.82, c3 23.94 / 23.82,
  reoccurring values 23.93 / 23.82, mean abs change 23.92 / 23.81, maximum 23.87 / 23.79, cid_ce 23.93 / 23.83,
  mean crossings 23.94 / 23.83, so none helps on its own either.
- **nbr, det:** need `research/cache/extra.npz`, which `extract_raw.py` + `prepare_extra.py` build from the raw
  VicRoads archives (about 19 GB, not committed).

### Hyperparameters (40-epoch runs, seed 1, validation MAE on 2024; `train.py --hp`)

| Change | Val MAE |
|---|---|
| defaults (3 layers, ff 256, embeddings 24/24/80, dropout 0.1, lr 1e-3) | **23.87** |
| feed-forward 512 | 24.01 |
| 4 layers | 23.88 (20% slower) |
| wider embeddings 32/32/120 | 23.99 |
| dropout 0 / 0.2 | 24.13 / 23.94 |

Nothing beat the defaults, so model size isn't the limit (most of the remaining error is counting noise).

### 5-fold cross-validation over all the data (24 Nov)

All windows from Jan 2014 to Nov 2026 are split into 5 consecutive blocks of about 2.6 years. Fold k trains on
the other 4 blocks and is scored on block k, which it never saw. Training windows within 4 weeks after a held-out
block are dropped (their lag / 4-week inputs would contain held-out values), scaling statistics come from the
training blocks only, and the last epoch is kept, so the held-out block isn't used to choose anything.

```
for k in 0 1 2 3 4; do
  python research/train.py --model staeformer --features ext+lag+wk4+diff+roll+cyc+event --epochs 80 \
      --schedule cosine --val-stride 4 --seed $k --fold $k/5 --tag cv5_f$k
done
python research/cv_report.py staeformer_ext-lag-wk4-diff-roll-cyc-event_cv5
```

| Fold | Held out | MAE |
|---|---|---|
| 0 | Jan 2014 → Aug 2016 | 25.03 |
| 1 | Aug 2016 → Mar 2019 | 23.88 |
| 2 | Mar 2019 → Oct 2021 | 23.55 |
| 3 | Oct 2021 → Apr 2024 | 24.06 |
| 4 | Apr 2024 → Nov 2026 | 23.43 |
| **all held-out predictions, 2014–2026** | | **23.99** (± 0.63 over folds) |

Pooled: 22.77 / 23.70 / 24.41 / 25.08 at 15 / 30 / 45 / 60 min, RMSE 40.88, MAPE 9.50%. On the 2025–2026 test
windows (held out in fold 4) the MAE is 23.61, RMSE 38.58, MAPE 8.91%, the same as the single models trained on
2014–2023 (23.60–23.67), so the model's accuracy is stable over time. MAE per year is 23.0–24.0 except
2014–2015 (25.2, 26.0) and 2023 (24.8). Weights: `outputs/weights/research/*_cv5_f{0..4}.pt`; full results in
`outputs/logs/research/staeformer_ext-lag-wk4-diff-roll-cyc-event_cv5_metrics.json`.

**Blending the 5 folds.** The final model is the plain average of the 5 fold models. Learned weights per model
change nothing (tested on the 3 seed models: 23.59 vs 23.59 on 2026). A LightGBM correction model on top
(`stack.py`: inputs = prediction, site, time of day, weekday, horizon; fitted on the other blocks' out-of-fold
predictions) makes it slightly worse: 23.99 → 24.08 overall, 23.61 → 23.75 on 2025–2026.

**Recent data matters per site and time of day.** A correction does help a model that hasn't seen the latest
years. The fold-4 model (trained up to Mar 2024) scores 23.71 on 2026, and 23.35 after a correction fitted on its
own Apr 2024 – Dec 2025 predictions (23.85 if fitted on other periods instead). Overall traffic volume doesn't
drift, but each site's time-of-day pattern shifts over the years, so the model should be trained on the latest
data available. Folds 0–3 include data up to Nov 2026; fold 4 doesn't.

## Gradient-boosted trees (`gbdt.py`, 24–25 Nov)

LightGBM, XGBoost and CatBoost on the same task, split and test values as `train.py` (every observed value in
the 2025–2026 test windows). One row per (time, site, horizon), 37 features known at prediction time: the last
12 values, same slot last week / yesterday, 4-week mean of the slot, "seasonal naive" (4-week mean + how far
the site is from it now), area-wide and nearby-site deviation, 24 h mean, calendar, weather, events, site id,
horizon. 6M training rows sampled from 2014–2023, early stopping on 2024, L1 (MAE) loss.

```
python research/gbdt.py --models lightgbm,xgboost,catboost
```

| Model | MAE | 15 / 30 / 45 / 60 min | RMSE | MAPE | Trees | Train time |
|---|---|---|---|---|---|---|
| LightGBM (CPU) | **24.83** | 23.61 / 24.56 / 25.26 / 25.91 | 41.55 | 9.21% | 5,000 (cap) | 11 min |
| XGBoost (GPU) | 24.99 | 23.74 / 24.69 / 25.42 / 26.10 | 42.15 | 9.22% | 8,703 | 2 min |
| CatBoost (CPU) | 25.10 | 23.82 / 24.79 / 25.54 / 26.24 | 42.18 | 9.27% | 15,125 | 145 min |
| Seasonal naive (no model) | 35.80 | | | | | |

All three beat every graph model and the original STAEformer (28.30), but not STAEformer with the best features
(23.6 single model, 23.49 3-seed average): a tree sees one site at a time, STAEformer attends across all 40.
Most important features (by gain) for all three: seasonal naive, 4-week mean of the slot, same slot last week,
yesterday. CatBoost's GPU version doesn't support the MAE loss properly (it barely trains), so it runs on the CPU.
The fitted models (`outputs/weights/research/lightgbm.txt` 118 MB, `xgboost.json` 317 MB, `catboost.cbm` 67 MB)
are too big for GitHub, so they aren't committed; the command above rebuilds them (metrics and logs are committed).

## Road graph (`road_graph.py`, work in progress)

Builds road distances and direct road neighbours between the 40 sites from OpenStreetMap
(for Part A edges and a road-based GNN adjacency). Needs `osmnx scikit-learn`.
On 23 Nov the public Overpass servers were overloaded, so the CSVs aren't built yet; rerun later.

## Files

| File | Contents |
|---|---|
| `prepare_data.py` | Shared dataset + distance-based adjacency (Gaussian kernel on lat/long) + external features |
| `external_features.py` | Calendar, lockdown and weather features |
| `road_graph.py` | OpenStreetMap road distances / neighbours between sites (WIP) |
| `train.py` | Training, evaluation and the model registry |
| `graph_wavenet.py` | Graph WaveNet |
| `gnn_models.py` | STGCN, AGCRN, MTGNN |
| `more_models.py` | STID, STAEformer |
| `more_features.py` | Calendar, semester, roadworks and time-series feature builders (lag/wk4/diff/roll/cyc) |
| `events.py` | MCG / Melbourne Park event history 2014–2026 and the `event` feature |
| `ensemble.py` | Averages the test predictions of several runs (e.g. 3 seeds) and scores them |
| `fe_loop.py`, `fe_loop/` | LLM feedback-loop feature search: sandbox, leakage test, LightGBM proxy, results |
| `extract_raw.py`, `prepare_extra.py` | Neighbour and detector arrays from the raw VicRoads archives (`nbr`, `det`) |
| `clean_faults.py`, `compare_clean.py` | Sensor-fault detection and the raw vs clean training comparison |
| `cv_report.py` | Out-of-fold results of the 5-fold cross-validation (`train.py --fold k/5`) |
| `stack.py` | LightGBM correction on the out-of-fold predictions (tested, not used) |
| `tsfresh_select.py`, `tsfresh_features.py` | tsfresh relevance test and the `tsf` feature group (tested, not used) |
| `gbdt.py` | LightGBM / XGBoost / CatBoost on 37 tabular features, same split and metrics |

## Results (test 2025–2026, averaged over the 4 horizons; full table in `outputs/logs/research/comparison.md`)

| Model | Type | MAE | RMSE | MAPE | Params | Train time |
|---|---|---|---|---|---|---|
| STAEformer | Transformer | **28.30** | **45.76** | **10.61%** | 842k | 138 min |
| Graph WaveNet | GNN | 28.70 | 46.28 | 10.91% | 294k | 56 min |
| AGCRN | GNN | 28.88 | 46.72 | 10.82% | 749k | 46 min |
| STGCN | GNN | 29.79 | 47.93 | 10.94% | 160k | 29 min |
| MTGNN | GNN | 29.86 | 47.87 | 11.24% | 163k | 15 min |
| STID | MLP | 31.05 | 50.93 | 11.62% | 61k | 2 min |
| LightGBM (37 features, see above) | Gradient-boosted trees | 24.83 | 41.55 | 9.21% | 5,000 trees | 11 min |
| XGBoost | Gradient-boosted trees | 24.99 | 42.15 | 9.22% | 8,703 trees | 2 min |
| CatBoost | Gradient-boosted trees | 25.10 | 42.18 | 9.27% | 15,125 trees | 145 min |
| Last value | Baseline | 58.74 | 91.46 | — | — | — |

## References

- Wu et al. (2019) *Graph WaveNet for Deep Spatial-Temporal Graph Modeling*, IJCAI — github.com/nnzhan/Graph-WaveNet
- Yu et al. (2018) *Spatio-Temporal Graph Convolutional Networks* (STGCN), IJCAI
- Bai et al. (2020) *Adaptive Graph Convolutional Recurrent Network* (AGCRN), NeurIPS
- Wu et al. (2020) *Connecting the Dots: Multivariate Time Series Forecasting with GNNs* (MTGNN), KDD
- Shao et al. (2022) *Spatial-Temporal Identity: A Simple yet Effective Baseline* (STID), CIKM
- Liu et al. (2023) *STAEformer: Spatio-Temporal Adaptive Embedding Makes Vanilla Transformer SOTA*, CIKM
