# Experiment log: traffic flow prediction with STAEformer (22–25 Nov 2026)

Everything we tried to get the best flow predictor for the 40 Boroondara SCATS sites, in the order we did it:
what we tried, the result, and whether we kept it. How to run things is in `README.md`; this file is the story.

## Final results

| Model | Test MAE 2025–26 | 15 / 30 / 45 / 60 min | RMSE | MAPE |
|---|---|---|---|---|
| **STAEformer + best features, 3 seeds averaged** | **23.49** | 22.50 / 23.27 / 23.82 / 24.38 | 38.37 | 8.86% |
| STAEformer + best features, 5-fold CV, every held-out prediction 2014–2026 | 23.99 (23.61 on 2025–26) | 22.77 / 23.70 / 24.41 / 25.08 | 40.88 | 9.50% |
| LightGBM, 37 tabular features | 24.83 | 23.61 / 24.56 / 25.26 / 25.91 | 41.55 | 9.21% |
| STAEformer, original (flow + time of day) | 28.30 | 25.33 / 27.36 / 29.13 / 31.40 | 45.76 | 10.61% |
| Last value (no model) | 58.74 | | | |

Best recipe: STAEformer with `--features ext+lag+wk4+diff+roll+cyc+event`, 80 epochs, cosine learning rate.
Compared with the original STAEformer that is 17% lower error, and the gain is the features, not the model.

## Setup (same for every experiment)

- **Task:** the last 12 steps (3 h) of flow at all 40 sites → the next 4 steps (15, 30, 45, 60 min).
- **Data:** `data/site_totals_15min_2014_2026.csv.gz`, 15-min totals per site, Jan 2014 – Nov 2026.
  Gaps up to 1 h are interpolated, longer ones use the site's weekday / time-of-day average; loss and metrics
  only use real (observed) values. MAPE skips flows under 10 vehicles.
- **Split by time:** train 2014–2023, validate 2024, test 2025-01-01 → 2026-11-16.
  Decisions were made on the 2024 validation MAE; the test MAE is reported but never used to choose anything.
- **Metrics:** MAE, RMSE, MAPE in vehicles per 15 min per site, averaged over the 4 horizons.
- **Quick comparisons** ("40-epoch runs"): 40 epochs × 60,000 windows, cosine schedule, seed 1.
  Two seeds of the same run differ by about ±0.15 MAE, so smaller differences are noise.
- **Hardware:** the first model comparison ran on an Apple M2; everything after on an RTX 5090
  (best STAEformer: 14 s per epoch, 19 min per 80-epoch run).

## 1. Which model? (6 models, same data and split)

Graph WaveNet, STGCN, AGCRN, MTGNN, STID and STAEformer, 30 epochs, input = flow + time of day.

| Model | Type | Test MAE | Params | Train time (M2) |
|---|---|---|---|---|
| STAEformer | Transformer | **28.30** | 842k | 138 min |
| Graph WaveNet | GNN | 28.70 | 294k | 56 min |
| AGCRN | GNN | 28.88 | 749k | 46 min |
| STGCN | GNN | 29.79 | 160k | 29 min |
| MTGNN | GNN | 29.86 | 163k | 15 min |
| STID | MLP | 31.05 | 61k | 2 min |
| LSTM / GRU (teammate's baselines, original pipeline, on `main`) | RNN | 34.42 / 34.47 | | |

**Kept:** STAEformer, which was best at every horizon. All later work is on STAEformer only.

## 2. External information (`ext`)

Six channels known in advance: day of week, public holiday, school holiday, COVID lockdown, rain, temperature
(Open-Meteo ERA5, hourly).

| Run | Val 2024 | Test |
|---|---|---|
| original STAEformer, 30 epochs | 27.66 | 28.30 |
| + ext, 30 epochs | 26.78 | 27.60 |
| + ext, 80 epochs, cosine schedule, 3 seeds averaged | 25.41–25.55 per seed | **26.41** |

**Kept.** The cosine schedule and longer training also helped.

## 3. Feature engineering (the big gain)

Each row is a 40-epoch run (seed 1), added on top of the row above unless it says otherwise.

| Features | Val 2024 | Test | Kept |
|---|---|---|---|
| ext | 25.64 | 26.82 | yes |
| + **lag**: same 15-min slot last week and yesterday | 24.70 | 24.70 | yes, the biggest single gain |
| + **wk4**: mean of the same slot over the last 4 weeks | 24.51 | 24.36 | yes |
| + **diff, roll, cyc**: recent changes, rolling 24 h / 7 day mean and sd, sin/cos of time | 24.19 | 24.03 | yes |
| + **event**: AFL / cricket at the MCG, Australian Open at Melbourne Park | **23.87** | **23.78** | yes |
| same, seed 2 | 23.78 | 23.72 | |

Tried and dropped:

| Features | Val 2024 | Test | Why dropped |
|---|---|---|---|
| fill flag (value was interpolated) on ext | 26.07 | 26.94 | worse than ext alone |
| days next to public holidays on ext | 26.08 | 27.01 | worse |
| university semesters on ext | 26.08 | 26.97 | worse |
| roadworks on ext | 26.02 | 26.99 | worse |
| lag + all four above | 24.88 | 24.79 | worse than lag alone |
| 6 h input window instead of 3 h (ext) | 27.34 at epoch 23 (3 h: 27.32) | | no better, 2× slower, stopped |

Single additions on top of lag (to see which part of the combination helps): wk4 24.51, roll 24.49,
event 24.60, cyc 24.68, diff 24.71. Each one alone is small; together they give 24.19, and events bring 23.87.
Events barely move the overall MAE, but on 2024 AFL nights the error at the western sites 2.5–4 h after the
start falls from 72 to 50.

## 4. Final training of the best recipe

80 epochs, cosine schedule, validation on every 2024 window, seeds 0–2, predictions averaged (`ensemble.py`).

| Run | Val 2024 | Test |
|---|---|---|
| seed 0 / 1 / 2 | 23.68 / 23.71 / 23.60 | 23.63 / 23.67 / 23.60 |
| **3 seeds averaged** | | **23.49** (RMSE 38.37, MAPE 8.86%) |

Weights: `outputs/weights/research/staeformer_ext-lag-wk4-diff-roll-cyc-event_final_s{0,1,2}.pt`.

## 5. More data: raw VicRoads detectors and neighbouring intersections

We downloaded the raw VicRoads SCATS archives (35 zips, 19 GB) to get data the site totals don't have.

| Added to the best features (40 epochs) | Val 2024 | Test | Kept |
|---|---|---|---|
| none (best features) | 23.87 | 23.78 | |
| **det**: the 8 busiest detectors of each site as separate channels | 24.19 | 24.12 | no |
| **nbr**: 72 neighbouring intersections as extra graph nodes | 24.17 | 24.05 | no (also 3.4× slower) |

More raw input made it worse: the site totals already carry what the model can use.

## 6. LLM feedback loop for new features (`fe_loop.py`)

An LLM (Claude Code) wrote candidate features as code over 2 rounds (13 features), each checked for leakage and
scored with a fast LightGBM proxy on the best model's 2024 errors. The two features that scored above noise
(fraction of observed inputs, steps since last observation) were then tested in STAEformer: 23.79 / 24.17 over
2 seeds (mean 23.98) vs 23.87 / 23.78 (mean 23.83) without. **Not kept.** Details in `fe_loop/README.md`.

## 7. How much error is left to remove?

An analysis of the final model's test errors:

- The remaining errors are close to random: correlation of an error with the one 15 min later is 0.03, and
  about 0 a day or a week later.
- Pure counting (Poisson) noise alone would give an MAE of about 14.8 (63% of 23.49); the noise floor estimated
  from the data is about 19 MAE, about 80% of the remaining error.
- A perfect area-wide feature (something affecting all sites at once) could remove at most 3.7%; a perfect
  per-day or per-site-day feature at most 0.4%. There's no slow drift in the overall level.
- One site (4035) causes 9% of all error.

Conclusion: new features have little room left, which matches everything we tried after step 4.

## 8. Sensor faults (`clean_faults.py`)

Found 0.56% of values that look like faults (dead detectors, spikes, stuck values, runs of zeros).

| Trained on | Scored on original labels | Scored on cleaned labels |
|---|---|---|
| raw data (final model) | **23.49** | **23.36** |
| cleaned data | 24.21 | 23.38 |

**Not kept:** training on cleaned data doesn't help. For the report, the final model is 23.36 if faulty
values are left out of the scoring.

## 9. Hyperparameter tuning (`train.py --hp`)

| Change (40 epochs, seed 1) | Val 2024 | Test |
|---|---|---|
| defaults: 3 layers, feed-forward 256, embeddings 24/24/80, dropout 0.1, lr 1e-3 | **23.87** | **23.78** |
| feed-forward 512 | 24.01 | 23.89 |
| 4 layers | 23.88 | 23.81 |
| wider embeddings 32/32/120 | 23.99 | 23.89 |
| dropout 0 | 24.13 | 24.01 |
| dropout 0.2 | 23.94 | 23.85 |

**Defaults kept:** model size and regularisation aren't the limit.

## 10. 5-fold cross-validation over all the data, 2014–2026

The single split never tests on old years, and never trains on 2025–26. To use and test all of the data with
one model type, we split every window from Jan 2014 to Nov 2026 into 5 consecutive blocks (about 2.6 years
each). Fold k trains on the other 4 blocks and is scored on block k, which it never saw:

- training windows whose inputs reach into the held-out block (up to 4 weeks after it) are dropped;
- the scaling statistics come from the training blocks only;
- the last epoch is kept, so the held-out block isn't used to choose anything.

| Fold | Held out | MAE |
|---|---|---|
| 0 | Jan 2014 → Aug 2016 | 25.03 |
| 1 | Aug 2016 → Mar 2019 | 23.88 |
| 2 | Mar 2019 → Oct 2021 | 23.55 |
| 3 | Oct 2021 → Apr 2024 | 24.06 |
| 4 | Apr 2024 → Nov 2026 | 23.43 |
| **all held-out predictions** | **2014–2026** | **23.99** (± 0.63 over folds) |

MAE per year:

| 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 25.24 | 25.96 | 23.34 | 23.87 | 23.90 | 23.49 | 23.65 | 23.77 | 23.02 | 24.81 | 23.67 | 23.52 | 23.71 |

On the 2025–26 test windows (held out by fold 4) the MAE is 23.61, the same as the single models trained on
2014–2023 (23.60–23.67): the accuracy is stable over time. 2014–15 and 2023 are the hardest years.
Re-running `cv_report.py` from the committed code reproduces these numbers exactly.
Weights: `outputs/weights/research/staeformer_ext-lag-wk4-diff-roll-cyc-event_cv5_f{0..4}.pt`.

## 11. Blending and stacking the models

- **Learned weights** for each model instead of a plain average: no change (23.59 vs 23.59 on 2026).
- **A LightGBM correction model on top** (`stack.py`; inputs: prediction, site, time of day, weekday, horizon),
  fitted on the other blocks' held-out predictions: slightly worse, 23.99 → 24.08 overall and
  23.61 → 23.75 on 2025–26. **Not kept;** the final model is the plain average.
- **Why a first test looked good:** a correction fitted on 2025 and scored on 2026 took the 3-seed models from
  23.59 to 23.20. That gain was recency, not stacking. The fold-4 model (trained up to Mar 2024) goes from
  23.71 to 23.35 on 2026 when corrected with its own Apr 2024 – Dec 2025 predictions, but only to 23.85 with
  older periods. Overall traffic volume doesn't drift, but each site's time-of-day pattern shifts slowly over
  the years, so **the model should be retrained on the latest data available.**

## 12. Gradient-boosted trees (`gbdt.py`)

Same task, split and test values, as a table: one row per (time, site, horizon) with 37 features known at
prediction time (last 12 values, same slot last week / yesterday, 4-week mean, "seasonal naive", area-wide and
nearby-site deviation, 24 h mean, calendar, weather, events, site, horizon). 6M training rows from 2014–2023,
early stopping on 2024, MAE loss.

| Model | Test MAE | RMSE | MAPE | Trees | Train time |
|---|---|---|---|---|---|
| LightGBM (CPU) | **24.83** | 41.55 | 9.21% | 5,000 (cap) | 11 min |
| XGBoost (GPU) | 24.99 | 42.15 | 9.22% | 8,703 | 2 min |
| CatBoost (CPU) | 25.10 | 42.18 | 9.27% | 15,125 | 145 min |
| Seasonal naive (4-week mean + current deviation) | 35.80 | | | | |

All three beat the original STAEformer (28.30) and every graph model, but not STAEformer with the best features
(23.49): a tree predicts one site at a time, while STAEformer looks at all 40 sites together. The most useful
features for all three were seasonal naive, the 4-week mean, same slot last week and yesterday, the same
information that gave STAEformer its big gain. CatBoost's GPU version doesn't support the MAE loss, so it ran
on the CPU.

## 13. tsfresh features (`tsfresh_select.py`, `tsfresh_features.py`)

tsfresh computed about 780 features on each site's last 24 h for 20,000 samples, compared with the 5-fold
held-out errors of 2014–2023. 181 of the 556 tested features pass tsfresh's relevance test (FDR 5%), but the
strongest correlation with the error is only 0.06. We took the best feature of each family (8 features; our
GPU version matches tsfresh's values exactly).

| Added to the best features (40 epochs) | Val 2024 | Test |
|---|---|---|
| none | 23.87 / 23.78 (seeds 1 / 2) | 23.78 / 23.72 |
| all 8 | 24.23 / 23.86 (seeds 1 / 2) | 24.09 / 23.79 |
| 20% quantile | 23.94 | 23.87 |
| change quantiles | 23.93 | 23.82 |
| maximum | 23.87 | 23.79 |
| mean absolute change | 23.92 | 23.81 |
| cid_ce (complexity) | 23.93 | 23.83 |
| c3 (non-linearity) | 23.94 | 23.82 |
| mean crossings | 23.94 | 23.83 |
| sum of reoccurring values | 23.93 | 23.82 |

**Not kept:** none helps, alone or together. They mostly repeat `roll` / `diff`, and all extra channels share
one small 16-number input layer inside STAEformer, so more channels crowd each other.

## 14. Speed limit (discussed, not tested)

A speed limit is fixed for each site, and STAEformer already learns a separate embedding for every site, so as
a flow feature it adds nothing. Where it matters is the flow → travel time conversion in `core/travel_time.py`,
which uses a flat 60 km/h as the assignment spec says; real limits per road would be an optional extension.

## What we learned

1. **Features beat models.** Swapping models changed the MAE by about 3; good features took STAEformer from
   28.30 to 23.49. Most of the gain came from "what happened at this time last week / yesterday / last 4 weeks".
2. **After that, the remaining error is mostly random.** Counting noise is about 80% of what's left, which is
   why more data (detectors, neighbours), LLM-proposed features, tsfresh, cleaning, tuning and stacking all failed
   to help.
3. **Tree models are a strong, fast baseline** (XGBoost: 24.99 in 2 minutes) but can't use the other 40 sites
   the way an attention model can.
4. **The model is stable over 13 years** (23.0–24.0 MAE most years), but each site's daily pattern shifts slowly,
   so retraining on recent data matters.
5. **Test discipline:** every decision was made on the 2024 validation set or held-out folds; the 2025–26 test
   set only reports results.
