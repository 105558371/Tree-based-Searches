# Experiment log: traffic flow prediction with STAEformer (22–25 Nov 2026)

Everything we tried to get the best flow predictor for the 40 Boroondara SCATS sites, in the order we did it:
what we tried, what we found, and whether we kept it. How to run things is in `README.md`; this file is the story.

## Final results

| Model | Test MAE 2025–26 | 15 / 30 / 45 / 60 min | RMSE | MAPE |
|---|---|---|---|---|
| **STAEformer + best features, 3 seeds averaged** | **23.49** | 22.50 / 23.27 / 23.82 / 24.38 | 38.37 | 8.86% |
| same, sensor faults left out of the scoring | 23.36 | | | |
| STAEformer + best features, 5-fold CV, every held-out prediction 2014–2026 | 23.99 (23.61 on 2025–26) | 22.77 / 23.70 / 24.41 / 25.08 | 40.88 | 9.50% |
| LightGBM, 37 tabular features | 24.83 | 23.61 / 24.56 / 25.26 / 25.91 | 41.55 | 9.21% |
| STAEformer, original (flow + time of day) | 28.30 | 25.33 / 27.36 / 29.13 / 31.40 | 45.76 | 10.61% |
| Last value (no model) | 58.74 | | | |

Best recipe: STAEformer with `--features ext+lag+wk4+diff+roll+cyc+event`, 80 epochs, cosine learning rate.
That is 17% lower error than the original STAEformer (22% at 60 min), and the gain is the features, not the model.

## Setup (same for every experiment)

- **Task:** the last 12 steps (3 h) of flow at all 40 sites → the next 4 steps (15, 30, 45, 60 min).
- **Data:** `data/site_totals_15min_2014_2026.csv.gz`, 15-min totals per site, Jan 2014 – Nov 2026.
  An average site carries about 425 vehicles per 15 min. Gaps are filled for the inputs (see the problems
  section), but loss and metrics only use real (observed) values. MAPE skips flows under 10 vehicles.
- **Split by time:** train 2014–2023, validate 2024, test 2025-01-01 → 2026-11-16.
  Decisions were made on the 2024 validation MAE; the test MAE is reported but never used to choose anything.
- **Metrics:** MAE, RMSE, MAPE in vehicles per 15 min per site, averaged over the 4 horizons.
- **Quick comparisons** ("40-epoch runs"): 40 epochs × 60,000 windows, cosine schedule, seed 1.
  Two seeds of the same run differ by about ±0.15 MAE, so smaller differences are noise.
- **Hardware:** the first model comparison ran on an Apple M2; everything after on an RTX 5090
  (best STAEformer: 14 s per epoch, 19 min per 80-epoch run).

## 0. Problems found in the data and code (review on 23 Nov)

| Problem | Effect | Status |
|---|---|---|
| **Site 4266 is placed in the Indian Ocean** (-28.37, 108.78) in `core/sites_aligned.json` and in the dataset. One of its rows in the 2006 file looks like (0, 0), which drags the average position. The official position is (-37.8237, 145.0448), Burwood/Auburn (`research/external/site_locations.csv`). | The distance graph is broken. The ~1,000 km outlier stretches the Gaussian kernel (width 765 km instead of 1.9 km), so the other 39 sites are all linked to each other with weight exactly 1.000 and 4266 is linked to nothing. With the right position only 33% of site pairs are linked, as intended. | **Open.** `prepare_data.py` still reads the wrong position. |
| ↳ who it affects | STGCN depends on the graph entirely, so its graph convolution just averaged all sites; Graph WaveNet partly recovers through its learned graph; the GCN in `core/tune.py` has the same problem; the tree models' "nearby-site deviation" feature is really an area-wide average. AGCRN, MTGNN, STID and STAEformer don't use the graph, so their results stand. | STGCN and Graph WaveNet should be retrained after the fix. |
| ↳ `road_graph.py` failing | Its OpenStreetMap download box ran from Western Australia to Melbourne because of 4266, which is the likely reason it failed (not "overloaded servers"). | Open, same fix. |
| Gap filling: `interpolate(limit=4)` also fills the first 4 steps of longer gaps (a 10-step gap gets 4 interpolated values), unlike the docstring and `docs/insights.md`. | Small: the loss and metrics ignore filled values. | Open. |
| `requirements.txt` is UTF-16 (from PowerShell `pip freeze >`) and misses `holidays`, so `prepare_data.py` fails on a fresh install. | Setup only. | Open. |
| `outputs/logs/best_params.json` (lr 0.01 for LSTM/GRU/GNN) comes from `core/tune.py`'s smoke test (about 1 s per run), and `tune.py` uses a different split, scaling, loss and horizon from `research/train.py`. | Those settings and LSTM/GRU numbers aren't comparable with the research table. | Open. |
| Travel time: `core/travel_time.py` uses the flow at the **end** site B, while `README.md` and `docs/insights.md` say the **start** site. | One of them is wrong; check against the spec. | Open. |

## 1. Which model? (6 models, same data and split)

Graph WaveNet, STGCN, AGCRN, MTGNN, STID and STAEformer, 30 epochs, input = flow + time of day.

| Model | Type | Test MAE | Params | Train time (M2) |
|---|---|---|---|---|
| STAEformer | Transformer | **28.30** | 842k | 138 min |
| Graph WaveNet | GNN | 28.70* | 294k | 56 min |
| AGCRN | GNN | 28.88 | 749k | 46 min |
| STGCN | GNN | 29.79* | 160k | 29 min |
| MTGNN | GNN | 29.86 | 163k | 15 min |
| STID | MLP | 31.05 | 61k | 2 min |
| LSTM / GRU (teammate's baselines, original pipeline, on `main`) | RNN | 34.42 / 34.47 | | |

\* trained on the broken distance graph (section 0), so these two may be underrated.

**Kept:** STAEformer, which was best at every horizon. All later work is on STAEformer only, so the other models
never got the better features or training recipe; part of STAEformer's final lead over them comes from that.

## 2. External information (`ext`) and the training recipe

Six channels known in advance: day of week, public holiday, school holiday, COVID lockdown, rain, temperature
(Open-Meteo ERA5, hourly).

| Run | Val 2024 | Test | Change |
|---|---|---|---|
| original STAEformer, 30 epochs, fixed lr | 27.66 | 28.30 | |
| + ext, same recipe | 26.78 | 27.60 | −0.70 |
| + cosine lr schedule, 80 epochs (seeds 0 / 1 / 2) | 25.55 / 25.41 / 25.55 | 26.68 / 26.66 / 26.75 | −0.93 |
| 3 seeds averaged | | **26.41** (60 min: 28.43, was 31.40) | −0.26 |

- **Why the recipe mattered:** with a fixed learning rate the validation MAE jumped between 27.7 and 29.2 from
  epoch to epoch and was still improving at epoch 29 when the 30-epoch run ended.
- **Converged at 80 epochs:** the last 20 epochs improved validation by only 0.08–0.12, the learning rate reaches
  0, and validation never went back up (train–validation gap 1.2–1.4, no overfitting). 120–150 epochs might
  add about 0.1.
- **Day of week is probably the main ext gain:** the original model only knew the time of day.
- **Caveat:** the ext model has 1.25M parameters against 842k (the extra embeddings widen every layer), so part
  of the gain may come from size.
- **6 h input window instead of 3 h (`--in-steps 24`):** best validation by epoch 23 was 27.34 against 26.69 for
  3 h (mean of epochs 19–23: 27.53 vs 26.92), and each epoch took twice as long. Briefly ahead around epoch 9,
  then 0.6–0.9 behind. Stopped at epoch 23 of 80. **Not kept.**

## 3. Where was the error? (analysis of the ext model, 2024 validation)

Done on the best ext model (seed 1, val 25.41) to decide which features to build.

| Possible cause | Finding | Worth a feature? |
|---|---|---|
| **Same time last week** | Mixing just 5% of last week's value into the predictions cut Jul–Dec MAE from 24.78 to 24.60 (mix tuned on Jan–Jun). | **Yes, best lead** |
| Gap-filled inputs | MAE 36.1 when any of the last 3 h was filled, 25.3 otherwise, but only 1.8% of all error. | small |
| Public holidays, Christmas period | Error relative to flow 7.8% and 7.3%, against 5.8% on normal weekdays; about 9% of all error. | small |
| Rain | Dry 6.0%, light rain 5.8%, heavy rain 6.1% of flow. | no |
| Days with a dead detector | Same relative error as other days. | no |
| Average weekly profile | Mixing it in gains nothing; the model already knows it. | no |
| Drift at particular sites | No site consistently off by more than 3%. | no |

Error scales with traffic volume: midday (10:00–15:59) carries 29% of all error and each peak period about 21%.

## 4. Feature engineering (the big gain)

Each row is a 40-epoch run (seed 1), added on top of the row above unless it says otherwise.

| Features | Val 2024 | Test | Kept |
|---|---|---|---|
| ext | 25.64 (seed 2: 25.91) | 26.82 | yes |
| + **lag**: same 15-min slot last week and yesterday | 24.70 (seed 2: 24.56) | 24.70 | yes, −1.14 (4.4%) over 2 seeds |
| + **wk4**: mean of the same slot over the last 4 weeks | 24.51 | 24.36 | yes |
| + **diff, roll, cyc**: recent changes, rolling 24 h / 7 day mean and sd, sin/cos of time and day of year | 24.19 | 24.03 | yes |
| + **event**: AFL / cricket at the MCG, Australian Open at Melbourne Park | **23.87** (seed 2: 23.78) | **23.78** | yes |

Single additions on top of lag: wk4 24.51, roll 24.49, event 24.60, cyc 24.68, diff 24.71. wk4 and roll were
borderline alone but clearly better together: the 4-week mean gives the usual level of each slot and the rolling
stats show how today differs from it. Day of year (in cyc) adds nothing on its own because the 4-week and 7-day
means already track the seasons.

Tried and dropped:

| Features (on ext) | Val 2024 | Test |
|---|---|---|
| fill flag (input value was interpolated) | 26.07 | 26.94 |
| days next to public holidays, bridge days, Christmas week, daylight saving | 26.08 | 27.01 |
| university semesters (Swinburne Hawthorn, Deakin Burwood) | 26.08 | 26.97 |
| roadworks: level-crossing removals, North East Link | 26.02 | 26.99 |
| lag + all four above | 24.88 | 24.79 (lag alone: 24.70) |

These four score about 0.3 worse than the 2-seed ext average (25.77). Three of them land within 0.01 of each
other because a one-channel change keeps the same starting weights and data order; ext's own 25.64 was a lucky
seed. Fill flags were probably too rare (2% of inputs) to be useful.

**Events data** (`events.py`, `external/mcg_events.csv`): 587 AFL games at the MCG (start time and crowd, AFL
Tables), 68 Big Bash nights and 50 Boxing Day Test days (Cricsheet), 185 Australian Open days (Wikipedia). After
big AFL games (crowd ≥ 50k), flow 2.5–4 h after the start is 12–14% above the same time 1–2 weeks earlier at the
western sites (3001 Barkers/Church/High +13.8%, 4812 Swan/Madden +12.9%, 4262 Burwood/Church +12.7%,
4821 Victoria/Burnley +12.3%), only +1–3% on Burke and Warrigal Rd, and +1–5% before the start. Events cover
1–2% of all 15-min slots, so the overall MAE barely moves, but on 2024 AFL nights the error at the western sites
in that window falls from 72 to 50. Four channels: crowd arriving, crowd leaving, cricket, tennis.
Caveats: the AFL crowd is the actual attendance (only the expected crowd is known in advance, so slightly
optimistic), and cricket start times are assumed (7:15 pm Big Bash, 10:30 am Test). Concerts and local festivals
had no dated records back to 2014.

## 5. Final training of the best recipe

80 epochs, cosine schedule, validation on every 2024 window, seeds 0–2, predictions averaged (`ensemble.py`).

| Run | Val 2024 | Test |
|---|---|---|
| seed 0 / 1 / 2 | 23.68 / 23.71 / 23.60 | 23.63 / 23.67 / 23.60 |
| **3 seeds averaged** | | **23.49** (RMSE 38.37, MAPE 8.86%) |

- **Biggest gain at 60 min:** 31.40 → 24.38 (−22%). Error now grows only 1.9 from 15 to 60 min, against 6.1
  before: knowing last week / yesterday matters most further ahead.
- **RMSE −16%** (45.76 → 38.37): fewer large misses, mostly around MCG games.
- **Averaging 3 seeds gains 0.14** over one seed; going from 40 to 80 epochs gained about 0.17 on validation.
  A single model (23.60–23.67) is nearly as good if simplicity matters.

Weights: `outputs/weights/research/staeformer_ext-lag-wk4-diff-roll-cyc-event_final_s{0,1,2}.pt`.

## 6. More data: raw VicRoads detectors and neighbouring intersections

We downloaded the raw VicRoads SCATS archives (35 zips, 19 GB) and extracted our 40 sites plus 191 signal sites
within 1 km (`extract_raw.py`, about 12.5M detector-day rows). `prepare_extra.py` kept 72 neighbouring
intersections (95–99% coverage every year) and the 8 busiest detectors at each site. 7 faulty detectors had
spikes up to 590× their normal spread, so values were capped at ±10 sd.

| Added to the best features (40 epochs) | Val 2024 | Test | Kept |
|---|---|---|---|
| none (best features) | 23.87 | 23.78 | |
| **det**: the 8 busiest detectors of each site as separate channels | 24.19 | 24.12 | no |
| **nbr**: 72 neighbouring intersections as extra graph nodes | 24.17 | 24.05 | no (also 3.4× slower) |

- **Detectors** add up to the site total the model already sees; split into lanes they add mostly noise (faulty
  loops, lane closures, renumbering). Even the training error got worse (23.95 vs 23.51).
- **Neighbours** carry little the model doesn't have. Cars take 1–3 minutes between intersections, so a
  neighbour's 15-min count arrives at the same time as the site's own, with no head start, and the remaining
  errors barely move together across sites (correlation 0.05).
- Only the heaviest way of adding neighbours was tested. A lighter version (1–2 summary inputs per site, such as
  the mean recent flow within 1 km) is untested.

## 7. LLM feedback loop for new features (`fe_loop.py`)

An LLM (Claude Code) wrote candidate features as code over 2 rounds (13 features, all passing a code whitelist,
process isolation and a leakage test that scrambles the future). Each was scored by how much it helped a LightGBM
model correct the best model's 2024 errors (fit Jan–Jun, scored Jul–Dec); a random feature scores −0.025, which
sets the noise floor. Only two scored above it (fraction of observed inputs +0.015, steps since last observation
+0.006). In STAEformer they gave 23.79 / 24.17 over 2 seeds (mean 23.98) against 23.83 without.
**Not kept.** The loop stopped because the remaining errors look like noise: correlation 0.017 one hour later and
about 0 a day or week later, 0.048 across sites, and the area-wide part explains 6% of their variance. The
scorer only sees 2024, so it can't credit features that help the model learn from older years (such as a
pre/post-COVID flag). Details in `fe_loop/README.md`.

## 8. How much error is left to remove?

| Estimate of the unavoidable error | MAE |
|---|---|
| random (Poisson) arrivals only | 14.8 (63% of 23.49) |
| measured from the data (robust estimate) | 19.2 |
| measured, including short-term trend changes (upper bound) | 21.1 |
| **our model** | **23.49** |

Real traffic is about twice as variable as random arrivals because cars come in platoons from upstream lights,
and a platoon can fall either side of a 15-min boundary. So **about 80% of the remaining error is noise** that no
feature or model can predict. With the unavoidable part at about 19–20, the avoidable error went from about 8.6
(original STAEformer) to about 3.8 (ours), so we removed roughly 55–60% of what could be removed.

The most error any perfect feature could remove: an area-wide one (weather, events, holidays) at most 3.7%;
one that knows which days are unusual 0%; which site-days are unusual (roadworks, closures) 0.4%; slow drift 0%
(overall bias under 0.3% in every quarter). A site's error barely predicts its error 15 min later (0.03).

- **One site dominates:** 4035 Burke/Barkers/Mont Albert carries about 2,400 vehicles per 15 min and causes 9% of
  all error, worst at night (22:00–06:00).
- **Broken test data:** site 2827 reports about 3,950 per 15 min on 19–23 Sep 2026, 10× normal. Its effect on the
  score is about 0.03.

## 9. Sensor faults (`clean_faults.py`)

Values that can't be real traffic, judged only from the site's past 4 and 26 weeks and the other sites at the
same moment (no future data). 0.56% of values are flagged:

| Fault | Share of values | Example |
|---|---|---|
| days undercounted by a dead detector | 0.52% | total below 85% of expected |
| implausible spikes | 0.04% | 2827 at 4,000+ on 19–21 Sep 2026; 800 cars at 3 am |
| runs of zeros | 0.04% | 0 for an hour where 50+ is normal |
| stuck values | <0.01% | the same count for 2 hours |

| 3-seed ensemble trained on | Scored on original labels | Scored on cleaned labels |
|---|---|---|
| raw data (final model) | **23.49** | **23.36** |
| cleaned data (`train.py --clean`) | 24.21 | 23.38 |

**Not kept:** on cleaned labels the two tie. On the original labels the cleaned model is much worse because most
faults are dead-detector days: the raw-trained model sees the low inputs and predicts low, which matches the
faulty label. A side effect also works against cleaning: removed days are filled with the site's 2014–2023
average, which is far off for sites whose level changed (2827's old average is about 3× its current traffic).
For the report, quote 23.49 on all observed test values and 23.36 with the 0.4% of faulty test values excluded.

## 10. Hyperparameter tuning (`train.py --hp`)

| Change (40 epochs, seed 1) | Val 2024 | Test |
|---|---|---|
| defaults: 3 layers, feed-forward 256, embeddings 24/24/80, dropout 0.1, lr 1e-3 | **23.87** | **23.78** |
| feed-forward 512 | 24.01 | 23.89 |
| 4 layers | 23.88 | 23.81 (20% slower) |
| wider embeddings 32/32/120 | 23.99 | 23.89 |
| dropout 0 | 24.13 | 24.01 |
| dropout 0.2 | 23.94 | 23.85 |

**Defaults kept:** model size and regularisation aren't the limit, which matches the noise analysis.
Not run (cancelled when we moved to 5-fold training): 8 heads, lr 2e-3, weight decay 0, batch 128, and a
backtest of training through 2022 and scoring 2024.

## 11. 5-fold cross-validation over all the data, 2014–2026

The single split never tests on old years and never trains on 2025–26. To use and test all of the data with one
model type, every window from Jan 2014 to Nov 2026 is split into 5 consecutive blocks (about 2.6 years each).
Fold k trains on the other 4 blocks and is scored on block k, which it never saw:

- training windows whose inputs reach into the held-out block (up to 4 weeks after it) are dropped;
- the scaling statistics come from the training blocks only;
- the last epoch is kept, so the held-out block isn't used to choose anything.

| Fold | Held out | MAE |
|---|---|---|
| 0 | Jan 2014 → Aug 2016 | 25.03 |
| 1 | Aug 2016 → Mar 2019 | 23.88 |
| 2 | Mar 2019 → Oct 2021 (COVID years) | 23.55 |
| 3 | Oct 2021 → Apr 2024 | 24.06 |
| 4 | Apr 2024 → Nov 2026 | 23.43 |
| **all held-out predictions** | **2014–2026** | **23.99** (± 0.63 over folds) |

| 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 25.24 | 25.96 | 23.34 | 23.87 | 23.90 | 23.49 | 23.65 | 23.77 | 23.02 | 24.81 | 23.67 | 23.52 | 23.71 |

- On the 2025–26 test windows (held out by fold 4) the MAE is 23.61 (RMSE 38.58, MAPE 8.91%), the same as the
  single models trained on 2014–2023 (23.60–23.67): the accuracy holds whichever period it's scored on.
- The COVID years are not harder (2020: 23.65, 2021: 23.77). 2014–15 and 2023 are, probably because of sensor
  problems and site changes; fold 0 also only had later years to learn from.
- **The final model is the plain average of the 5 fold models.** That average can't be scored fairly on any
  period (4 of the 5 models trained on every block), so the number to quote is the cross-validated 23.99.
- Re-running `cv_report.py` from the committed code reproduces these numbers exactly.

Weights: `outputs/weights/research/staeformer_ext-lag-wk4-diff-roll-cyc-event_cv5_f{0..4}.pt`.

## 12. Blending and stacking the models

- **Learned blend weights can't be fitted on the folds:** each window has exactly one held-out prediction (the
  other 4 models trained on it). Weighting by fold score would also be wrong, since each fold was scored on a
  different period. The median gives the same as the mean for 5 near-identical models.
- **Learned weights on the 3 seed models** (all held out on 2025–26): no change (23.59 vs 23.59 on 2026).
- **A LightGBM correction model on top** (`stack.py`; inputs: prediction, site, time of day, weekday, horizon),
  fitted on the other blocks' held-out predictions: worse in all 5 blocks, 23.99 → 24.08 overall and
  23.61 → 23.75 on 2025–26. **Not kept.**
- **Why a first test looked good:** a correction fitted on 2025 and scored on 2026 took the 3-seed models from
  23.59 to 23.20. That was recency, not stacking. The fold-4 model (trained up to Mar 2024) goes from 23.71 to
  23.35 on 2026 when corrected with its own Apr 2024 – Dec 2025 predictions, but gets worse (23.85) when corrected
  with older periods.
- **So traffic does drift, per site and time of day.** The overall level doesn't, but each site's daily pattern
  shifts over the years; a model that misses the latest 1–2 years loses about 0.35 MAE. **Retrain on the latest
  data.** Folds 0–3 include data up to Nov 2026; only fold 4 stops at Mar 2024. For predictions after Nov 2026,
  averaging folds 0–3 only might help slightly (about 0.05, not measurable without future data).

## 13. Gradient-boosted trees (`gbdt.py`)

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

- **The algorithm matters little:** all three are within 0.3 of each other with the same features.
- **They beat every graph model and the original STAEformer by about 3.5 MAE,** but those models never got
  these features. STAEformer with our features is still about 1.3 better than the best tree: a tree predicts one
  site at a time, while STAEformer looks at all 40 sites and the whole 3 h input together.
- **Same signals as STAEformer:** all three rank seasonal naive, the 4-week mean, same slot last week and yesterday
  highest by gain (XGBoost's top 5 also has the last value and public holiday). LightGBM's default ranking (by
  split count) favours continuous inputs like temperature and is misleading.
- **Training notes:** LightGBM used 4,998 of its 5,000 trees, so it was still improving; a 20,000-tree rerun was
  stopped after 2 h because it competed for the CPU with STAEformer runs. XGBoost scored 25.00 capped at 5,000
  trees and 24.99 with 8,703. CatBoost was gaining only 0.01 per 1,000 trees by the end. CatBoost's GPU version
  doesn't support the MAE loss properly, so it ran on the CPU.
- **Caveat:** the "nearby-site deviation" feature uses the distance graph, which is broken (section 0), so it is
  really an equal-weight average over the other sites.
- The fitted model files (118 / 317 / 67 MB) are too big for GitHub; `gbdt.py` rebuilds them.

## 14. tsfresh features (`tsfresh_select.py`, `tsfresh_features.py`)

tsfresh computed about 780 features on each site's last 24 h for 20,000 samples, compared with the 5-fold
held-out errors of 2014–2023 (no 2024–26 labels used to choose). 181 of the 556 tested features pass tsfresh's
relevance test (FDR 5%), but the strongest correlation with the error is only 0.06. We took the best feature of
each family (8 features; our GPU version matches tsfresh's values exactly).

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

**Not kept:** none helps alone (0.00–0.07 worse), and together they're about 0.2 worse. Three reasons:

- tsfresh's test looks at one feature at a time, not at what it adds to the existing inputs, and these mostly
  repeat them (maximum and quantile ≈ `roll`; mean absolute change and cid_ce ≈ `diff` and the raw inputs).
- Every extra channel goes through one shared layer that compresses them into 16 numbers; the parameter count grew
  by exactly 128 (8 × 16). The weak channels crowd the strong ones (last week, 4-week mean).
- Some have extreme values (c3 multiplies three values), which disturbs training even after clipping at ±10 sd.

Passing a statistical relevance test isn't enough: a feature has to improve the full model.

## 15. Speed limit (discussed, not tested)

- **Not as a flow feature:** a speed limit is fixed for each site, and STAEformer already learns a separate
  embedding per site. School-zone limits follow from time of day, weekday and school holidays, which the model
  already has. Only a change over time would add anything, and VicRoads publishes current zones, not when they
  changed (the roadworks feature, the closest test, didn't help).
- **Where it matters:** `core/travel_time.py` uses a flat 60 km/h, as the "Traffic Flow to Travel Time Conversion
  v1.0" spec says, while Boroondara's roads range from 50 to 70–80 km/h. Real limits (OpenStreetMap `maxspeed` or
  VicRoads speed zones) could be an optional extension, keeping 60 km/h as the default.

## What we learned

1. **Features beat models.** Swapping models changed the MAE by about 3; good features took STAEformer from
   28.30 to 23.49. Most of the gain came from "what happened at this time last week / yesterday / last 4 weeks".
2. **After that, the remaining error is mostly random.** Counting noise is about 80% of what's left, which is
   why more data (detectors, neighbours), LLM-proposed features, tsfresh, cleaning, tuning and stacking all failed
   to help. A "relevant-looking" feature has to be tested in the full model.
3. **Tree models are a strong, fast baseline** (XGBoost: 24.99 in 2 minutes) but can't use the other 40 sites
   the way an attention model can.
4. **The model is stable over 13 years** (23.0–24.0 MAE most years, COVID included), but each site's daily
   pattern shifts slowly, so retraining on recent data matters.
5. **Test discipline:** every decision was made on the 2024 validation set or held-out folds; the 2025–26 test
   set only reports results.

## Still open

1. Fix the site 4266 position in `prepare_data.py`, then retrain STGCN and Graph WaveNet (and the GCN in
   `core/tune.py`), and rerun `road_graph.py`.
2. Decide whether travel time uses the start or the end site's flow (code and docs disagree).
3. `requirements.txt`: convert to UTF-8 and add `holidays`.
4. Re-run `core/tune.py` properly (the saved parameters are from a smoke test).
5. Optional: per-road speed limits in `core/travel_time.py`; a lighter neighbour-site feature.
