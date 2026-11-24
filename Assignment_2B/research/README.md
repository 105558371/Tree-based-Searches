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
| 6 h input window instead of 3 h | worse | no |

- **events:** they barely change the overall MAE, but on 2024 AFL nights the error at the western sites
  2.5–4 h after the start falls from 72 to 50.
- **Where the remaining error comes from:** counting noise accounts for about 80% of the 23.49 (estimated
  floor of about 19 MAE from the data). A perfect area-wide feature could remove at most 3.7%; a perfect
  per-day or per-site-day feature at most 0.4%.
- **Sensor faults** (`clean_faults.py`, `train.py --clean`): 0.56% of values (dead detectors, spikes, stuck
  values, zero runs). Training on cleaned data didn't help (23.38 vs 23.36 on cleaned labels), so it isn't used.
- **nbr, det:** need `research/cache/extra.npz`, which `extract_raw.py` + `prepare_extra.py` build from the raw
  VicRoads archives (about 19 GB, not committed).

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

## Results (test 2025–2026, averaged over the 4 horizons; full table in `outputs/logs/research/comparison.md`)

| Model | Type | MAE | RMSE | MAPE | Params | Train time |
|---|---|---|---|---|---|---|
| STAEformer | Transformer | **28.30** | **45.76** | **10.61%** | 842k | 138 min |
| Graph WaveNet | GNN | 28.70 | 46.28 | 10.91% | 294k | 56 min |
| AGCRN | GNN | 28.88 | 46.72 | 10.82% | 749k | 46 min |
| STGCN | GNN | 29.79 | 47.93 | 10.94% | 160k | 29 min |
| MTGNN | GNN | 29.86 | 47.87 | 11.24% | 163k | 15 min |
| STID | MLP | 31.05 | 50.93 | 11.62% | 61k | 2 min |
| Last value | Baseline | 58.74 | 91.46 | — | — | — |

## References

- Wu et al. (2019) *Graph WaveNet for Deep Spatial-Temporal Graph Modeling*, IJCAI — github.com/nnzhan/Graph-WaveNet
- Yu et al. (2018) *Spatio-Temporal Graph Convolutional Networks* (STGCN), IJCAI
- Bai et al. (2020) *Adaptive Graph Convolutional Recurrent Network* (AGCRN), NeurIPS
- Wu et al. (2020) *Connecting the Dots: Multivariate Time Series Forecasting with GNNs* (MTGNN), KDD
- Shao et al. (2022) *Spatial-Temporal Identity: A Simple yet Effective Baseline* (STID), CIKM
- Liu et al. (2023) *STAEformer: Spatio-Temporal Adaptive Embedding Makes Vanilla Transformer SOTA*, CIKM
