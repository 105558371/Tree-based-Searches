# Work log: Assignment 2B (TBRGS) — Step 1 & 2 progress
Last updated: 2026-11-22

## 1. Status
| Item | Status |
|---|---|
| Data cleaning (site totals 2014-2026, Oct 2006) | Done (`data/`) |
| Coordinate alignment (global offset) | Offset found and checked; `coords_mapping.py` must apply it (see 5) |
| Volume -> travel time (`core/travel_time.py`) | Written, sign bug fixed (see 6) |
| Preprocessing (`core/data_processing.py`) | Tested on synthetic data |
| Models, trainer, tuning | Written and syntax-checked, **not yet run**: do the smoke test (section 7) |
| Final training, test results, 15+ test cases | Not started |
| Part A integration, top-5 routes, GUI | Not started |

## 2. Repo map
```
core/
  coords_mapping.py   SCATS lat/lon -> corrected GUI coordinates (JSON/CSV)
  check_map.py        folium map to verify alignment (open core/coord_check.html)
  travel_time.py      flow -> speed -> edge travel time
  data_processing.py  CSV -> scaled train/val/test + scaler + adjacency (.npz)
  models.py           RNN (LSTM/GRU) and GCN classes, LAYOUT table
  trainer.py          shared fit / evaluate / save_run
  tune.py             grid search -> best_params.json
data/                 site totals (.csv.gz) + filter/make scripts (see data/README.md)
outputs/              logs/<tag>/ and weights/<tag>/
```
`models.py` and `trainer.py` are libraries. Only run `data_processing.py` and `tune.py`.

## 3. Data
- 40 Boroondara sites only. The raw VicRoads zips (~18 GB) cover all Victoria, but `filter_sites.py` keeps the 40 sites from the Oct 2006 file.
- `site_totals_15min_2014_2026.csv.gz`: one row per site per day, `V00-V95` = vehicles per 15 min, summed over detectors. No data exists for 2007-2013.
- `site_totals_15min_oct2006.csv.gz`: assignment data in the same layout (the script that made it is not in the repo).
- Cleaning done: dates normalised, duplicates dropped, faulty (-1/blank) counts become blanks (~2%).
- 10 days are missing entirely (listed in `data/README.md`).

## 4. Preprocessing design (`data_processing.py`)
- Chronological split 70/15/15, no shuffling. Test is ~2024-26.
- Imputation: gaps up to 2 h are interpolated; longer gaps and missing days use the **train-only** median for that weekday and 15-min slot. For example, a missing Tuesday at 8 AM is filled with what a typical Tuesday at 8 AM looks like.
- Scaling: per-site min-max from the **train split only**. The constants are saved so predictions convert back to vehicles per 15 min.
- Val/test include the last `lookback` rows of the previous split, so no windows are lost.
- Windows are built lazily (`Windows` class). Each sample is x `(sites, lookback)`, y `(sites,)`. Horizon is fixed at 1 (next 15 min).
- The `.npz` also stores the GNN adjacency matrix (Gaussian kernel on haversine distance) and `node_order`.

## 5. Coordinates
- The dataset coordinates are shifted from the real intersections. Fix = global offset **dlat = +0.0015, dlon = +0.0013** (~165 m N, 115 m E).
- Derived from sites 2000, 3682, 3120 (agree within ~3 m). Checked on 3126, 3127 (residual 6-25 m).
- Verify any change with `python core/check_map.py`. Click real intersections on the OSM map (`LatLngPopup`) to read true coordinates.
- Open: make sure the kept `coords_mapping.py` applies the offset (the shared version only cleans and snaps) and that `check_map.py` reads its output keys (`gui_lat`/`gui_lon`).

## 6. Travel time
`time = distance / speed * 3600 + 30 s`, with `speed` from `flow = -1.4648375*s^2 + 93.75*s`.
- Speed is capped at 60 km/h for flow <= 351 veh/h, and 32 km/h at capacity (1500 veh/h). 15-min counts are multiplied by 4.
- Only the under-capacity (green) branch is used. The red branch can't be chosen from flow alone, and the PDF assumes every segment is under capacity.
- Sanity checks: `flow_to_speed(351)` = 60, `flow_to_speed(1500)` = 32.
- **Conflict to settle with the tutor:** the assignment brief says flow at site **B** (destination); the conversion PDF says flow at the **starting** site. The function takes a flow, so the caller chooses. Document the choice.
- Distance is straight-line (haversine), which underestimates road distance.

## 7. Running
```
python -m core.data_processing --tag smoke --max-days 60
python -m core.tune --npz data/processed/scats_smoke_lb12.npz --tag smoke --lrs 1e-3 --hidden 32 --layers 1 --epochs 2
```
Real run: tune on a 2-year subset, train on the full data.
```
python -m core.data_processing --tag tune --max-days 730
python -m core.tune --npz data/processed/scats_tune_lb12.npz --tag research
python -m core.data_processing --tag research
```
Then a per-model script (`train_lstm.py` etc.) reads `outputs/logs/research/best_params.json`, trains with early stopping, and saves weights, history and metrics.

## 8. Models and comparison
| Name | Input | Spatial awareness |
|---|---|---|
| `lstm`, `gru` | one site's history | none |
| `lstm_multi`, `gru_multi` (commented out, optional testing)| all sites' histories at once | implicit (learned correlations) |
| `gnn` | all sites + distance graph | explicit (distance-based adjacency) |
- Any design gives all 40 predictions in one forward pass, so "all sites at once" is not an efficiency question. The difference is whether a site sees its neighbours.
- Protocol so results are comparable: same preprocessing `.npz`, same split, RMSE in **vehicles per 15 min**, tune on val only, **test once** at the end. Give each model a similar tuning budget and note any difference.
- Tuning methods may differ per teammate, but each must cover learning rate and hidden units (assignment requirement).

## 9. Insights so far (for the report)
- Coordinate shift is systematic, not random: one global offset removes ~95% of the error.
- The 30 s intersection delay dominates short edges. Each edge costs driving time plus 30 s. A 1 km edge takes 60 s at 60 km/h and 112 s at 32 km/h, so heavy traffic adds about 52 s. A 0.5 km edge only gains about 26 s from traffic, which is less than the fixed 30 s. On short edges the fixed delay can matter as much as the prediction, so the route with the fewest intersections often wins. (PREDICTION ONLY, NOT RESULT).
- Test data (2024-26) is far from the training start. COVID and test shift: train includes 2020-21, when traffic was abnormal. Test is 2025-26, when patterns may differ (work from home, roadworks). The model may do worse on test than on val because the data changed, not because the model is bad.
- The split is chronological, 70/15/15. Models predict the future from the past, so test come after train. Shuffling would let the model see the future. The split works out roughly as:
  - train: 2014 to end 2022
  - val: 2023 to end 2024
  - test: 2025 to Nov 2026
- 2006 data covers only 1-3 approaches at 15 sites, so its totals are lower. Don't merge with 2014+ without normalising.
- Detector counts change over time (e.g. 4035 ~4x its 2006 total). A jump may be hardware, not traffic.
- Training loss is MSE on 0-1 scaled data. Val/test RMSE is in vehicles per 15 min. They have different units, don't plot them on one axis without saying so.
- Tuning runs are short, so the ranking is approximate. Close results are effectively ties.

### Big note on the data:
The raw 2006 file shows what these terms mean.

**Approach:**
An approach is one road arriving at an intersection, and the 2006 file has one row per approach per site. Site 2000 (Warrigal Rd / Toorak Rd) has 4 approaches: Warrigal Rd from the north, Burwood Hwy from the east, Warrigal Rd from the south and Toorak Rd from the west. Summing the approaches gives the total traffic through that intersection.

**Why only 15 sites:**
Those 15 are the sites where the 2006 file lists fewer approaches than a normal intersection:
- 24 of the other 25 sites have 4 approaches (one has 6).
- The 15 have only 1-3. For example, site 3126 (Canterbury Rd / Warrigal Rd) has 3 approaches in 2006: Canterbury east, Warrigal south and Canterbury west. The Warrigal north approach is missing.

I don't know why those approaches are missing. Possible reasons are that the 2006 dataset only included some approaches at certain sites, or that a detector was missing or not provided. The raw file doesn't say.

**Detector:**
A detector is a sensor loop in the road that counts vehicles. The 2014+ VicRoads data gives one row per detector, and an approach can have several (one per lane, for example). `make_site_totals.py` sums all detectors at a site into one total.

**Smaller:"**
For the 15 sites, the 2006 total leaves out the missing approaches, so it is lower than the true intersection total. The 2014+ total includes every detector at the site. The same intersection therefore looks smaller in 2006 than in 2014+, only because less of it was counted.

**Jumps:**
If you join the 2006 series to the 2014+ series for one of those sites, the count suddenly steps up at the join. The same happens inside the 2014+ data if a site gets extra detectors (site 4035 is about 4× its 2006 total). A model can read a step like that as a real traffic increase, when it's only a change in what is counted. This is why I suggested not mixing the two datasets for those sites, and checking `N_DETECTORS` before trusting a sudden change.

## 10. Open problems
- **The models predict the next 15 min from the previous 12 intervals (3 h). In the GUI, what does the user choose as the departure time? For example a date/time within the dataset, so the model uses the 3 h before it as history?**
- **Decide to work with the entire Victoria state (2006, 2014-2026) or 40 Boroondara sites (2006, 2014-2026) only.**
- Run the smoke test and fix any bugs in `models.py` / `trainer.py` / `tune.py`.
- Agree on one preprocessing output across the team (this repo's `.npz` or another), then freeze it.
- Flow at site A or B for edge cost (tutor).
- Build the graph edges (which sites are adjacent). The dataset has no adjacency. Options: derive from `Location` strings in the 2006 file, or k-nearest neighbours.
- Top-5 routes needs k-shortest paths (e.g. Yen's) on top of Part A.
- Part A code not yet integrated (nodes = SCATS numbers, cost = travel time, A* heuristic = straight-line distance / 60 km/h).
- Decide how to treat the 2006 data (baseline only, or excluded).
- 15+ test cases and logged results are still to do.
- GUI: confirm framework (web map vs canvas).

## 11. Things to pay attention to
- Run from the repo root. Paths are relative.
- Needs `core/__init__.py` for `python -m core.xxx`.
- Don't adjust anything based on **test** results. Val only, then test once.
- Save the scaler. Without `data_min`/`data_range`, you cannot compute RMSE in vehicles.
- RTX 50-series GPUs need PyTorch 2.7+ with CUDA 12.8+ (check `torch.cuda.is_available()`).
- Tuning on the full dataset takes hours. Use the 2-year subset, then train on full data.
- Batch size counts time windows, so a per-site model sees `batch x 40` sequences per step.
- Dropout is fixed at 0.1 (not tuned). State it in the report.
