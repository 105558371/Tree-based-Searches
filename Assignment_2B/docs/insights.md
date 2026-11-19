# Insights (for the Part B report)

## Data
- Only 2014-2026 data is available. Nothing exists for 2007-2013, so there is a gap before the 2006 file.
- About 2% of 15-min values are blank (faulty detectors) and 10 days are missing. Imputation method: TBD.
- Detector counts change over time (e.g. site 4035 is about 4x its 2006 total), so a jump in
  counts may be a hardware change, not real traffic. Check `N_DETECTORS`.
- 15 sites have lower 2006 totals because 2006 covers only 1-3 approaches. Don't mix 2006 with
  2014+ data for these sites without normalising.

## Coordinates
- The dataset coordinates are consistently shifted by about (+0.0015, +0.0013) degrees. Offsets
  from 3 reference sites agreed within about 3 m, so one global shift was used.
- After correction, residual error at checked sites is 6-25 m, with one outlier of about 70 m (4273).
- A 70 m error on a roughly 1 km edge changes travel time by about 5 s, small compared with the 30 s delay.

## Travel time
- Distance is straight-line (haversine), which underestimates road distance. Real routes are longer.
- Flow is taken at the start site only, per the spec. Above capacity (1500 veh/h) speed is clamped to 32 km/h.
- With the spec's 30 s per intersection, the delay dominates short edges, so route choice can
  be driven more by hop count than by predicted traffic.

## Models and tuning
- Grid: lr {1e-3, 5e-3, 1e-2} x hidden {32, 64, 128}, for LSTM, GRU and GNN. Results: TBD
- The GNN predicts all sites jointly while LSTM/GRU predict one site at a time, so setups are not identical.
- The GNN adjacency is distance-based, not the real road network. Using the Part A edges is a possible improvement.
- Best parameters per model: TBD. RMSE comparison: TBD.
