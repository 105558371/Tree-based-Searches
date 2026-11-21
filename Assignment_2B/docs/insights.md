# Insights (for the Part B report)

## Data
- Only 2014-2026 data is available. Nothing exists for 2007-2013, so there is a gap before the 2006 file.
- About 2% of 15-min values are blank (faulty detectors) and 10 days are missing. Imputation method: TBD.
- Detector counts change over time (e.g. site 4035 is about 4x its 2006 total), so a jump in
  counts may be a hardware change, not real traffic. Check `N_DETECTORS`.
- 15 sites have lower 2006 totals because 2006 covers only 1-3 approaches. Don't mix 2006 with
  2014+ data for these sites without normalising.
- Handling missing data: Uses linear interpolation for short gaps (< 2 hrs), but for long gaps, it fills with the median for the same weekday and time of day.
For example, a missing Tuesday at 8 AM is filled with what a typical Tuesday at 8 AM looks like. This will improve model accuracy.

## Coordinates
- The dataset coordinates are consistently shifted by about (+0.0015, +0.0013) degrees. Offsets
  from 3 reference sites agreed within about 3 m, so one global shift was used.
- After correction, residual error at checked sites is 6-25 m, with one outlier of about 70 m (4273).
- A 70 m error on a roughly 1 km edge changes travel time by about 5 s, small compared with the 30 s delay.
- `fix_signs()` explicitly handles the swapped/inverted signs that often plague VicRoads data (making sure lat is negative and lon is positive for Melbourne). It also allows for a global offset (dlat, dlon) and specific manual overrides (coord_overrides.json) to fix known bad sites.

## Travel time
- Distance is straight-line (haversine), which underestimates road distance. Real routes are longer.
- Flow is taken at the start site only, per the spec. Above capacity (1500 veh/h) speed is clamped to 32 km/h.
- With the spec's 30 s per intersection, the delay dominates short edges, so route choice can
  be driven more by hop count than by predicted traffic.
- Using Site B's volume works well as long as traffic is flowing freely (the green branch). However, if a severe bottleneck blocks intersection B, Site B's detector will count very few cars crossing the stop line—tricking the green-branch formula into thinking traffic is light and fast (60 km/h), when in reality cars are stuck in a queue behind B.

## Models and tuning
- LSTM, GRU, and GNN: Given historical SCATS traffic counts (V00–V95), these models predict future traffic volume at each SCATS site/intersection. GNNs are especially helpful because they account for neighboring nodes on the graph, while LSTMs and GRUs capture temporal patterns over time. What they output are predicted traffic flows, not a sequence of roads to drive.
- Grid: lr {1e-3, 5e-3, 1e-2} x hidden {32, 64, 128}, for LSTM, GRU and GNN. Results: TBD
- The GNN predicts all sites jointly while LSTM/GRU predict one site at a time, so setups are not identical.
- The GNN adjacency is distance-based, not the real road network. Using the Part A edges is a possible improvement.
- Best parameters per model: TBD. RMSE comparison: TBD.

## Example (not real data):
- A grid search over the intersection of learning rates [1e-3, 5e-3, 1e-2] and hidden dimensions [64, 128] revealed that an LR of 0.001 with 64 hidden units provided the best validation RMSE for both LSTM and GRU, while the GNN required a higher learning rate of 0.005 to effectively propagate spatial messages...
