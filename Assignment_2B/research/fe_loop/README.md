# LLM feedback-loop feature engineering

`research/fe_loop.py` runs a CAAFE-style loop (Hollmann et al., 2023): an LLM writes candidate
features as code, the script checks and scores them, and the history goes back to the LLM.
In this run the LLM was Claude Code (in-session); the same proposal format works for any LLM.

Checks per proposal: AST whitelist (numpy/pandas/math only), separate process with timeout,
shape check, leakage test (scramble future traffic/weather; earlier rows must not change).
Score: MAE drop (vehicles / 15 min) when the feature is added to a LightGBM model that corrects
the current best STAEformer's 2024 validation residuals (fit Jan-Jun, scored Jul-Dec).
Noise floor (random feature): -0.025.

## Results (base model ext+lag+wk4+diff+roll+cyc+event, val MAE 23.87 / 23.78 over 2 seeds)

| Round | Feature | Proxy score |
|---|---|---|
| 1 | analog_week, analog_day (level-adjusted lags) | -0.013, -0.005 |
| 1 | lag_regime_mismatch, regime_matched_lastweek | -0.016, -0.008 |
| 1 | median_8w, neighbor_anomaly | -0.022, -0.027 |
| 2 | input_observed_frac | **+0.015** |
| 2 | steps_since_observed | +0.006 |
| 2 | day_volume_ratio, rain_3h, level_shift | -0.033, -0.035, -0.029 |
| 2 | docklands_leave, docklands_arrive (AFL at Marvel Stadium) | -0.003, 0.000 |

Real STAEformer check (40 epochs) with the two positives (`--features ...+llm`, `selected.json`):
23.79 / 24.17 (2 seeds, mean 23.98) vs 23.87 / 23.78 without (mean 23.83) -> not adopted.

Why the loop stopped: the base model's residuals are close to white noise - autocorrelation
0.017 one hour later and ~0 one day / week later, mean cross-site correlation 0.048, and the
area-wide component explains 6% of residual variance. Features built from past traffic, calendar,
weather or events have little left to explain; further gains need new data (neighbouring sites,
per-detector counts). The proxy only sees 2024, so it cannot score features that help the model
learn from older years (e.g. a pre/post-COVID flag); those need a full training run.
