"""Which tsfresh features explain what STAEformer still gets wrong?

1. Errors: the 5-fold out-of-fold predictions (each window predicted by the fold model that never saw it),
   restricted to 2014-2023, so neither the 2024 validation year nor the 2025-2026 test labels are used.
2. Sample 20,000 (window, site) pairs; target = the error averaged over the 4 horizons, in site standard deviations.
3. tsfresh (EfficientFCParameters, ~780 features) on each site's last 24 h of normalised flow.
4. tsfresh's relevance test (one test per feature, Benjamini-Yekutieli FDR 5%) -> relevant features, ranked.

Usage (from Assignment_2B/): python research/tsfresh_select.py
Writes research/cache/tsfresh_relevance.csv
"""
import numpy as np
import pandas as pd
import torch
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters
from tsfresh.feature_selection.relevance import calculate_relevance_table
from tsfresh.utilities.dataframe_functions import impute
import train
from cv_report import fold_predictions

RUN, K, WINDOW, SAMPLES = "staeformer_ext-lag-wk4-diff-roll-cyc-event_cv5", 5, 96, 20_000


def main():
    d = np.load(train.ROOT / "cache" / "traffic.npz", allow_pickle=True)
    times, flow = d["times"].astype("datetime64[m]"), d["flow"]
    rows = times < train.VAL_START
    mean, std = flow[rows].mean(0), flow[rows].std(0) + 1e-6
    norm = (flow - mean) / std

    err, win = [], []
    for k in range(K):
        p, w, labels = fold_predictions(f"{RUN}_f{k}", k, K, "cuda")
        keep = times[w] < train.VAL_START
        if not keep.any():
            continue
        idx = torch.as_tensor(w[keep], device="cuda")
        i_out = idx[:, None] + torch.arange(train.HORIZON, device="cuda")
        y, m = labels["flow"][i_out], labels["mask"][i_out]
        e = torch.where(m, y - p[torch.as_tensor(np.flatnonzero(keep), device="cuda")], torch.nan)
        err.append(e.cpu().numpy()); win.append(w[keep])
    err, win = np.concatenate(err), np.concatenate(win)             # [W, 4, N] out-of-fold errors, window starts
    target = np.nanmean(err, 1) / std                                 # [W, N], in site standard deviations
    ok = np.isfinite(err).all(1)                                      # all 4 horizons observed
    wi, si = np.nonzero(ok)
    pick = np.random.default_rng(0).choice(len(wi), SAMPLES, replace=False)
    wi, si = wi[pick], si[pick]
    print(f"{ok.sum():,} usable (window, site) pairs from 2014-2023, sampled {SAMPLES:,}", flush=True)

    t = win[wi]
    series = norm[t[:, None] + np.arange(-WINDOW, 0)[None, :], si[:, None]]   # [S, 96] last 24 h of the site
    long = pd.DataFrame({"id": np.repeat(np.arange(SAMPLES), WINDOW), "time": np.tile(np.arange(WINDOW), SAMPLES),
                         "flow": series.ravel()})
    X = extract_features(long, column_id="id", column_sort="time", default_fc_parameters=EfficientFCParameters(),
                         n_jobs=16, disable_progressbar=True)
    X = impute(X)
    X = X.loc[:, X.std() > 0]
    y = pd.Series(target[wi, si], index=X.index)
    print(f"tsfresh: {X.shape[1]} non-constant features", flush=True)
    rel = calculate_relevance_table(X, y, ml_task="regression", fdr_level=0.05, n_jobs=16)
    rel = rel.sort_values("p_value")
    corr = X.corrwith(y, method="spearman")
    rel["spearman_with_error"] = corr.reindex(rel.index).round(4)
    rel.to_csv(train.ROOT / "cache" / "tsfresh_relevance.csv")
    print(f"relevant at FDR 5%: {int(rel['relevant'].sum())} of {len(rel)}")
    print(rel.head(25)[["p_value", "relevant", "spearman_with_error"]].to_string())


if __name__ == "__main__":
    main()
