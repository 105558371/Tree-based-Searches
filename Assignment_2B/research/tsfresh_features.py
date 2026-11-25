"""The `tsf` feature group: tsfresh features over each site's last 24 h, for every site and 15-min step.

Chosen by tsfresh_select.py (tsfresh's relevance test against the 5-fold out-of-fold errors, 2014-2023):
the most relevant feature of each family. tsfresh itself would take days on 18M windows, so the same
definitions are computed here on the GPU over sliding windows; verify() checks them against tsfresh.
The value at step i uses the window of steps i-95 .. i (inclusive), so train.py can read it at any input step.

Usage (from Assignment_2B/): python research/tsfresh_features.py  -> research/cache/tsf.npy [T, 40, 8] float32
(standardised on 2014-2023, clipped to +-10 sd)
"""
import numpy as np
import torch
import train

WINDOW = 96
NAMES = ["quantile__q_0.2", 'change_quantiles__f_agg_"mean"__isabs_True__qh_0.4__ql_0.0', "maximum",
         "mean_abs_change", "cid_ce__normalize_False", "c3__lag_3", "number_crossing_m__m_1",
         "sum_of_reoccurring_values"]


def window_features(x):
    """x [B, 96] -> [B, 8], tsfresh's definitions."""
    d = x[:, 1:] - x[:, :-1]
    q02, q04 = torch.quantile(x, 0.2, dim=1), torch.quantile(x, 0.4, dim=1)
    inband = x <= q04[:, None]                                    # pd.qcut(x, [0, 0.4]) == 0
    pair = inband[:, 1:] & inband[:, :-1]
    n_pair = pair.sum(1)
    cq = torch.where((n_pair > 0) & (q04 > x.min(1).values),      # tsfresh returns 0 if no pair or equal edges
                     (d.abs() * pair).sum(1) / n_pair.clamp(min=1), torch.zeros_like(q04))
    c3 = (x[:, 6:] * x[:, 3:-3] * x[:, :-6]).mean(1)
    pos = x > 1
    cross = (pos[:, 1:] != pos[:, :-1]).sum(1).float()
    s = torch.sort(x, dim=1).values
    same = s[:, 1:] == s[:, :-1]                                   # s[k] == s[k+1]
    first = same & torch.cat([torch.ones_like(same[:, :1]), ~same[:, :-1]], 1)   # first pair of each repeated run
    reocc = (s[:, :-1] * first).sum(1)
    return torch.stack([q02, cq, x.max(1).values, d.abs().mean(1), (d ** 2).sum(1).sqrt(), c3, cross, reocc], 1)


def normalised_flow():
    d = np.load(train.ROOT / "cache" / "traffic.npz", allow_pickle=True)
    flow, times = d["flow"], d["times"].astype("datetime64[m]")
    rows = times < train.VAL_START
    mean, std = flow[rows].mean(0), flow[rows].std(0) + 1e-6
    std = np.where(std < 1, 1.0, std)
    return ((flow - mean) / std).astype(np.float32), rows


def build(device="cuda", chunk=4096):
    norm, rows = normalised_flow()
    T, N = norm.shape
    x = torch.as_tensor(norm, device=device)
    out = torch.zeros(T, N, len(NAMES), device=device)
    for a in range(WINDOW - 1, T, chunk):
        b = min(a + chunk, T)
        win = x[a - WINDOW + 1:b].unfold(0, WINDOW, 1)              # [b-a, N, 96]
        out[a:b] = window_features(win.reshape(-1, WINDOW)).reshape(b - a, N, -1)
    out = out.cpu().numpy()
    tr = out[WINDOW - 1:][rows[WINDOW - 1:]]
    mu, sd = tr.mean((0, 1)), tr.std((0, 1)) + 1e-6
    out = np.clip((out - mu) / sd, -10, 10).astype(np.float32)
    out[:WINDOW - 1] = 0
    return out


def verify(n=300):
    from tsfresh.feature_extraction import feature_calculators as fc
    norm, _ = normalised_flow()
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(n):
        i, s = rng.integers(WINDOW, len(norm)), rng.integers(norm.shape[1])
        w = norm[i - WINDOW + 1:i + 1, s].astype(np.float64)
        ref = [fc.quantile(w, 0.2), fc.change_quantiles(w, 0.0, 0.4, True, "mean"), fc.maximum(w), fc.mean_abs_change(w),
               fc.cid_ce(w, False), fc.c3(w, 3), fc.number_crossing_m(w, 1), fc.sum_of_reoccurring_values(w)]
        got = window_features(torch.as_tensor(w[None], dtype=torch.float64))[0].numpy()
        worst = max(worst, np.max(np.abs(got - np.array(ref, dtype=float)) / (1 + np.abs(ref))))
    return worst


if __name__ == "__main__":
    print(f"largest relative difference to tsfresh on 300 random windows: {verify():.2e}")
    f = build()
    np.save(train.ROOT / "cache" / "tsf.npy", f)
    print("saved cache/tsf.npy", f.shape)
