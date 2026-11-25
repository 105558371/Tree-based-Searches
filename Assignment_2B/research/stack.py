"""Second-stage correction model on top of the 5-fold STAEformer (stacking).

STAEformer makes small systematic errors per site and time of day. A LightGBM model (L1 loss) learns them from
the out-of-fold predictions: inputs = STAEformer prediction, site, time of day and weekday of the target slot,
horizon; target = what STAEformer got wrong (actual - prediction).

Honest evaluation: every out-of-fold prediction comes from a fold model that never saw that window, and the
correction model for block k is fitted only on the other 4 blocks, so block k is still unseen.
Result (24 Nov): no gain (23.99 -> 24.08 pooled, 23.61 -> 23.75 on 2025-2026), so it isn't used. A correction only
helps a model that hasn't seen the latest 1-2 years (see README, "Blending the 5 folds").

Usage (from Assignment_2B/): python research/stack.py staeformer_ext-lag-wk4-diff-roll-cyc-event_cv5
Writes outputs/logs/research/<name>_stack_metrics.json.
"""
import argparse, json
import numpy as np
import torch
import lightgbm as lgb
import train, ensemble
from cv_report import fold_predictions

FEATURES = ["pred", "site", "tod", "dow", "horizon"]
PARAMS = dict(objective="l1", n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=200,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
MAX_ROWS = 2_000_000          # rows sampled to fit each correction model


def table(pred, wins, times):
    """One row per (window, horizon, site): the features above, as a [W, 4, N, 5] array."""
    W, H, N = pred.shape
    t_out = times[wins][:, None] + np.arange(H) * np.timedelta64(15, "m")            # target slot times [W, 4]
    tod = (t_out - t_out.astype("datetime64[D]")).astype(int) // 15
    dow = (t_out.astype("datetime64[D]").astype(int) + 3) % 7                         # 0 = Monday
    b = lambda a: np.broadcast_to(a, (W, H, N))
    return np.stack([pred, b(np.arange(N)[None, None, :]), b(tod[:, :, None]), b(dow[:, :, None]),
                     b(np.arange(H)[None, :, None])], -1).astype(np.float32)


def fit(X, r, seed=0):
    k = np.random.default_rng(seed).choice(len(X), min(len(X), MAX_ROWS), replace=False)
    return lgb.LGBMRegressor(**PARAMS, random_state=seed).fit(X[k], r[k], categorical_feature=[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    times = np.load(train.ROOT / "cache" / "traffic.npz", allow_pickle=True)["times"].astype("datetime64[m]")

    preds, wins = [], []
    for k in range(a.folds):
        p, w, labels = fold_predictions(f"{a.name}_f{k}", k, a.folds, device)
        preds.append(p.cpu().numpy()); wins.append(w)
    idx = torch.as_tensor(np.concatenate(wins), device=device)
    i_out = idx[:, None] + torch.arange(train.HORIZON, device=device)
    y, m = labels["flow"][i_out].cpu().numpy(), labels["mask"][i_out].cpu().numpy()
    pred, wins_all = np.concatenate(preds), np.concatenate(wins)
    X = table(pred, wins_all, times)
    block = np.concatenate([np.full(len(w), k) for k, w in enumerate(wins)])

    corrected = pred.copy()
    for k in range(a.folds):                       # correction for block k fitted on the other blocks only
        tr = m & (block != k)[:, None, None]
        te = m & (block == k)[:, None, None]
        model = fit(X[tr], (y - pred)[tr], seed=k)
        corrected[te] = np.clip(pred[te] + model.predict(X[te]), 0, None)
        print(f"block {k}: MAE {np.abs(pred - y)[te].mean():.3f} -> {np.abs(corrected - y)[te].mean():.3f}", flush=True)

    def metrics(p, sel):
        out = {}
        for h in range(train.HORIZON):
            mm = m[:, h] & sel[:, None]
            e, yy = (p[:, h] - y[:, h])[mm], y[:, h][mm]
            big = yy >= 10
            out[f"{15 * (h + 1)}min"] = {"MAE": round(float(np.abs(e).mean()), 3), "RMSE": round(float(np.sqrt((e ** 2).mean())), 3),
                                         "MAPE": round(float(100 * np.mean(np.abs(e[big]) / yy[big])), 2)}
        return out
    allw, test = np.ones(len(wins_all), bool), times[wins_all] >= train.TEST_START
    out = {"features": FEATURES, "params": PARAMS}
    for label, sel in [("all_2014_2026", allw), ("test_2025_2026", test)]:
        for kind, p in [("staeformer", pred), ("staeformer+correction", corrected)]:
            out[f"{label} | {kind}"] = s = metrics(p, sel)
            print(f"{label:15s} {kind:22s} MAE {ensemble.avg_mae(s):.3f} | "
                  + " ".join(f"{h} {v['MAE']:.2f}" for h, v in s.items())
                  + f" | RMSE {np.mean([v['RMSE'] for v in s.values()]):.2f} | MAPE {np.mean([v['MAPE'] for v in s.values()]):.2f}%")
    (train.LOGS / f"{a.name}_stack_metrics.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
