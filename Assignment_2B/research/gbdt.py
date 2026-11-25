"""Gradient-boosted trees (LightGBM, XGBoost, CatBoost) on the same task, split and metrics as train.py.

Task: from the last 3 h of flow, predict each site's flow 15/30/45/60 min ahead.
Each row is one (time t, site, horizon h); the target is the flow at slot s = t + h (h = 0..3 -> 15..60 min).
Features (all known at time t; calendar, events and seasonal values are read at the target slot):
    last 12 values of the site, change over the last step and hour, mean of the last 24 h
    same slot last week / yesterday, mean of the same slot over the last 4 weeks (wk4)
    "seasonal naive": wk4 at the target slot + how far the site currently is from its own wk4
    area level (all sites now vs their wk4) and nearby sites' deviation (distance-weighted, from the graph)
    time of day, weekday, public/school holiday, lockdown, rain, temperature, MCG/tennis events
    site id (categorical), horizon, whether the last value was observed or gap-filled
Training rows: a random sample of observed targets from 2014-2023; 2024 is used for early stopping;
test = every observed value in the 2025-2026 test windows (the same windows train.py scores).
L1 (MAE) loss for all three models, up to --trees trees with early stopping on 2024 (200 rounds).
CatBoost runs on the CPU: its GPU version doesn't support the MAE loss properly (the model barely moves).

Usage (from Assignment_2B/): python research/gbdt.py [--models lightgbm,xgboost,catboost] [--train-rows 6000000] [--trees 20000]
Writes outputs/logs/research/<model>_metrics.json and the model to outputs/weights/research/.
"""
import argparse, json, time
import numpy as np
import pandas as pd
import train, events

WEEK, DAY, H, IN = 672, 96, train.HORIZON, 12
FIRST = IN + 4 * WEEK                       # same first window as train.py with wk4 features


def base_arrays():
    d = np.load(train.ROOT / "cache" / "traffic.npz", allow_pickle=True)
    flow, mask = d["flow"].astype(np.float32), d["mask"]
    times = d["times"].astype("datetime64[m]")
    T, N = flow.shape
    wk4 = np.full_like(flow, np.nan)
    wk4[4 * WEEK:] = np.mean([flow[4 * WEEK - j * WEEK:T - j * WEEK] for j in range(1, 5)], axis=0)
    c = np.concatenate([np.zeros((1, N), np.float32), np.cumsum(flow, 0, dtype=np.float64)]).astype(np.float64)
    roll24 = np.full_like(flow, np.nan)
    roll24[DAY:] = ((c[DAY:T] - c[:T - DAY]) / DAY).astype(np.float32)          # mean of [i-96, i)
    dev = flow - wk4
    area = flow.sum(1) / np.nansum(wk4, 1).clip(min=1)                          # whole area now vs usual
    adj = d["adj"].astype(np.float32).copy(); np.fill_diagonal(adj, 0)
    adj /= adj.sum(1, keepdims=True).clip(min=1e-6)
    nbr = np.nan_to_num(dev) @ adj.T                                             # nearby sites' deviation
    idx = pd.DatetimeIndex(d["times"])
    cal = np.stack([idx.hour * 4 + idx.minute // 15, idx.dayofweek, d["ext_public_holiday"], d["ext_school_holiday"],
                    d["ext_lockdown"]], 1).astype(np.float32)
    weather = np.stack([d["ext_rain_mm"], d["ext_temp_c"]], 1).astype(np.float32)
    ev = events.event_features(idx)
    return dict(flow=flow, mask=mask, times=times, wk4=wk4, roll24=roll24, dev=dev, area=area, nbr=nbr,
                cal=cal, weather=weather, ev=ev, N=N, T=T)


NAMES = ([f"last_{k}" for k in range(1, IN + 1)] +
         ["diff_1", "diff_4", "roll24", "lastweek_slot", "yesterday_slot", "wk4_slot", "wk4_now", "dev_now",
          "seasonal_naive", "area_now", "nbr_dev_now", "observed_last",
          "tod", "dow", "public_holiday", "school_holiday", "lockdown", "rain", "temp",
          "afl_arrive", "afl_leave", "cricket", "tennis", "site", "horizon"])


def features(a, t, n, h):
    """Rows (t = first target index of the window, site n, horizon h) -> X [R, F] float32, y [R]."""
    s = t + h
    f, last = a["flow"], t - 1
    cols = [f[t - k, n] for k in range(1, IN + 1)]
    cols += [f[last, n] - f[t - 2, n], f[last, n] - f[t - 5, n], a["roll24"][t, n],
             f[s - WEEK, n], f[s - DAY, n], a["wk4"][s, n], a["wk4"][last, n], a["dev"][last, n],
             a["wk4"][s, n] + a["dev"][last, n], a["area"][last], a["nbr"][last, n], a["mask"][last, n]]
    cols += list(a["cal"][s].T) + list(a["weather"][last].T) + list(a["ev"][s].T) + [n, h]
    return np.stack(cols, 1).astype(np.float32), f[s, n]


def rows(a, windows, rng=None, limit=None):
    """All observed (window, site, horizon) targets of these windows, or a random sample of `limit` of them."""
    t = np.repeat(windows, a["N"] * H)
    n = np.tile(np.repeat(np.arange(a["N"]), H), len(windows))
    h = np.tile(np.arange(H), len(windows) * a["N"])
    keep = a["mask"][t + h, n]
    t, n, h = t[keep], n[keep], h[keep]
    if limit and len(t) > limit:
        k = np.sort(rng.choice(len(t), limit, replace=False))
        t, n, h = t[k], n[k], h[k]
    return t, n, h


def metrics(pred, y, h):
    out = {}
    for k in range(H):
        e, yy = pred[h == k] - y[h == k], y[h == k]
        big = yy >= 10
        out[f"{15 * (k + 1)}min"] = {"MAE": round(float(np.abs(e).mean()), 3), "RMSE": round(float(np.sqrt((e ** 2).mean())), 3),
                                     "MAPE": round(float(100 * np.mean(np.abs(e[big]) / yy[big])), 2)}
    return out


TREES, PATIENCE = 20000, 200


def fit_lightgbm(Xtr, ytr, Xva, yva):
    import lightgbm as lgb
    m = lgb.LGBMRegressor(objective="l1", n_estimators=TREES, learning_rate=0.05, num_leaves=255, min_child_samples=100,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8, n_jobs=-1, verbose=-1,
                          importance_type="gain")
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], categorical_feature=[NAMES.index("site")],
          callbacks=[lgb.early_stopping(PATIENCE, verbose=False), lgb.log_evaluation(1000)])
    return m, m.predict, m.best_iteration_, lambda p: m.booster_.save_model(p + ".txt")


def fit_xgboost(Xtr, ytr, Xva, yva):
    import xgboost as xgb
    m = xgb.XGBRegressor(objective="reg:absoluteerror", n_estimators=TREES, learning_rate=0.05, max_depth=10,
                         min_child_weight=100, subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                         device="cuda", early_stopping_rounds=PATIENCE, max_bin=256, importance_type="gain")
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return m, m.predict, m.best_iteration, lambda p: m.save_model(p + ".json")


def fit_catboost(Xtr, ytr, Xva, yva):
    from catboost import CatBoostRegressor, Pool
    cat = [NAMES.index("site")]
    as_pool = lambda X, y=None: Pool(pd.DataFrame(X, columns=NAMES).astype({"site": int}), y, cat_features=cat)
    m = CatBoostRegressor(loss_function="MAE", iterations=TREES, learning_rate=0.1, depth=8, thread_count=-1,
                          od_type="Iter", od_wait=PATIENCE, verbose=500,
                          train_dir=str(train.ROOT / "cache" / "catboost_info"))   # not the working directory
    m.fit(as_pool(Xtr, ytr), eval_set=as_pool(Xva, yva), use_best_model=True)
    return m, lambda X: m.predict(as_pool(X)), m.get_best_iteration(), lambda p: m.save_model(p + ".cbm")


FITTERS = {"lightgbm": ("LightGBM", fit_lightgbm), "xgboost": ("XGBoost", fit_xgboost), "catboost": ("CatBoost", fit_catboost)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="lightgbm,xgboost,catboost")
    ap.add_argument("--train-rows", type=int, default=6_000_000)
    ap.add_argument("--val-rows", type=int, default=1_000_000)
    ap.add_argument("--trees", type=int, default=20000)
    a = ap.parse_args()
    global TREES
    TREES = a.trees
    rng = np.random.default_rng(0)
    t0 = time.time()
    arr = base_arrays()
    starts = np.arange(FIRST, arr["T"] - H + 1)
    tt = arr["times"][starts]
    split = {"train": starts[tt < train.VAL_START], "val": starts[(tt >= train.VAL_START) & (tt < train.TEST_START)],
             "test": starts[tt >= train.TEST_START]}
    Xtr, ytr = features(arr, *rows(arr, split["train"], rng, a.train_rows))
    Xva, yva = features(arr, *rows(arr, split["val"], rng, a.val_rows))
    te = rows(arr, split["test"])
    Xte, yte = features(arr, *te)
    print(f"features: train {len(ytr):,} rows, val {len(yva):,}, test {len(yte):,} (all observed test values) | "
          f"{len(NAMES)} features | {time.time() - t0:.0f}s", flush=True)

    sn = Xte[:, NAMES.index("seasonal_naive")]
    print(f"{'seasonal naive (wk4 + current deviation)':42s} test MAE {np.mean([v['MAE'] for v in metrics(np.nan_to_num(sn).clip(0), yte, te[2]).values()]):.3f}")
    for key in a.models.split(","):
        name, fitter = FITTERS[key]
        t1 = time.time()
        model, predict, best, save = fitter(Xtr, ytr, Xva, yva)
        minutes = (time.time() - t1) / 60
        test = metrics(np.clip(predict(Xte), 0, None), yte, te[2])
        imp = getattr(model, "feature_importances_", None)
        top = [NAMES[i] for i in np.argsort(imp)[::-1][:8]] if imp is not None else []
        out = {"model": name, "features": NAMES, "test_2025_2026": test, "best_iteration": int(best),
               "train_minutes": round(minutes, 1), "train_rows": len(ytr), "top_features": top}
        (train.LOGS / f"{key}_metrics.json").write_text(json.dumps(out, indent=2))
        save(str(train.WEIGHTS / key))
        print(f"{name:42s} test MAE {np.mean([v['MAE'] for v in test.values()]):.3f} | "
              + " ".join(f"{k} {v['MAE']:.2f}" for k, v in test.items())
              + f" | RMSE {np.mean([v['RMSE'] for v in test.values()]):.2f} | MAPE {np.mean([v['MAPE'] for v in test.values()]):.2f}%"
              + f" | {best} trees, {minutes:.1f} min | top: {', '.join(top[:5])}", flush=True)


if __name__ == "__main__":
    main()
