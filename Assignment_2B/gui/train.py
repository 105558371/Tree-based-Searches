"""Train LSTM, GRU and CNN over a small hyperparameter grid.
Writes results/metrics.csv (test error of every run) and models/<name>.pt (best run by validation error).
Usage: python train.py [--models lstm gru cnn] [--quick]"""
import argparse
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tbrgs.config import find_dataset, load_config
from tbrgs.data import load_scats, make_windows, split_bounds
from tbrgs.models import count_params, predict, train_model

BASE = Path(__file__).parent


def baselines(series, a, b, end):
    """Error of two models that need no training, on the validation and test targets:
    persistence (next flow = last observed flow) and seasonal mean (average of the same time of day in training)."""
    out = {}
    for name in ("Persistence", "Seasonal mean"):
        err = {"va": [], "te": []}
        for s in series.values():
            tod_mean = s[:a].reshape(-1, 96).mean(axis=0)
            for key, (lo, hi) in {"va": (a, b), "te": (b, end)}.items():
                idx = np.arange(max(lo, 1), min(hi, len(s)))
                pred = s[idx - 1] if name == "Persistence" else tod_mean[idx % 96]
                err[key].append(pred - s[idx])
        va, te = np.concatenate(err["va"]), np.concatenate(err["te"])
        out[name] = {"model": name, "hidden_units": np.nan, "learning_rate": np.nan, "window": np.nan, "epochs": 0,
                     "val_rmse": float(np.sqrt(np.mean(va ** 2))), "rmse": float(np.sqrt(np.mean(te ** 2))),
                     "mae": float(np.mean(np.abs(te))), "params": 0, "train_seconds": 0.0}
    return list(out.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["lstm", "gru", "cnn"])
    ap.add_argument("--quick", action="store_true", help="few sites and epochs, to check that everything runs")
    args = ap.parse_args()
    cfg = load_config(BASE / "config.json")
    data_path = find_dataset(BASE, cfg)
    if data_path is None:
        raise SystemExit("Dataset not found. Put the Boroondara file (.xlsx, .xls or .csv) in the data folder.")
    ds = load_scats(data_path)
    series = dict(list(ds.series.items())[:5]) if args.quick else ds.series
    a, b, end = split_bounds(len(ds.dates), cfg["train_frac"], cfg["val_frac"])
    scale = float(max(s[:a].max() for s in series.values())) or 1.0
    grid = cfg["grid"]
    rows = baselines(series, a, b, end)
    windows = {}                                     # the windows depend only on the window length, so build them once

    def get_data(win):
        if win not in windows:
            windows[win] = {}
            for n, (lo, hi) in {"tr": (0, a), "va": (a, b), "te": (b, end)}.items():
                x, y = make_windows(series, win, lo, hi)
                x[:, :, 0] /= scale                  # inputs and labels are both in units of `scale`
                windows[win][n] = (x, y / scale)
        return windows[win]

    for kind in args.models:
        best = None
        for hid, lr, win, ep in itertools.product(grid["hidden_units"], grid["learning_rate"], grid["window"], grid["epochs"]):
            ep = 2 if args.quick else ep
            data = get_data(win)
            t0 = time.time()
            net, hist = train_model(kind, *data["tr"], *data["va"], hidden=hid, lr=lr, epochs=ep)
            secs = time.time() - t0
            err = lambda n: (predict(net, data[n][0]) - data[n][1]) * scale   # back to vehicles per 15 minutes
            val_rmse, test = float(np.sqrt(np.mean(err("va") ** 2))), err("te")
            rows.append({"model": kind.upper(), "hidden_units": hid, "learning_rate": lr, "window": win, "epochs": ep,
                         "val_rmse": val_rmse, "rmse": float(np.sqrt(np.mean(test ** 2))),
                         "mae": float(np.mean(np.abs(test))), "params": count_params(net), "train_seconds": round(secs, 1),
                         "train_rmse": float(np.sqrt(np.mean(err("tr") ** 2))),      # to compare with val_rmse (overfitting check)
                         "best_epoch": int(np.argmin([h[1] for h in hist])) + 1 if hist else 0})
            print(rows[-1])
            if best is None or val_rmse < best[0]:
                best = (val_rmse, {"state": net.state_dict(), "kind": kind, "in_size": 3, "hidden": hid,
                                   "window": win, "scale": scale})
        (BASE / cfg["models_dir"]).mkdir(exist_ok=True)
        torch.save(best[1], BASE / cfg["models_dir"] / f"{kind}.pt")
    (BASE / "results").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(BASE / "results" / "metrics.csv", index=False)
    print("Saved results/metrics.csv and models/")


if __name__ == "__main__":
    main()
