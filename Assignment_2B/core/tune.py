"""Grid search over learning rate, hidden units and layers for each model.
Every model gets the same grid, epochs and seed, so the tuning effort is equal.
Usage: python -m core.tune --npz data/processed/scats_research_lb12.npz --tag research"""
import argparse
import itertools
import json
import time
from pathlib import Path

import pandas as pd

from core.data_processing import load_dataset
from core.models import LAYOUT, build_model
from core.trainer import fit, get_device


def main():
    ap = argparse.ArgumentParser(description="Hyperparameter tuning (validation split only).")
    ap.add_argument("--npz", required=True)
    ap.add_argument("--tag", default="research", help="output sub-folder, e.g. basic / research")
    ap.add_argument("--models", nargs="+", default=["lstm", "gru", "gnn"], choices=list(LAYOUT))
    ap.add_argument("--lrs", nargs="+", type=float, default=[5e-4, 1e-3, 5e-3])
    ap.add_argument("--hidden", nargs="+", type=int, default=[64, 128, 256])
    ap.add_argument("--layers", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=8, help="short runs: only a ranking is needed")
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    ds, device = load_dataset(a.npz), get_device()
    out = Path("outputs/logs") / a.tag
    out.mkdir(parents=True, exist_ok=True)
    rows, curves = [], {}

    for name, lr, hidden, layers in itertools.product(a.models, a.lrs, a.hidden, a.layers):
        run_id = f"{name}_lr{lr}_h{hidden}_l{layers}"
        print(f"[{run_id}] on {device}")
        t0 = time.time()
        model = build_model(name, ds, hidden, layers)
        hist = fit(model, ds, LAYOUT[name], lr, a.epochs, a.patience, a.batch, a.seed,
                   device, verbose=False)
        curves[run_id] = hist
        rows.append({"model": name, "lr": lr, "hidden": hidden, "layers": layers,
                     "batch": a.batch, "best_val_rmse": min(e["val_rmse"] for e in hist),
                     "epochs_run": len(hist), "seconds": round(time.time() - t0, 1)})
        print(f"    best val RMSE {rows[-1]['best_val_rmse']:.2f} veh/15min")

    res = pd.DataFrame(rows).sort_values("best_val_rmse")
    res.to_csv(out / "tuning_results.csv", index=False)
    (out / "tuning_curves.json").write_text(json.dumps(curves))
    best = {}
    for m in a.models:
        r = res[res.model == m].iloc[0]
        best[m] = {"lr": float(r.lr), "hidden": int(r.hidden), "layers": int(r.layers),
                   "batch": int(r.batch)}
    (out / "best_params.json").write_text(json.dumps(best, indent=2))
    print("best params:", json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
