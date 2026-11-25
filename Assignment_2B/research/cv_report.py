"""Out-of-fold results of the blocked K-fold cross-validation (train.py --fold k/K).

Every window from Jan 2014 to Nov 2026 is predicted by the one fold model that did not train on it.
Reports the held-out MAE of each fold, the pooled out-of-fold MAE / RMSE / MAPE, the MAE per year,
and the out-of-fold MAE on the original 2025-2026 test windows (comparable with earlier test results).

Usage (from Assignment_2B/): python research/cv_report.py staeformer_ext-lag-wk4-diff-roll-cyc-event_cv5 [--folds 5]
    reads runs <name>_f0 .. <name>_f{K-1}; writes outputs/logs/research/<name>_metrics.json
"""
import argparse, json
import numpy as np
import torch
import train, ensemble


def fold_predictions(run, k, K, device):
    """Rebuild fold k exactly as it was trained and predict its held-out windows."""
    args = json.loads((train.LOGS / f"{run}_metrics.json").read_text())["args"]
    train.IN_STEPS = args.get("in_steps", 12)
    train.CLEAN = args.get("clean", False)
    train.MODEL_HP = train.parse_hp(args.get("hp", ""))
    train.TSF_COLS = args.get("tsf_cols", "")
    train.FOLD = (k, K)                      # same held-out block and scaling statistics as in training
    data, split, adj = train.load(device, args["features"])
    in_dim = train.batch(data, torch.as_tensor(split["val"][:1], device=device))[0].shape[1]
    model = train.MODELS[args["model"]][1](data["flow"].shape[1], adj, in_dim).to(device)
    model.load_state_dict(torch.load(train.WEIGHTS / f"{run}.pt", map_location=device))
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(split["val"]), 512):
            idx = torch.as_tensor(split["val"][i:i + 512], device=device)
            out.append(train.predict(model, data, idx).clamp(min=0))
    return torch.cat(out), split["val"], {"flow": data["flow"], "mask": data["mask"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    times = np.load(train.ROOT / "cache" / "traffic.npz", allow_pickle=True)["times"].astype("datetime64[m]")
    show = lambda m: f"MAE {ensemble.avg_mae(m):6.2f} | " + " ".join(f"{h} {v['MAE']:.2f}" for h, v in m.items())

    preds, wins, folds = [], [], {}
    for k in range(a.folds):
        p, w, labels = fold_predictions(f"{a.name}_f{k}", k, a.folds, device)
        m = ensemble.score(p, labels, w)
        folds[k] = {"held_out": f"{times[w[0]]} -> {times[w[-1]]}", "metrics": m}
        print(f"fold {k}  {folds[k]['held_out']:40s} {show(m)}")
        preds.append(p); wins.append(w)
    p, w = torch.cat(preds), np.concatenate(wins)
    pooled = ensemble.score(p, labels, w)              # labels (raw flow + mask) are the same for every fold
    fold_mae = [ensemble.avg_mae(f["metrics"]) for f in folds.values()]
    print(f"pooled out-of-fold {'':33s} {show(pooled)} | RMSE {np.mean([v['RMSE'] for v in pooled.values()]):.2f}"
          f" | MAPE {np.mean([v['MAPE'] for v in pooled.values()]):.2f}%")
    print(f"fold MAE mean {np.mean(fold_mae):.2f} +- {np.std(fold_mae, ddof=1):.2f} (sd over folds)")

    years = times[w].astype("datetime64[Y]").astype(int) + 1970
    per_year = {}
    for y in np.unique(years):
        sel = torch.as_tensor(np.flatnonzero(years == y), device=p.device)
        per_year[int(y)] = round(ensemble.avg_mae(ensemble.score(p[sel], labels, w[years == y])), 3)
    print("MAE per year:", per_year)
    sel = times[w] >= train.TEST_START
    test = ensemble.score(p[torch.as_tensor(np.flatnonzero(sel), device=p.device)], labels, w[sel])
    print(f"original test period 2025-2026 (out-of-fold) {show(test)}")

    out = {"model": f"{a.folds}-fold blocked CV", "runs": [f"{a.name}_f{k}" for k in range(a.folds)],
           "folds": folds, "pooled_out_of_fold": pooled,
           "fold_mae_mean": float(np.mean(fold_mae)), "fold_mae_sd": float(np.std(fold_mae, ddof=1)),
           "mae_per_year": per_year, "test_2025_2026_out_of_fold": test}
    (train.LOGS / f"{a.name}_metrics.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
