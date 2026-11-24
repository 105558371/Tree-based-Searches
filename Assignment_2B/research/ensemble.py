"""Average the test predictions of several runs saved by train.py (e.g. STAEformer trained with
different seeds) and score the average with the same masked MAE / RMSE / MAPE as train.py.

Usage: python research/ensemble.py RUN [RUN ...] --name NAME
    RUN   a run name as saved by train.py, e.g. staeformer_ext_cos_s1 (weights + metrics json)
Writes outputs/logs/research/<NAME>_metrics.json.
"""
import argparse, json
import torch
import train


def member_predictions(run, device, cache):
    """Rebuild a run from the settings in its metrics json and predict every test window."""
    args = json.loads((train.LOGS / f"{run}_metrics.json").read_text())["args"]
    features = args.get("features", "base")
    train.IN_STEPS = args.get("in_steps", 12)          # read by batch() and the model builders
    train.CLEAN = args.get("clean", False)             # read by load()
    key = (features, train.CLEAN)
    if key not in cache:
        cache[key] = train.load(device, features)
    data, split, adj = cache[key]
    in_dim = train.batch(data, torch.as_tensor(split["test"][:1], device=device))[0].shape[1]
    model = train.MODELS[args["model"]][1](data["flow"].shape[1], adj, in_dim).to(device)
    model.load_state_dict(torch.load(train.WEIGHTS / f"{run}.pt", map_location=device))
    model.eval()
    out = []
    with torch.no_grad():
        for k in range(0, len(split["test"]), 512):
            idx = torch.as_tensor(split["test"][k:k + 512], device=device)
            out.append(train.predict(model, data, idx).clamp(min=0))
    return torch.cat(out), data, split["test"]


def score(pred, data, windows):
    """Same metrics as train.evaluate: observed values only, MAPE skips flow < 10."""
    idx = torch.as_tensor(windows, device=pred.device)
    i_out = idx[:, None] + torch.arange(train.HORIZON, device=pred.device)
    y, m = data["flow"][i_out], data["mask"][i_out].float()
    e, pm = (pred - y) * m, m * (y >= 10)
    mae = e.abs().sum((0, 2)) / m.sum((0, 2))
    rmse = ((e ** 2).sum((0, 2)) / m.sum((0, 2))).sqrt()
    mape = 100 * (e.abs() / y.clamp(min=1) * pm).sum((0, 2)) / pm.sum((0, 2))
    return {f"{15 * (h + 1)}min": {"MAE": round(mae[h].item(), 3), "RMSE": round(rmse[h].item(), 3),
                                   "MAPE": round(mape[h].item(), 2)} for h in range(train.HORIZON)}


def avg_mae(metrics):
    return sum(v["MAE"] for v in metrics.values()) / len(metrics)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    cache, preds, members = {}, [], {}
    for run in a.runs:
        p, data, windows = member_predictions(run, device, cache)
        members[run] = score(p, data, windows)
        preds.append(p)
        print(f"{run:32s} test avg MAE {avg_mae(members[run]):.2f}")
    test = score(torch.stack(preds).mean(0), data, windows)
    print(f"{'ensemble (' + str(len(preds)) + ' runs)':32s} test avg MAE {avg_mae(test):.2f}")
    out = {"model": f"Ensemble of {len(preds)}", "runs": a.runs, "test_2025_2026": test, "members": members}
    (train.LOGS / f"{a.name}_metrics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(test, indent=2))


if __name__ == "__main__":
    main()
