"""Train and evaluate one spatio-temporal GNN on the shared traffic dataset.

Task: from the last 12 steps (3 h) of flow at all 40 sites, predict the next
4 steps (15, 30, 45, 60 min). Train 2014-2023, validate 2024, test 2025-2026.
Loss and metrics only count values that were really observed (mask).

Usage: python train.py --model {gwn,stgcn,agcrn,mtgnn,stid,staeformer} [--epochs 30] [--samples-per-epoch 60000]
Writes outputs/weights/research/<model>.pt and outputs/logs/research/<model>_metrics.json.
"""
import argparse, json, pathlib, time
import numpy as np
import torch
from graph_wavenet import GraphWaveNet, transition
from gnn_models import AGCRN, MTGNN, STGCN
from more_models import STAEformer, STID

ROOT = pathlib.Path(__file__).resolve().parent
WEIGHTS = ROOT.parent / "outputs" / "weights" / "research"
LOGS = ROOT.parent / "outputs" / "logs" / "research"
IN_STEPS, HORIZON = 12, 4
VAL_START, TEST_START = np.datetime64("2024-01-01"), np.datetime64("2025-01-01")


def load(device):
    d = np.load(ROOT / "cache" / "traffic.npz", allow_pickle=True)
    flow, mask, times = d["flow"], d["mask"], d["times"].astype("datetime64[m]")
    train_rows = times < VAL_START
    mean, std = flow[train_rows].mean(0), flow[train_rows].std(0) + 1e-6   # per site
    tod = ((times - times.astype("datetime64[D]")).astype(int) / 1440.0).astype(np.float32)

    # window t uses inputs [t-12, t) and targets [t, t+4); split by first target time
    starts = np.arange(IN_STEPS, len(flow) - HORIZON + 1)
    split = {"train": starts[times[starts] < VAL_START],
             "val": starts[(times[starts] >= VAL_START) & (times[starts] < TEST_START)],
             "test": starts[times[starts] >= TEST_START]}
    t = lambda a: torch.as_tensor(a, device=device)
    data = dict(norm=t(((flow - mean) / std).astype(np.float32)), flow=t(flow), mask=t(mask),
                tod=t(tod), mean=t(mean.astype(np.float32)), std=t(std.astype(np.float32)))
    return data, split, torch.as_tensor(d["adj"])


MODELS = {
    "gwn": ("Graph WaveNet", lambda n, adj: GraphWaveNet(n, [transition(adj), transition(adj.T)], horizon=HORIZON)),
    "stgcn": ("STGCN", lambda n, adj: STGCN(n, adj, horizon=HORIZON, in_steps=IN_STEPS)),
    "agcrn": ("AGCRN", lambda n, adj: AGCRN(n, horizon=HORIZON)),
    "mtgnn": ("MTGNN", lambda n, adj: MTGNN(n, horizon=HORIZON, in_steps=IN_STEPS)),
    "stid": ("STID", lambda n, adj: STID(n, horizon=HORIZON, in_steps=IN_STEPS)),
    "staeformer": ("STAEformer", lambda n, adj: STAEformer(n, horizon=HORIZON, in_steps=IN_STEPS)),
}


def batch(data, idx):
    """Build input [B, 2, N, 12] (normalised flow + time of day) and targets [B, 4, N]."""
    i_in = idx[:, None] + torch.arange(-IN_STEPS, 0, device=idx.device)
    i_out = idx[:, None] + torch.arange(HORIZON, device=idx.device)
    x = data["norm"][i_in]                                    # [B, 12, N]
    tod = data["tod"][i_in][..., None].expand_as(x)
    x = torch.stack([x, tod], dim=1).transpose(2, 3)          # [B, 2, N, 12]
    return x, data["flow"][i_out], data["mask"][i_out]


def predict(model, data, idx):
    return model(batch(data, idx)[0]) * data["std"] + data["mean"]


def masked_mae(pred, y, m):
    m = m.float()
    return (torch.abs(pred - y) * m).sum() / m.sum().clamp(min=1)


@torch.no_grad()
def evaluate(model, data, windows, device, bs=128):
    """MAE, RMSE and MAPE per horizon, on observed values only (MAPE skips flow < 10)."""
    model.eval()
    sums = torch.zeros(4, HORIZON, device=device)   # abs, sq, pct, count(for mape)
    count = torch.zeros(HORIZON, device=device)
    for k in range(0, len(windows), bs):
        idx = torch.as_tensor(windows[k:k + bs], device=device)
        _, y, m = batch(data, idx)
        p = predict(model, data, idx).clamp(min=0)
        m = m.float()
        e = (p - y) * m
        pm = m * (y >= 10)
        sums[0] += e.abs().sum((0, 2)); sums[1] += (e ** 2).sum((0, 2))
        sums[2] += (e.abs() / y.clamp(min=1) * pm).sum((0, 2)); sums[3] += pm.sum((0, 2))
        count += m.sum((0, 2))
    mae, rmse, mape = sums[0] / count, (sums[1] / count).sqrt(), 100 * sums[2] / sums[3]
    return {f"{15 * (h + 1)}min": {"MAE": round(mae[h].item(), 3), "RMSE": round(rmse[h].item(), 3),
                                   "MAPE": round(mape[h].item(), 2)} for h in range(HORIZON)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, default="gwn")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--samples-per-epoch", type=int, default=60000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=6)
    args = ap.parse_args()

    torch.manual_seed(0); np.random.seed(0)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    data, split, adj = load(device)
    name, build = MODELS[args.model]
    model = build(data["flow"].shape[1], adj).to(device)
    ckpt = WEIGHTS / f"{args.model}.pt"
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    val_sub = split["val"][::8]     # every 2 h of 2024, for fast validation
    print(f"{name} | device {device} | params {sum(p.numel() for p in model.parameters()):,} | "
          f"windows train {len(split['train']):,} val {len(split['val']):,} test {len(split['test']):,}")

    WEIGHTS.mkdir(parents=True, exist_ok=True); LOGS.mkdir(parents=True, exist_ok=True)
    best, bad, history = float("inf"), 0, []
    for epoch in range(1, args.epochs + 1):
        model.train(); t0 = time.time(); losses = []
        # each epoch trains on a fresh random sample of training windows
        picks = np.random.choice(split["train"], args.samples_per_epoch, replace=False)
        for k in range(0, len(picks), args.batch_size):
            idx = torch.as_tensor(picks[k:k + args.batch_size], device=device)
            _, y, m = batch(data, idx)
            loss = masked_mae(predict(model, data, idx), y, m)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); losses.append(loss.item())
        val = evaluate(model, data, val_sub, device)
        val_mae = np.mean([v["MAE"] for v in val.values()])
        history.append({"epoch": epoch, "train_mae": float(np.mean(losses)), "val_mae": float(val_mae)})
        print(f"epoch {epoch:2d} | train MAE {np.mean(losses):7.2f} | val MAE {val_mae:7.2f} | "
              f"{time.time() - t0:5.0f}s", flush=True)
        if val_mae < best:
            best, bad = val_mae, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= args.patience:
                print("early stop"); break

    model.load_state_dict(torch.load(ckpt, map_location=device))
    test = evaluate(model, data, split["test"], device)
    out = {"model": name, "test_2025_2026": test, "best_val_mae": best,
           "history": history, "args": vars(args)}
    (LOGS / f"{args.model}_metrics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(test, indent=2))


if __name__ == "__main__":
    main()
