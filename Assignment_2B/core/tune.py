"""Grid search over learning rate and hidden units for LSTM, GRU, and GNN.
Usage: python -m core.tune --sites 10 --days 180 --epochs 10"""
import argparse, itertools, json, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data import DataLoader, TensorDataset
from core.travel_time import haversine_km

VCOLS = [f"V{i:02d}" for i in range(96)]
LOOKBACK = 12                                   # 12 x 15 min = 3 h history
LRS = [1e-3, 5e-3, 1e-2]
HIDDEN = [32, 64, 128]
LOG_DIR = Path("outputs/logs")

class RNNNet(nn.Module):                        # swap for models/ classes if available
    def __init__(self, kind, hidden):
        super().__init__()
        self.rnn = (nn.LSTM if kind == "lstm" else nn.GRU)(1, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1])

class GCNNet(nn.Module):
    """2-layer graph conv over sites. Input (B, N, T) -> predicts next step for all N sites (B, N)."""
    def __init__(self, A_hat, hidden):
        super().__init__()
        self.register_buffer("A", torch.tensor(A_hat, dtype=torch.float32))
        self.l1 = nn.Linear(LOOKBACK, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)
    def forward(self, x):
        h = torch.relu(self.A @ self.l1(x))
        h = torch.relu(self.A @ self.l2(h))
        return self.out(h).squeeze(-1)

def load_graph_data(path, n_sites, n_days):
    df = pd.read_csv(path)
    sites = sorted(df["SCATS Number"].unique())[:n_sites]
    df = df[df["SCATS Number"].isin(sites)]
    # only days present at every site, so the site series stay time-aligned
    dates = sorted(set.intersection(*[set(df[df["SCATS Number"] == s].Date) for s in sites]))[-n_days:]
    cols = []
    for s in sites:
        d = df[(df["SCATS Number"] == s) & df.Date.isin(dates)].sort_values("Date")
        v = pd.Series(d[VCOLS].to_numpy(float).ravel()).interpolate(limit_direction="both").to_numpy()
        cols.append(v)
    M = np.stack(cols, axis=1)                                    # (time, N)

    # adjacency: Gaussian kernel on haversine distance, symmetric normalisation
    xy = df.groupby("SCATS Number")[["NB_LATITUDE", "NB_LONGITUDE"]].mean().loc[sites].to_numpy()
    n = len(sites)
    D = np.array([[haversine_km(*xy[i], *xy[j]) for j in range(n)] for i in range(n)])
    A = np.exp(-(D / D[D > 0].mean()) ** 2)
    A[A < 0.1] = 0
    dinv = 1 / np.sqrt(A.sum(1))
    A_hat = A * dinv[:, None] * dinv[None, :]

    w = sliding_window_view(M, LOOKBACK + 1, axis=0)             # (B, N, T+1)
    cut = int(len(w) * 0.8)
    tr, va = w[:cut], w[cut:]
    scale = tr.max()
    f = lambda a: (torch.tensor(a[:, :, :-1] / scale, dtype=torch.float32),
                   torch.tensor(a[:, :, -1] / scale, dtype=torch.float32))
    return (*f(tr), *f(va), scale), A_hat

def load_data(path, n_sites, n_days):
    df = pd.read_csv(path)
    sites = sorted(df["SCATS Number"].unique())[:n_sites]
    tr, va = [], []
    for s in sites:
        d = df[df["SCATS Number"] == s].sort_values("Date").tail(n_days)
        v = pd.Series(d[VCOLS].to_numpy(float).ravel()).interpolate(limit_direction="both").to_numpy()
        w = sliding_window_view(v, LOOKBACK + 1)
        cut = int(len(w) * 0.8)                 # chronological split per site
        tr.append(w[:cut]); va.append(w[cut:])
    tr, va = np.concatenate(tr), np.concatenate(va)
    scale = tr.max()
    f = lambda a: (torch.tensor(a[:, :-1] / scale, dtype=torch.float32).unsqueeze(-1),
                   torch.tensor(a[:, -1:] / scale, dtype=torch.float32))
    return *f(tr), *f(va), scale

def run(model, lr, data, epochs, dev):
    Xtr, ytr, Xva, yva, scale = data
    model = model.to(dev)
    opt, loss_fn = torch.optim.Adam(model.parameters(), lr=lr), nn.MSELoss()
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=256, shuffle=True)
    hist = []
    for ep in range(epochs):
        model.train(); total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad(); loss = loss_fn(model(xb), yb); loss.backward(); opt.step()
            total += loss.item() * len(xb)
        model.eval()
        with torch.no_grad():
            pred = model(Xva.to(dev)).cpu()
        hist.append({"epoch": ep + 1, "train_loss": total / len(Xtr),
                     "val_rmse": ((pred - yva) ** 2).mean().sqrt().item() * scale})
    return hist

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/site_totals_15min_2014_2026.csv.gz")
    ap.add_argument("--sites", type=int, default=10)
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--epochs", type=int, default=10)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    LOG_DIR.mkdir(exist_ok=True)
    rnn_data = load_data(a.data, a.sites, a.days)
    gnn_data, A_hat = load_graph_data(a.data, a.sites, a.days)
    rows, curves = [], {}
    for kind, lr, h in itertools.product(["lstm", "gru", "gnn"], LRS, HIDDEN):
        t0 = time.time()
        if kind == "gnn":
            model, data = GCNNet(A_hat, h), gnn_data
        else:
            model, data = RNNNet(kind, h), rnn_data
        hist = run(model, lr, data, a.epochs, dev)
        key = f"{kind}_lr{lr}_h{h}"
        curves[key] = hist
        rows.append({"model": kind, "lr": lr, "hidden": h, "final_val_rmse": hist[-1]["val_rmse"],
                     "best_val_rmse": min(e["val_rmse"] for e in hist), "seconds": round(time.time() - t0, 1)})
        print(rows[-1], flush=True)
    res = pd.DataFrame(rows).sort_values("best_val_rmse")
    res.to_csv(LOG_DIR / "tuning_results.csv", index=False)
    (LOG_DIR / "tuning_curves.json").write_text(json.dumps(curves))
    best = {m: res[res.model == m].iloc[0][["lr", "hidden"]].to_dict() for m in ["lstm", "gru", "gnn"]}
    (LOG_DIR / "best_params.json").write_text(json.dumps(best, indent=2))
    print("best:", best)
