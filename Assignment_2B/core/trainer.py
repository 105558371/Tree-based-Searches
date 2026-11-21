"""Shared training and evaluation loop. The same code trains every model, so comparisons are fair."""
import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_input(x, layout):
    """x: (batch, n_sites, lookback) -> the shape a model expects."""
    if layout == "graph":                      # GNN: sites are nodes
        return x
    if layout == "multi":                      # multivariate RNN: sites are features
        return x.transpose(1, 2)               # (batch, lookback, n_sites)
    return x.reshape(-1, x.shape[-1], 1)       # per-site RNN: every site is its own sequence


def forward(model, x, layout):
    """Always returns (batch, n_sites), whatever the layout."""
    return model(to_input(x, layout)).reshape(len(x), -1)


@torch.no_grad()
def evaluate(model, ds, split, layout, device=None, batch=256):
    """RMSE / MAE in vehicles per 15 min on 'train', 'val' or 'test'."""
    device = device or get_device()
    model.eval()
    preds, targets = [], []
    for x, y in getattr(ds, split).batches(batch):
        preds.append(forward(model, x.to(device), layout).cpu().numpy())
        targets.append(y.numpy())
    pred = np.maximum(ds.to_vehicles(np.concatenate(preds)), 0)   # flow can't be negative
    true = ds.to_vehicles(np.concatenate(targets))
    err = pred - true
    return {"rmse": float(np.sqrt((err ** 2).mean())), "mae": float(np.abs(err).mean())}


def fit(model, ds, layout, lr, epochs, patience=10, batch=64, seed=42, device=None, verbose=True):
    """Train with Adam + MSE, early-stop on val RMSE, restore the best weights.
    batch counts time windows (each holds all sites). Returns the per-epoch history."""
    device = device or get_device()
    set_seed(seed)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    gen = torch.Generator().manual_seed(seed)
    history, best_rmse, best_state, bad = [], float("inf"), None, 0

    for epoch in range(1, epochs + 1):
        model.train()
        total, count = 0.0, 0
        for x, y in ds.train.batches(batch, shuffle=True, generator=gen):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(forward(model, x, layout), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(x)
            count += len(x)
        val = evaluate(model, ds, "val", layout, device)
        history.append({"epoch": epoch, "train_loss": total / count,   # scaled units
                        "val_rmse": val["rmse"], "val_mae": val["mae"]})  # vehicles / 15 min
        if verbose:
            print(f"  epoch {epoch:3d} | train loss {total / count:.5f} | val RMSE {val['rmse']:.2f}")
        if val["rmse"] < best_rmse - 1e-6:
            best_rmse, bad = val["rmse"], 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return history


def save_run(name, model, history, metrics=None, tag="research", root="outputs"):
    """Weights -> outputs/weights/<tag>/, history + metrics -> outputs/logs/<tag>/."""
    wdir, ldir = Path(root) / "weights" / tag, Path(root) / "logs" / tag
    wdir.mkdir(parents=True, exist_ok=True)
    ldir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), wdir / f"{name}.pt")
    pd.DataFrame(history).to_csv(ldir / f"{name}_history.csv", index=False)
    if metrics:
        (ldir / f"{name}_metrics.json").write_text(json.dumps(metrics, indent=2))
