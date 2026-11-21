"""Turn the site-total CSV into model-ready arrays (nothing here is model-specific).

Pipeline: wide rows -> complete 15-min grid -> chronological split -> impute -> scale -> save.
Imputation and scaling statistics come from the TRAIN split only, so val/test never leak.
"""
import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

VCOLS = [f"V{i:02d}" for i in range(96)]
CSV = "data/site_totals_15min_2014_2026.csv.gz"
OUT_DIR = "data/processed"
MAX_GAP = 8  # steps (2 h): shorter gaps are interpolated, longer ones use the typical profile


# ---------- preprocessing ----------
def wide_to_series(df):
    """One row per site/day (V00-V95) -> DataFrame (15-min timestamp x site), NaN where missing."""
    df = df.drop_duplicates(["SCATS Number", "Date"]).copy()
    df["Date"] = pd.to_datetime(df["Date"])
    days = pd.date_range(df.Date.min(), df.Date.max(), freq="D")
    idx = pd.date_range(days[0], periods=len(days) * 96, freq="15min")
    cols = {site: g.set_index("Date")[VCOLS].reindex(days).to_numpy(float).ravel()
            for site, g in df.groupby("SCATS Number")}
    return pd.DataFrame(cols, index=idx)


def slot_key(idx):
    """(weekday, 15-min slot of the day): the key for 'typical traffic at this time'."""
    return [idx.dayofweek, idx.hour * 4 + idx.minute // 15]


def impute(part, profile, col_median):
    """Short gaps: linear interpolation. Long gaps / missing days: train median for that
    weekday and time slot. Last resort: the site's overall train median."""
    part = part.interpolate(limit=MAX_GAP, limit_area="inside")
    fill = profile.reindex(pd.MultiIndex.from_arrays(slot_key(part.index)))
    fill.index = part.index
    return part.fillna(fill).fillna(col_median).fillna(0.0)


def build_adjacency(lats, lons):
    """Normalised Gaussian-kernel adjacency from haversine distances (used by the GNN)."""
    lat, lon = np.radians(lats)[:, None], np.radians(lons)[:, None]
    a = (np.sin((lat.T - lat) / 2) ** 2
         + np.cos(lat) * np.cos(lat.T) * np.sin((lon.T - lon) / 2) ** 2)
    D = 6371.0 * 2 * np.arcsin(np.sqrt(a))
    A = np.exp(-(D / D[D > 0].mean()) ** 2)
    A[A < 0.1] = 0
    d = 1 / np.sqrt(A.sum(1))
    return A * d[:, None] * d[None, :]


def prepare_dataset(csv_path=CSV, out_dir=OUT_DIR, tag="research", lookback=12,
                    max_days=None, train_frac=0.7, val_frac=0.15):
    """Build and save the shared dataset. max_days keeps only the latest N days (quick tests)."""
    raw = pd.read_csv(csv_path, low_memory=False)
    series = wide_to_series(raw)
    if max_days:
        series = series.iloc[-max_days * 96:]
    sites = list(series.columns)
    coords = raw.groupby("SCATS Number")[["NB_LATITUDE", "NB_LONGITUDE"]].mean().loc[sites]

    a, b = int(len(series) * train_frac), int(len(series) * (train_frac + val_frac))
    parts = series.iloc[:a], series.iloc[a:b], series.iloc[b:]          # chronological, no shuffle
    profile = parts[0].groupby(slot_key(parts[0].index)).median()
    col_median = parts[0].median()
    train, val, test = (impute(p, profile, col_median).to_numpy(np.float32) for p in parts)

    data_min = train.min(0)                                             # per-site min-max, train only
    data_range = train.max(0) - data_min
    data_range[data_range == 0] = 1.0
    train, val, test = ((x - data_min) / data_range for x in (train, val, test))
    # val/test start with the last `lookback` rows of the previous split, so no windows are lost
    val_full = np.concatenate([train[-lookback:], val])
    test_full = np.concatenate([val[-lookback:], test])

    path = Path(out_dir) / f"scats_{tag}_lb{lookback}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, train=train.astype(np.float32), val=val_full.astype(np.float32),
        test=test_full.astype(np.float32), data_min=data_min, data_range=data_range,
        node_order=np.asarray(sites), lookback=lookback,
        adjacency=build_adjacency(coords.NB_LATITUDE.to_numpy(), coords.NB_LONGITUDE.to_numpy()))
    print(f"saved {path} | {len(sites)} sites | imputed {series.isna().mean().mean():.1%} of values")
    return path


# ---------- loading ----------
class Windows:
    """Sliding windows over a (time, sites) array, built lazily (no copy of the whole dataset).
    Each sample: x = last `lookback` values of every site (N, L), y = next value (N,)."""

    def __init__(self, series, lookback):
        self.w = torch.as_tensor(series).unfold(0, lookback + 1, 1)   # (S, N, L+1) view

    def __len__(self):
        return self.w.shape[0]

    def batches(self, batch, shuffle=False, generator=None):
        idx = torch.randperm(len(self), generator=generator) if shuffle else torch.arange(len(self))
        for i in range(0, len(self), batch):
            w = self.w[idx[i:i + batch]]
            yield w[:, :, :-1], w[:, :, -1]


@dataclass
class Dataset:
    train: Windows
    val: Windows
    test: Windows
    data_min: np.ndarray       # per-site scaling constants, for converting back to vehicles
    data_range: np.ndarray
    node_order: np.ndarray     # SCATS numbers, in the column order of every array
    adjacency: np.ndarray      # (N, N), for the GNN
    lookback: int

    @property
    def n_sites(self):
        return len(self.node_order)

    def to_vehicles(self, scaled):
        """(..., N) scaled values -> vehicles per 15 min."""
        return scaled * self.data_range + self.data_min


def load_dataset(path):
    z = np.load(path)
    lb = int(z["lookback"])
    return Dataset(Windows(z["train"], lb), Windows(z["val"], lb), Windows(z["test"], lb),
                   z["data_min"], z["data_range"], z["node_order"], z["adjacency"], lb)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build the shared model-ready dataset.")
    ap.add_argument("--csv", default=CSV)
    ap.add_argument("--tag", default="research")
    ap.add_argument("--lookback", type=int, default=12)
    ap.add_argument("--max-days", type=int, default=None)
    a = ap.parse_args()
    prepare_dataset(a.csv, OUT_DIR, a.tag, a.lookback, a.max_days)
