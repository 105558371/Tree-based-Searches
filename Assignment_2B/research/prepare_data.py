"""Turn the data/ site totals into model-ready arrays, shared by every model
(LSTM, GRU, Graph WaveNet) so they all train and test on exactly the same data.

Output (research/cache/traffic.npz):
    flow       [T, N] vehicles per 15 min, gaps filled (model input)
    mask       [T, N] True where the count was really observed (used for loss/metrics)
    times      [T]    timestamps, 15-minute steps
    sites, lat, lon   [N] SCATS site numbers and coordinates
    adj        [N, N] distance-based adjacency (Gaussian kernel), for graph models

Gap filling: gaps up to 1 hour are linearly interpolated; longer gaps use that
site's average for the same weekday and time of day (from the training years only).
"""
import pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
SRC = ROOT.parent / "data" / "site_totals_15min_2014_2026.csv.gz"
OUT = ROOT / "cache" / "traffic.npz"

TRAIN_END = "2023-12-31 23:45"   # train 2014-2023, validate 2024, test 2025-2026
VCOLS = [f"V{i:02d}" for i in range(96)]


def load_wide():
    df = pd.read_csv(SRC, parse_dates=["Date"])
    long = df.melt(id_vars=["SCATS Number", "Date"], value_vars=VCOLS,
                   var_name="slot", value_name="flow")
    long["time"] = long.Date + pd.to_timedelta(long.slot.str[1:].astype(int) * 15, unit="min")
    wide = long.pivot_table(index="time", columns="SCATS Number", values="flow", aggfunc="first")
    full = pd.date_range(wide.index.min(), wide.index.max(), freq="15min")
    coords = df.groupby("SCATS Number")[["NB_LATITUDE", "NB_LONGITUDE"]].first()
    return wide.reindex(full), coords.loc[wide.columns]


def fill_gaps(wide):
    filled = wide.interpolate(limit=4, limit_area="inside")
    # weekly profile (weekday x time of day) per site, from training years only
    train = wide.loc[:TRAIN_END]
    key = lambda idx: idx.dayofweek * 96 + idx.hour * 4 + idx.minute // 15
    profile = train.groupby(key(train.index)).mean()
    fallback = profile.reindex(key(filled.index)).set_index(filled.index)
    return filled.fillna(fallback).fillna(0)


def gaussian_adjacency(lat, lon, threshold=0.1):
    lat, lon = np.radians(lat), np.radians(lon)
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2
    dist_km = 2 * 6371 * np.arcsin(np.sqrt(a))
    sigma = dist_km[np.triu_indices_from(dist_km, 1)].std()
    adj = np.exp(-(dist_km / sigma) ** 2)
    adj[adj < threshold] = 0
    return adj.astype(np.float32)


def main():
    wide, coords = load_wide()
    mask = wide.notna().to_numpy()
    flow = fill_gaps(wide).to_numpy(np.float32)
    lat, lon = coords.NB_LATITUDE.to_numpy(), coords.NB_LONGITUDE.to_numpy()
    OUT.parent.mkdir(exist_ok=True)
    np.savez_compressed(OUT, flow=flow, mask=mask, times=wide.index.to_numpy(),
                        sites=wide.columns.to_numpy(), lat=lat, lon=lon,
                        adj=gaussian_adjacency(lat, lon))
    print(f"{OUT}: {flow.shape[0]} steps x {flow.shape[1]} sites, "
          f"{100 * (1 - mask.mean()):.2f}% filled")


if __name__ == "__main__":
    main()
