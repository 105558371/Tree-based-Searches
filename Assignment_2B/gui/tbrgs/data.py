"""Load the Boroondara SCATS flow data and build training windows.
Expected layout: one row per site/approach/day with columns 'SCATS Number', 'Location', latitude and
longitude columns, 'Date' and V00..V95 (vehicles per 15-minute interval)."""
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

INTERVALS = 96  # 24 h x 4


@dataclass
class Dataset:
    sites: pd.DataFrame      # site_id, name, lat, lon
    dates: list              # sorted list of datetime.date
    series: dict             # site_id -> cleaned 1-D array, len(dates) * 96, starting at 00:00 of dates[0]


def clean_series(x):
    """Negative values become missing, missing values are interpolated (edges filled), leftovers are 0."""
    s = pd.Series(np.asarray(x, dtype=float))
    s[s < 0] = np.nan
    return s.interpolate(limit_direction="both").fillna(0).to_numpy()


def _read_raw(path):
    if str(path).lower().endswith(".csv"):
        return pd.read_csv(path, header=None, dtype=object)
    xl = pd.ExcelFile(path)
    return xl.parse("Data" if "Data" in xl.sheet_names else xl.sheet_names[0], header=None, dtype=object)


def load_scats(path) -> Dataset:
    raw = _read_raw(path)
    hdr = next((i for i in range(min(len(raw), 30))
                if "scats number" in [str(v).strip().lower() for v in raw.iloc[i]]), None)
    if hdr is None:
        raise ValueError("Header row containing 'SCATS Number' not found in the dataset.")
    df = raw.iloc[hdr + 1:].copy()
    df.columns = [str(c).strip() for c in raw.iloc[hdr]]
    col = lambda key: next(c for c in df.columns if key in c.lower())
    vcols = [c for c in df.columns if re.fullmatch(r"V\d{2}", c)]
    if len(vcols) != INTERVALS:
        raise ValueError(f"Expected {INTERVALS} columns V00..V95, found {len(vcols)}.")
    df["site"] = pd.to_numeric(df[col("scats number")], errors="coerce")
    df["day"] = pd.to_datetime(df[col("date")], dayfirst=True, errors="coerce").dt.date
    df = df.dropna(subset=["site", "day"])
    df[vcols] = df[vcols].apply(pd.to_numeric, errors="coerce")
    # Average the approaches of a site. Summing them gives flows far above the 1500 veh/h of the conversion
    # (43% of site-hours on the Boroondara data), while the average stays in its range (0.1%).
    daily = df.groupby(["site", "day"])[vcols].mean()
    dates = sorted(df["day"].unique())
    lat, lon, loc = col("latitude"), col("longitude"), col("location")
    num = lambda s: pd.to_numeric(s, errors="coerce")
    info = df.groupby("site").agg(name=(loc, "first"), lat=(lat, lambda s: num(s).replace(0, np.nan).median()),
                                  lon=(lon, lambda s: num(s).replace(0, np.nan).median())).reset_index()
    sites = info.rename(columns={"site": "site_id"}).dropna(subset=["lat", "lon"])
    sites["site_id"] = sites["site_id"].astype(int)
    series = {}
    for sid in sites.site_id:
        block = daily.loc[sid].reindex(dates)                          # missing days become NaN rows
        series[int(sid)] = clean_series(block.to_numpy(dtype=float).reshape(-1))
    return Dataset(sites.reset_index(drop=True), dates, series)


def features(flow, offset=0):
    """(T,) flows starting at interval index `offset` -> (T, 3): flow, sin(time of day), cos(time of day)."""
    tod = (np.arange(len(flow)) + offset) % INTERVALS / INTERVALS * 2 * np.pi
    return np.stack([flow, np.sin(tod), np.cos(tod)], axis=1)


def make_windows(series, window, start, end):
    """Samples whose target index lies in [start, end). X: (N, window, 3) unscaled, y: (N,)."""
    xs, ys = [], []
    for s in series.values():
        idx = np.arange(max(start, window), min(end, len(s)))
        if len(idx) == 0:
            continue
        win = np.lib.stride_tricks.sliding_window_view(features(s), (window, 3))[:, 0]
        xs.append(win[idx - window]); ys.append(s[idx])
    return np.concatenate(xs), np.concatenate(ys)


def split_bounds(n_days, train_frac, val_frac):
    a, b = int(n_days * train_frac) * INTERVALS, int(n_days * (train_frac + val_frac)) * INTERVALS
    return a, b, n_days * INTERVALS
