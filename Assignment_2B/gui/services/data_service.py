"""Read-only dataset catalog and historical replay boundaries (Melbourne wall time)."""
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

GUI = Path(__file__).resolve().parents[1]
ROOT = GUI.parent
ASSIGNMENT, RESEARCH = "assignment_2006", "research_2014_2026"
DATASETS = {ASSIGNMENT: "site_totals_15min_oct2006.csv.gz",
            RESEARCH: "site_totals_15min_2014_2026.csv.gz"}
SLOTS = [f"V{i:02}" for i in range(96)]
STEP = pd.Timedelta(minutes=15)
HORIZONS = (15, 30, 45, 60)


class Unavailable(ValueError):
    """Required artifacts or observations are unavailable for this selection."""


def config():
    cfg = json.loads((GUI / "config.json").read_text())
    cfg.update(json.loads((GUI / "replay.json").read_text()))
    valid_times = {f"{h:02}:{m:02}" for h in range(24) for m in range(0, 60, 15)}
    if (cfg["dataset"] not in DATASETS or cfg["horizon"] not in HORIZONS
            or cfg["time"] not in valid_times or not set(cfg["presets"].values()) <= valid_times
            or not 1 <= cfg["k_routes"] <= 5):
        raise ValueError("Invalid replay defaults in GUI configuration.")
    return cfg


def artifact(paths):
    for name in paths:
        path = (GUI / name).resolve()
        if path.is_file():
            return path
    raise Unavailable("Missing " + " or ".join(paths))


def validate_days(rows):
    ids = rows["SCATS Number"]
    dates = pd.DatetimeIndex(rows.Date)
    if (rows.empty or ids.isna().any() or not pd.api.types.is_integer_dtype(ids)
            or (ids <= 0).any() or dates.hasnans or not dates.equals(dates.normalize())
            or rows.duplicated(["SCATS Number", "Date"]).any()):
        raise ValueError("Invalid site ID, date or duplicate site/day in dataset.")


def check_schema(columns):
    if [c for c in columns if c.startswith("V")] != SLOTS:
        raise ValueError("Expected exactly V00 through V95 in chronological order.")


@lru_cache(maxsize=2)
def catalog(dataset):
    path = ROOT / "data" / DATASETS[dataset]
    check_schema(pd.read_csv(path, nrows=0).columns)
    rows = pd.read_csv(path, usecols=["SCATS Number", "Date"], parse_dates=["Date"])
    validate_days(rows)
    ids = sorted(rows["SCATS Number"].unique())
    sites = pd.read_json(ROOT / "core/sites_aligned.json").set_index("site").loc[ids].reset_index()
    sites = sites.rename(columns={"site": "site_id"}).assign(name="")
    # The research notes identify a bad coordinate for 4266. Correct only the UI lookup by ID.
    flagged = sites.out_of_bounds.astype(bool)
    if flagged.any():
        official = pd.read_csv(ROOT / "research/external/site_locations.csv").set_index("site")
        sites.loc[flagged, ["lat", "lon"]] = official.loc[sites.loc[flagged, "site_id"], ["lat", "lon"]].to_numpy()
    if sites.site_id.duplicated().any() or not np.isfinite(sites[["lat", "lon"]]).all().all():
        raise ValueError("Invalid aligned site coordinates.")
    return sites, sorted(rows.Date.dt.date.unique())


def wide_counts(rows):
    check_schema(rows.columns)
    validate_days(rows)
    days = pd.date_range(rows.Date.min(), rows.Date.max())
    ids = sorted(rows["SCATS Number"].unique())
    index = pd.MultiIndex.from_product([days, ids], names=["Date", "SCATS Number"])
    counts = rows.set_index(["Date", "SCATS Number"])[SLOTS].reindex(index).to_numpy(float)
    if np.isinf(counts).any() or (counts < 0).any():
        raise ValueError("Invalid observed vehicle count.")
    values = counts.reshape(len(days), len(ids), 96).transpose(0, 2, 1).reshape(-1, len(ids))
    return pd.DataFrame(values, index=pd.date_range(days[0], periods=len(values), freq=STEP), columns=ids)


@dataclass
class ReplayData:
    times: pd.DatetimeIndex
    ids: tuple
    flow: np.ndarray
    mask: np.ndarray
    adj: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    profile: np.ndarray
    external: dict

    def position(self, origin, history, steps=4):
        ts = pd.Timestamp(origin)
        i = self.times.get_indexer([ts])[0]
        if ts.tzinfo is not None or i < history or i + steps > len(self.times):
            raise ValueError("Time must be aligned, in coverage, with sufficient input and target intervals.")
        if ts < pd.Timestamp("2025-01-01"):
            raise ValueError("Choose a date in the held-out research period.")
        if not self.mask[i-history:i].any():
            raise Unavailable("No observed traffic in the input history.")
        return i

    def causal(self, start, stop):
        idx = self.times[start:stop]
        slots = idx.day_of_week * 96 + idx.hour * 4 + idx.minute // 15
        return np.where(self.mask[start:stop], self.flow[start:stop], self.profile[slots])


@lru_cache(maxsize=1)
def load_cache(path, stamp, ids):
    with np.load(path, allow_pickle=False) as cache:
        times = pd.DatetimeIndex(cache["times"])
        sites = tuple(cache["sites"].tolist())
        flow, mask, adj = cache["flow"], cache["mask"], cache["adj"]
        ext = {k: cache[k] for k in cache.files if k.startswith("ext_")}
    if (sites != tuple(sorted(set(ids))) or sites != ids or len(times) < 2
            or times[0] != times[0].floor("15min") or not ((times[1:] - times[:-1]) == STEP).all()
            or flow.shape != (len(times), len(ids)) or mask.shape != flow.shape or mask.dtype != bool
            or adj.shape != (len(ids), len(ids)) or not np.isfinite(adj).all()
            or not np.isfinite(flow).all() or (flow < 0).any()
            or any(v.shape != (len(times),) for v in ext.values())):
        raise ValueError("Incompatible traffic cache schema, site order, mask or timestamps.")
    train = times < "2024-01-01"
    if not train.any():
        raise ValueError("Cache lacks training-period scaling data.")
    training = flow[train]
    mean, std = training.mean(0), np.maximum(training.std(0) + 1e-6, 1.0)
    idx = times[train]
    slots = idx.day_of_week * 96 + idx.hour * 4 + idx.minute // 15
    observed = pd.DataFrame(np.where(mask[train], training, np.nan))
    profile = observed.groupby(slots).mean().reindex(range(672)).fillna(0).to_numpy(np.float32)
    return ReplayData(times, sites, flow, mask, adj, mean, std, profile, ext)
