"""Single-window CPU replay. No training module state, model registry mutations or writes."""
import json
import pickle
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from .data_service import (RESEARCH, STEP, HORIZONS, ROOT, Unavailable, artifact, catalog,
                           config, load_cache)
from .metrics_service import BASE, ENSEMBLE, LOGS, MEMBERS, RECIPE, WEIGHTS


def model_choices(dataset, advanced=False):
    if dataset == RESEARCH:
        return (*BASE, ENSEMBLE, *MEMBERS) if advanced else BASE[:3]
    return ("lstm", "gru", "cnn")


def horizons(dataset, run, cfg):
    return () if availability(dataset, run, cfg) else HORIZONS


def manifest(run):
    if run not in (*BASE, *MEMBERS, ENSEMBLE):
        raise Unavailable("No verified inference recipe for this model.")
    path = LOGS / f"{run}_metrics.json"
    if not path.is_file():
        raise Unavailable(f"Missing run metadata: {path.name}")
    return json.loads(path.read_text())


def members(run):
    if run != ENSEMBLE:
        return (run,)
    names = tuple(manifest(run).get("runs", []))
    if names != MEMBERS:
        raise Unavailable("Ensemble requires the recorded final s0, s1 and s2 members.")
    return names


def availability(dataset, run, cfg):
    if dataset != RESEARCH:
        return "Assignment replay requires verified October 2006 checkpoints, preprocessing and evaluation evidence."
    try:
        artifact(cfg["traffic"])
        for member in members(run):
            doc = manifest(member)
            args = doc.get("args", {})
            kind = "staeformer" if member in MEMBERS else member
            features = RECIPE if member in MEMBERS else "base"
            if (args.get("model", kind) != kind or args.get("features", "base") != features
                    or doc.get("features", "base") != features or args.get("clean") or args.get("fold")
                    or type(args.get("in_steps", 12)) is not int or args.get("in_steps", 12) < 1
                    or args.get("tsf_cols") or args.get("hp") or args.get("train_end", "2024-01-01") != "2024-01-01"):
                raise Unavailable(f"Unsupported checkpoint recipe: {member}")
            if not (WEIGHTS / f"{member}.pt").is_file():
                raise Unavailable(f"Missing checkpoint: {member}.pt")
            if member in MEMBERS and not (ROOT / "research/external/mcg_events.csv").is_file():
                raise Unavailable("Missing local mcg_events.csv.")
    except (OSError, ValueError) as exc:
        return str(exc)
    return ""


@lru_cache(maxsize=1)
def external_channels(cache_path, stamp, ids):
    from Assignment_2B.research import more_features
    data = load_cache(cache_path, stamp, ids)
    ext = data.external
    train = data.times < "2024-01-01"
    def standardize(values):
        return (values - values[train].mean()) / values[train].std()
    channels = [ext["ext_dow"] / 7, ext["ext_public_holiday"], ext["ext_school_holiday"],
                ext["ext_lockdown"], standardize(np.log1p(ext["ext_rain_mm"])), standardize(ext["ext_temp_c"])]
    return np.column_stack([*channels, more_features.cyclical_features(data.times)]).astype(np.float32)


def inputs(data, i, steps, advanced=False, exo=None):
    # Keep measured traffic strictly before i, including in lagged/rolling channels.
    week = 7 * 96
    start = i - steps - (4 * week if advanced else 0)
    if start < 0 or i + 4 > len(data.times) or steps < 1:
        raise ValueError("Insufficient history or forecast coverage.")
    norm = (data.causal(start, i) - data.mean) / data.std
    idx = data.times[i-steps:i]
    tod = np.asarray((idx.hour * 60 + idx.minute) / 1440, dtype=np.float32)
    shape = (steps, len(data.ids))
    broadcast = lambda a: np.broadcast_to(a[:, None], shape)
    cols = [norm[-steps:], broadcast(tod)]
    if advanced:
        from Assignment_2B.research import more_features, events
        cols.extend(broadcast(c) for c in exo[i-steps:i].T)
        j = np.arange(len(norm) - steps, len(norm))
        cols.extend(norm[j + 4 - lag] for lag in (week, 96))
        cols.append(np.mean([norm[j + 4 - k * week] for k in range(1, 5)], axis=0, dtype=np.float64))
        cols.extend(norm[j] - norm[j-lag] for lag in (1, week))
        day_mean, day_std = more_features.rolling_mean_std(norm, 96)
        week_mean, _ = more_features.rolling_mean_std(norm, week)
        cols.extend([day_mean[j], day_std[j], week_mean[j]])
        event = events.event_features(data.times[i-steps+4:i+4])
        cols.extend(broadcast(c) for c in event.T)
    x = np.stack(cols).transpose(0, 2, 1)[None].astype(np.float32)
    if not np.isfinite(x).all():
        raise Unavailable("Input channels contain missing or nonfinite values.")
    return x


@lru_cache(maxsize=10)
def network(run, weight_stamp, cache_path, cache_stamp, ids, steps, channels):
    import torch
    from Assignment_2B.research import gnn_models, graph_wavenet, more_models, rnn_models
    data = load_cache(cache_path, cache_stamp, ids)
    state = torch.load(WEIGHTS / f"{run}.pt", map_location="cpu", weights_only=True)
    n = len(ids)
    if run in ("lstm", "gru"):
        hidden = state["fc.weight"].shape[1]
        net = rnn_models.RNNForecaster(run, in_dim=channels, hidden=hidden)
    elif run == "gwn":
        adj = torch.as_tensor(data.adj)
        supports = [graph_wavenet.transition(a) for a in (adj, adj.T)]
        if not torch.allclose(torch.stack(supports), state["fixed"], atol=1e-5):
            raise Unavailable("Graph WaveNet site support differs from its checkpoint.")
        net = graph_wavenet.GraphWaveNet(n, supports, in_dim=channels)
    elif run == "stgcn":
        net = gnn_models.STGCN(n, torch.as_tensor(data.adj), in_dim=channels, in_steps=steps)
    elif run == "agcrn":
        net = gnn_models.AGCRN(n, in_dim=channels)
    elif run == "mtgnn":
        net = gnn_models.MTGNN(n, in_dim=channels, in_steps=steps)
    elif run == "stid":
        net = more_models.STID(n, in_steps=steps)
    elif run == "staeformer" or run in MEMBERS:
        net = more_models.STAEformer(n, in_steps=steps, exo_dim=channels-3 if channels > 2 else 0)
    else:
        raise Unavailable("Unsupported architecture.")
    net.load_state_dict(state, strict=True)
    return net.eval()


@dataclass
class Forecast:
    dataset: str
    model: str
    origin: pd.Timestamp
    ids: tuple
    flows: np.ndarray
    actual: np.ndarray
    history: pd.DataFrame

    @property
    def horizons(self):
        return tuple(15 * (i+1) for i in range(len(self.flows)))

    def interval(self, horizon):
        if horizon not in self.horizons:
            raise ValueError("Unsupported forecast horizon.")
        start = self.origin + STEP * self.horizons.index(horizon)
        return start, start + STEP

    def snapshot(self, horizon):
        self.interval(horizon)
        return dict(zip(self.ids, self.flows[self.horizons.index(horizon)], strict=True))


def original_units(raw, data):
    if raw.shape != (1, len(HORIZONS), len(data.ids)) or not np.isfinite(raw).all():
        raise Unavailable("Invalid checkpoint output shape or nonfinite prediction.")
    flows = raw[0] * data.std + data.mean
    if not np.isfinite(flows).all():
        raise Unavailable("Nonfinite inverse-normalized flow.")
    return np.maximum(flows, 0)


def forecast(dataset, run, origin, cfg=None):
    cfg = config() if cfg is None else cfg
    reason = availability(dataset, run, cfg)
    if reason:
        raise Unavailable(reason)
    import torch
    origin = pd.Timestamp(origin)
    sites, days = catalog(dataset)
    if origin.date() not in days:
        raise ValueError("Date absent from the recorded dataset.")
    path = artifact(cfg["traffic"])
    key = (path, path.stat().st_mtime_ns, tuple(sites.site_id))
    data = load_cache(*key)
    outputs = []
    for member in members(run):
        advanced = member in MEMBERS
        steps = int(manifest(member).get("args", {}).get("in_steps", 12))
        i = data.position(origin, steps + (4 * 672 if advanced else 0))
        if not data.mask[i-steps:i].any():
            raise Unavailable("The entire recent input window is missing.")
        exo = external_channels(*key) if advanced else None
        x = inputs(data, i, steps, advanced, exo)
        weights = WEIGHTS / f"{member}.pt"
        try:
            net = network(member, weights.stat().st_mtime_ns, *key, steps, x.shape[1])
            with torch.inference_mode():
                raw = net(torch.from_numpy(x)).numpy()
            outputs.append(original_units(raw, data))
        except (KeyError, RuntimeError, OSError, EOFError, pickle.UnpicklingError, IndexError) as exc:
            raise Unavailable(f"Incompatible checkpoint {member}: {exc}") from exc
    observed = np.where(data.mask[i:i+4], data.flow[i:i+4], np.nan)
    history = pd.DataFrame(np.where(data.mask[i-steps:i], data.flow[i-steps:i], np.nan),
                           index=data.times[i-steps:i], columns=data.ids)
    return Forecast(dataset, run, origin, data.ids, np.mean(outputs, axis=0), observed, history)
