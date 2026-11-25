from datetime import date
import json

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from services import data_service as ds, metrics_service as metrics, model_service as models, routing_service as routing

IDS = (2000, 2100, 2200, 3002)
EDGES = [(2000, 2100, 2.), (2100, 3002, 1.), (2000, 2200, 2.5), (2200, 3002, 1.)]


@pytest.fixture
def sites():
    return pd.DataFrame({"site_id": IDS, "lat": [-37.8, -37.81, -37.82, -37.83],
                         "lon": [145., 145.01, 145.02, 145.03], "name": [""]*4})


@pytest.fixture
def graph(sites):
    return routing.read_graph(pd.DataFrame(EDGES, columns=["from_site", "to_site", "road_km"]), sites)


def controlled(run, when):
    origin = pd.Timestamp(when)
    values = {3: (0, 0, 0, 0), 8: (0, 375, 0, 0), 17: (0, 0, 375, 375)}[origin.hour]
    flows = np.array(values)[None, :] + np.arange(4)[:, None] * 100 + (10 if run == "gru" else 0)
    actual = flows.astype(float) + 1
    actual[1, 0] = np.nan
    history = pd.DataFrame(np.zeros((12, 4)), columns=IDS,
                           index=pd.date_range(end=origin-ds.STEP, periods=12, freq=ds.STEP))
    return models.Forecast(ds.RESEARCH, run, origin, IDS, flows, actual, history)


@pytest.fixture
def snapshot():
    return lambda time="08:00": controlled("lstm", f"2025-01-01 {time}")


@pytest.fixture
def replay():
    t = pd.date_range("2025-01-01", periods=32, freq="15min")
    raw = np.arange(len(t)*4, dtype=np.float32).reshape(-1, 4)
    return ds.ReplayData(t, IDS, raw, np.ones_like(raw, bool), np.eye(4), np.ones(4), np.ones(4)*2,
                         np.ones((672, 4))*8, {})


@pytest.fixture
def real_inputs():
    pytest.importorskip("torch")
    cfg = ds.config()
    try:
        path = ds.artifact(cfg["traffic"])
    except ds.Unavailable as exc:
        pytest.skip(str(exc))
    sites, days = ds.catalog(ds.RESEARCH)
    key = (path, path.stat().st_mtime_ns, tuple(sites.site_id))
    data = ds.load_cache(*key)
    first = data.times.searchsorted(pd.Timestamp("2025-01-01"))
    # Select an observed window to compare directly against the original research batch.
    complete = data.mask[first:].all(axis=1)
    spans = np.flatnonzero(np.convolve(complete.astype(int), np.ones(12, int), "valid") == 12)
    assert len(spans), "No completely observed held-out reference window"
    return cfg, data, key, first + int(spans[0]) + 12


@pytest.fixture
def smoke(real_inputs, monkeypatch):
    import torch
    from Assignment_2B.research.rnn_models import RNNForecaster
    cfg, data, key, i = real_inputs
    def run(name, architecture):
        if not (metrics.WEIGHTS / f"{name}.pt").exists():
            pytest.skip(f"Missing {name} checkpoint")
        forecast = models.forecast(ds.RESEARCH, name, data.times[i], cfg)
        net = models.network(name, (metrics.WEIGHTS / f"{name}.pt").stat().st_mtime_ns, *key, 12, 2)
        assert (type(net.rnn).__name__ if isinstance(net, RNNForecaster) else type(net).__name__) == architecture
        assert not net.training and next(net.parameters()).device.type == "cpu"
        # Reproduce train.predict on this small, wholly observed slice, without calling load or evaluate.
        monkeypatch.syspath_prepend(str(ds.ROOT / "research"))
        from Assignment_2B.research import train
        times = data.times[i-12:i+4]
        flow = torch.from_numpy(data.flow[i-12:i+4].copy())
        ref = dict(flow=flow, mask=torch.from_numpy(data.mask[i-12:i+4]), mean=torch.from_numpy(data.mean),
                   std=torch.from_numpy(data.std), tod=torch.tensor((times.hour*60+times.minute)/1440, dtype=torch.float32))
        ref["norm"] = (flow - ref["mean"]) / ref["std"]
        idx = torch.tensor([12])
        assert train.batch(ref, idx)[0].shape == (1, 2, len(data.ids), 12)
        with torch.inference_mode():
            reference = train.predict(net, ref, idx).clamp(min=0).numpy()[0]
        np.testing.assert_allclose(forecast.flows, reference, rtol=2e-5, atol=2e-4)
        assert forecast.ids == data.ids and forecast.flows.shape == (4, len(data.ids))
        assert np.isfinite(forecast.flows).all() and (forecast.flows >= 0).all()
    return run


@pytest.fixture
def ui(monkeypatch, sites, graph, tmp_path):
    calls = []
    monkeypatch.setattr(ds, "catalog", lambda dataset: (sites, [date(2006, 10, 1) if dataset == ds.ASSIGNMENT else date(2025, 1, 1)]))
    monkeypatch.setattr(routing, "road_graph", lambda cfg, selected: graph)
    def availability(dataset, model, cfg):
        if dataset != ds.RESEARCH:
            return "October 2006 checkpoints unavailable."
        return "" if model in ("lstm", "gru") else "Checkpoint unavailable for this model."
    monkeypatch.setattr(models, "availability", availability)
    def predict(dataset, model, when, cfg):
        assert dataset == ds.RESEARCH
        calls.append((model, when))
        return controlled(model, when)
    monkeypatch.setattr(models, "forecast", predict)
    report = {"model": "LSTM", "test_2025_2026": {f"{h}min": {"MAE": h+1., "RMSE": h+2., "MAPE": h+3.}
                                                   for h in ds.HORIZONS}, "history": []}
    for run in ("lstm", "gru", "gwn"):
        (tmp_path / f"{run}_metrics.json").write_text(json.dumps(report))
    read_records = metrics.records
    monkeypatch.setattr(metrics, "records", lambda dataset: read_records(dataset, tmp_path))
    return AppTest.from_file(str(ds.GUI / "app.py"), default_timeout=30).run(), calls
