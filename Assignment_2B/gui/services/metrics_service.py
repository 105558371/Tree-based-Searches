"""Recorded, final research test scores. Cross-validation artifacts use a different protocol."""
import json
from math import isfinite

import numpy as np
import pandas as pd

from .data_service import HORIZONS, RESEARCH, ROOT

LOGS = ROOT / "outputs/logs/research"
WEIGHTS = ROOT / "outputs/weights/research"
RECIPE = "ext+lag+wk4+diff+roll+cyc+event"
FINAL = "staeformer_" + RECIPE.replace("+", "-") + "_final_"
ENSEMBLE = FINAL + "ens3"
MEMBERS = tuple(FINAL + f"s{i}" for i in range(3))
BASE = ("lstm", "gru", "gwn", "staeformer", "stgcn", "agcrn", "mtgnn", "stid")
LABELS = {"gwn": "Graph WaveNet", "staeformer": "STAEformer", ENSEMBLE: "STAEformer ensemble (3 seeds)"}


def label(run):
    return LABELS.get(run, run.upper() if run in BASE else run)


def records(dataset, folder=LOGS):
    if dataset != RESEARCH:
        return {}
    result = {}
    for path in sorted(folder.glob("*_metrics.json")):
        run = path.name.removesuffix("_metrics.json")
        if "_cv5" in run:
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        if "test_2025_2026" in doc:
            result[run] = doc
    return result


def scores(docs, runs, metric):
    rows = []
    for run in runs:
        doc = docs[run]
        features = doc.get("features", RECIPE if run == ENSEMBLE else "base")
        if isinstance(features, list):
            features = f"{len(features)} tabular features"
        for h in HORIZONS:
            value = ((doc.get("test_2025_2026") or {}).get(f"{h}min") or {}).get(metric)
            valid = type(value) in (int, float) and isfinite(value) and value >= 0
            rows.append((label(run), h, value if valid else np.nan, features))
    return pd.DataFrame(rows, columns=["Model", "Horizon (min)", metric, "Features"])


def sample_errors(actual, predicted, mask):
    actual, predicted = np.asarray(actual)[mask], np.asarray(predicted)[mask]
    error = predicted - actual
    big = actual >= 10
    return {"MAE": np.abs(error).mean() if len(error) else np.nan,
            "RMSE": np.sqrt(np.mean(error ** 2)) if len(error) else np.nan,
            "MAPE": 100 * np.abs(error[big] / actual[big]).mean() if big.any() else np.nan}
