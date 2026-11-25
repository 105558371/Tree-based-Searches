import json
import numpy as np
import pytest
from services import data_service as ds, metrics_service as m


def test_recorded_json_and_masked_errors():
    source = json.loads((m.LOGS / "lstm_metrics.json").read_text())["test_2025_2026"]
    docs = m.records(ds.RESEARCH)
    table = m.scores(docs, ["lstm"], "RMSE")
    assert list(table["Horizon (min)"]) == list(ds.HORIZONS)
    assert list(table.RMSE) == [source[f"{h}min"]["RMSE"] for h in ds.HORIZONS]
    assert not m.records(ds.ASSIGNMENT) and all("cv5" not in run for run in docs)
    assert m.scores({"x": {"test_2025_2026": {"15min": {"MAE": None}}}}, ["x"], "MAE").MAE.isna().all()
    score = m.sample_errors([5, 20, 0], [7, 16, 500], np.array([True, True, False]))
    assert score == pytest.approx(dict(MAE=3, RMSE=10**0.5, MAPE=20))
