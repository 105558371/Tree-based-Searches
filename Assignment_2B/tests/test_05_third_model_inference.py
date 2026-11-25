import numpy as np
import pytest
from services import model_service as m


@pytest.mark.integration
def test_real_graph_wavenet(smoke, real_inputs):
    smoke("gwn", "GraphWaveNet")
    _, _, key, _ = real_inputs
    stamp = (m.WEIGHTS / "gwn.pt").stat().st_mtime_ns
    with pytest.raises(RuntimeError, match="size mismatch"):
        m.network("gwn", stamp, *key, 12, 3)


@pytest.mark.integration
def test_real_ensemble_average(real_inputs):
    cfg, data, _, i = real_inputs
    for run in m.MEMBERS:
        if not (m.WEIGHTS / f"{run}.pt").exists():
            pytest.skip(f"Missing {run}.pt")
    origin = data.times[i]
    actual = m.forecast(m.RESEARCH, m.ENSEMBLE, origin, cfg)
    members = [m.forecast(m.RESEARCH, run, origin, cfg).flows for run in m.MEMBERS]
    np.testing.assert_allclose(actual.flows, np.mean(members, axis=0), rtol=1e-6)


def test_missing_member_and_invalid_model_output(monkeypatch, tmp_path, replay):
    monkeypatch.setattr(m, "WEIGHTS", tmp_path)
    (tmp_path / f"{m.MEMBERS[0]}.pt").touch()
    assert m.MEMBERS[1] in m.availability(m.RESEARCH, m.ENSEMBLE, {"traffic": [str(tmp_path / f"{m.MEMBERS[0]}.pt")]})
    for raw in (np.full((1, 4, 4), np.nan), np.zeros((1, 3, 4))):
        with pytest.raises(m.Unavailable):
            m.original_units(raw, replay)
