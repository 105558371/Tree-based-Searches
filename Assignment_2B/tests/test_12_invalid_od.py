import pandas as pd
import pytest
from services import routing_service as r


def test_exceptional_endpoints(graph, snapshot, monkeypatch, sites):
    f = snapshot()
    for a, b in [("", 3002), (2000, ""), (1, 3002), (2000, 1)]:
        with pytest.raises(ValueError, match="Unknown"):
            r.routes(graph, f, a, b, 1)
    same = r.routes(graph, f, 2000, 2000, 5)
    assert len(same) == 1 and same[0]["site_ids"] == (2000,) and same[0]["travel_time_seconds"] == 0
    assert not r.routes(graph, f, 3002, 2000, 5)
    with pytest.raises(r.Unavailable):
        r.routes(None, f, 2000, 3002, 1)
    monkeypatch.setattr(r.algos, "MAX_NODES", 2)
    with pytest.raises(r.SearchLimit):
        r.routes(graph, f, 2000, 3002, 1)
    frame = pd.DataFrame([(2000, 3002, 2)]*2, columns=["from_site", "to_site", "road_km"])
    with pytest.raises(ValueError, match="duplicate"):
        r.read_graph(frame, sites)
