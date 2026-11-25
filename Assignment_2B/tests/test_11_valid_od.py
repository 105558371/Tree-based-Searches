from itertools import pairwise
import pytest
from services import routing_service as r, model_service as m, data_service as ds


def check(graph, f):
    rows = r.routes(graph, f, 2000, 3002, 5)
    assert rows
    for row in rows:
        path = row["site_ids"]
        assert path[0] == 2000 and path[-1] == 3002
        assert all(e in graph for e in pairwise(path))
        assert row["distance_km"] == sum(graph[e][0] for e in pairwise(path))
        assert row["travel_time_seconds"] == sum(e["Seconds"] for e in row["edge_details"])


def test_valid_fixture(graph, snapshot):
    check(graph, snapshot())


@pytest.mark.integration
def test_real_od():
    cfg = ds.config()
    for key in ("roads", "traffic"):
        try:
            ds.artifact(cfg[key])
        except ds.Unavailable as exc:
            pytest.skip(str(exc))
    sites, _ = ds.catalog(ds.RESEARCH)
    check(r.road_graph(cfg, sites), m.forecast(ds.RESEARCH, "lstm", "2025-01-01 08:00", cfg))
