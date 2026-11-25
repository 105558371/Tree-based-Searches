import math
import pytest
from services import routing_service as r


def test_conversion_and_direction(snapshot):
    assert r.edge_cost(2, 0) == 150
    assert r.edge_cost(2, 375) == 255
    f = snapshot()
    graph = {(2000, 2100): (2., None), (2100, 2000): (2., None)}
    assert r.routes(graph, f, 2000, 2100, 1)[0]["travel_time_seconds"] == 255
    assert r.routes(graph, f, 2100, 2000, 1)[0]["travel_time_seconds"] == 150
    for km, flow in [(0, 1), (-1, 0), (2, -1), (math.inf, 1), (1, math.nan)]:
        with pytest.raises(ValueError):
            r.edge_cost(km, flow)
