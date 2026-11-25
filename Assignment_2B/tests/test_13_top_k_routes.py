from itertools import pairwise
from unittest.mock import patch
import networkx as nx
import pytest
from services import routing_service as r


def test_top_k_matches_oracle(snapshot):
    f = snapshot("03:00")
    graph = {(a,b): (1., None) for a in f.ids for b in f.ids if a != b}
    oracle = nx.DiGraph()
    oracle.add_weighted_edges_from((a, b, 90) for a, b in graph)
    saved = dict(graph)
    costs = sorted(nx.path_weight(oracle, p, "weight") for p in nx.all_simple_paths(oracle, 2000, 3002))
    with patch.object(r.algos, "uniform_cost_tree_search", wraps=r.algos.uniform_cost_tree_search) as search:
        for k in (1, 5):
            rows = r.routes(graph, f, 2000, 3002, k)
            assert [x["travel_time_seconds"] for x in rows] == costs[:k]
            paths = [x["site_ids"] for x in rows]
            assert len(set(paths)) == k
            assert all(len(set(p)) == len(p) and p[0] == 2000 and p[-1] == 3002
                       and all(e in graph for e in pairwise(p)) for p in paths)
        assert search.call_count > 5 and saved == graph
    assert len(r.routes({(2000, 3002): (1., None)}, f, 2000, 3002, 5)) == 1
    for k in (0, 6, 1.5, True):
        with pytest.raises(ValueError):
            r.routes(graph, f, 2000, 3002, k)
