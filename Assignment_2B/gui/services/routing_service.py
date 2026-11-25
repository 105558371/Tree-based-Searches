"""Directed road snapshot and Yen spur paths using the unmodified Part A UCS."""
import heapq
import json
import math
from itertools import pairwise

import pandas as pd
from Assignment_2A.src import algos
from Assignment_2B.core.travel_time import flow_to_speed, haversine_km, travel_time_seconds

from .data_service import Unavailable, artifact


class SearchLimit(ValueError):
    pass


def read_graph(frame, sites):
    pos = sites.set_index("site_id")
    required = {"from_site", "to_site", "road_km"}
    if frame.empty or not required <= set(frame):
        raise ValueError("Expected a directed road edge list with road_km.")
    graph = {}
    for row in frame.itertuples():
        a, b, km = row.from_site, row.to_site, row.road_km
        if (a not in pos.index or b not in pos.index or a == b or (a, b) in graph
                or not isinstance(km, (float, int)) or not math.isfinite(km) or km <= 0):
            raise ValueError("Invalid or duplicate directed road edge.")
        geometry = None
        raw = getattr(row, "geometry", None)
        if pd.notna(raw):
            line = json.loads(raw)
            geometry = line["coordinates"]
            if (line.get("type") != "LineString" or len(geometry) < 2
                    or any(len(p) != 2 or not all(map(math.isfinite, p)) for p in geometry)
                    or any(haversine_km(pos.at[s, "lat"], pos.at[s, "lon"], y, x) > 0.1
                           for s, (x, y) in [(a, geometry[0]), (b, geometry[-1])])):
                raise ValueError("Road geometry does not match the directed endpoints.")
        graph[int(a), int(b)] = (float(km), geometry)
    return graph


def road_graph(cfg, sites):
    path = artifact(cfg["roads"])
    return read_graph(pd.read_csv(path), sites)


def edge_cost(km, flow):
    if not math.isfinite(km) or km <= 0 or not math.isfinite(flow) or flow < 0:
        raise ValueError("Distance must be positive and destination flow nonnegative and finite.")
    return travel_time_seconds(km, flow)


def shortest(nodes, adjacency, origin, dest):
    path, created = algos.uniform_cost_tree_search(nodes, adjacency, origin, {dest})
    if path is None and created >= algos.MAX_NODES:
        raise SearchLimit("A2A UCS reached its node limit. Search is incomplete.")
    return tuple(path) if path else None


def top_paths(nodes, edges, origin, dest, k):
    adjacency = {s: {} for s in nodes}
    for (a, b), seconds in edges.items():
        adjacency[a][b] = seconds
    first = shortest(nodes, adjacency, origin, dest)
    if first is None:
        return []
    accepted, seen, candidates = [first], {first}, []
    while len(accepted) < k:
        previous = accepted[-1]
        for i, spur in enumerate(previous[:-1]):
            root = previous[:i+1]
            blocked = {(p[i], p[i+1]) for p in accepted if len(p) > i+1 and p[:i+1] == root}
            removed = set(root[:-1])
            restricted = {a: {b: w for b, w in nbrs.items() if b not in removed and (a, b) not in blocked}
                          for a, nbrs in adjacency.items() if a not in removed}
            tail = shortest(nodes, restricted, spur, dest)
            if tail:
                path = root[:-1] + tail
                if path not in seen:
                    heapq.heappush(candidates, (sum(edges[e] for e in pairwise(path)), path))
                    seen.add(path)
        if not candidates:
            break
        accepted.append(heapq.heappop(candidates)[1])
    return sorted(accepted, key=lambda p: (sum(edges[e] for e in pairwise(p)), p))


def routes(graph, forecast, origin, destination, k, horizon=15):
    if origin not in forecast.ids or destination not in forecast.ids:
        raise ValueError("Unknown origin or destination site.")
    if type(k) is not int or not 1 <= k <= 5:
        raise ValueError("Select between 1 and 5 routes.")
    if graph is None:
        raise Unavailable("Directed road graph unavailable.")
    flows = forecast.snapshot(horizon)
    details, costs = {}, {}
    for (a, b), (km, geometry) in graph.items():
        flow = float(flows[b])
        seconds = edge_cost(km, flow)
        costs[a, b] = seconds
        details[a, b] = {"From": a, "To": b, "Road km": km, "B flow (veh/15 min)": flow,
                         "Speed (km/h)": flow_to_speed(4 * flow), "Seconds": seconds}
    result = []
    for path in top_paths(forecast.ids, costs, origin, destination, k):
        edges = list(pairwise(path))
        rows = [details[e] for e in edges]
        geometry = [p for e in edges for p in graph[e][1]] if edges and all(graph[e][1] for e in edges) else None
        result.append(dict(site_ids=path, edge_details=rows, distance_km=sum(r["Road km"] for r in rows),
                           travel_time_seconds=sum(costs[e] for e in edges), geometry=geometry,
                           model_id=forecast.model, dataset_id=forecast.dataset,
                           target_interval=forecast.interval(horizon), algorithm="A2A UCS / Yen",
                           status="no travel" if len(path) == 1 else "ok"))
    return result
