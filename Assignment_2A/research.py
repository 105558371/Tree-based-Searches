import math
from itertools import permutations
from algos import uniform_cost_tree_search

MAX_DESTINATIONS = 8    # 8! = 40,320 orders


def path_cost(edges, path):
    return sum(edges[a][b] for a, b in zip(path, path[1:]))


def visit_all_destinations(nodes, edges, origin, destinations):
    """Research: cheapest walk visiting EVERY destination, using UCS legs and trying every order."""
    targets = sorted(set(destinations) - {origin})
    if not targets:
        return [origin], 1
    if len(targets) > MAX_DESTINATIONS:
        raise ValueError(f"RESEARCH supports up to {MAX_DESTINATIONS} destinations "
                         f"besides the origin; this file has {len(targets)}")

    # legs[(start, end)] = (cost, path)
    legs = {}
    nodes_created = 0
    for start in [origin] + targets:
        for end in targets:
            if end == start:
                continue
            leg, created = uniform_cost_tree_search(nodes, edges, start, {end})
            nodes_created += created
            if leg is not None:
                legs[(start, end)] = (path_cost(edges, leg), leg)

    best_stops = None
    best_cost = math.inf
    for order in permutations(targets):
        stops = [origin] + list(order)
        total = 0
        for start, end in zip(stops, stops[1:]):
            if (start, end) not in legs:
                total = math.inf
                break
            total += legs[(start, end)][0]

        if total < best_cost:
            best_stops = stops
            best_cost = total

    if best_stops is None:
        return None, nodes_created

    walk = [origin]
    for start, end in zip(best_stops, best_stops[1:]):
        walk += legs[(start, end)][1][1:]

    return walk, nodes_created
