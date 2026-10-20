import math

def get_sorted_neighbors(edges, state):
    return sorted(list(edges.get(state, {}).keys()))


def calculate_heuristic(state, destinations, nodes):
    x1, y1 = nodes[state]
    shortest = math.inf
    for d in destinations:
        x2, y2 = nodes[d]
        shortest = min(shortest, math.hypot(x1 - x2, y1 - y2))
    return shortest
