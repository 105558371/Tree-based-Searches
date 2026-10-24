import math

MAX_NODES = 1_000_000    # Give up after this many nodes


def get_sorted_neighbors(edges, state):
    return sorted(edges.get(state, {}))


def calculate_heuristic(state, destinations, nodes):
    x1, y1 = nodes[state]
    shortest = math.inf
    for d in destinations:
        x2, y2 = nodes[d]
        shortest = min(shortest, math.hypot(x1 - x2, y1 - y2))
    return shortest


def heuristic_table(nodes, destinations):
    """h(n) for every node, computed once."""
    return {node: calculate_heuristic(node, destinations, nodes) for node in nodes}


def same_cost(value):
    """Round so float sums still tie."""
    return round(value, 9)


# Paths are linked tuples (node, parent_path) so children never copy the path.

def on_path(path, node):
    """True if node is already on this branch."""
    while path is not None:
        if path[0] == node:
            return True
        path = path[1]
    return False


def path_to_list(path):
    result = []
    while path is not None:
        result.append(path[0])
        path = path[1]
    return result[::-1]
