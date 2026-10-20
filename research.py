import heapq
import math
from itertools import permutations


def dijkstra(edges, source):
    """
    Graph search (explored set): cheapest cost from source to every reachable node.
    Returns (cost, parent, nodes_created):
      cost:   dict, node -> cheapest cost from source
      parent: dict, node -> previous node on its cheapest path
      nodes_created: number of items added to the frontier
    """
    cost = {source: 0}
    parent = {source: None}
    frontier = [(0, source)]    # (cost, node): ties go to the smaller node ID
    explored = set()
    nodes_created = 1

    while frontier:
        current_cost, node = heapq.heappop(frontier)
        if node in explored:    # Outdated entry, a cheaper one was already used
            continue
        explored.add(node)

        for neighbor, edge_cost in edges[node].items():
            new_cost = current_cost + edge_cost
            if neighbor not in cost or new_cost < cost[neighbor]:
                cost[neighbor] = new_cost
                parent[neighbor] = node
                heapq.heappush(frontier, (new_cost, neighbor))
                nodes_created += 1

    return cost, parent, nodes_created


def build_path(parent, end):
    """Follow parents back from end to the source, then reverse."""
    path = []
    node = end
    while node is not None:
        path.append(node)
        node = parent[node]
    return path[::-1]


def visit_all_destinations(nodes, edges, origin, destinations):
    """
    Research: cheapest walk from the origin that visits EVERY destination, in any order.
    1. Run Dijkstra from the origin and from each destination.
    2. Try every visiting order and keep the one with the lowest total cost.
    3. Join the legs of the best order into one path.
    Returns (path, nodes_created). path is None if some destination cannot be visited.
    nodes_created is the total over all Dijkstra runs.
    """
    targets = sorted(destinations - {origin})    # The origin is already visited
    if not targets:
        return [origin], 1

    # Step 1: shortest[start] = (cost, parent) from one Dijkstra run
    shortest = {}
    nodes_created = 0
    for start in [origin] + targets:
        cost, parent, created = dijkstra(edges, start)
        shortest[start] = (cost, parent)
        nodes_created += created

    # Step 2: orders are tried in ascending node ID, and "<" keeps the first on ties
    best_stops = None
    best_cost = math.inf
    for order in permutations(targets):
        stops = [origin] + list(order)
        total = 0
        for start, end in zip(stops, stops[1:]):
            total += shortest[start][0].get(end, math.inf)    # Unreachable leg = infinite

        if total < best_cost:
            best_stops = stops
            best_cost = total

    if best_stops is None:
        return None, nodes_created

    # Step 3: join the legs, dropping each leg's first node (it ends the previous leg)
    walk = [origin]
    for start, end in zip(best_stops, best_stops[1:]):
        walk += build_path(shortest[start][1], end)[1:]

    return walk, nodes_created
