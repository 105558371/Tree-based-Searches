"""Top-k route search. Adapts the A* tree search of Part A (no explored set, cycle check on the
current path, goal tested when a node is removed from the frontier). Instead of stopping at the first
goal it keeps going, so goals come out in order of cost, and it keeps the first k sufficiently
different routes."""
import heapq
import itertools


def _edges(path):
    return {(path[i], path[i + 1]) for i in range(len(path) - 1)}


def _distinct(path, found, max_overlap):
    e = _edges(path)
    return all(len(e & _edges(p)) / max(1, min(len(e), len(p) - 1)) <= max_overlap for p, _ in found)


def top_k_routes(graph, h, origin, dest, k, max_nodes=200_000, max_overlap=0.7):
    """graph: node -> list of (neighbour, cost); h: admissible heuristic. Returns [(path, cost)]."""
    tie = itertools.count()
    frontier = [(h(origin), next(tie), 0.0, origin, (origin,))]
    found, created = [], 0
    while frontier and len(found) < k and created < max_nodes:
        _, _, g, node, path = heapq.heappop(frontier)
        if node == dest:
            if _distinct(path, found, max_overlap):
                found.append((list(path), g))
            continue
        for nxt, cost in graph.get(node, []):
            if nxt in path:
                continue
            created += 1
            heapq.heappush(frontier, (g + cost + h(nxt), next(tie), g + cost, nxt, path + (nxt,)))
    return found
