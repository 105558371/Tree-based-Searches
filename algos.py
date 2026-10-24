import math
import heapq
from collections import deque
from utils import (MAX_NODES, get_sorted_neighbors, heuristic_table, same_cost,
                   on_path, path_to_list)

# Each search returns (path, nodes_created); path is None if no solution or MAX_NODES hit.
# Tree search: no explored set, only skips nodes already on the current path.
# Heap items are (priority, state, counter, ...): ties go to smaller ID, then oldest.


def depth_first_tree_search(nodes, edges, origin, destinations):
    """DFS: LIFO stack."""
    frontier = [(origin, (origin, None))]
    nodes_created = 1

    while frontier:
        state, path = frontier.pop()

        if state in destinations:
            return path_to_list(path), nodes_created

        # Push in reverse so the smallest node ID is popped first
        for neighbor in reversed(get_sorted_neighbors(edges, state)):
            if on_path(path, neighbor):
                continue
            frontier.append((neighbor, (neighbor, path)))
            nodes_created += 1
            if nodes_created >= MAX_NODES:
                return None, nodes_created

    return None, nodes_created


def breadth_first_tree_search(nodes, edges, origin, destinations):
    """BFS: FIFO queue."""
    frontier = deque([(origin, (origin, None))])
    nodes_created = 1

    while frontier:
        state, path = frontier.popleft()

        if state in destinations:
            return path_to_list(path), nodes_created

        for neighbor in get_sorted_neighbors(edges, state):
            if on_path(path, neighbor):
                continue
            frontier.append((neighbor, (neighbor, path)))
            nodes_created += 1
            if nodes_created >= MAX_NODES:
                return None, nodes_created

    return None, nodes_created


def greedy_best_first_tree_search(nodes, edges, origin, destinations):
    """GBFS: ordered by h(n)."""
    h = heuristic_table(nodes, destinations)
    counter = 0
    frontier = [(same_cost(h[origin]), origin, counter, (origin, None))]
    nodes_created = 1

    while frontier:
        _, state, _, path = heapq.heappop(frontier)

        if state in destinations:
            return path_to_list(path), nodes_created

        for neighbor in get_sorted_neighbors(edges, state):
            if on_path(path, neighbor):
                continue
            counter += 1
            heapq.heappush(frontier, (same_cost(h[neighbor]), neighbor, counter, (neighbor, path)))
            nodes_created += 1
            if nodes_created >= MAX_NODES:
                return None, nodes_created

    return None, nodes_created


def uniform_cost_tree_search(nodes, edges, origin, destinations):
    """CUS1 (UCS): ordered by g(n)."""
    counter = 0
    frontier = [(0.0, origin, counter, 0.0, (origin, None))]
    nodes_created = 1

    while frontier:
        _, state, _, g, path = heapq.heappop(frontier)

        if state in destinations:
            return path_to_list(path), nodes_created

        for neighbor in get_sorted_neighbors(edges, state):
            if on_path(path, neighbor):
                continue
            counter += 1
            new_g = g + edges[state][neighbor]
            heapq.heappush(frontier, (same_cost(new_g), neighbor, counter, new_g, (neighbor, path)))
            nodes_created += 1
            if nodes_created >= MAX_NODES:
                return None, nodes_created

    return None, nodes_created


def a_star_tree_search(nodes, edges, origin, destinations):
    """AS: ordered by f(n) = g(n) + h(n)."""
    h = heuristic_table(nodes, destinations)
    counter = 0
    frontier = [(same_cost(h[origin]), origin, counter, 0.0, (origin, None))]
    nodes_created = 1

    while frontier:
        _, state, _, g, path = heapq.heappop(frontier)

        if state in destinations:
            return path_to_list(path), nodes_created

        for neighbor in get_sorted_neighbors(edges, state):
            if on_path(path, neighbor):
                continue
            new_g = g + edges[state][neighbor]
            counter += 1
            heapq.heappush(frontier, (same_cost(new_g + h[neighbor]), neighbor, counter, new_g, (neighbor, path)))
            nodes_created += 1
            if nodes_created >= MAX_NODES:
                return None, nodes_created

    return None, nodes_created


def iterative_deepening_a_star(nodes, edges, origin, destinations):
    """
    CUS2 (IDA*): fewest moves. h(n) = ceil(distance to goal / longest edge).
    Uses a stack, not recursion, so long paths can't hit the recursion limit.
    """
    nodes_created = 1
    h = heuristic_table(nodes, destinations)

    # No single move can cover more distance than the longest edge
    longest_edge = 0
    for source in edges:
        for target in edges[source]:
            x1, y1 = nodes[source]
            x2, y2 = nodes[target]
            longest_edge = max(longest_edge, math.hypot(x1 - x2, y1 - y2))

    def moves_left(state):
        if longest_edge == 0:
            return 0
        return math.ceil(h[state] / longest_edge - 1e-9)    # Tolerance for float rounding

    bound = moves_left(origin)
    while bound != math.inf:
        next_bound = math.inf
        stack = [(origin, 0, (origin, None))]

        while stack:
            state, moves, path = stack.pop()
            f = moves + moves_left(state)

            if f > bound:
                next_bound = min(next_bound, f)
                continue
            if state in destinations:
                return path_to_list(path), nodes_created

            for neighbor in reversed(get_sorted_neighbors(edges, state)):
                if on_path(path, neighbor):
                    continue
                stack.append((neighbor, moves + 1, (neighbor, path)))
                nodes_created += 1
                if nodes_created >= MAX_NODES:
                    return None, nodes_created

        bound = next_bound

    return None, nodes_created
