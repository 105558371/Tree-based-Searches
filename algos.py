import math
import heapq
from collections import deque
from utils import get_sorted_neighbors, calculate_heuristic

# Every search returns (path, nodes_created); path is None if there is no solution.
# Tree search: no explored set. The only check is "neighbor in path", which skips
# cycles on the current branch and never looks at other branches.
# Priority queue items are (priority, state, counter, ...) so ties go to the smaller
# node ID first, then to the node that was added first.


def depth_first_tree_search(nodes, edges, origin, destinations):
    """DFS: LIFO stack. Frontier items: (state, path)."""
    frontier = [(origin, [origin])]
    nodes_created = 1

    while frontier:
        state, path = frontier.pop()

        if state in destinations:
            return path, nodes_created

        # Push in reverse so the smallest node ID is popped first
        for neighbor in reversed(get_sorted_neighbors(edges, state)):
            if neighbor in path:
                continue
            frontier.append((neighbor, path + [neighbor]))
            nodes_created += 1

    return None, nodes_created


def breadth_first_tree_search(nodes, edges, origin, destinations):
    """BFS: FIFO queue. Frontier items: (state, path)."""
    frontier = deque([(origin, [origin])])
    nodes_created = 1

    while frontier:
        state, path = frontier.popleft()

        if state in destinations:
            return path, nodes_created

        for neighbor in get_sorted_neighbors(edges, state):
            if neighbor in path:
                continue
            frontier.append((neighbor, path + [neighbor]))
            nodes_created += 1

    return None, nodes_created


def greedy_best_first_tree_search(nodes, edges, origin, destinations):
    """GBFS: ordered by h(n). Frontier items: (h, state, counter, path)."""
    counter = 0
    origin_h = calculate_heuristic(origin, destinations, nodes)
    frontier = [(origin_h, origin, counter, [origin])]
    nodes_created = 1

    while frontier:
        _, state, _, path = heapq.heappop(frontier)

        if state in destinations:
            return path, nodes_created

        for neighbor in get_sorted_neighbors(edges, state):
            if neighbor in path:
                continue
            h = calculate_heuristic(neighbor, destinations, nodes)
            counter += 1
            heapq.heappush(frontier, (h, neighbor, counter, path + [neighbor]))
            nodes_created += 1

    return None, nodes_created


def uniform_cost_tree_search(nodes, edges, origin, destinations):
    """CUS1 (UCS): ordered by g(n). Frontier items: (g, state, counter, path)."""
    counter = 0
    frontier = [(0.0, origin, counter, [origin])]
    nodes_created = 1

    while frontier:
        g, state, _, path = heapq.heappop(frontier)

        if state in destinations:
            return path, nodes_created

        for neighbor in get_sorted_neighbors(edges, state):
            if neighbor in path:
                continue
            counter += 1
            new_g = g + edges[state][neighbor]
            heapq.heappush(frontier, (new_g, neighbor, counter, path + [neighbor]))
            nodes_created += 1

    return None, nodes_created


def a_star_tree_search(nodes, edges, origin, destinations):
    """AS: ordered by f(n) = g(n) + h(n). Frontier items: (f, state, counter, g, path)."""
    counter = 0
    origin_h = calculate_heuristic(origin, destinations, nodes)
    frontier = [(origin_h, origin, counter, 0.0, [origin])]
    nodes_created = 1

    while frontier:
        _, state, _, g, path = heapq.heappop(frontier)

        if state in destinations:
            return path, nodes_created

        for neighbor in get_sorted_neighbors(edges, state):
            if neighbor in path:
                continue
            new_g = g + edges[state][neighbor]
            new_f = new_g + calculate_heuristic(neighbor, destinations, nodes)
            counter += 1
            heapq.heappush(frontier, (new_f, neighbor, counter, new_g, path + [neighbor]))
            nodes_created += 1

    return None, nodes_created


def iterative_deepening_a_star(nodes, edges, origin, destinations):
    """
    CUS2 (IDA*): finds the path with the fewest moves. Each edge counts as 1 move.
    h(n) = ceil(distance to nearest destination / longest edge), which never
    overestimates the number of moves left.
    """
    nodes_created = 1

    # No single move can cover more distance than the longest edge
    longest_edge = 0
    for source in edges:
        for target in edges[source]:
            x1, y1 = nodes[source]
            x2, y2 = nodes[target]
            length = math.hypot(x1 - x2, y1 - y2)
            if length > longest_edge:
                longest_edge = length

    def moves_left(state):
        if longest_edge == 0:
            return 0
        distance = calculate_heuristic(state, destinations, nodes)
        return math.ceil(distance / longest_edge - 1e-9)    # Tolerance for float rounding

    def search(state, moves, bound, path):
        """Returns (next_bound, path); path is None if no goal was found within bound."""
        nonlocal nodes_created
        f = moves + moves_left(state)

        if f > bound:
            return f, None
        if state in destinations:
            return f, path

        next_bound = math.inf
        for neighbor in get_sorted_neighbors(edges, state):
            if neighbor in path:
                continue
            nodes_created += 1
            result, found_path = search(neighbor, moves + 1, bound, path + [neighbor])
            if found_path is not None:
                return result, found_path
            next_bound = min(next_bound, result)

        return next_bound, None

    bound = moves_left(origin)
    while bound != math.inf:
        bound, path = search(origin, 0, bound, [origin])
        if path is not None:
            return path, nodes_created

    return None, nodes_created
