import sys
from parser import parse_graph
from algos import depth_first_tree_search, breadth_first_tree_search, greedy_best_first_tree_search, a_star_tree_search, uniform_cost_tree_search,iterative_deepening_a_star
from research import visit_all_destinations

METHODS = {
    "DFS": depth_first_tree_search,
    "BFS": breadth_first_tree_search,
    "GBFS": greedy_best_first_tree_search,
    "AS": a_star_tree_search,
    "CUS1": uniform_cost_tree_search,
    "CUS2": iterative_deepening_a_star,
    "RESEARCH": visit_all_destinations,    # Research: visit every destination
}


def main():
    if len(sys.argv) != 3:
        sys.exit("Usage: python search.py <filename> <method>\n"
                 f"Methods supported: {', '.join(METHODS)}")

    filename = sys.argv[1]
    method = sys.argv[2].upper()

    if method not in METHODS:
        sys.exit(f"Error: Unknown method '{method}'. Choose from {', '.join(METHODS)}.")

    try:
        nodes, edges, origin, destinations = parse_graph(filename)
    except FileNotFoundError:
        sys.exit(f"Error: File '{filename}' not found.")
    except (OSError, ValueError) as error:
        sys.exit(f"Error: {error}")

    path, nodes_created = METHODS[method](nodes, edges, origin, destinations)

    print(f"{filename} {method}")
    if path:
        print(f"{path[-1]} {nodes_created}")
        print(" ".join(str(node) for node in path))
    else:
        print(f"NONE {nodes_created}")
        print("No path found.")


if __name__ == "__main__":
    main()
