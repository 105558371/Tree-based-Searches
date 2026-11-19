import sys
from parser import parse_graph
from algos import depth_first_tree_search, breadth_first_tree_search, greedy_best_first_tree_search, a_star_tree_search, uniform_cost_tree_search, iterative_deepening_a_star
from research import visit_all_destinations
from utils import MAX_NODES

METHODS = {
    "DFS": (depth_first_tree_search, "depth-first search"),
    "BFS": (breadth_first_tree_search, "breadth-first search"),
    "GBFS": (greedy_best_first_tree_search, "greedy best-first search"),
    "AS": (a_star_tree_search, "A* search"),
    "CUS1": (uniform_cost_tree_search, "uniform-cost search (uninformed, cheapest path)"),
    "CUS2": (iterative_deepening_a_star, "IDA* (informed, fewest moves)"),
    "RESEARCH": (visit_all_destinations, "visit every destination, cheapest order"),
}


def usage():
    lines = ["Usage: python search.py <filename> <method>", "Methods:"]
    lines += [f"  {code:<9}{description}" for code, (_, description) in METHODS.items()]
    lines.append("Example: python search.py PathFinder-test.txt AS")
    return "\n".join(lines)


def main():
    if len(sys.argv) != 3:
        sys.exit(usage())

    filename = sys.argv[1]
    method = sys.argv[2].upper()

    if method not in METHODS:
        sys.exit(f"Error: unknown method '{sys.argv[2]}'.\n\n{usage()}")

    try:
        nodes, edges, origin, destinations = parse_graph(filename)
    except FileNotFoundError:
        sys.exit(f"Error: file '{filename}' not found. Check the name and the folder you are running from.")
    except IsADirectoryError:
        sys.exit(f"Error: '{filename}' is a folder, not a problem file.")
    except UnicodeDecodeError:
        sys.exit(f"Error: '{filename}' is not a plain text file.")
    except (OSError, ValueError) as error:
        sys.exit(f"Error: {error}")

    search, _ = METHODS[method]
    try:
        path, nodes_created = search(nodes, edges, origin, destinations)
    except ValueError as error:
        sys.exit(f"Error: {error}")

    print(f"{filename} {method}")
    if path:
        print(f"{path[-1]} {nodes_created}")
        print(" ".join(str(node) for node in path))
    elif nodes_created >= MAX_NODES:
        print(f"NONE {nodes_created}")
        print(f"No path found. The search stopped after creating {MAX_NODES:,} nodes.")
    else:
        print(f"NONE {nodes_created}")
        print("No path found.")


if __name__ == "__main__":
    main()
