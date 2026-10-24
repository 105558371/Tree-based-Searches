# Tree-Based Search: Route Finding

COS30019 Introduction to AI, Assignment 2A.

The program finds a route from an origin node to one of the destination nodes on a 2D graph, using six tree-based search methods.

## Requirements

- Python 3.10 or later (standard library only, nothing to install)
- Tested from the Windows command prompt (`cmd`)

## How to run

```
python search.py <filename> <method>
```

Example:

```
python search.py tests/t00_assignment.txt AS
```

Output:

```
tests/t00_assignment.txt AS
4 8
2 1 4
```

- Line 1: the file name and method
- Line 2: the destination reached and the number of nodes created
- Line 3: the path from the origin to that destination

If no destination can be reached, line 2 starts with `NONE` and line 3 says `No path found.`

## Methods

| Method | Name | Finds |
|---|---|---|
| `DFS` | Depth-first search | A path (not always the best) |
| `BFS` | Breadth-first search | The path with the fewest moves |
| `GBFS` | Greedy best-first search | A path, guided by distance to the goal |
| `AS` | A* search | The cheapest path |
| `CUS1` | Uniform-cost search (uninformed) | The cheapest path |
| `CUS2` | IDA* (informed) | The path with the fewest moves |
| `RESEARCH` | Visit all destinations | The cheapest route that visits every destination |

Method names are not case-sensitive.

## Input file format

```
Nodes:
1: (4,1)
2: (2,2)
3: (4,4)
Edges:
(2,1): 4
(2,3): 4
Origin:
2
Destinations:
1; 3
```

- `1: (4,1)` is node 1 at x = 4, y = 1
- `(2,1): 4` is a one-way edge from node 2 to node 1 with cost 4
- Destinations are separated by `;`. Reaching any one of them counts as success.
- A value can also go on the same line as its heading, e.g. `Origin: 2`

## How the searches behave

- **Ties:** when all else is equal, the smaller node number is expanded first, then the node that was added first.
- **Tree search:** there is no explored set. A node is only skipped if it is already on the current path, which stops loops.
- **Heuristic:** straight-line distance from a node to the nearest destination.
- **Node limit:** a search stops after creating 1,000,000 nodes and reports no path.

## Testing

```
python run_tests.py
```

This runs every method on every file in `tests/` and checks the results against `tests/expected.csv`. Add `--table` to print how many nodes each method created.

See [`tests/README.md`](tests/README.md) for what each test checks and its expected result.

## Files

| File | Purpose |
|---|---|
| `search.py` | Command-line entry point |
| `parser.py` | Reads and checks the problem file |
| `algos.py` | DFS, BFS, GBFS, A*, CUS1 and CUS2 |
| `research.py` | Visit-all-destinations method |
| `utils.py` | Shared helpers (heuristic, path handling) |
| `run_tests.py` | Runs all tests |
| `tests/` | Test problem files and expected results |
