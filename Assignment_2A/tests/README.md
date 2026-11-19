# Test cases

Run everything with:

```
python run_tests.py            # PASS/FAIL for every test and method
python run_tests.py --table    # also prints the nodes-created table below
```

`expected.csv` holds the expected goal, nodes created and path for every test and method (26 tests × 7 methods = 182 checks). Expected paths for the contrast tests (t20–t26, t29) were worked out by hand first, then confirmed by the program.

## Edge cases (does the program behave correctly?)

| Test | Scenario | Expected outcome |
|---|---|---|
| t00_assignment | The sample problem from the brief | AS finds 2 1 4 (cost 10). DFS reaches the other destination: 2 1 3 5. |
| t01_direct_edge | Origin connected straight to the goal | Every method: 1 2 |
| t02_simple_path | A single chain 1 → 2 → 3 → 4 | Every method: 1 2 3 4 |
| t03_two_paths | Cheap route (cost 2) vs expensive route (cost 10) | Every method: 1 2 4 |
| t04_multi_dest | Two destinations, one unreachable | Every method reaches 3. RESEARCH reports no path, because 4 cannot be visited. |
| t05_origin_is_dest | The origin is also a destination | Every method returns 1 with 1 node created |
| t06_unreachable | No route to the goal | Every method: no path found |
| t07_cycle | Graph with a cycle | No method loops forever; all find 1 2 3 4 |
| t08_self_loop | A node with an edge to itself | The self-loop is skipped; all find 1 2 3 |
| t09_two_components | Two separate parts; goal in the origin's part | Every method: 1 2 |
| t11_one_way | One-way and expensive return edges | Every method: 3 2 1 |
| t12_asymmetric | Different costs in each direction | Every method: 1 2 3 |
| t13_tie_ids | Two equal routes, ties broken by node number | DFS/BFS take the smaller ID (1 2 4 5); GBFS/AS follow the heuristic (1 3 4 5) |
| t16_gbfs_trap | GBFS heads toward the goal | GBFS happens to find the cheapest route (1 4 3). See t22 for a real trap. Edge (4,3) costs less than its straight-line length. |
| t17_astar_optimal | Cheap 3-move route vs expensive 2-move route | AS/CUS1 find the cheapest (1 2 4 5, cost 3); BFS/CUS2 find the fewest moves (1 3 5, cost 10) |
| t19_grid_3x3 | 3×3 grid, corner to corner | Every method except DFS finds a 4-move route; DFS wanders through all 9 cells (8 moves). Node counts differ (see table). |
| t27_long_chain | 1,500 nodes in a line | Every method finds the full path. The old recursive CUS2 crashed here with RecursionError. |
| t28_node_cap | 11 fully connected nodes, goal unreachable (millions of routes) | Every method stops at 1,000,000 nodes and reports no path |
| t29_decimals | Decimal coordinates and costs | AS/CUS1 find 1 2 3 (cost 2.75); BFS/CUS2 take the direct edge 1 3 (cost 3.0) |

## Contrast tests (what each search is good and bad at)

| Test | What it shows | Expected outcome |
|---|---|---|
| t20_dfs_wrong_branch | **DFS is not optimal.** It follows the smaller node ID down a long branch. | DFS: 1 2 3 4 5 (4 moves, cost 8). BFS, GBFS, AS, CUS1, CUS2: 1 5 (1 move, cost 4). |
| t21_bfs_not_cheapest | **Fewest moves is not cheapest.** | BFS, GBFS, CUS2: 1 4 (cost 10). AS, CUS1: 1 2 3 4 (cost 6). |
| t22_gbfs_trap | **GBFS can be fooled.** The node closest to the goal leads to a cost-50 edge. | GBFS: 1 2 4 (cost 55). AS, CUS1: 1 3 4 (cost 18). |
| t23_inadmissible_h | **A\* needs h to never overestimate.** Edges 1→2→4 cost 1 each but are 20+ units long. | AS: 1 3 4 (cost 10), not optimal. CUS1 (no heuristic): 1 2 4 (cost 2). |
| t24_dfs_dead_end | **DFS can waste work.** A 31-node dead-end subtree sits on the smaller-ID side. | DFS creates 33 nodes; BFS 5; AS, GBFS, CUS1 3. All find 1 40. |
| t25_bfs_wide | **BFS uses a lot of memory on wide trees.** Branching factor 6, goal at depth 3. | BFS and CUS1 create 259 nodes; AS 103; CUS2 25; DFS and GBFS 19. All find 1 2 8 44. |
| t26_grid_5x5 | **The heuristic saves work.** 5×5 grid, corner to corner. | Shortest is 8 moves. CUS1 creates 1,933 nodes, BFS 1,077, AS 425, CUS2 74, GBFS 22. DFS creates 41 but wanders through every cell (24 moves). |

## Nodes created by each method

| Test | DFS | BFS | GBFS | AS | CUS1 | CUS2 | RESEARCH |
|---|---|---|---|---|---|---|---|
| t00_assignment | 7 | 10 | 6 | 8 | 11 | 5 | 39 |
| t01_direct_edge | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| t02_simple_path | 4 | 4 | 4 | 4 | 4 | 4 | 4 |
| t03_two_paths | 4 | 5 | 4 | 4 | 4 | 6 | 4 |
| t04_multi_dest | 3 | 3 | 3 | 3 | 3 | 3 | 8 |
| t05_origin_is_dest | 1 | 1 | 1 | 1 | 1 | 1 | 3 |
| t06_unreachable | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| t07_cycle | 4 | 4 | 4 | 4 | 4 | 6 | 4 |
| t08_self_loop | 3 | 3 | 3 | 3 | 3 | 3 | 3 |
| t09_two_components | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| t11_one_way | 3 | 3 | 3 | 3 | 3 | 3 | 3 |
| t12_asymmetric | 3 | 3 | 3 | 3 | 3 | 3 | 3 |
| t13_tie_ids | 5 | 7 | 6 | 6 | 7 | 11 | 7 |
| t16_gbfs_trap | 5 | 6 | 5 | 5 | 5 | 5 | 5 |
| t17_astar_optimal | 5 | 6 | 5 | 5 | 5 | 5 | 5 |
| t19_grid_3x3 | 13 | 31 | 10 | 27 | 37 | 10 | 37 |
| t20_dfs_wrong_branch | 6 | 4 | 3 | 3 | 5 | 3 | 5 |
| t21_bfs_not_cheapest | 5 | 4 | 3 | 5 | 5 | 3 | 5 |
| t22_gbfs_trap | 4 | 5 | 4 | 5 | 5 | 6 | 5 |
| t23_inadmissible_h | 4 | 5 | 4 | 4 | 4 | 6 | 4 |
| t24_dfs_dead_end | 33 | 5 | 3 | 3 | 3 | 3 | 3 |
| t25_bfs_wide | 19 | 259 | 19 | 103 | 259 | 25 | 259 |
| t26_grid_5x5 | 41 | 1077 | 22 | 425 | 1933 | 74 | 1933 |
| t27_long_chain | 1500 | 1500 | 1500 | 1500 | 1500 | 1500 | 1500 |
| t28_node_cap | 1000000 | 1000000 | 1000000 | 1000000 | 1000000 | 1000000 | 1000000 |
| t29_decimals | 4 | 4 | 3 | 4 | 4 | 3 | 4 |

Notes on counting: "nodes created" is the origin plus every node added to the frontier. CUS2 (IDA\*) re-creates nodes on each deepening pass, and RESEARCH adds up its uniform-cost runs for every leg.
