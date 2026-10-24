"""Checks every method on every test against tests/expected.csv. --table prints node counts."""
import csv
import os
import sys
import time
from parser import parse_graph
from search import METHODS

TEST_DIR = "tests"
EXPECTED_FILE = os.path.join(TEST_DIR, "expected.csv")


def load_expected():
    expected = {}
    with open(EXPECTED_FILE, newline="") as file:
        for row in csv.DictReader(file):
            expected[(row["test"], row["method"])] = (row["goal"], row["nodes_created"], row["path"])
    return expected


def run_one(filename, method):
    nodes, edges, origin, destinations = parse_graph(filename)
    search, _ = METHODS[method]
    path, nodes_created = search(nodes, edges, origin, destinations)
    if not path:
        return "NONE", str(nodes_created), ""
    return str(path[-1]), str(nodes_created), " ".join(str(node) for node in path)


def main():
    expected = load_expected()
    tests = sorted(name for name in os.listdir(TEST_DIR) if name.endswith(".txt"))
    results = {}
    failures = []

    for test in tests:
        for method in METHODS:
            start = time.perf_counter()
            actual = run_one(os.path.join(TEST_DIR, test), method)
            seconds = time.perf_counter() - start
            results[(test, method)] = actual
            want = expected.get((test, method))
            status = "PASS" if want == actual else ("NEW " if want is None else "FAIL")
            if status != "PASS":
                failures.append((test, method, want, actual))
            print(f"{status}  {test:<28} {method:<9} goal={actual[0]:<5} nodes={actual[1]:<8} {seconds:6.2f}s")

    print()
    for test, method, want, actual in failures:
        print(f"{test} {method}: expected {want}, got {actual}")
    print(f"{len(results) - len(failures)} of {len(results)} checks passed")

    if "--table" in sys.argv:
        print("\nNodes created (lower is better):")
        print("| Test | " + " | ".join(METHODS) + " |")
        print("|---|" + "---|" * len(METHODS))
        for test in tests:
            cells = [results[(test, method)][1] for method in METHODS]
            print(f"| {test[:-4]} | " + " | ".join(cells) + " |")

    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
