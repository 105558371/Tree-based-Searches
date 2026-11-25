# Tests

Each file in `cases/` describes one test case (ID, scenario, expected result). `run_tests.py` runs all of them:

    python tests/run_tests.py

Statuses: PASS, FAIL, SKIP. SKIP means the needed dataset, trained models or package is missing.
Without the dataset the route and GUI tests run on the built-in demo system; for the report, run the
tests with the real dataset and trained models (`python train.py` first). Results go to `results.csv`.
