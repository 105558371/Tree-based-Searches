# TBRGS: Traffic-based Route Guidance System (COS30019 Assignment 2B)

Predicts traffic flow at SCATS sites in Boroondara with LSTM, GRU and CNN models, turns the predicted flow
into link travel times, and finds up to five routes with an A* search adapted from Part A. A Streamlit GUI
shows the routes on a map.

## Layout

    app.py            Streamlit GUI
    train.py          trains the models over a hyperparameter grid, writes results/metrics.csv and models/
    plot_results.py   draws the comparison figures from results/metrics.csv
    config.json       defaults (origin, destination, models, k, data paths, training grid)
    tbrgs/            data.py (loading and windows), models.py, predict.py, traffic.py (flow to time),
                      routing.py (top-k A*), backend.py (system used by the GUI and tests), demo.py, config.py
    tests/            cases/*.json (18 test cases), run_tests.py
    data/             put the Boroondara dataset (.xlsx, .xls or .csv) here; optional edges.csv (columns a,b)

## Run (Windows, Python 3.11 or later)

    py -3.11 -m venv .venv
    .venv\Scripts\python.exe -m pip install -r requirements.txt
    .venv\Scripts\python.exe train.py
    .venv\Scripts\python.exe plot_results.py
    .venv\Scripts\python.exe tests\run_tests.py
    .venv\Scripts\python.exe -m streamlit run app.py

If `pip install` fails with "No such file or directory" on a very long path (Windows limit of 260 characters), create the
environment in a short folder instead, for example `py -3.11 -m venv C:\tbrgs_env`, and use
`C:\tbrgs_env\Scripts\python.exe` in place of `.venv\Scripts\python.exe`.

Without the dataset the GUI starts in demo mode (demo sites and flows) and says so.

## Assumptions to check

- Dataset layout: columns `SCATS Number`, `Location`, latitude and longitude, `Date`, `V00` to `V95`.
- The flow to speed constants in `tbrgs/traffic.py` must match Traffic Flow to Travel Time Conversion v1.0.
- Without `data/edges.csv`, each site is linked to its three nearest sites.
