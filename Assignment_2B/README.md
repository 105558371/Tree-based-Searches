# TBRGS: Traffic-based Route Guidance System

Predicts traffic flow at 40 Boroondara SCATS sites (LSTM, GRU, GNN), converts predicted flow to
travel time, and runs the Part A search algorithms to return up to 5 routes between two sites.

## Folder structure
```
Assignment_2B/
├── core/
│   ├── coords_mapping.py   # clean SCATS lat/lon, apply map correction, export GUI payload
│   ├── check_map.py        # plots sites on OpenStreetMap to verify alignment
│   ├── travel_time.py      # flow -> speed -> travel time
│   ├── tune.py             # hyperparameter search (lr, hidden units) for LSTM/GRU/GNN
│   ├── sites_aligned.json  # corrected site coordinates
│   └── coord_check.html    # generated map (output of check_map.py)
├── data/                   # site-total CSVs + preprocessing scripts (see data/README.md)
├── models/                 # model definitions
├── outputs/
│   ├── logs/               # best_params.json, tuning_curves.json, tuning_results.csv
│   └── weights/            # saved weights (basic/, research/)
├── gui/                    # GUI
├── tests/                  # test cases and results
├── docs/                   # insights for the report
└── README.md
```

## Setup
```
pip install torch pandas numpy folium
```

## Usage
```
python core/coords_mapping.py          # build corrected coordinates
python core/check_map.py               # open core/coord_check.html to verify
python -m core.tune --sites 10 --days 180 --epochs 10
```
Quick smoke test for tuning: `python -m core.tune --sites 2 --days 30 --epochs 2`

## Key design decisions
- **Coordinates:** dataset lat/lon is shifted from the real intersections. A global correction of
  dlat=+0.0015, dlon=+0.0013 (about 165 m N, 115 m E) was derived from reference intersections
  and checked on further sites (residual mostly under 25 m).
- **Travel time:** speed from the spec's flow-speed curve (60 km/h cap up to 351 veh/h, 32 km/h at
  capacity), plus 30 s per intersection. Flow is the predicted count at the start site, x4 for veh/h.
- **Data:** 2014-2026 VicRoads site totals, 15-min intervals (see `data/README.md`).

## Status
- [x] Data processing
- [x] Coordinate alignment
- [x] Travel-time function
- [x] Tuning script
- [ ] Final training, weights and logs for LSTM, GRU, GNN
- [ ] 15+ test cases
- [ ] GUI and Part A integration

See [Insights](docs/insights.md) for findings.
