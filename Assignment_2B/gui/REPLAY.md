# TBRGS historical replay

Launch the three-page application from the repository root in PowerShell, using the project environment. The original GUI dependencies are already installed in this checkout. The additions are `holidays` for offline calendar preparation and `pytest` for the test suite.

```powershell
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install -r Assignment_2B/gui/requirements-gui.txt -r Assignment_2B/tests/requirements-test.txt
& $py -m streamlit run Assignment_2B/gui/app.py --server.address=127.0.0.1
```

On a fresh environment, install the existing `Assignment_2B/gui/requirements.txt` first. The original prototype and its existing tests are preserved. Its separate launch command is `& $py -m streamlit run Assignment_2B/gui/app.py -- --prototype`. Its documented demo and nearest-neighbour behavior apply only to that entry point.

## What each page uses

- **Route Guidance** uses the selected model's first output for the departure slot. Select up to five routes, swap the endpoints or compare model ETAs. Blue marks the selected route and gray marks alternatives. The panel lists the directed edges, destination counts, road kilometres, estimated speeds and seconds. The default map uses local coordinates. “Load online street map” enables the attributed street basemap.
- **Traffic Forecast** runs one CPU inference window on submission. Select a site or a horizon to inspect the same result. Map thresholds stay fixed across dates and models. Green represents fewer than 150 vehicles per site per 15 minutes, amber represents 150 to below 375, and red represents 375 or more. These are descriptive volume bands. The chart shows observed history and the predictions for the following slots. Missing observations appear as gaps.
- **Model Evaluation** reads final test JSON files and can replay one sample against recorded observations. The default comparison includes LSTM, GRU, Graph WaveNet and the final three-seed STAEformer ensemble. Advanced experiments include individual seed runs and evaluation-only boosted trees. Fold scores and tuning records use different protocols and are excluded. The October 2006 view exposes the older prototype grid CSV separately because that file lacks dataset provenance.

## Artifacts and configuration

`config.json` supplies existing origin, destination and route-count defaults. `replay.json` selects the dataset, model, forecast time, presets, default horizon and artifact search paths relative to the GUI directory. The initial research selection uses 08:00 on the first recorded held-out date. A selection change never writes either configuration file.

**Traffic cache.** The app reads an existing compatible research cache or the GUI-owned cache. To prepare the latter explicitly from local files, run the following once. The command requires the processed research CSV and the bundled hourly weather CSV. It uses the research filling, Gaussian adjacency and external feature routines. Publication refuses to replace an existing file. The ensemble also requires the local MCG events CSV.

```powershell
& $py -B Assignment_2B/gui/prepare_replay.py
```

This checkout's GUI cache covers 451,488 consecutive quarter-hours across 40 sites. The model adjacency retains the coordinates used for training. For display, the catalog uses the official location by ID for the flagged outlier SCATS 4266, as documented in `research/EXPERIMENTS.md`. The map labels use site IDs to avoid the conflicting names reported for some intersections.

**Road network.** Request the verified direct road-neighbour export from the graph owner and place a copy in `Assignment_2B/gui/cache/road_edges.csv`, or use the existing research export listed in `replay.json`. The required columns are `from_site,to_site,road_km`. Optional `geometry` is a quoted GeoJSON LineString in [longitude, latitude] order, from A to B, with endpoints within 100 m of the displayed sites. Plain edges appear dashed. Check the documented 4266 coordinate issue with the owner before producing an OSM bounding box. Graph and weather downloads are unnecessary for normal startup.

**October 2006 integration.** The processed assignment dataset is available for catalog browsing. The model owner must supply trained weights, exact architecture and scaling, site order, target definitions and horizons, held-out observations/mask, and evaluation results with dataset provenance. The prototype trains on approach averages while the provided processed files contain site totals. Confirm the training target before enabling a 2006 adapter. The GUI calls `model_choices(dataset)`, `availability(dataset, run, cfg)`, `horizons(dataset, run, cfg)` and `forecast(dataset, run, origin, cfg)` in `services/model_service.py`. A compatible adapter returns `Forecast` with its actual number of steps, ordered site IDs, original-unit [step, site] predictions, masked actuals and observed history. Add dataset-specific recorded scores to `metrics_service.records` only after the protocol is verified.

## Replay and routing assumptions

Times use the dataset's naive Australia/Melbourne wall clock. Every recorded day has 96 labeled slots, including DST transition days. No machine-timezone conversion occurs. At the forecast boundary 08:00, input history ends before 08:00. The +15 prediction covers 08:00–08:15 and +60 covers 08:45–09:00. Route departure at 08:00 uses +15.

Base models consume flow and time of day over 12 slots. Advanced final seeds also need four weeks of prior history. Scaling uses the original research training years. For replay, a missing measured input uses its training weekday/slot profile (zero only where the training profile itself has no observation), avoiding interpolation from a later target. A wholly missing recent input window is unavailable. Actual traffic after the cutoff is used only for display. Archived weather and recorded AFL attendance in the advanced recipe make this a retrospective comparison. Recorded full-test scores retain their original preprocessing and observation mask.

The ensemble reconstructs all three final seeds from its saved manifest and averages their nonnegative predictions after per-site inverse scaling. The base checkpoint tests compare single-window outputs to the protected research batch/predict code on fully observed inputs. No training, tuning or full-period inference is part of startup, replay, or these tests.

For A→B, the adapter passes B's 15-minute count directly to the original `core.travel_time.travel_time_seconds`. The core estimates hourly volume by multiplying by four, applies its speed curve and includes one 30-second intersection delay per edge. The same flow snapshot applies throughout the trip. Yen's spur search calls the original Part A uniform-cost tree search on directed graph copies. The node cap produces an incomplete-search message. Valid identical endpoints produce a zero-distance, zero-time result. Times shown are predictions. The official conversion document, tutor approval for the third model and measured travel-time validation remain outstanding.

The scenario matrix, commands and execution evidence are in [the test report](../tests/README.md).
