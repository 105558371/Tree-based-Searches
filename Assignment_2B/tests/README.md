# Assignment 2B test execution report (with test result for reference)

On 26 November 2026, pytest collected 19 checks across exactly 16 new scenario files. **18 passed, 0 failed and 1 skipped.** Every scenario file had at least one passing check. TC11's real route could not run because neither configured location contains the directed road graph. The protected Assignment 2A runner separately passed 182 of 182 checks.

| Evidence group | Passed | Failed | Skipped | What ran |
|---|---:|---:|---:|---|
| Offline service/data/routing checks | 11 | 0 | 0 | Nine controlled service/graph checks, plus TC01 and TC06 which also read real artifacts |
| Offline Streamlit GUI interactions | 3 | 0 | 0 | Actual pages and widgets, deterministic forecasts, directed fixture graph and temporary metrics JSON |
| Real CPU model integration | 4 | 0 | 0 | LSTM, GRU, Graph WaveNet, and three-seed STAEformer ensemble |
| Real model plus road network | 0 | 0 | 1 | Missing direct road-neighbour CSV for 2000 → 3002 |
| Existing Assignment 2A regression | 182 | 0 | 0 | Read-only original runner and expected.csv |

The new suite ran in the existing Python 3.12 project environment, using CPU PyTorch 2.15.0, NumPy 2.5.5, Pandas 3.1.1, Streamlit 1.67.0 and pytest 9.2.0. The only final warnings are two Pandas deprecations from the existing research cyclical-feature helper. The earlier sandboxed run lacked permission to create pytest's temporary fixtures. The final suites used the normal Windows temporary directory.

## Reproduce from the repository root

```powershell
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install -r Assignment_2B/gui/requirements-gui.txt -r Assignment_2B/tests/requirements-test.txt
# Only when a compatible research cache is absent
& $py -B Assignment_2B/gui/prepare_replay.py
$env:OMP_NUM_THREADS = "2"
& $py -B -m pytest Assignment_2B/tests -m "not integration" -q
& $py -B -m pytest Assignment_2B/tests -m integration -q -rs
$stamp = Get-Date -Format "yyyyMMdd-HHmmss-fff"
& $py -B -m pytest Assignment_2B/tests -q -rs "--junitxml=Assignment_2B/tests/results/junit-$stamp.xml"
```

To reproduce one row, substitute its file for `Assignment_2B/tests` in the pytest command. The recorded per-check outcomes are in [JUnit XML](results/junit-20261126-005209.xml). The matching console evidence is available for the [offline suite](results/not-integration-20261126-005209.txt), [integration suite](results/integration-20261126-005209.txt), and [combined suite](results/all-20261126-005209.txt).

| ID / executable file | Input and expected outcome | Method and actual result | Status |
|---|---|---|---|
| [TC01](test_01_data_validation.py) | Two unordered sites, 96 slots, invalid schema, duplicate rows and negative counts | Fixture sorted to 2000, 3002 and 23:45 mapped to V95. Real October catalog loaded. Research checkpoint rejected for 2006. | PASS |
| [TC02](test_02_forecast_window.py) | First/last valid windows, short history, misaligned time, masked past, modified future targets | Four correct interval boundaries. Invalid windows rejected. Inputs stayed identical after changing future and masked traffic. | PASS |
| [TC03](test_03_lstm_inference.py) | Saved LSTM on an observed held-out window | Strict CPU load, original LSTM architecture, two channels and four outputs for 40 ordered sites. Inverse-scaled values matched research predict. | PASS |
| [TC04](test_04_gru_inference.py) | Same real window, saved GRU | Confirmed GRU dispatch and original-unit agreement with research predict. | PASS |
| [TC05](test_05_third_model_inference.py) | Real Graph WaveNet and final ensemble, incompatible channels, missing member, invalid output | Strict Graph WaveNet matched research predict. Three member forecasts matched the manifest ensemble average. Incompatible dimensions, missing weights, NaNs and invalid output shapes were rejected. | 3 PASS |
| [TC06](test_06_model_evaluation.py) | Actual LSTM test JSON plus masked pairs (5,7), (20,16), excluded (0,500) | All four displayed RMSE values matched the JSON. Missing scores stayed missing. Sample MAE 3, RMSE √10 and MAPE 20%. | PASS |
| [TC07](test_07_travel_time_conversion.py) | 2 km with 0 or 375 vehicles per 15 min | Core returned 150 or 255 seconds according to end-site B. Reversing direction changed cost. Invalid distances and flows rejected. | PASS |
| [TC08](test_08_off_peak_routing.py) | Directed diamond, 03:00 low-flow fixture | Fastest 2000 → 2100 → 3002, 240 s. | PASS |
| [TC09](test_09_morning_peak_routing.py) | Same diamond, high 08:00 flow at 2100 | Fastest changed to 2000 → 2200 → 3002, 270 s. Alternative 345 s. | PASS |
| [TC10](test_10_evening_peak_routing.py) | 17:00 fixture with higher flows at 2200 and 3002 | Recalculated 2000 → 2100 → 3002, 292.5 s. | PASS |
| [TC11](test_11_valid_od.py) | 2000 → 3002 on fixture and real directed graph | Fixture endpoints and distance/time sums passed. Real model-to-road-network run skipped with missing graph path. | 1 PASS, 1 SKIP |
| [TC12](test_12_invalid_od.py) | Blank/unknown IDs, identical endpoints, reverse-only route, missing graph, UCS cap and duplicate edge | Distinct outcomes for invalid input, no travel, no path, unavailable, and incomplete search. | PASS |
| [TC13](test_13_top_k_routes.py) | Cyclic directed graph with tied costs, k=1 or 5, single path and invalid k | Unique simple paths and costs matched NetworkX exhaustive enumeration. Verified calls to original UCS and intact input edges. | PASS |
| [TC14](test_14_gui_route_guidance.py) | AppTest navigation, defaults, swap, preset, k, submit, alternative and dataset change | Exactly three pages. Route selection updated map layer payload, ETA, distance and edge table. Same-site state showed zero ETA. Changing to unavailable data cleared route paths and summary. | PASS |
| [TC15](test_15_gui_traffic_forecast.py) | AppTest selected GRU, site 2100 and +45 then unavailable model | Marker, summary, chart data and steps matched fixture flow. Cached forecast reused for horizon/site changes. Unavailable selection cleared output. | PASS |
| [TC16](test_16_gui_model_evaluation.py) | Temporary JSON, choose two models, RMSE, +45 and change dataset | Table and chart showed 47 for both fixture models. Missing history and unavailable sample were visible. One explicit sample call showed a masked gap. | PASS |

**Additional real GUI smoke.** A separate Streamlit AppTest run used the production services and the real cache. On 2025-01-01 at 08:00, the LSTM's first prediction for SCATS 970 was 135.69 vehicles/15 min versus 117 observed. Both forecast and evaluation pages returned the same four predicted/observed pairs. The route page displayed the missing-road-graph state. This single window supplies functional evidence. The research test scores displayed on Page 3 came from existing JSON. [Console output](results/gui-real-20261126.txt)

**Assignment 2A regression.** From the repository root in PowerShell, the original runner can be reproduced with the following commands. Its working directory is required for the original relative tests path. [182-check console output](results/a2a-20261126.txt)

```powershell
$env:PYTHONPATH = (Resolve-Path .).Path
Push-Location Assignment_2A
..\.venv\Scripts\python.exe -B src/run_tests.py
Pop-Location
```

## Manual browser checklist

| Desktop check | Status | Available evidence |
|---|---|---|
| Zoom and pan the map | NOT RUN | Browser access host unavailable. The existing Edge window was minimized. |
| Hover site and route tooltips | NOT RUN | The marker and selected-line payloads were checked in AppTest. |
| Select and highlight an alternative route visually | NOT RUN | TC14 checked selection, map data and summary using the directed fixture. |
| Inspect desktop layout and unavailable-state rendering | NOT RUN | Real GUI smoke exercised all three pages and the graph warning through Streamlit. |

**Assignment readiness remains blocked.** Supply the directed road-neighbour graph and verified October 2006 checkpoints, training/evaluation provenance, preprocessing and target units. Obtain tutor approval for the third model and verify the conversion against the official Canvas document. The research checkpoint smokes establish 2014–2026 replay. They provide no October 2006 model or real end-to-end ETA evidence.
