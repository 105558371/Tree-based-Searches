# 2B traffic data (SCATS, Boroondara)

Extra SCATS volume data for the 40 sites in the assignment's October 2006 file.
Source: VicRoads "Traffic Signal Volume Data" (data.vic.gov.au), 1 Jan 2014 – 16 Nov 2026.
No public data exists for 2007–2013.

## Files

| File | What it is |
|---|---|
| `site_totals_15min_2014_2026.csv.gz` | One row per site per day, `V00`–`V95` = total vehicles across all detectors per 15 min (187,657 rows) |
| `site_totals_15min_oct2006.csv.gz` | The assignment's Oct 2006 data in the same layout (approaches summed per site) |
| `filter_sites.py` | Streams the raw VicRoads zips and keeps only the 40 sites |
| `make_site_totals.py` | Builds the site-total files from the filtered data |

Columns: `SCATS Number, NB_LATITUDE, NB_LONGITUDE, Date, N_DETECTORS, N_DEAD_DETECTORS, V00…V95`
(lat/long taken from the 2006 file; the 2006 file has `N_APPROACHES` instead of the detector columns).

## Cleaning already applied

- Dates normalised to `YYYY-MM-DD` (2020 and 2022 had some `D/MM/YYYY` rows)
- Duplicate rows dropped (June 2021 was published twice; 2016-09-25 was in two files)
- `-1` / blank counts are detector faults. A detector faulty all day is left out of that day's
  total (`N_DEAD_DETECTORS`); otherwise a faulty detector leaves that 15-min total blank (~2% of values)

## Still to decide

- How to fill the blank intervals, and the 10 missing days:
  2015-03-06, 2017-07-18, 2019-12-12, 2020-06-03, 2020-08-07, 2020-09-25,
  2022-12-31, 2023-07-31, 2024-09-30, 2026-08-31
- 2006 only covers 1–3 approaches at 15 sites, so their 2006 totals are lower than the new data:
  2820, 2825, 3122, 3126, 3127, 3180, 3685, 4030, 4051, 4262, 4272, 4273, 4324, 4335, 4812
- Detector counts change over time (e.g. 4035 ~4x its 2006 total) — check `N_DETECTORS` before trusting jumps

## Rebuilding

Raw zips (~18 GB) aren't in the repo. Download them from the dataset page, then:

```
python filter_sites.py raw/ filtered/
python make_site_totals.py filtered/ Scats_Data_October_2006.xls site_totals_15min_2014_2026.csv.gz
```
