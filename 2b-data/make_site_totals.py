"""Sum per-detector SCATS counts into one total per site per 15-minute interval,
in the same wide layout as the 2006 assignment data (one row per site per day, V00-V95).

Rules:
- dates in either YYYY-MM-DD or D/MM/YYYY form are normalised
- exact duplicate site/day/detector rows are dropped (keep first)
- -1 and blank counts are faults; a detector faulty for the whole day is excluded
  from that day's total (counted in N_DEAD_DETECTORS); otherwise any faulty
  detector in an interval leaves that site's interval blank rather than undercounted
"""
import glob, sys
import numpy as np
import pandas as pd

VCOLS = [f"V{i:02d}" for i in range(96)]

def load(path):
    d = pd.read_csv(path, low_memory=False, dtype={"QT_INTERVAL_COUNT": str})
    s = d.QT_INTERVAL_COUNT.str.split(" ").str[0]
    dmy = s.str.contains("/")
    day = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    day[dmy] = pd.to_datetime(s[dmy], format="%d/%m/%Y")
    d["Date"] = day
    return d.drop_duplicates(["NB_SCATS_SITE", "Date", "NB_DETECTOR"])

def site_totals(d):
    V = d[VCOLS].to_numpy(float)
    V[V < 0] = np.nan
    dead = np.isnan(V).all(axis=1)
    V[dead] = 0
    keys = [d.NB_SCATS_SITE.to_numpy(), d.Date.to_numpy()]
    tot = pd.DataFrame(V, columns=VCOLS).groupby(keys).sum(min_count=1)
    tot[pd.DataFrame(np.isnan(V), columns=VCOLS).groupby(keys).any().to_numpy()] = np.nan
    meta = pd.DataFrame({"N_DETECTORS": ~dead, "N_DEAD_DETECTORS": dead}).groupby(keys).sum()
    out = meta.join(tot)
    out.index.names = ["SCATS Number", "Date"]
    return out.reset_index()

if __name__ == "__main__":
    filtered_dir, coords_xls, out_path = sys.argv[1:4]
    coords = (pd.read_excel(coords_xls, "Data", header=1)
                .groupby("SCATS Number")[["NB_LATITUDE", "NB_LONGITUDE"]].mean())
    parts = [site_totals(load(f)) for f in sorted(glob.glob(f"{filtered_dir}/*.csv.gz"))]
    out = pd.concat(parts)
    # the same day can appear in two archives (e.g. 2016-09-25); keep the first
    out = out.drop_duplicates(["SCATS Number", "Date"]).sort_values(["SCATS Number", "Date"])
    out = out.join(coords, on="SCATS Number")
    cols = ["SCATS Number", "NB_LATITUDE", "NB_LONGITUDE", "Date",
            "N_DETECTORS", "N_DEAD_DETECTORS"] + VCOLS
    out[cols].to_csv(out_path, index=False, date_format="%Y-%m-%d", compression="gzip")
    print(out_path, len(out), "rows")
