"""Clean SCATS lat/long, apply map corrections, and prepare for Streamlit."""
import json
from pathlib import Path
import pandas as pd

DATA = Path("data/site_totals_15min_2014_2026.csv.gz")
OVERRIDES = Path("core/coord_overrides.json")   # manual fixes: {"2000": [lat, lon]}
OUT = Path("core/sites_aligned.json")
BBOX = (-37.87, -37.77, 145.00, 145.13)         # approx Boroondara: lat_min, lat_max, lon_min, lon_max

def load_sites(path=DATA):
    df = pd.read_csv(path, usecols=["SCATS Number", "NB_LATITUDE", "NB_LONGITUDE"])
    s = df.groupby("SCATS Number")[["NB_LATITUDE", "NB_LONGITUDE"]].mean().reset_index()
    s.columns = ["site", "lat", "lon"]
    return s

def fix_signs(s):
    """Melbourne is lat<0, lon>0; fix flipped signs and swapped columns."""
    swap = s.lat.abs() > 90
    s.loc[swap, ["lat", "lon"]] = s.loc[swap, ["lon", "lat"]].to_numpy()
    s["lat"] = -s.lat.abs()
    s["lon"] = s.lon.abs()
    return s

def apply_corrections(s, dlat=0.0, dlon=0.0, overrides_path=OVERRIDES):
    """Global offset first, then per-site manual overrides (found by checking against the map)."""
    s = s.copy()
    s["lat"] += dlat
    s["lon"] += dlon
    if overrides_path.exists():
        for site, (lat, lon) in json.loads(overrides_path.read_text()).items():
            s.loc[s.site == int(site), ["lat", "lon"]] = [lat, lon]
    return s

def flag_outliers(s):
    la0, la1, lo0, lo1 = BBOX
    s["out_of_bounds"] = ~(s.lat.between(la0, la1) & s.lon.between(lo0, lo1))
    return s

if __name__ == "__main__":
    # Apply corrections and flag any points outside the Boroondara bounding box
    s = flag_outliers(apply_corrections(fix_signs(load_sites()), dlat=0.0015, dlon=0.0013))
    # Save to JSON for Streamlit to consume
    s.to_json(OUT, orient="records", indent=2)
    print(f"{len(s)} sites -> {OUT}; out of bounds: {s[s.out_of_bounds].site.tolist()}")
