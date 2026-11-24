"""Pull the 40 study sites plus their neighbouring signals out of the raw VicRoads volume zips.

Neighbours (feature group "nbr"): signalised intersections (INT) and pedestrian signals (POS)
within NEIGHBOUR_KM of any study site, from the "Victorian Traffic Signals" site list.
Rows are kept per detector, so the same output also feeds the per-detector features ("det").

Usage (from Assignment_2B/):
    python research/extract_raw.py research/cache/vicroads_raw research/cache/filtered_ext
Only zips listed as "done" in <raw_dir>/download.log are processed; finished outputs are skipped,
so it can be rerun as more downloads complete.
"""
import pathlib, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "data"))
import filter_sites                                     # noqa: E402  (data/filter_sites.py)

SIGNALS = ROOT / "cache" / "victorian_traffic_signals.csv"
SIGNALS_URL = ("https://opendata.transport.vic.gov.au/dataset/923af458-363d-469f-bc5e-84746a80b9a2/resource/"
               "1036a318-8867-4dc6-bd84-fc3bcd6cbdb9/download/victorian_traffic_signals.csv")
STUDY = ROOT / "external" / "site_locations.csv"
NEIGHBOURS = ROOT / "external" / "neighbour_sites.csv"
NEIGHBOUR_KM = 1.0


def neighbour_sites():
    """Write external/neighbour_sites.csv and return study + neighbour site numbers."""
    if not SIGNALS.exists():
        import urllib.request
        SIGNALS.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(SIGNALS_URL, SIGNALS)
    s = pd.read_csv(SIGNALS, encoding="utf-8-sig")
    study = pd.read_csv(STUDY)
    p1, p2 = np.radians(s.LATITUDE.to_numpy())[:, None], np.radians(study.lat.to_numpy())[None]
    dlon = np.radians(study.lon.to_numpy())[None] - np.radians(s.LONGITUDE.to_numpy())[:, None]
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2) ** 2
    s["nearest_km"] = (2 * 6371 * np.arcsin(np.sqrt(a))).min(1).round(3)
    nb = s[~s.SITE_NO.isin(study.site) & s.TYPE.isin(["INT", "POS"]) & (s.nearest_km <= NEIGHBOUR_KM)]
    nb = nb.rename(columns={"SITE_NO": "site", "SITE_NAME": "name", "TYPE": "type",
                            "LATITUDE": "lat", "LONGITUDE": "lon"})[["site", "name", "type", "lat", "lon", "nearest_km"]]
    nb.sort_values("site").to_csv(NEIGHBOURS, index=False)
    return set(study.site) | set(nb.site)


def run_one(args):
    src, out_dir, sites = args
    filter_sites.SITES = sites                          # read_csvs filters on this module global
    return str(filter_sites.filter_zip(pathlib.Path(src), pathlib.Path(out_dir)))


if __name__ == "__main__":
    raw_dir, out_dir = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    sites = neighbour_sites()
    print(f"{len(sites)} sites (40 study + {len(sites) - 40} neighbours within {NEIGHBOUR_KM} km)", flush=True)
    log = raw_dir / "download.log"
    done = [l.split(" ", 1)[1].strip() for l in log.read_text().splitlines() if l.startswith("done ")]
    todo = [raw_dir / f for f in sorted(set(done))
            if not (out_dir / (pathlib.Path(f).stem + "_boroondara.csv.gz")).exists()]
    print(f"{len(todo)} zips to process", flush=True)
    with ProcessPoolExecutor(max_workers=6) as pool:
        for out in pool.map(run_one, [(str(z), str(out_dir), sites) for z in todo]):
            print("wrote", out, flush=True)
