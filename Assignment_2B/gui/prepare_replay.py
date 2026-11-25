"""Explicit offline preparation of a new GUI-owned cache. Never replaces an existing cache."""
import sys
import tempfile
import os
from pathlib import Path

import numpy as np
import pandas as pd

from services.data_service import GUI, ROOT, DATASETS, RESEARCH, wide_counts


def main():
    output = GUI / "cache/traffic.npz"
    if output.exists():
        raise SystemExit(f"Already exists: {output}")
    weather = ROOT / "research/external/weather_hourly_boroondara.csv.gz"
    if not weather.is_file():
        raise SystemExit(f"Supply local weather first: {weather}")
    # The original preparation script uses sibling imports. Scope them to this CLI import.
    sys.path.insert(0, str(ROOT / "research"))
    try:
        import external_features
        from prepare_data import fill_gaps, gaussian_adjacency
    finally:
        sys.path.pop(0)
    rows = pd.read_csv(ROOT / "data" / DATASETS[RESEARCH], parse_dates=["Date"])
    wide = wide_counts(rows)
    coords = rows.groupby("SCATS Number")[["NB_LATITUDE", "NB_LONGITUDE"]].first().loc[wide.columns]
    del rows
    lat, lon = coords.to_numpy().T
    from unittest.mock import patch
    with patch.object(external_features, "fetch_weather", lambda *_: pd.read_csv(weather, parse_dates=["time"])):
        ext = {f"ext_{k}": v for k, v in external_features.build_features(wide.index).items()}
    output.parent.mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".npz", delete=False) as tmp:
        temp = Path(tmp.name)
    try:
        np.savez_compressed(temp, flow=fill_gaps(wide).to_numpy(np.float32), mask=wide.notna().to_numpy(),
                            times=wide.index.to_numpy(), sites=wide.columns.to_numpy(), lat=lat, lon=lon,
                            adj=gaussian_adjacency(lat, lon), **ext)
        # Hard-link publication is atomic and fails if another process has already finished.
        os.link(temp, output)
    finally:
        temp.unlink(missing_ok=True)
    print(f"{output.name}: {len(wide)} quarter-hours, {len(wide.columns)} sites")


if __name__ == "__main__":
    main()
