"""Configuration with built-in defaults; a missing or invalid config.json falls back to them."""
import json
from pathlib import Path

DEFAULTS = {
    "default_origin": 2000, "default_destination": 3002,
    "default_models": ["LSTM", "GRU"], "available_models": ["LSTM", "GRU", "CNN"],
    "k_routes": 5, "speed_limit_kmh": 60, "intersection_delay_s": 30,
    "data_path": "data/Scats Data October 2006.xls",   # Boroondara dataset (xlsx, xls or csv)
    "edges_path": "data/edges.csv",           # optional: columns a,b (SCATS ids of connected sites)
    "nearest_neighbours": 3,                  # used when edges_path does not exist
    "models_dir": "models",
    "google_maps_api_key": "",                # optional: draw on the real Google Maps (needs a key)
    "window": 12, "train_frac": 0.7, "val_frac": 0.1,
    "grid": {"hidden_units": [32, 64], "learning_rate": [0.001, 0.01], "window": [8, 12], "epochs": [10]},
}


def load_config(path) -> dict:
    cfg = dict(DEFAULTS)
    try:
        cfg.update(json.loads(Path(path).read_text(encoding="utf-8")))
    except FileNotFoundError:
        pass
    except json.JSONDecodeError:
        cfg["_warning"] = "config.json is not valid JSON; built-in defaults are used."
    return cfg


def find_dataset(base, cfg):
    """The configured data file if it exists, else a .xlsx, .xls or .csv file in the same folder
    (edges.csv excluded; a file whose name contains "scats data" comes first), else None."""
    path = Path(base) / cfg["data_path"]
    if path.exists():
        return path
    skip = Path(cfg["edges_path"]).name.lower()
    files = [f for pattern in ("*.xlsx", "*.xls", "*.csv") for f in path.parent.glob(pattern) if f.name.lower() != skip]
    files.sort(key=lambda f: ("scats data" not in f.name.lower(), f.name.lower()))
    return files[0] if files else None
