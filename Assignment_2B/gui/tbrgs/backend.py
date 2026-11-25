"""System facade used by the GUI and the tests: TBRGS(sites, neighbours, flow_fn).route(...)."""
from pathlib import Path

import pandas as pd

from .config import find_dataset
from .routing import top_k_routes
from .traffic import haversine_km, travel_time_s


def build_neighbours(sites: pd.DataFrame, edges_path=None, nearest=3) -> dict:
    """Connected sites: from edges.csv (columns a,b) if it exists, else each site to its nearest ones."""
    ids = [int(s) for s in sites.site_id]
    nb = {s: set() for s in ids}
    if edges_path and Path(edges_path).exists():
        for a, b in pd.read_csv(edges_path)[["a", "b"]].itertuples(index=False):
            if int(a) in nb and int(b) in nb:
                nb[int(a)].add(int(b)); nb[int(b)].add(int(a))
    else:
        pos = {int(r.site_id): (r.lat, r.lon) for r in sites.itertuples()}
        for a in ids:
            near = sorted((s for s in ids if s != a), key=lambda s: haversine_km(*pos[a], *pos[s]))[:nearest]
            for b in near:
                nb[a].add(b); nb[b].add(a)
    return {a: sorted(v) for a, v in nb.items()}


class TBRGS:
    def __init__(self, sites, neighbours, flow_fn, days, delay_s=30.0, speed_kmh=60.0):
        self.sites, self.neighbours, self.flow_fn, self.days = sites.reset_index(drop=True), neighbours, flow_fn, days
        self.delay_s, self.speed = delay_s, speed_kmh
        self.pos = {int(r.site_id): (float(r.lat), float(r.lon)) for r in self.sites.itertuples()}

    def _km(self, a, b):
        return haversine_km(*self.pos[a], *self.pos[b])

    def route(self, origin, dest, model, when, k=5):
        """Up to k routes, fastest first: dicts with path, coords ([lon, lat]), km, minutes."""
        if not 1 <= k <= 5:
            raise ValueError("Number of routes must be between 1 and 5.")
        for s in (origin, dest):
            if s not in self.pos:
                raise ValueError(f"Unknown SCATS site: {s}")
        if origin == dest:
            raise ValueError("Origin and destination must be different.")
        flows = self.flow_fn(model, when)
        graph = {a: [(b, travel_time_s(self._km(a, b), flows.get(b, 0.0), 0.0 if b == dest else self.delay_s, self.speed))
                     for b in nbrs] for a, nbrs in self.neighbours.items()}
        h = lambda n: self._km(n, dest) / self.speed * 3600
        out = []
        for path, cost in top_k_routes(graph, h, origin, dest, k):
            out.append({"path": tuple(path), "coords": [[self.pos[n][1], self.pos[n][0]] for n in path],
                        "km": sum(self._km(path[i], path[i + 1]) for i in range(len(path) - 1)), "minutes": cost / 60})
        return out


def load_system(cfg, base_dir):
    """Returns (system, notice). Real data if the dataset exists, else a demo system with a notice."""
    base = Path(base_dir)
    data_path = find_dataset(base, cfg)
    if data_path is None:
        from .demo import demo_system
        return demo_system(cfg), "Demo mode: no dataset found in the data folder (copy Scats Data October 2006.xls there), so sites, flows and routes are demo data."
    from .data import load_scats
    from .predict import Predictor
    ds = load_scats(data_path)
    nb = build_neighbours(ds.sites, base / cfg["edges_path"], cfg["nearest_neighbours"])
    pred = Predictor(ds, base / cfg["models_dir"])
    return TBRGS(ds.sites, nb, pred.flows, [d.isoformat() for d in ds.dates], cfg["intersection_delay_s"], cfg["speed_limit_kmh"]), ""
