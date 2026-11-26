"""Road network between the 40 SCATS sites, from OpenStreetMap (drivable roads only).

For every ordered pair of sites it finds the shortest drivable route (one-way streets respected,
so A->B and B->A can differ). A pair is a *direct* road neighbour when that route does not pass
through another site on the way. Direct neighbours are the natural edges for the Part A search
graph and a road-based adjacency for the GNNs (instead of straight-line distance).

Needs osmnx + scikit-learn (not in requirements.txt; only needed to rebuild the CSVs).
Site coordinates: core/sites_aligned.json (offset-corrected by coords_mapping.py).

Outputs (research/external/):
    road_distance_km.csv  40 x 40 shortest road distance, row = from, column = to
    road_edges.csv        direct neighbours: from_site, to_site, road_km, straight_km
"""
import json, math, pathlib
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
SITES = ROOT.parent / "core" / "sites_aligned.json"
OUT = ROOT / "external"
CACHE = ROOT / "cache" / "boroondara_drive.graphml"
MARGIN_DEG = 0.01          # ~1 km of extra road network around the sites
PASS_THROUGH_M = 80        # a route passing this close to another site goes through it


def haversine_m(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371000 * np.arcsin(np.sqrt(a))



def load_network(lats, lons):
    print("Latitude min:", lats.min(), flush=True)
    print("Latitude max:", lats.max(), flush=True)
    print("Longitude min:", lons.min(), flush=True)
    print("Longitude max:", lons.max(), flush=True)

    bbox = (
        lons.min() - MARGIN_DEG,
        lats.min() - MARGIN_DEG,
        lons.max() + MARGIN_DEG,
        lats.max() + MARGIN_DEG
    )

    print("Bounding box:", bbox, flush=True)

    if not (
        -38.1 < lats.min() <= lats.max() < -37.6
        and 144.8 < lons.min() <= lons.max() < 145.4
    ):
        raise ValueError("Invalid SCATS coordinates for Boroondara")

    if CACHE.exists():
        print("Loading cached road graph...")
        return ox.load_graphml(CACHE)

    ox.settings.log_console = True
    ox.settings.use_cache = True

    print("Downloading road network...", flush=True)

    g = ox.graph_from_bbox(bbox, network_type="drive")

    CACHE.parent.mkdir(exist_ok=True)
    ox.save_graphml(g, CACHE)

    print("Road network downloaded successfully!")

    return g


def main():
    sites = pd.DataFrame(json.load(open(SITES))).sort_values("site").reset_index(drop=True)
    lats, lons = sites.lat.to_numpy(), sites.lon.to_numpy()
    g = load_network(lats, lons)
    nodes = ox.distance.nearest_nodes(g, lons, lats)
    snap_m = [haversine_m(la, lo, g.nodes[n]["y"], g.nodes[n]["x"]) for la, lo, n in zip(lats, lons, nodes)]
    print(f"road network: {g.number_of_nodes():,} nodes, {g.number_of_edges():,} edges | "
          f"sites snapped within {max(snap_m):.0f} m (median {np.median(snap_m):.0f} m)")

    n = len(sites)
    dist = np.full((n, n), np.nan)
    edges = []
    for i in range(n):
        lengths, paths = nx.single_source_dijkstra(g, nodes[i], weight="length")
        for j in range(n):
            if i == j or nodes[j] not in lengths:
                continue
            dist[i, j] = lengths[nodes[j]] / 1000
            path = paths[nodes[j]]
            py = np.array([g.nodes[p]["y"] for p in path]); px = np.array([g.nodes[p]["x"] for p in path])
            others = [k for k in range(n) if k not in (i, j)]
            through = any(haversine_m(lats[k], lons[k], py, px).min() < PASS_THROUGH_M for k in others)
            if not through:
                edges.append({"from_site": sites.site[i], "to_site": sites.site[j],
                              "road_km": round(dist[i, j], 3),
                              "straight_km": round(haversine_m(lats[i], lons[i], lats[j], lons[j]) / 1000, 3)})

    OUT.mkdir(exist_ok=True)
    pd.DataFrame(dist.round(3), index=sites.site, columns=sites.site).to_csv(OUT / "road_distance_km.csv")
    e = pd.DataFrame(edges)
    e.to_csv(OUT / "road_edges.csv", index=False)
    deg = e.groupby("from_site").size().reindex(sites.site, fill_value=0)
    print(f"direct road neighbours: {len(e)} directed edges | per site min {deg.min()} median {deg.median():.0f} "
          f"max {deg.max()} | road/straight distance ratio median {(e.road_km / e.straight_km).median():.2f}")


if __name__ == "__main__":
    main()
