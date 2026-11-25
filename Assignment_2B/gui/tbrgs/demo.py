"""Demo and test system: a few sites and synthetic flows (busier at 07:00-09:00 and 16:00-18:00)."""
import math

import pandas as pd

from .backend import TBRGS, build_neighbours

SITES = [(2000, "WARRIGAL_RD/TOORAK_RD", -37.8497, 145.0990), (3002, "DENMARK_ST/BARKERS_RD", -37.8087, 145.0285),
         (9001, "DEMO_SITE_1", -37.8300, 145.0800), (9002, "DEMO_SITE_2", -37.8200, 145.0600),
         (9003, "DEMO_SITE_3", -37.8350, 145.0400), (9004, "DEMO_SITE_4", -37.8150, 145.0450)]
MODEL_FACTOR = {"LSTM": 1.0, "GRU": 1.05, "CNN": 0.95}


def demo_flows(model, when):
    h = when.hour + when.minute / 60
    peak = max(math.exp(-((h - 8) / 1.5) ** 2), math.exp(-((h - 17) / 1.5) ** 2))
    return {s[0]: (150 + 700 * peak + 40 * (i % 3)) * MODEL_FACTOR.get(model, 1.0) for i, s in enumerate(SITES)}


def demo_system(cfg=None, neighbours=None):
    cfg = cfg or {}
    sites = pd.DataFrame(SITES, columns=["site_id", "name", "lat", "lon"])
    nb = neighbours or build_neighbours(sites, None, cfg.get("nearest_neighbours", 3))
    return TBRGS(sites, nb, demo_flows, ["2006-10-30"], cfg.get("intersection_delay_s", 30), cfg.get("speed_limit_kmh", 60))
