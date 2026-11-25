"""Find sensor faults in the 40 study sites' 15-min totals (values that are observed but not real traffic).

The rules use only the past (previous 4 weeks) and the other sites at the same moment, so the cleaned series
could be produced live and cleaning the model inputs leaks nothing from the future.

    ref4, ref26  median of site s at the same weekday and time over the previous 4 weeks (+-15 min, 12 values)
                 and over the previous 26 weeks
    area[t]      median over sites of flow / ref at time t  (holidays, lockdowns, storms, events: all sites move)
    expect       area * ref, for each reference;  hi / lo = the larger / smaller of the two expectations
                 (so a reference period that was itself broken cannot cause false alarms)

Fault rules (a flagged value becomes a gap: mask False, filled like every other gap):
    spike      flow > 3 * hi and flow - hi > 100 and flow > 1.5 * the site's 99th percentile for that weekday
               and time over the training years (2014-2023): physically implausible jump at one site. The
               last condition stops a site that read low for months from being flagged when it recovers.
    stuck      the same non-zero value 8+ times in a row (2 h) where lo > 20
    zeros      0 for 4+ steps in a row (1 h) where lo > 50 (a dead site reporting zeros)
    dead_det   a site-day whose total excludes dead detectors (N_DEAD_DETECTORS > 0) and is < 85 % of
               lo over the day: the dead detector(s) carried real traffic, so the day is undercounted
"""
import pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
SITE_TOTALS = ROOT.parent / "data" / "site_totals_15min_2014_2026.csv.gz"
WEEK = 672
TRAIN_END = pd.Timestamp("2024-01-01")


def run_lengths(eq):
    """eq [T, N] bool: for each True cell, the length of the run of consecutive True values it belongs to."""
    out = np.zeros(eq.shape, dtype=np.int32)
    for j in range(eq.shape[1]):
        e = np.concatenate([[False], eq[:, j], [False]]).astype(np.int8)
        d = np.diff(e); starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
        for a, b in zip(starts, ends):
            out[a:b, j] = b - a
    return out


def ref_4wk(wide):
    """median of the same weekday and time (+-15 min) over the previous 4 weeks (12 values)."""
    T, N = wide.shape
    past = [np.vstack([np.full((w * WEEK + k, N), np.nan), wide[:-(w * WEEK + k)]])
            for w in range(1, 5) for k in (-1, 0, 1)]
    with np.errstate(all="ignore"):
        return np.nanmedian(np.stack(past), axis=0)


def ref_26wk(wide):
    """median of the same weekday and time over the previous 26 weeks (at least 8 observed)."""
    T, N = wide.shape
    W = -(-T // WEEK)
    pad = np.full((W * WEEK, N), np.nan); pad[:T] = wide
    cube = pd.DataFrame(pad.reshape(W, WEEK * N))                      # one row per week
    med = cube.rolling(26, min_periods=8).median().shift(1).to_numpy()
    return med.reshape(W * WEEK, N)[:T]


def slot_ceiling(wide, times, q=99):
    """per site and weekday x time-of-day: q-th percentile over the training years (2014-2023)."""
    key = (times.dayofweek * 96 + times.hour * 4 + times.minute // 15).to_numpy()
    tr = times < TRAIN_END
    out = np.full((WEEK, wide.shape[1]), np.nan)
    for k in range(WEEK):
        with np.errstate(all="ignore"):
            out[k] = np.nanpercentile(wide[tr & (key == k)], q, axis=0)
    return out[key]


def expectation(wide, ref):
    """ref scaled by the area-wide factor at each time (median over sites of flow / ref)."""
    with np.errstate(all="ignore"):
        area = np.nanmedian(np.where(ref > 20, wide / ref, np.nan), axis=1)
    return np.where(np.isfinite(area), area, 1.0)[:, None] * ref


def find_faults(wide, times, sites):
    """wide [T, N] float with NaN for missing; returns dict of [T, N] bool fault masks."""
    T, N = wide.shape
    e4, e26 = expectation(wide, ref_4wk(wide)), expectation(wide, ref_26wk(wide))
    # a broken reference period (e.g. a site reading near zero for weeks) must not create false alarms:
    # high-side rules use the larger expectation, low-side rules the smaller one
    hi, lo = np.fmax(e4, e26), np.fmin(e4, e26)
    ok = np.isfinite(wide) & np.isfinite(e4) & np.isfinite(e26)
    expect = hi

    f = {}
    ceil = slot_ceiling(wide, times)
    with np.errstate(invalid="ignore"):
        f["spike"] = ok & (wide > 3 * hi) & (wide - hi > 100) & ~(wide <= 1.5 * ceil)

    same = np.zeros((T, N), dtype=bool)
    same[1:] = (wide[1:] == wide[:-1]) & (wide[1:] > 0)
    same[:-1] |= same[1:]                                              # mark both ends of each equal pair
    f["stuck"] = (run_lengths(same) >= 8) & ok & (lo > 20)

    f["zeros"] = (run_lengths(np.nan_to_num(wide, nan=-1) == 0) >= 4) & ok & (lo > 50)

    meta = pd.read_csv(SITE_TOTALS, usecols=["SCATS Number", "Date", "N_DEAD_DETECTORS"], parse_dates=["Date"])
    dead = meta[meta.N_DEAD_DETECTORS > 0]
    day0 = times[0].normalize()
    f["dead_det"] = np.zeros((T, N), dtype=bool)
    col = {s: j for j, s in enumerate(sites)}
    for r in dead.itertuples():
        if r._1 not in col:
            continue
        a = (r.Date - day0).days * 96
        if a < 0 or a + 96 > T:
            continue
        j = col[r._1]
        v, e = wide[a:a + 96, j], lo[a:a + 96, j]
        m = np.isfinite(v) & np.isfinite(e)
        if m.sum() >= 48 and v[m].sum() < 0.85 * e[m].sum():
            f["dead_det"][a:a + 96, j] = np.isfinite(v)
    return f, expect


if __name__ == "__main__":
    d = np.load(ROOT / "cache" / "traffic.npz", allow_pickle=True)
    times = pd.DatetimeIndex(d["times"]); sites = d["sites"]
    wide = np.where(d["mask"], d["flow"], np.nan).astype(np.float64)
    f, expect = find_faults(wide, times, sites)
    obs = np.isfinite(wide).sum()
    anyf = np.zeros_like(d["mask"])
    for k, v in f.items():
        anyf |= v
        print(f"{k:9s} {v.sum():9,d} values ({100 * v.sum() / obs:.3f}% of observed)  sites affected {int((v.sum(0) > 0).sum())}")
    print(f"total     {anyf.sum():9,d} values ({100 * anyf.sum() / obs:.3f}%)")
    yr = times.year.to_numpy()
    print("by year %:", {y: round(100 * anyf[yr == y].sum() / max(np.isfinite(wide[yr == y]).sum(), 1), 2) for y in range(2014, 2027)})
    j = list(sites).index(2827); s = (times >= "2026-09-17") & (times < "2026-09-26")
    print("2827 Sep 17-25 2026 flagged:", int(anyf[s, j].sum()), "of", int(np.isfinite(wide[s, j]).sum()), "observed")
    np.save(ROOT / "cache" / "fault_mask.npy", anyf)
