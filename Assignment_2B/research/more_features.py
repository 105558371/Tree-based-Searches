"""Extra input features for train.py --features (no packages beyond numpy/pandas).

Global (same value for every site), one value per 15-min step:
    cal    day before / after a Victorian public holiday, bridge day (a working day between two
           non-working days), Christmas-New Year week (24 Dec - 1 Jan), daylight saving in force,
           daylight-saving change day                                                    -> 6 channels
    uni    university semester in session (Swinburne Hawthorn; Deakin Burwood runs similar dates) -> 1

    cyc    sin/cos of time of day, day of week and day of year                             -> 6

Per site:
    works  a major road/rail construction project within WORKS_RADIUS_KM of the site     -> 1
    Standard time-series transforms of the normalised flow (built in train.py with the helpers below):
    wk4 (same slot averaged over 4 weeks), diff (step change, change vs last week),
    roll (24 h rolling mean/std, 7-day rolling mean). All use past values only.

Site coordinates: external/site_locations.csv (official VicRoads "Victorian Traffic Signals" list).
"""
import pathlib
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
SITE_LOCATIONS = ROOT / "external" / "site_locations.csv"
MELBOURNE = ZoneInfo("Australia/Melbourne")
WORKS_RADIUS_KM = 1.5

# Semesters: start on the first Monday of March / August and run ~16 weeks incl. exams.
# Checked against Swinburne's published dates: S1 2026 starts 2 Mar, S2 2025 runs 4 Aug - 22 Nov,
# S2 2026 starts 3 Aug (swinburne.edu.au/courses/intakes, international holiday letter 2026).
# Earlier years follow the same pattern but are not individually verified.
SEMESTER_WEEKS = 16

# Major construction near the 40 sites: (name, lat, lon, start, end, source). Dates are inclusive.
# Locations are the nearest signalised intersection in the VicRoads site list.
WORKS = [
    ("Burke Rd Glen Iris level crossing removal (final works)", -37.8530, 145.0526, "2016-01-01", "2016-01-31",
     "bigbuild.vic.gov.au: boom gates removed and lowered line opened January 2016"),
    ("Toorak Rd Kooyong level crossing removal (final works)", -37.8454, 145.0450, "2020-04-01", "2020-05-31",
     "bigbuild.vic.gov.au: crossing removed April 2020"),
    ("Union Rd / Mont Albert Rd level crossing removals", -37.8205, 145.0983, "2020-12-02", "2023-05-29",
     "bigbuild.vic.gov.au: community update 2 Dec 2020; Union station opened 29 May 2023"),
    ("North East Link - Bulleen Rd interchange", -37.7796, 145.0786, "2022-10-26", "2026-12-31",
     "bigbuild.vic.gov.au: 'Getting ready for tunnelling, Bulleen' 26 Oct 2022; completion 2028"),
    ("Eastern Freeway upgrades, Hoddle St - Burke Rd (Chandler Hwy)", -37.7920, 145.0298, "2025-10-01", "2026-12-31",
     "bigbuild.vic.gov.au: works on Hoddle St - Burke Rd section to begin late 2025"),
    ("Eastern Freeway upgrades, Hoddle St - Burke Rd (Burke Rd ramps)", -37.7853, 145.0635, "2025-10-01", "2026-12-31",
     "bigbuild.vic.gov.au: works on Hoddle St - Burke Rd section to begin late 2025"),
]


def cyclical_features(times):
    """[T, 6] float32: sin/cos of time of day, day of week and day of year."""
    tod = (np.asarray(times.hour) * 60 + np.asarray(times.minute)) / 1440
    dow, doy = np.asarray(times.dayofweek) / 7, (np.asarray(times.dayofyear) - 1) / 365.25
    return np.stack([f(2 * np.pi * a) for a in (tod, dow, doy) for f in (np.sin, np.cos)], axis=1).astype(np.float32)


def rolling_mean_std(x, window):
    """Causal rolling mean and std of x [T, N] over the last `window` steps, current step included."""
    x = x.astype(np.float64)
    c1 = np.vstack([np.zeros((1, x.shape[1])), np.cumsum(x, axis=0)])
    c2 = np.vstack([np.zeros((1, x.shape[1])), np.cumsum(x ** 2, axis=0)])
    hi = np.arange(1, len(x) + 1); lo = np.maximum(hi - window, 0); n = (hi - lo)[:, None]
    mean = (c1[hi] - c1[lo]) / n
    std = np.sqrt(np.maximum((c2[hi] - c2[lo]) / n - mean ** 2, 0))
    return mean.astype(np.float32), std.astype(np.float32)


def seasonal_mean(x, period, k):
    """Row s = mean of x[s - period], ..., x[s - k*period] (e.g. same slot over the last k weeks)."""
    out, cnt = np.zeros(x.shape, dtype=np.float64), np.zeros((len(x), 1))
    for j in range(1, k + 1):
        out[j * period:] += x[:-j * period]; cnt[j * period:] += 1
    return (out / np.maximum(cnt, 1)).astype(np.float32)


def shifted_diff(x, lag):
    """Row s = x[s] - x[s - lag] (0 for the first `lag` rows)."""
    out = np.zeros_like(x)
    out[lag:] = x[lag:] - x[:-lag]
    return out


def _haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))


def _per_step(times, day_values):
    """Spread a per-day array (indexed from the first day) onto the 15-min steps."""
    day_idx = (times.normalize() - times[0].normalize()).days
    return np.asarray(day_values)[day_idx]


def calendar_features(times, public_holiday):
    """times: DatetimeIndex of 15-min steps; public_holiday: bool per step -> [T, 6] float32."""
    days = pd.date_range(times[0].normalize(), times[-1].normalize(), freq="D")
    hol = pd.Series(public_holiday, index=times.normalize()).groupby(level=0).max().reindex(days, fill_value=False)
    hol = hol.to_numpy(bool)
    nonwork = hol | (days.dayofweek >= 5)
    prev = lambda a: np.r_[False, a[:-1]]
    nxt = lambda a: np.r_[a[1:], False]
    day_before, day_after = nxt(hol) & ~hol, prev(hol) & ~hol
    bridge = ~nonwork & prev(nonwork) & nxt(nonwork)
    xmas = ((days.month == 12) & (days.day >= 24)) | ((days.month == 1) & (days.day == 1))
    offset = np.array([MELBOURNE.utcoffset(datetime(d.year, d.month, d.day, 12)).total_seconds() for d in days])
    dst = offset > offset.min()
    dst_change = np.r_[False, offset[1:] != offset[:-1]]
    cols = [day_before, day_after, bridge, xmas, dst, dst_change]
    return np.stack([_per_step(times, c) for c in cols], axis=1).astype(np.float32)


def semester_features(times):
    """[T, 1] float32: 1 inside a university semester (teaching + exams)."""
    days = pd.date_range(times[0].normalize(), times[-1].normalize(), freq="D")
    on = np.zeros(len(days), dtype=bool)
    for year in range(days.year.min(), days.year.max() + 1):
        for month in (3, 8):
            first = pd.Timestamp(year, month, 1)
            start = first + pd.Timedelta(days=(7 - first.dayofweek) % 7)      # first Monday
            on |= (days >= start) & (days < start + pd.Timedelta(weeks=SEMESTER_WEEKS))
    return _per_step(times, on)[:, None].astype(np.float32)


def works_features(times, sites):
    """[T, N] float32: 1 while a WORKS project within WORKS_RADIUS_KM of the site is active."""
    loc = pd.read_csv(SITE_LOCATIONS).set_index("site").loc[sites]
    out = np.zeros((len(times), len(sites)), dtype=np.float32)
    for _, lat, lon, start, end, _ in WORKS:
        near = _haversine_km(loc.lat.to_numpy(), loc.lon.to_numpy(), lat, lon) <= WORKS_RADIUS_KM
        active = (times >= pd.Timestamp(start)) & (times < pd.Timestamp(end) + pd.Timedelta(days=1))
        out[np.ix_(np.asarray(active), near)] = 1.0
    return out


if __name__ == "__main__":
    d = np.load(ROOT / "cache" / "traffic.npz", allow_pickle=True)
    t = pd.DatetimeIndex(d["times"])
    cal = calendar_features(t, d["ext_public_holiday"])
    daily = lambda a: a.reshape(-1, 96)[:, 0].sum() if len(a) % 96 == 0 else a[::96].sum()
    names = ["day before holiday", "day after holiday", "bridge day", "Christmas week", "daylight saving", "DST change day"]
    print(", ".join(f"{n}: {daily(cal[:, i]):.0f} days" for i, n in enumerate(names)))
    print(f"semester: {daily(semester_features(t)[:, 0]):.0f} days")
    w = works_features(t, d["sites"])
    for name, lat, lon, *_ in WORKS:
        loc = pd.read_csv(SITE_LOCATIONS).set_index("site").loc[d["sites"]]
        near = loc.index[_haversine_km(loc.lat.to_numpy(), loc.lon.to_numpy(), lat, lon) <= WORKS_RADIUS_KM].tolist()
        print(f"{name}: sites {near}")
    print(f"works flag on for {100 * w.mean():.1f}% of site-steps")
