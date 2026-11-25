"""Build the arrays for feature groups "nbr" and "det" from extract_raw.py output.

Output research/cache/extra.npz, on the same 15-min time axis as cache/traffic.npz:
    nbr_flow  [T, M]     total vehicles per 15 min at M neighbouring intersections, gaps filled
    nbr_mask  [T, M]     True where observed
    nbr_sites [M]
    det       [T, 40, K] the study sites' top-K detectors (by training-period volume), each
                         standardised on the training years, clipped to +-10 sd; unobserved
                         values are 0 (= mean)
    det_ids   [40, K]    detector numbers used (-1 where a site has fewer than K)

Gap filling matches prepare_data.py: gaps up to 1 h interpolated, longer ones from the site's
weekday x time-of-day average over the training years.
Usage (from Assignment_2B/): python research/prepare_extra.py [--types INT]
"""
import argparse, glob, pathlib, sys
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "data"))
import make_site_totals                                 # noqa: E402  (data/make_site_totals.py)

FILTERED = ROOT / "cache" / "filtered_ext"
OUT = ROOT / "cache" / "extra.npz"
TRAIN_END = np.datetime64("2024-01-01")
VCOLS = [f"V{i:02d}" for i in range(96)]
K = 8
DET_CLIP = 10.0      # standardised detector values are clipped to +-10 sd (affects ~0.007% of values)


def fill_gaps(wide, times):
    """Same rule as prepare_data.fill_gaps, on a [T, M] array."""
    df = pd.DataFrame(wide, index=times)
    filled = df.interpolate(limit=4, limit_area="inside")
    train = df.loc[:TRAIN_END - np.timedelta64(15, "m")]
    key = lambda idx: idx.dayofweek * 96 + idx.hour * 4 + idx.minute // 15
    profile = train.groupby(key(train.index)).mean()
    fallback = profile.reindex(key(filled.index)).set_index(filled.index)
    return filled.fillna(fallback).fillna(0).to_numpy(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", default="INT", help="neighbour signal types to use, e.g. INT or INT,POS")
    a = ap.parse_args()
    base = np.load(ROOT / "cache" / "traffic.npz", allow_pickle=True)
    times = pd.DatetimeIndex(base["times"]); study = base["sites"]
    day0, n_days = times[0].normalize(), len(times) // 96
    nb = pd.read_csv(ROOT / "external" / "neighbour_sites.csv")
    nb = nb[nb.type.isin(a.types.split(","))]

    rows = []
    for f in sorted(glob.glob(str(FILTERED / "*.csv.gz"))):
        d = make_site_totals.load(f)                     # parses both date formats, drops duplicates
        d = d.loc[d.NB_SCATS_SITE.isin(set(study) | set(nb.site)), ["NB_SCATS_SITE", "Date", "NB_DETECTOR"] + VCOLS]
        rows.append(d.astype({c: np.float32 for c in VCOLS}))          # ~12M rows for 2014-2026: keep it lean
    d = pd.concat(rows).drop_duplicates(["NB_SCATS_SITE", "Date", "NB_DETECTOR"])
    d = d[(d.Date >= day0) & (d.Date < day0 + pd.Timedelta(days=n_days))]
    print(f"{len(d):,} detector-day rows, {d.Date.min().date()} .. {d.Date.max().date()}")

    # neighbours: site totals with the same fault rules as data/make_site_totals.py
    tot = make_site_totals.site_totals(d[d.NB_SCATS_SITE.isin(nb.site)])
    nbr_sites = np.array(sorted(tot["SCATS Number"].unique()))
    cube = np.full((n_days, len(nbr_sites), 96), np.nan, dtype=np.float32)
    di = (tot.Date - day0).dt.days.to_numpy(); si = np.searchsorted(nbr_sites, tot["SCATS Number"].to_numpy())
    cube[di, si] = tot[VCOLS].to_numpy(np.float32)
    wide = cube.transpose(0, 2, 1).reshape(-1, len(nbr_sites))
    nbr_mask = ~np.isnan(wide)
    print(f"neighbours: {len(nbr_sites)} sites with data, {100 * nbr_mask.mean():.1f}% of values observed")

    # study sites: top-K detectors by mean volume over the training years
    s = d[d.NB_SCATS_SITE.isin(study)].copy()
    V = s[VCOLS].to_numpy(np.float32); V[V < 0] = np.nan
    s["day_mean"] = np.nanmean(V, axis=1)
    train = s[s.Date < TRAIN_END] if (s.Date < TRAIN_END).any() else s     # fallback while only recent data exists
    rank = train.groupby(["NB_SCATS_SITE", "NB_DETECTOR"]).day_mean.mean().sort_values(ascending=False)
    det_ids = np.full((len(study), K), -1)
    for i, site in enumerate(study):
        top = rank.loc[site].index[:K] if site in rank.index.get_level_values(0) else []
        det_ids[i, :len(top)] = sorted(top)
    det = np.full((n_days, len(study), K, 96), np.nan, dtype=np.float32)
    site_i = {x: i for i, x in enumerate(study)}
    k_of = {(study[i], det_ids[i, k]): k for i in range(len(study)) for k in range(K) if det_ids[i, k] >= 0}
    keep = np.array([(r, q) in k_of for r, q in zip(s.NB_SCATS_SITE, s.NB_DETECTOR)])
    sk = s[keep]
    det[(sk.Date - day0).dt.days.to_numpy(), sk.NB_SCATS_SITE.map(site_i).to_numpy(),
        np.array([k_of[(r, q)] for r, q in zip(sk.NB_SCATS_SITE, sk.NB_DETECTOR)])] = V[keep]
    det = det.transpose(0, 3, 1, 2).reshape(-1, len(study), K)            # [T, 40, K]
    tr = times < TRAIN_END
    ref = det[tr] if np.isfinite(det[tr]).any() else det
    mu, sd = np.nanmean(ref, axis=0), np.nanstd(ref, axis=0) + 1e-6
    # clip: a few faulty detectors (e.g. 3685/12, 4262/16) produce spikes of 100-200 sd
    det = np.clip(np.nan_to_num((det - mu) / sd, nan=0.0), -DET_CLIP, DET_CLIP).astype(np.float32)
    print(f"detectors: {np.mean(det_ids >= 0) * 100:.0f}% of {K} slots used per site")

    np.savez_compressed(OUT, nbr_flow=fill_gaps(wide, times), nbr_mask=nbr_mask, nbr_sites=nbr_sites,
                        det=det, det_ids=det_ids)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
