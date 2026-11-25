"""Train and evaluate one spatio-temporal GNN on the shared traffic dataset.

Task: from the last 12 steps (3 h) of flow at all 40 sites, predict the next
4 steps (15, 30, 45, 60 min). Train 2014-2023, validate 2024, test 2025-2026.
Loss and metrics only count values that were really observed (mask).

--features ext adds day of week, public/school holidays, COVID lockdowns, rain and temperature
(external_features.py) as extra input channels; results are saved with an "_ext" suffix.
More groups can be added with "+", e.g. --features ext+lag+fill (saved as "_ext-lag-fill"):
    lag    flow at the same time last week and yesterday (per site)
    fill   1 where an input value was gap-filled rather than observed (per site)
    cal    holiday-adjacent days, bridge days, Christmas week, daylight saving (more_features.py)
    uni    university semester in session (more_features.py)
    works  major road/rail construction near the site (per site, more_features.py)
    wk4, diff, roll, cyc   standard time-series transforms (more_features.py)
    nbr    neighbouring intersections as extra input nodes (cache/extra.npz from prepare_extra.py;
           loss and metrics stay on the 40 study sites; not for gwn/stgcn)
    det    the study sites' top-8 detectors as extra per-site channels (cache/extra.npz)
    event  AFL / cricket at the MCG and the Australian Open (events.py), aligned with the target
           slots because fixtures are published in advance (AFL crowd is the actual attendance)
    llm    features selected by the LLM feedback loop (fe_loop.py; list in fe_loop/selected.json)
    tsf    8 tsfresh features of each site's last 24 h (tsfresh_features.py -> cache/tsf.npy);
           --tsf-cols 0,5 keeps only those columns (order of tsfresh_features.NAMES)

Usage: python train.py --model {gwn,stgcn,agcrn,mtgnn,stid,staeformer} [--features {base,ext}] [--epochs 30]
Writes outputs/weights/research/<model>[_ext][_tag].pt and outputs/logs/research/<model>[_ext][_tag]_metrics.json.

Optional recipe flags (defaults reproduce the original runs):
    --schedule cosine   linear warmup (--warmup-epochs) then cosine decay of the LR to 0
    --val-stride 1      validate on every 2024 window instead of every 8th
    --in-steps N        history length in 15-min steps (test targets stay the same)
    --seed, --tag       repeat runs without overwriting each other
    --clean             treat the sensor faults found by clean_faults.py (cache/fault_mask.npy) as gaps:
                        excluded from loss and metrics, filled like other gaps in the inputs ("_clean" in the name)
    --hp k=v,...        STAEformer size/regularisation, e.g. --hp ff=512,layers=4,dropout=0.05
    --weight-decay      Adam L2 penalty (default 1e-4)
    --train-end DATE    train on windows before DATE (default 2024-01-01). After 2024-01-01 the 2024
                        windows are also trained on (2025-01-01 = refit on 2014-2024): validation is then
                        in-sample, so the last epoch is kept instead of the best one
    --init RUN          start from that run's weights (fine-tuning) instead of a random init
    --fold k/K          blocked K-fold cross-validation on all of 2014-2024: fold k (0-based) validates on the
                        k-th of K equal consecutive blocks and trains on the rest. Training windows whose
                        targets or input history (incl. lag/wk4 features) touch the held-out block are dropped.
                        The test period (2025-2026) is never used; average the K fold models with ensemble.py
"""
import argparse, json, math, pathlib, time
import numpy as np
import pandas as pd
import torch
import events
import more_features
from graph_wavenet import GraphWaveNet, transition
from gnn_models import AGCRN, MTGNN, STGCN
from more_models import STAEformer, STID

ROOT = pathlib.Path(__file__).resolve().parent
WEIGHTS = ROOT.parent / "outputs" / "weights" / "research"
LOGS = ROOT.parent / "outputs" / "logs" / "research"
IN_STEPS, HORIZON = 12, 4
CLEAN = False           # --clean: sensor faults (cache/fault_mask.npy) become gaps
VAL_START, TEST_START = np.datetime64("2024-01-01"), np.datetime64("2025-01-01")
TRAIN_END = VAL_START   # --train-end: training windows have their first target before this
MODEL_HP = {}           # --hp: STAEformer keyword arguments
FOLD = None             # --fold: (k, K) for blocked K-fold cross-validation on 2014-2024
TSF_COLS = ""           # --tsf-cols: which tsfresh columns the "tsf" group uses ("" = all)


EXT_FLAGS = ["public_holiday", "school_holiday", "lockdown"]
# --features is "base" or "+"-joined groups, e.g. "ext+lag+fill". Any group beyond ext needs ext,
# because STAEformer reads day of week from the first ext channel.
FEATURE_GROUPS = ["ext", "lag", "fill", "cal", "uni", "works", "wk4", "diff", "roll", "cyc", "nbr", "det", "event", "llm", "tsf"]
GRAPH_MODELS = {"gwn", "stgcn"}     # use the 40-site distance graph, so they can't take "nbr" nodes
LAGS = [672, 96]        # same time last week, same time yesterday (15-min steps)
WEEK = 672
HISTORY = {"lag": WEEK, "wk4": 4 * WEEK, "diff": WEEK, "roll": WEEK, "tsf": 96}   # steps of history each group needs


def parse_features(features):
    groups = [] if features == "base" else features.split("+")
    bad = [g for g in groups if g not in FEATURE_GROUPS]
    if bad or (groups and groups[0] != "ext"):
        raise SystemExit(f"--features must be 'base' or 'ext[+group...]' with groups from {FEATURE_GROUPS}")
    return groups


def parse_hp(text):
    """'ff=512,dropout=0.05' -> {'ff': 512, 'dropout': 0.05}"""
    num = lambda v: float(v) if any(c in v for c in ".e") else int(v)
    return {k: num(v) for k, v in (kv.split("=") for kv in text.split(","))} if text else {}


def fill_gaps(wide):
    """Same rule as prepare_data.fill_gaps: gaps up to 1 h interpolated, longer ones from the site's
    weekday x time-of-day average over the training years."""
    filled = wide.interpolate(limit=4, limit_area="inside")
    train = wide.loc[:VAL_START - np.timedelta64(15, "m")]
    key = lambda idx: idx.dayofweek * 96 + idx.hour * 4 + idx.minute // 15
    profile = train.groupby(key(train.index)).mean()
    return filled.fillna(profile.reindex(key(filled.index)).set_index(filled.index)).fillna(0)


def load(device, features="base"):
    groups = parse_features(features)
    d = np.load(ROOT / "cache" / "traffic.npz", allow_pickle=True)
    flow, mask, times = d["flow"], d["mask"], d["times"].astype("datetime64[m]")
    if CLEAN:   # sensor faults become gaps: no loss or metric on them, inputs filled as in prepare_data.py
        mask = mask & ~np.load(ROOT / "cache" / "fault_mask.npy")
        wide = pd.DataFrame(np.where(mask, flow, np.nan), index=pd.DatetimeIndex(d["times"]))
        flow = fill_gaps(wide).to_numpy(np.float32)
    observed, n_study = mask, flow.shape[1]
    extra = np.load(ROOT / "cache" / "extra.npz") if {"nbr", "det"} & set(groups) else None
    if "nbr" in groups:
        # neighbouring intersections become extra input nodes; their targets are masked out,
        # so loss and metrics still cover only the 40 study sites
        flow = np.concatenate([flow, extra["nbr_flow"]], axis=1)
        observed = np.concatenate([mask, extra["nbr_mask"]], axis=1)
        mask = np.concatenate([mask, np.zeros_like(extra["nbr_mask"])], axis=1)
    pad = lambda a: np.pad(a, [(0, 0), (0, flow.shape[1] - a.shape[1])])   # study-only array -> all nodes

    # window t uses inputs [t-12, t) and targets [t, t+4); split by first target time.
    # Windows start once every group has the history it needs, so lagged indices stay >= 0.
    first = IN_STEPS + max([HISTORY.get(g, 0) for g in groups] + [0])
    starts = np.arange(first, len(flow) - HORIZON + 1)
    split = {"train": starts[times[starts] < TRAIN_END],
             "val": starts[(times[starts] >= VAL_START) & (times[starts] < TEST_START)],
             "test": starts[times[starts] >= TEST_START]}
    train_rows = times < VAL_START
    if FOLD:    # blocked K-fold over every window, 2014 to Nov 2026
        k, K = FOLD
        edges = np.linspace(0, len(starts), K + 1).astype(int)
        val = starts[edges[k]:edges[k + 1]]
        # purge: a training window's targets must end before the block, and its inputs
        # (back to `first` steps, which covers lag/wk4) must start after the block's last target
        keep = (starts + HORIZON <= val[0]) | (starts - first > val[-1] + HORIZON)
        # "test" = the held-out windows in the original test period (2025-2026), if this block has any
        split = {"train": starts[keep], "val": val, "test": val[times[val] >= TEST_START]}
        train_rows = np.ones(len(times), bool)
        train_rows[val[0]:val[-1] + HORIZON] = False      # scaling statistics from the training blocks only
        print(f"fold {k}/{K}: held out {times[val[0]]} -> {times[val[-1]]}")
    mean, std = flow[train_rows].mean(0), flow[train_rows].std(0) + 1e-6   # per site
    std = np.where(std < 1, 1.0, std)                  # guard for nodes with (almost) no training data
    tod = ((times - times.astype("datetime64[D]")).astype(int) / 1440.0).astype(np.float32)
    t = lambda a: torch.as_tensor(a, device=device)
    norm = ((flow - mean) / std).astype(np.float32)
    data = dict(norm=t(norm), flow=t(flow), mask=t(mask),
                tod=t(tod), mean=t(mean.astype(np.float32)), std=t(std.astype(np.float32)))
    # per-site channels as (array [T, N], offset): batch() reads array[input step + offset].
    # Offset HORIZON - lag lines the last 4 input steps up with the 4 target slots one lag earlier.
    site = []
    if "lag" in groups:
        site += [(data["norm"], HORIZON - lag) for lag in LAGS]
    if "fill" in groups:
        site.append((t((~observed).astype(np.float32)), 0))   # 1 where the input value was gap-filled
    if "works" in groups:
        site.append((t(pad(more_features.works_features(pd.DatetimeIndex(d["times"]), d["sites"]))), 0))
    if "wk4" in groups:      # same slot averaged over the previous 4 weeks, at the target slots
        site.append((t(more_features.seasonal_mean(norm, WEEK, 4)), HORIZON))
    if "diff" in groups:     # change since the previous step, and versus the same time last week
        site += [(t(more_features.shifted_diff(norm, 1)), 0), (t(more_features.shifted_diff(norm, WEEK)), 0)]
    if "roll" in groups:     # level and volatility over the past day, level over the past week
        m24, s24 = more_features.rolling_mean_std(norm, 96)
        site += [(t(m24), 0), (t(s24), 0), (t(more_features.rolling_mean_std(norm, WEEK)[0]), 0)]
    if "det" in groups:      # the study sites' top detectors (standardised), 0 for neighbour nodes
        site += [(t(pad(extra["det"][:, :, k])), 0) for k in range(extra["det"].shape[2])]
    if "event" in groups:    # MCG / Melbourne Park events are scheduled ahead, so read them at the target slots
        ev = t(events.event_features(pd.DatetimeIndex(d["times"])))
        site += [(ev[:, c:c + 1].expand(-1, flow.shape[1]), HORIZON) for c in range(ev.shape[1])]
    if "tsf" in groups:      # tsfresh features over the 24 h ending at each input step, study sites only
        tsf = np.load(ROOT / "cache" / "tsf.npy")
        cols = [int(c) for c in TSF_COLS.split(",")] if TSF_COLS else range(tsf.shape[2])
        site += [(t(pad(tsf[:, :, k])), 0) for k in cols]
    if "llm" in groups:      # features chosen by the LLM feedback loop (fe_loop.py -> fe_loop/selected.json)
        for f in json.loads((ROOT / "fe_loop" / "selected.json").read_text()):
            arr = np.load(ROOT / "cache" / "fe_loop" / f"{f['name']}.npy")
            arr = t(arr[:, None]).expand(-1, flow.shape[1]) if arr.ndim == 1 else t(pad(arr))
            site.append((arr, 0 if f["alignment"] == "input" else HORIZON))
    data["site"] = site
    if "ext" in groups:
        # [T, 6]: dow/7, three 0/1 flags, log rain and temperature standardised on the training years
        rain, temp = np.log1p(d["ext_rain_mm"]), d["ext_temp_c"]
        z = lambda a: (a - a[train_rows].mean()) / a[train_rows].std()
        exo = [np.stack([d["ext_dow"] / 7.0] + [d[f"ext_{k}"].astype(np.float32) for k in EXT_FLAGS]
                        + [z(rain), z(temp)], axis=1)]
        idx = pd.DatetimeIndex(d["times"])
        if "cal" in groups:
            exo.append(more_features.calendar_features(idx, d["ext_public_holiday"]))
        if "uni" in groups:
            exo.append(more_features.semester_features(idx))
        if "cyc" in groups:
            exo.append(more_features.cyclical_features(idx))
        data["exo"] = t(np.concatenate(exo, axis=1).astype(np.float32))
    return data, split, torch.as_tensor(d["adj"])


# builders take (num_nodes, adjacency, in_dim); in_dim is 2 (base) or 8 (ext)
MODELS = {
    "gwn": ("Graph WaveNet", lambda n, adj, c: GraphWaveNet(n, [transition(adj), transition(adj.T)], in_dim=c, horizon=HORIZON)),
    "stgcn": ("STGCN", lambda n, adj, c: STGCN(n, adj, in_dim=c, horizon=HORIZON, in_steps=IN_STEPS)),
    "agcrn": ("AGCRN", lambda n, adj, c: AGCRN(n, in_dim=c, horizon=HORIZON)),
    "mtgnn": ("MTGNN", lambda n, adj, c: MTGNN(n, in_dim=c, horizon=HORIZON, in_steps=IN_STEPS)),
    "stid": ("STID", lambda n, adj, c: STID(n, horizon=HORIZON, in_steps=IN_STEPS)),   # uses flow + time of day only
    "staeformer": ("STAEformer", lambda n, adj, c: STAEformer(n, horizon=HORIZON, in_steps=IN_STEPS, exo_dim=c - 3 if c > 2 else 0,
                                                              **MODEL_HP)),
}


def batch(data, idx):
    """Build input [B, C, N, 12] (normalised flow, time of day, then any ext features) and targets [B, 4, N]."""
    i_in = idx[:, None] + torch.arange(-IN_STEPS, 0, device=idx.device)
    i_out = idx[:, None] + torch.arange(HORIZON, device=idx.device)
    x = data["norm"][i_in]                                    # [B, 12, N]
    tod = data["tod"][i_in][..., None].expand_as(x)
    x = torch.stack([x, tod], dim=1).transpose(2, 3)          # [B, 2, N, 12]
    if "exo" in data:                                         # same values for every site
        e = data["exo"][i_in].permute(0, 2, 1)[:, :, None, :]  # [B, C, 1, 12]
        x = torch.cat([x, e.expand(-1, -1, x.size(2), -1)], dim=1)
    # per-site channels, each [B, 12, N]; every index read is still before t (no future data)
    site = [arr[i_in + off] for arr, off in data.get("site", [])]
    if site:
        x = torch.cat([x, torch.stack(site, dim=1).transpose(2, 3)], dim=1)
    return x, data["flow"][i_out], data["mask"][i_out]


def predict(model, data, idx):
    return model(batch(data, idx)[0]) * data["std"] + data["mean"]


def masked_mae(pred, y, m):
    m = m.float()
    return (torch.abs(pred - y) * m).sum() / m.sum().clamp(min=1)


@torch.no_grad()
def evaluate(model, data, windows, device, bs=128):
    """MAE, RMSE and MAPE per horizon, on observed values only (MAPE skips flow < 10)."""
    model.eval()
    sums = torch.zeros(4, HORIZON, device=device)   # abs, sq, pct, count(for mape)
    count = torch.zeros(HORIZON, device=device)
    for k in range(0, len(windows), bs):
        idx = torch.as_tensor(windows[k:k + bs], device=device)
        _, y, m = batch(data, idx)
        p = predict(model, data, idx).clamp(min=0)
        m = m.float()
        e = (p - y) * m
        pm = m * (y >= 10)
        sums[0] += e.abs().sum((0, 2)); sums[1] += (e ** 2).sum((0, 2))
        sums[2] += (e.abs() / y.clamp(min=1) * pm).sum((0, 2)); sums[3] += pm.sum((0, 2))
        count += m.sum((0, 2))
    mae, rmse, mape = sums[0] / count, (sums[1] / count).sqrt(), 100 * sums[2] / sums[3]
    return {f"{15 * (h + 1)}min": {"MAE": round(mae[h].item(), 3), "RMSE": round(rmse[h].item(), 3),
                                   "MAPE": round(mape[h].item(), 2)} for h in range(HORIZON)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, default="gwn")
    ap.add_argument("--features", default="base", help=f"'base' or 'ext[+group...]', groups: {FEATURE_GROUPS[1:]}")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--samples-per-epoch", type=int, default=60000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--schedule", choices=["none", "cosine"], default="none")
    ap.add_argument("--warmup-epochs", type=float, default=1.0)
    ap.add_argument("--val-stride", type=int, default=8)
    ap.add_argument("--in-steps", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--hp", default="")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--train-end", default="2024-01-01")
    ap.add_argument("--init", default="")
    ap.add_argument("--fold", default="", help="k/K, e.g. 0/5")
    ap.add_argument("--tsf-cols", default="")
    args = ap.parse_args()

    global IN_STEPS, CLEAN, TRAIN_END, MODEL_HP, FOLD, TSF_COLS
    TSF_COLS = args.tsf_cols
    IN_STEPS = args.in_steps        # read by load(), batch() and the model builders
    CLEAN = args.clean
    TRAIN_END = np.datetime64(args.train_end)
    MODEL_HP = parse_hp(args.hp)
    FOLD = tuple(map(int, args.fold.split("/"))) if args.fold else None
    if MODEL_HP and args.model != "staeformer":
        raise SystemExit("--hp is only for --model staeformer")
    # keep the last epoch when 2024 is trained on (validation would be in-sample), and in K-fold
    # (so the held-out block is never used to choose anything)
    refit = TRAIN_END > VAL_START or bool(FOLD)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True     # TF32 matmuls: ~25% faster on recent NVIDIA GPUs
    if args.model in GRAPH_MODELS and "nbr" in parse_features(args.features):
        raise SystemExit(f"{args.model} uses the 40-site distance graph and can't take --features ...+nbr")
    data, split, adj = load(device, args.features)
    name, build = MODELS[args.model]
    in_dim = batch(data, torch.as_tensor(split["train"][:1], device=device))[0].shape[1]
    model = build(data["flow"].shape[1], adj, in_dim).to(device)
    if args.init:
        model.load_state_dict(torch.load(WEIGHTS / f"{args.init}.pt", map_location=device))
    run = (args.model + ("" if args.features == "base" else "_" + args.features.replace("+", "-"))
           + ("_clean" if args.clean else "") + (f"_{args.tag}" if args.tag else ""))
    ckpt = WEIGHTS / f"{run}.pt"
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(args.samples_per_epoch / args.batch_size)
    total, warmup = args.epochs * steps_per_epoch, max(1.0, args.warmup_epochs * steps_per_epoch)
    lr_factor = lambda s: (min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total))
                           if args.schedule == "cosine" else 1.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    val_sub = split["val"][::args.val_stride]     # default: every 2 h of 2024, for fast validation
    print(f"{name} | features {args.features} | device {device} | params {sum(p.numel() for p in model.parameters()):,} | "
          f"windows train {len(split['train']):,} val {len(split['val']):,} test {len(split['test']):,}"
          + (" | K-fold: last epoch kept" if FOLD else " | refit: val is in-sample, last epoch kept" if refit else ""))

    WEIGHTS.mkdir(parents=True, exist_ok=True); LOGS.mkdir(parents=True, exist_ok=True)
    best, bad, history = float("inf"), 0, []
    for epoch in range(1, args.epochs + 1):
        model.train(); t0 = time.time(); losses = []
        # each epoch trains on a fresh random sample of training windows
        picks = np.random.choice(split["train"], args.samples_per_epoch, replace=False)
        for k in range(0, len(picks), args.batch_size):
            idx = torch.as_tensor(picks[k:k + args.batch_size], device=device)
            _, y, m = batch(data, idx)
            loss = masked_mae(predict(model, data, idx), y, m)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step(); losses.append(loss.item())
        val = evaluate(model, data, val_sub, device)
        val_mae = np.mean([v["MAE"] for v in val.values()])
        lr_now = opt.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train_mae": float(np.mean(losses)), "val_mae": float(val_mae), "lr": lr_now})
        print(f"epoch {epoch:2d} | train MAE {np.mean(losses):7.2f} | val MAE {val_mae:7.2f} | "
              f"lr {lr_now:.1e} | {time.time() - t0:5.0f}s", flush=True)
        if refit:
            best = val_mae
            torch.save(model.state_dict(), ckpt)
        elif val_mae < best:
            best, bad = val_mae, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= args.patience:
                print("early stop"); break

    model.load_state_dict(torch.load(ckpt, map_location=device))
    test = evaluate(model, data, split["test"], device) if len(split["test"]) else None
    out = {"model": name, "features": args.features, "test_2025_2026": test, "best_val_mae": best,
           "history": history, "args": vars(args)}
    if FOLD:    # every held-out window (per-epoch validation may use only every n-th)
        out["heldout"] = evaluate(model, data, split["val"], device)
    (LOGS / f"{run}_metrics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(test, indent=2))


if __name__ == "__main__":
    main()
