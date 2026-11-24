"""LLM feedback-loop feature engineering (CAAFE-style) for the STAEformer traffic model.

Each round an LLM writes candidate features as Python code (JSON file, see PROPOSAL FORMAT);
this script checks, scores and records them, and the history is fed back for the next round.

    python research/fe_loop.py context              # task + data description + where the model errs
    python research/fe_loop.py eval <round.json>    # score a round of proposals, append to history
    python research/fe_loop.py history              # all results so far (the feedback)

PROPOSAL FORMAT (list of objects):
    {"name": "snake_case", "alignment": "input" | "target", "hypothesis": "...",
     "code": "def feature(ctx):\n    ...\n    return arr"}       # arr: [T, N] per site or [T] global
alignment "input":  row s may use traffic/weather rows <= s only (read at the last input step)
alignment "target": row s may use traffic/weather rows <= s - HORIZON, plus calendar/events at s
                    (read at the target slot; for things known in advance)

Checks: AST whitelist (numpy/pandas/math only, no files/system/dunder access), run in a separate
process with a timeout, shape/finite check, and a leakage test (recompute with all traffic and
weather from a cut-off onwards scrambled; rows before the allowed limit must not change).

Score: the current best STAEformer's 2024 validation predictions are fixed. A LightGBM model learns
to correct their residuals from base inputs (prediction, horizon, site, time of day, day of week,
last value) on Jan-Jun 2024 and is scored on Jul-Dec 2024. score = MAE drop (vehicles per 15 min)
when the candidate is added. A random-noise control gives the noise floor.
"""
import ast, json, multiprocessing as mp, pathlib, sys, tempfile, time
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
LOOP = ROOT / "fe_loop"
CACHE = ROOT / "cache" / "fe_loop"
HISTORY = LOOP / "history.jsonl"
BASE_RUN = "staeformer_ext-lag-wk4-diff-roll-cyc-event_final_s0"
BASE_FEATURES = "ext+lag+wk4+diff+roll+cyc+event"
HORIZON, WEEK, STEPS_PER_DAY = 4, 672, 96
WINDOW_STRIDE = 4            # use every 4th validation window for the proxy (speed)
TIMEOUT_S = 180

ALLOWED_IMPORTS = {"numpy", "pandas", "math"}
BANNED_NAMES = {"open", "exec", "eval", "compile", "__import__", "globals", "locals", "getattr", "setattr",
                "delattr", "input", "breakpoint", "vars", "dir", "memoryview", "help", "exit", "quit"}
BANNED_ATTRS = {"system", "popen", "remove", "unlink", "rmdir", "rmtree", "to_csv", "to_pickle", "to_parquet",
                "to_json", "to_excel", "to_hdf", "to_sql", "save", "savez", "savez_compressed", "savetxt",
                "tofile", "load", "loadtxt", "fromfile", "read_csv", "read_pickle", "read_json", "memmap"}


# ------------------------------------------------------------------ context for feature code

def build_context():
    d = np.load(ROOT / "cache" / "traffic.npz", allow_pickle=True)
    times = pd.DatetimeIndex(d["times"])
    flow, observed = d["flow"].astype(np.float32), d["mask"]
    train = times < "2024-01-01"
    mean, std = flow[train].mean(0), flow[train].std(0) + 1e-6
    sites = pd.read_csv(ROOT / "external" / "site_locations.csv").set_index("site").loc[d["sites"]].reset_index()
    lat, lon = np.radians(sites.lat.to_numpy()), np.radians(sites.lon.to_numpy())
    a = (np.sin((lat[:, None] - lat[None]) / 2) ** 2
         + np.cos(lat[:, None]) * np.cos(lat[None]) * np.sin((lon[:, None] - lon[None]) / 2) ** 2)
    return {
        "flow": ((flow - mean) / std).astype(np.float32),     # [T, N] z-scored (training years), gaps filled
        "raw": flow,                                          # [T, N] vehicles per 15 min, gaps filled
        "observed": observed,                                 # [T, N] True where really measured
        "times": times,                                       # DatetimeIndex, 15-min, Melbourne local time
        "sites": sites[["site", "name", "lat", "lon"]],       # N rows, same order as the columns
        "dist_km": (2 * 6371 * np.arcsin(np.sqrt(a))).astype(np.float32),   # [N, N]
        "rain_mm": d["ext_rain_mm"].astype(np.float32), "temp_c": d["ext_temp_c"].astype(np.float32),
        "public_holiday": d["ext_public_holiday"].astype(bool), "school_holiday": d["ext_school_holiday"].astype(bool),
        "lockdown": d["ext_lockdown"].astype(bool),
        "events": pd.read_csv(ROOT / "external" / "mcg_events.csv", parse_dates=["start"]),
        "HORIZON": HORIZON, "WEEK": WEEK, "STEPS_PER_DAY": STEPS_PER_DAY, "train_rows": train,
    }


CONTEXT_DOC = """ctx keys (T = 451,488 15-min steps 2014-01-01..2026-11-16, N = 40 sites):
  flow [T,N] float32 z-scored flow (per-site mean/std of 2014-2023), gaps filled; raw [T,N] vehicles/15 min
  observed [T,N] bool; times DatetimeIndex (local time); sites DataFrame(site,name,lat,lon); dist_km [N,N]
  rain_mm, temp_c [T]; public_holiday, school_holiday, lockdown [T] bool; train_rows [T] bool (< 2024)
  events DataFrame(kind in afl|bbl|test|tennis, start, crowd, note) for the MCG / Melbourne Park
  HORIZON=4, WEEK=672, STEPS_PER_DAY=96
The model already sees: last 12 steps of flow, time of day, day of week, holidays, lockdown, rain, temp,
flow 1 week and 1 day earlier at the target slots, 4-week same-slot mean, step diffs, 24 h / 7 d rolling
stats, sin/cos time of day/week/year, MCG event crowd windows. Fit statistics on train_rows only."""


# ------------------------------------------------------------------ safety checks

def check_code(code):
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            if any(m.split(".")[0] not in ALLOWED_IMPORTS for m in mods):
                raise ValueError(f"import not allowed: {mods}")
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            raise ValueError(f"name not allowed: {node.id}")
        elif isinstance(node, ast.Attribute) and (node.attr.startswith("_") or node.attr in BANNED_ATTRS):
            raise ValueError(f"attribute not allowed: {node.attr}")
    if not any(isinstance(n, ast.FunctionDef) and n.name == "feature" for n in tree.body):
        raise ValueError("code must define feature(ctx)")


def _child(code, ctx, out_path):
    import math
    ns = {"np": np, "pd": pd, "math": math}
    exec(compile(code, "<feature>", "exec"), ns)            # code already passed check_code
    np.save(out_path, np.asarray(ns["feature"](ctx), dtype=np.float32))


def run_feature(code, ctx):
    """Run feature(ctx) in a forked process (ctx shared copy-on-write) with a timeout."""
    with tempfile.TemporaryDirectory() as tmp:
        out = pathlib.Path(tmp) / "f.npy"
        p = mp.get_context("fork").Process(target=_child, args=(code, ctx, str(out)))
        p.start(); p.join(TIMEOUT_S)
        if p.is_alive():
            p.kill(); raise TimeoutError(f"feature took more than {TIMEOUT_S}s")
        if p.exitcode != 0 or not out.exists():
            raise RuntimeError(f"feature raised an error (exit code {p.exitcode})")
        return np.load(out)


def leakage_check(code, ctx, arr, alignment):
    """Scramble traffic + weather from row k on; rows before the allowed limit must be unchanged."""
    rng = np.random.default_rng(0)
    T = len(ctx["times"]); k = ctx["times"].get_loc(pd.Timestamp("2024-06-03 08:00"))
    bad = dict(ctx)
    for key in ("flow", "raw"):
        a = ctx[key].copy(); a[k:] = rng.normal(size=a[k:].shape).astype(a.dtype) * a.std() + a.mean(); bad[key] = a
    o = ctx["observed"].copy(); o[k:] = rng.random(o[k:].shape) < 0.5; bad["observed"] = o
    for key in ("rain_mm", "temp_c"):
        a = ctx[key].copy(); a[k:] = rng.permutation(a[k:]); bad[key] = a
    arr2 = np.nan_to_num(run_feature(code, bad), nan=0.0, posinf=0.0, neginf=0.0)
    limit = k if alignment == "input" else k + HORIZON
    return np.allclose(arr[:limit], arr2[:limit], atol=1e-5)


# ------------------------------------------------------------------ proxy evaluator

def base_predictions():
    """The current best model's 2024 validation predictions (cached)."""
    path = CACHE / "base_preds.npz"
    if path.exists():
        return dict(np.load(path))
    import torch, train
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data, split, adj = train.load(dev, BASE_FEATURES)
    in_dim = train.batch(data, torch.as_tensor(split["val"][:1], device=dev))[0].shape[1]
    model = train.MODELS["staeformer"][1](data["flow"].shape[1], adj, in_dim).to(dev)
    model.load_state_dict(torch.load(train.WEIGHTS / f"{BASE_RUN}.pt", map_location=dev)); model.eval()
    W = split["val"][::WINDOW_STRIDE]
    with torch.no_grad():
        P = torch.cat([train.predict(model, data, torch.as_tensor(W[i:i + 2048], device=dev)).clamp(min=0).cpu()
                       for i in range(0, len(W), 2048)]).numpy()
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(path, W=W, P=P)
    return {"W": W, "P": P}


class Proxy:
    def __init__(self, ctx):
        bp = base_predictions()
        W, P = bp["W"], bp["P"]                                   # P [W, 4, N]
        n_w, N = len(W), P.shape[2]
        tgt = W[:, None] + np.arange(HORIZON)                     # target rows [W, 4]
        raw, obs, times = ctx["raw"], ctx["observed"], ctx["times"]
        mean, std = raw[ctx["train_rows"]].mean(0), raw[ctx["train_rows"]].std(0) + 1e-6
        y, m = raw[tgt], obs[tgt]                                 # [W, 4, N]
        w_i, h_i, n_i = np.meshgrid(np.arange(n_w), np.arange(HORIZON), np.arange(N), indexing="ij")
        keep = m.ravel()
        self.w, self.h, self.n = w_i.ravel()[keep], h_i.ravel()[keep], n_i.ravel()[keep]
        self.tgt_row = tgt[self.w, self.h]; self.last_row = W[self.w] - 1
        self.y, self.p, self.std = y.ravel()[keep], P.ravel()[keep], std[self.n]
        self.resid = (self.y - self.p) / self.std
        tt = times[self.tgt_row]
        self.base_X = np.column_stack([(self.p - mean[self.n]) / self.std, self.h, self.n,
                                       tt.hour * 4 + tt.minute // 15, tt.dayofweek,
                                       ctx["flow"][self.last_row, self.n]]).astype(np.float32)
        self.first_half = np.asarray(tt < pd.Timestamp("2024-07-01"))
        self.hour = np.asarray(tt.hour)
        self.model_mae = np.abs(self.y - self.p)[~self.first_half].mean()
        self.base_mae, self.base_err = self._fit(self.base_X)

    def _fit(self, X):
        import lightgbm as lgb
        a, b = self.first_half, ~self.first_half
        reg = lgb.LGBMRegressor(objective="l1", n_estimators=300, learning_rate=0.05, num_leaves=63,
                                min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.9,
                                random_state=0, verbose=-1, n_jobs=32)
        reg.fit(X[a], self.resid[a], categorical_feature=[2])
        err = np.abs(self.y[b] - (self.p[b] + reg.predict(X[b]) * self.std[b]))
        return err.mean(), err

    def column(self, arr, alignment):
        row = self.last_row if alignment == "input" else self.tgt_row
        return arr[row] if arr.ndim == 1 else arr[row, self.n]

    def score(self, arr, alignment):
        col = self.column(arr, alignment).astype(np.float32)
        mae, err = self._fit(np.column_stack([self.base_X, col]))
        hours = self.hour[~self.first_half]
        by = {name: float(self.base_err[sel].mean() - err[sel].mean())
              for name, sel in [("night 0-5", hours < 6), ("am peak 6-9", (hours >= 6) & (hours < 10)),
                                ("midday 10-15", (hours >= 10) & (hours < 16)), ("pm peak 16-19", (hours >= 16) & (hours < 20)),
                                ("evening 20-23", hours >= 20)]}
        return float(self.base_mae - mae), by


# ------------------------------------------------------------------ commands

def describe_errors(ctx):
    bp = base_predictions(); proxy = Proxy(ctx)
    e = np.abs(proxy.y - proxy.p); tt = ctx["times"][proxy.tgt_row]
    df = pd.DataFrame({"err": e, "y": proxy.y, "site": ctx["sites"].site.to_numpy()[proxy.n],
                       "hour": tt.hour, "dow": tt.dayofweek, "h": proxy.h})
    print(CONTEXT_DOC)
    print(f"\nCurrent best model ({BASE_FEATURES}) on 2024 validation (every {WINDOW_STRIDE}th window): "
          f"MAE {e.mean():.2f}; Jul-Dec {proxy.model_mae:.2f}; proxy with base inputs only {proxy.base_mae:.2f}")
    for col in ["hour", "dow", "h"]:
        g = df.groupby(col).agg(MAE=("err", "mean"), flow=("y", "mean"))
        print(f"\nMAE by {col}:\n" + " ".join(f"{i}:{r.MAE:.1f}" for i, r in g.iterrows()))
    s = df.groupby("site").agg(MAE=("err", "mean"), flow=("y", "mean"))
    s["rel"] = s.MAE / s.flow
    names = ctx["sites"].set_index("site").name
    print("\nworst sites by relative error:\n" + "\n".join(
        f"  {i} {names[i][:30]:30s} MAE {r.MAE:5.1f} flow {r.flow:5.0f} rel {r.rel:.3f}"
        for i, r in s.sort_values("rel", ascending=False).head(8).iterrows()))
    top = df.nlargest(int(len(df) * 0.01), "err")
    print(f"\nlargest 1% errors: mean {top.err.mean():.0f} veh; by hour " +
          top.hour.value_counts().sort_index().to_string().replace("\n", ", "))


def evaluate(round_file):
    proposals = json.loads(pathlib.Path(round_file).read_text())
    ctx = build_context(); proxy = Proxy(ctx)
    rnd = pathlib.Path(round_file).stem
    print(f"proxy baseline: Jul-Dec 2024 MAE {proxy.model_mae:.3f} (model) -> {proxy.base_mae:.3f} (base proxy)")
    noise, _ = proxy.score(np.random.default_rng(1).normal(size=ctx["flow"].shape).astype(np.float32), "input")
    print(f"noise floor (random feature): {noise:+.4f}\n")
    CACHE.mkdir(parents=True, exist_ok=True); LOOP.mkdir(exist_ok=True)
    for p in proposals:
        rec = {"round": rnd, "name": p["name"], "alignment": p["alignment"], "hypothesis": p["hypothesis"],
               "noise_floor": noise, "time": time.strftime("%Y-%m-%d %H:%M")}
        try:
            check_code(p["code"])
            t0 = time.time(); arr = run_feature(p["code"], ctx)
            if arr.shape not in [(len(ctx["times"]),), ctx["flow"].shape]:
                raise ValueError(f"bad shape {arr.shape}")
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            if not leakage_check(p["code"], ctx, arr, p["alignment"]):
                raise ValueError("LEAKAGE: values before the allowed row change when future traffic/weather changes")
            score, by = proxy.score(arr, p["alignment"])
            np.save(CACHE / f"{p['name']}.npy", arr)
            rec.update(status="ok", score=round(score, 4), by_period={k: round(v, 3) for k, v in by.items()},
                       seconds=round(time.time() - t0, 1), shape=list(arr.shape))
            verdict = "KEEP?" if score > 2 * max(noise, 0.005) else "no gain"
            print(f"{p['name']:32s} {score:+.4f}  {verdict:8s} " + " ".join(f"{k.split()[0]}:{v:+.2f}" for k, v in by.items()))
        except Exception as e:
            rec.update(status="error", error=str(e)[:300])
            print(f"{p['name']:32s} ERROR {str(e)[:120]}")
        rec["code"] = p["code"]
        with open(HISTORY, "a") as f:
            f.write(json.dumps(rec) + "\n")


def show_history():
    if not HISTORY.exists():
        print("no history yet"); return
    rows = [json.loads(l) for l in HISTORY.read_text().splitlines()]
    for r in sorted(rows, key=lambda r: -r.get("score", -9)):
        s = f"{r['score']:+.4f}" if r["status"] == "ok" else "ERROR "
        print(f"{r['round']:9s} {r['name']:32s} {r['alignment']:6s} {s}  {r.get('error', r['hypothesis'])[:90]}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "history"
    if cmd == "context":
        describe_errors(build_context())
    elif cmd == "eval":
        evaluate(sys.argv[2])
    else:
        show_history()
