"""Run all test cases in tests/cases/*.json. Usage: python tests/run_tests.py
Each case has an id, scenario, expected result and the name of its check below. Status is PASS, FAIL
or SKIP (SKIP means a needed dataset, trained model or package is missing). Without the dataset the
route and GUI tests run on the built-in demo system, shown in the Target column.
Results are also written to tests/results.csv."""
import csv
import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from tbrgs.backend import load_system  # noqa: E402
from tbrgs.config import DEFAULTS, find_dataset, load_config  # noqa: E402
from tbrgs.demo import demo_system  # noqa: E402

CFG = load_config(BASE / "config.json")
O, D = CFG["default_origin"], CFG["default_destination"]
_SYS = {}


class Skip(Exception):
    pass


def system():
    if not _SYS:
        _SYS["s"], notice = load_system(CFG, BASE)
        _SYS["target"] = "demo" if notice else "real"
    return _SYS["s"]


def when(hhmm):
    return datetime.fromisoformat(f"{system().days[-1]}T{hhmm}")


def flows(model, hhmm):
    try:
        return system().flow_fn(model, when(hhmm))
    except ValueError as e:
        if "not trained" in str(e):
            raise Skip(str(e))
        raise


def need_torch():
    try:
        import torch  # noqa: F401
    except ImportError:
        raise Skip("PyTorch not installed")


def real_dataset():
    from tbrgs.data import load_scats
    path = find_dataset(BASE, CFG)
    if path is None:
        raise Skip("dataset not found")
    return load_scats(path)


def tiny_training(kind):
    need_torch()
    from tbrgs.data import make_windows
    from tbrgs.models import train_model
    t = np.arange(6 * 96)
    series = {i: 100 + 60 * np.sin(2 * np.pi * t / 96 + i) for i in range(3)}
    tr, va = make_windows(series, 8, 0, 4 * 96), make_windows(series, 8, 4 * 96, 6 * 96)
    scaled = lambda xy: (np.concatenate([xy[0][:, :, :1] / 200, xy[0][:, :, 1:]], axis=2), xy[1] / 200)
    (xt, yt), (xv, yv) = scaled(tr), scaled(va)
    return train_model(kind, xt, yt, xv, yv, hidden=16, lr=0.01, epochs=5, seed=1)[1]


def check_train(kind):
    h = tiny_training(kind)
    assert h[-1][0] < h[0][0], f"training loss did not fall: {h[0][0]:.4f} -> {h[-1][0]:.4f}"


# ----- checks -----
def t01():
    ds = real_dataset()
    for s, x in ds.series.items():
        assert len(x) == len(ds.dates) * 96, f"site {s}: {len(x)} values for {len(ds.dates)} days"
        assert not np.isnan(x).any(), f"site {s} has NaN"


def t02():
    from tbrgs.data import make_windows
    s = {1: np.arange(100.0)}
    x, y = make_windows(s, 12, 0, 100)
    assert len(y) == 88 and y[0] == 12 and np.array_equal(x[0, :, 0], np.arange(12))
    x, y = make_windows(s, 12, 50, 100)
    assert y.min() == 50 and np.array_equal(x[0, :, 0], np.arange(38, 50))


def t03():
    from tbrgs.data import clean_series
    c = clean_series([5, -3, np.nan, 7, np.nan])
    assert not np.isnan(c).any() and (c >= 0).all()


def t04(): check_train("lstm")
def t05(): check_train("gru")
def t06(): check_train("cnn")


def t07():
    ds = real_dataset()
    f = flows("LSTM", "08:00")
    for s, v in f.items():
        assert v >= 0 and v <= 4 * 1.5 * ds.series[s].max(), f"site {s}: {v:.0f} veh/h outside the observed range"


def t08():
    real_dataset()
    peak, off = flows("LSTM", "08:00"), flows("LSTM", "03:00")
    assert all(v >= 0 for v in off.values())
    share = np.mean([off[s] < peak[s] for s in peak])
    assert share >= 0.6, f"only {share:.0%} of sites are lower at 03:00 than at 08:00"


def t09():
    a, b = tiny_training("gru"), tiny_training("gru")
    assert np.allclose(a, b, atol=1e-6), "two runs with the same seed differ"


def usable_model():
    for m in CFG["available_models"]:
        try:
            system().flow_fn(m, when("08:00"))
            return m
        except ValueError as e:
            if "not trained" not in str(e):
                raise
    raise Skip("no trained model found")


def t10():
    r = system().route(O, D, usable_model(), when("08:00"), 5)
    assert 1 <= len(r) <= 5
    assert all(x["path"][0] == O and x["path"][-1] == D for x in r)
    assert all(r[i]["minutes"] <= r[i + 1]["minutes"] for i in range(len(r) - 1))


def t11():
    m = usable_model()
    peak = system().route(O, D, m, when("08:00"), 1)[0]["minutes"]
    off = system().route(O, D, m, when("03:00"), 1)[0]["minutes"]
    assert peak >= off, f"peak {peak:.1f} min < off-peak {off:.1f} min"


def raises(fn):
    try:
        fn()
    except ValueError as e:
        assert str(e)
        return
    raise AssertionError("no ValueError raised")


def t12(): raises(lambda: system().route(O, O, CFG["available_models"][0], when("08:00"), 3))
def t13():
    m = CFG["available_models"][0]
    raises(lambda: system().route(9999, D, m, when("08:00"), 3))
    raises(lambda: system().route(O, 9999, m, when("08:00"), 3))


def t14():
    nb = {2000: [9001, 9002], 3002: [9002], 9001: [2000], 9002: [2000, 3002], 9003: [], 9004: []}
    assert demo_system(CFG, nb).route(2000, 9004, "LSTM", datetime(2006, 10, 30, 8), 3) == []


def t15():
    m0 = CFG["available_models"][0]
    raises(lambda: system().route(O, D, m0, when("08:00"), 0))      # rejected before any prediction
    raises(lambda: system().route(O, D, m0, when("08:00"), 6))
    m = usable_model()
    assert 1 <= len(system().route(O, D, m, when("08:00"), 1)) <= 1
    assert 1 <= len(system().route(O, D, m, when("08:00"), 5)) <= 5


def t16():
    for m in CFG["available_models"]:
        try:
            assert system().route(O, D, m, when("08:00"), 3), f"{m} returned no routes"
        except ValueError as e:
            if "not trained" in str(e):
                raise Skip(str(e))
            raise


def t17():
    with tempfile.TemporaryDirectory() as d:
        assert load_config(Path(d) / "missing.json")["k_routes"] == DEFAULTS["k_routes"]
        bad = Path(d) / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        c = load_config(bad)
        assert c["k_routes"] == DEFAULTS["k_routes"] and "_warning" in c


def t18():
    try:
        import app
    except ImportError as e:
        raise Skip(f"{e.name} not installed")
    s = system()
    routes = s.route(O, D, usable_model(), when("08:00"), 3)
    layers = app.build_deck(s.sites, O, D, routes, 0).layers
    rows = lambda l: l.data.to_dict("records") if hasattr(l.data, "to_dict") else l.data
    tags = {r["tag"]: (r["lon"], r["lat"]) for l in layers if l.type == "TextLayer" for r in rows(l) if "tag" in r}
    po, pd_ = s.pos[O], s.pos[D]
    assert tags == {"A": (po[1], po[0]), "B": (pd_[1], pd_[0])}, f"markers: {tags}"
    drawn = {r["route"] for l in layers if l.type == "PathLayer" for r in rows(l)}
    assert drawn == set(range(len(routes))), f"routes drawn: {sorted(drawn)}"


def main():
    out = []
    for f in sorted((BASE / "tests" / "cases").glob("T*.json")):
        case = json.loads(f.read_text(encoding="utf-8"))
        try:
            globals()[case["check"]]()
            status, msg = "PASS", ""
        except Skip as e:
            status, msg = "SKIP", str(e)
        except ImportError as e:
            status, msg = "SKIP", f"{e.name} not installed"
        except Exception as e:  # noqa: BLE001
            status, msg = "FAIL", f"{type(e).__name__}: {e}"
        out.append((case["id"], case["area"], status, _SYS.get("target", "-"), msg))
        print(f"{case['id']} {case['area']:<6} {status:<5} {msg}")
    with open(BASE / "tests" / "results.csv", "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows([("id", "area", "status", "target", "message"), *out])
    n = lambda s: sum(r[2] == s for r in out)
    print(f"\n{n('PASS')} passed, {n('FAIL')} failed, {n('SKIP')} skipped (route tests ran on the {_SYS.get('target', '-')} system)")
    sys.exit(1 if n("FAIL") else 0)


if __name__ == "__main__":
    main()
