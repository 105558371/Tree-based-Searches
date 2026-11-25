"""Major events at the MCG / Melbourne Park precinct (just west of Boroondara), 2014-2026.

Traffic check (research notes): after MCG AFL games with 50k+ crowds, flow 2.5-4 h after the
start is +8-14% at the western study sites (Swan/Madden, Barkers/Church, Burwood/Church,
Victoria/Burnley) versus the same time 1-2 weeks earlier, fading to +1-3% at Burke and Warrigal Rd.

Sources (downloaded into cache/events/, summarised in external/mcg_events.csv):
    AFL     afltables.com/afl/seas/<year>.html - every MCG game: local start time and crowd
    Cricket cricsheet.org (bbl_male_json.zip, tests_male_json.zip) - BBL and Test days at the MCG
    Tennis  en.wikipedia.org "<year> Australian Open" infobox - tournament dates (Melbourne Park)

Features (feature group "event", global, known in advance so train.py aligns them with the
target slots), one value per 15-min step:
    afl_arrive  crowd / 100k from 2 h before the start until the start
    afl_leave   crowd / 100k from 2.5 h to 4 h after the start (crowd leaving)
    cricket     1 on BBL nights at the MCG (start 19:15 assumed; Cricsheet has no start times)
                and Boxing Day Test days (10:30 start), from 2 h before the start to 4.5 h after
    tennis      1 from 10:00 to midnight on Australian Open days
Usage: python research/events.py   (rebuilds external/mcg_events.csv from the cached sources)
"""
import glob, html, json, pathlib, re, urllib.request, zipfile
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
CACHE = ROOT / "cache" / "events"
EVENTS = ROOT / "external" / "mcg_events.csv"
YEARS = range(2014, 2027)
UA = {"User-Agent": "traffic-research-script/1.0"}


def _get(url, path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as r:
            path.write_bytes(r.read())
    return path


def afl_games():
    rows = []
    for y in YEARS:
        page = _get(f"https://afltables.com/afl/seas/{y}.html", CACHE / "afl" / f"{y}.html").read_text("latin-1")
        for r in re.findall(r"(?is)<tr[^>]*>(.*?)</tr>", page):
            txt = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", r))).strip()
            m = re.search(r"\w{3} (\d{1,2}-\w{3}-\d{4}) (\d{1,2}:\d{2} [AP]M).*?Att: ([\d,]+) Venue: (M\.C\.G\.|Docklands)$", txt)
            if m:   # first time on the row is local Melbourne time
                start = pd.to_datetime(f"{m.group(1)} {m.group(2)}", format="%d-%b-%Y %I:%M %p")
                kind, note = ("afl", "AFL at the MCG") if m.group(4) == "M.C.G." else ("afl_docklands", "AFL at Docklands")
                rows.append((kind, start, int(m.group(3).replace(",", "")), note))
    return rows


def cricket_days():
    rows = []
    for kind, f in [("bbl", "bbl_male_json.zip"), ("test", "tests_male_json.zip")]:
        z = zipfile.ZipFile(_get(f"https://cricsheet.org/downloads/{f}", CACHE / f.replace("_male_json", "")))
        for n in z.namelist():
            if not n.endswith(".json"):
                continue
            info = json.loads(z.read(n))["info"]
            if "Melbourne Cricket Ground" not in info.get("venue", ""):
                continue
            for day in info.get("dates", []):
                if str(YEARS.start) <= day[:4] < str(YEARS.stop):
                    t = "19:15" if kind == "bbl" else "10:30"
                    rows.append((kind, pd.Timestamp(f"{day} {t}"), np.nan, " v ".join(info.get("teams", []))))
    return rows


def tennis_days():
    rows = []
    for y in YEARS:
        url = ("https://en.wikipedia.org/w/api.php?action=query&prop=revisions&rvprop=content&rvslots=main"
               f"&format=json&formatversion=2&titles={y}_Australian_Open")
        page = json.loads(_get(url, CACHE / "ao" / f"{y}.json").read_text())
        text = page["query"]["pages"][0]["revisions"][0]["slots"]["main"]["content"]
        date = re.search(r"(?m)^\s*\|\s*date\s*=\s*(.+)$", text).group(1)
        # "13–26 January 2014" or "19 January – 1 February 2015"
        m = re.match(r"(\d{1,2})\s*(\w+)?\s*[–-]\s*(\d{1,2})\s+(\w+)\s+(\d{4})", date.strip())
        d1, m1, d2, m2, yr = m.groups()
        start, end = pd.Timestamp(f"{d1} {m1 or m2} {yr}"), pd.Timestamp(f"{d2} {m2} {yr}")
        rows += [("tennis", day + pd.Timedelta(hours=10), np.nan, "Australian Open")
                 for day in pd.date_range(start, end)]
    return rows


def load_events():
    if not EVENTS.exists():
        ev = pd.DataFrame(afl_games() + cricket_days() + tennis_days(), columns=["kind", "start", "crowd", "note"])
        ev.sort_values("start").to_csv(EVENTS, index=False)
    return pd.read_csv(EVENTS, parse_dates=["start"])


def event_features(times):
    """times: DatetimeIndex of 15-min steps (local time) -> [T, 4] float32."""
    ev = load_events()
    out = np.zeros((len(times), 4), dtype=np.float32)
    pos = lambda ts: np.clip(times.searchsorted(ts), 0, len(times))

    def mark(col, start, end, value):
        a, b = pos(start), pos(end)
        out[a:b, col] = np.maximum(out[a:b, col], value)

    for e in ev.itertuples():
        h = lambda x: pd.Timedelta(hours=x)
        if e.kind == "afl":
            mark(0, e.start - h(2), e.start, e.crowd / 1e5)
            mark(1, e.start + h(2.5), e.start + h(4), e.crowd / 1e5)
        elif e.kind in ("bbl", "test"):
            mark(2, e.start - h(2), e.start + h(4.5), 1.0)
        elif e.kind == "tennis":
            mark(3, e.start, e.start.normalize() + h(24), 1.0)
    return out


if __name__ == "__main__":
    if EVENTS.exists():
        EVENTS.unlink()
    ev = load_events()
    print(ev.groupby("kind").size().to_dict(), "| rows written to", EVENTS)
    t = pd.date_range("2014-01-01", "2026-11-16 23:45", freq="15min")
    f = event_features(t)
    print("share of 15-min steps flagged:", dict(zip(["afl_arrive", "afl_leave", "cricket", "tennis"],
                                                     (f > 0).mean(0).round(4))))
