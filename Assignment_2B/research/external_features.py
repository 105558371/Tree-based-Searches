"""Calendar and weather features from outside the SCATS data, one value per 15-min step.

Features (all known at prediction time):
    dow             day of week, 0 = Monday
    public_holiday  Victorian public holiday (python `holidays` package, subdiv VIC)
    school_holiday  not inside a Victorian government school term
    lockdown        inside one of Melbourne's six COVID-19 lockdowns
    rain_mm         hourly precipitation, Boroondara (Open-Meteo historical / ERA5)
    temp_c          hourly 2 m temperature, Boroondara (Open-Meteo historical / ERA5)

Sources:
    school terms  https://www.vic.gov.au/school-term-dates-and-holidays-victoria
    lockdowns     Victorian Government announcements, summarised at
                  https://en.wikipedia.org/wiki/COVID-19_lockdowns_in_Melbourne
    weather       https://open-meteo.com/en/docs/historical-weather-api
"""
import json, pathlib, urllib.request
import numpy as np
import pandas as pd
import holidays

ROOT = pathlib.Path(__file__).resolve().parent
WEATHER_CACHE = ROOT / "external" / "weather_hourly_boroondara.csv.gz"
BOROONDARA = (-37.83, 145.06)    # centre of the 40 sites

# (start, end) inclusive, Victorian government school terms
SCHOOL_TERMS = [
    ("2014-01-28", "2014-04-04"), ("2014-04-22", "2014-06-27"), ("2014-07-14", "2014-09-19"), ("2014-10-06", "2014-12-19"),
    ("2015-01-28", "2015-03-27"), ("2015-04-13", "2015-06-26"), ("2015-07-13", "2015-09-18"), ("2015-10-05", "2015-12-18"),
    ("2016-01-27", "2016-03-24"), ("2016-04-11", "2016-06-24"), ("2016-07-11", "2016-09-16"), ("2016-10-03", "2016-12-20"),
    ("2017-01-30", "2017-03-31"), ("2017-04-18", "2017-06-30"), ("2017-07-17", "2017-09-22"), ("2017-10-09", "2017-12-22"),
    ("2018-01-29", "2018-03-29"), ("2018-04-16", "2018-06-29"), ("2018-07-16", "2018-09-21"), ("2018-10-08", "2018-12-21"),
    ("2019-01-29", "2019-04-05"), ("2019-04-23", "2019-06-28"), ("2019-07-15", "2019-09-20"), ("2019-10-07", "2019-12-20"),
    ("2020-01-28", "2020-03-24"), ("2020-04-14", "2020-06-26"), ("2020-07-13", "2020-09-18"), ("2020-10-05", "2020-12-18"),
    ("2021-01-27", "2021-04-01"), ("2021-04-19", "2021-06-25"), ("2021-07-12", "2021-09-17"), ("2021-10-04", "2021-12-17"),
    ("2022-01-28", "2022-04-08"), ("2022-04-26", "2022-06-24"), ("2022-07-11", "2022-09-16"), ("2022-10-03", "2022-12-20"),
    ("2023-01-27", "2023-04-06"), ("2023-04-24", "2023-06-23"), ("2023-07-10", "2023-09-15"), ("2023-10-02", "2023-12-20"),
    ("2024-01-29", "2024-03-28"), ("2024-04-15", "2024-06-28"), ("2024-07-15", "2024-09-20"), ("2024-10-07", "2024-12-20"),
    ("2025-01-28", "2025-04-04"), ("2025-04-22", "2025-07-04"), ("2025-07-21", "2025-09-19"), ("2025-10-06", "2025-12-19"),
    ("2026-01-27", "2026-04-02"), ("2026-04-20", "2026-06-26"), ("2026-07-13", "2026-09-18"), ("2026-10-05", "2026-12-18"),
]

# Melbourne's six COVID-19 lockdowns (stay-at-home orders), inclusive
LOCKDOWNS = [
    ("2020-03-31", "2020-05-12"), ("2020-07-09", "2020-10-27"), ("2021-02-13", "2021-02-17"),
    ("2021-05-28", "2021-06-10"), ("2021-07-16", "2021-07-27"), ("2021-08-05", "2021-10-21"),
]


def in_ranges(days, ranges):
    out = np.zeros(len(days), dtype=bool)
    for start, end in ranges:
        out |= (days >= np.datetime64(start)) & (days <= np.datetime64(end))
    return out


def fetch_weather(start, end):
    """Hourly rain and temperature for Boroondara, cached as a csv.gz after the first download."""
    if WEATHER_CACHE.exists():
        return pd.read_csv(WEATHER_CACHE, parse_dates=["time"])
    url = ("https://archive-api.open-meteo.com/v1/archive"
           f"?latitude={BOROONDARA[0]}&longitude={BOROONDARA[1]}&start_date={start}&end_date={end}"
           "&hourly=temperature_2m,precipitation&timezone=Australia%2FMelbourne")
    with urllib.request.urlopen(url, timeout=120) as r:
        hourly = json.load(r)["hourly"]
    df = pd.DataFrame({"time": pd.to_datetime(hourly["time"]),
                       "temp_c": hourly["temperature_2m"], "rain_mm": hourly["precipitation"]})
    WEATHER_CACHE.parent.mkdir(exist_ok=True)
    df.to_csv(WEATHER_CACHE, index=False, compression="gzip")
    return df


def build_features(times):
    """times: DatetimeIndex of 15-min steps -> dict of [T] arrays."""
    days = times.normalize().to_numpy().astype("datetime64[D]")
    vic = holidays.Australia(subdiv="VIC", years=range(times.year.min(), times.year.max() + 1))
    hol_days = np.array(sorted(vic), dtype="datetime64[D]")

    weather = fetch_weather(times.min().strftime("%Y-%m-%d"), times.max().strftime("%Y-%m-%d"))
    # local clock time; daylight-saving repeats are averaged, gaps filled from the neighbouring hour
    weather = weather.groupby("time").mean().reindex(times.floor("h")).ffill().bfill()

    return {
        "dow": times.dayofweek.to_numpy().astype(np.int64),
        "public_holiday": np.isin(days, hol_days),
        "school_holiday": ~in_ranges(days, SCHOOL_TERMS),
        "lockdown": in_ranges(days, LOCKDOWNS),
        "rain_mm": weather["rain_mm"].to_numpy(np.float32),
        "temp_c": weather["temp_c"].to_numpy(np.float32),
    }


if __name__ == "__main__":
    t = pd.date_range("2014-01-01", "2026-11-16 23:45", freq="15min")
    f = build_features(t)
    daily = lambda a: a.reshape(-1, 96)[:, 0]
    print(f"{len(t)} steps | public holidays {daily(f['public_holiday']).sum()} days | "
          f"school holidays {daily(f['school_holiday']).sum()} days | lockdown {daily(f['lockdown']).sum()} days | "
          f"rain>0 {np.mean(f['rain_mm'] > 0):.1%} of steps | temp {f['temp_c'].min():.1f}..{f['temp_c'].max():.1f} C")
