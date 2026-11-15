"""Download the inputs of the grid-load task into data/grid_load/ (open data, no keys).

- elia_load.parquet:        Elia ods001, Belgian grid load every 15 min since 2015, with Elia's own day-ahead forecast.
- weather.parquet:          Open-Meteo archive, hourly *measured* (reanalysis) weather for central Belgium since 2015.
- weather_forecast.parquet: Open-Meteo Previous Runs API, the *forecast* for the same hours and variables as it was
                            issued 1 and 2 days earlier: what was knowable when Elia published its day-ahead forecast.
                            Columns <var>_previous_day1 / _previous_day2 from ECMWF IFS 0.25° (archived since Mar 2024),
                            plus the same with suffix _best_match from Open-Meteo's default model mix (since Jan 2024).

Re-running refreshes all three (about 30 MB in total). Only some: python get_data.py weather_forecast
"""
import datetime as dt
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import mlkit  # noqa: E402

import pandas as pd  # noqa: E402

ELIA = "https://opendata.elia.be/api/explore/v2.1/catalog/datasets/ods001/exports/parquet"
VARS = "temperature_2m,shortwave_radiation,cloud_cover,wind_speed_10m"
WHERE = "latitude=50.85&longitude=4.35&timezone=UTC"
METEO = f"https://archive-api.open-meteo.com/v1/archive?{WHERE}&start_date=2015-01-01&end_date={{end}}&hourly={VARS}"
FORECAST = ("https://previous-runs-api.open-meteo.com/v1/forecast?" + WHERE + "&start_date={start}&end_date={end}&hourly="
            + ",".join(f"{v}_previous_day{d}" for d in (1, 2) for v in VARS.split(",")) + "&models={model}")
FORECAST_START = dt.date(2024, 1, 1)  # the Previous Runs archive starts in Jan-Mar 2024; earlier hours come back empty
# ECMWF IFS is the main source: one model, the same family as the ERA5 archive the models train on. best_match (the
# default mix) is kept as a check; its day-1 runs read ~1.2 °C warm against the archive since mid-2024.
FORECAST_MODELS = {"ecmwf_ifs025": "", "best_match": "_best_match"}


def fetch(url, tries=4):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "brussels-transit-ml"})
            with urllib.request.urlopen(req, timeout=300) as r:
                return r.read()
        except OSError:
            if i == tries - 1:
                raise
            time.sleep(10 * (i + 1))


def hourly(url):
    df = pd.DataFrame(json.loads(fetch(url))["hourly"]).rename(columns={"time": "hour"})
    df["hour"] = pd.to_datetime(df["hour"], utc=True)
    return df


def elia_load():
    (data / "elia_load.parquet").write_bytes(fetch(ELIA))
    elia = pd.read_parquet(data / "elia_load.parquet")
    print(f"elia_load: {len(elia):,} rows, {elia['datetime'].min()} -> {elia['datetime'].max()}")


def weather():
    end = (dt.date.today() - dt.timedelta(days=6)).isoformat()  # the archive lags a few days behind today
    w = hourly(METEO.format(end=end))
    w.to_parquet(data / "weather.parquet", index=False)
    print(f"weather: {len(w):,} hours, {w.hour.min()} -> {w.hour.max()}")


def weather_forecast():
    models = []
    for model, suffix in FORECAST_MODELS.items():
        parts, start = [], FORECAST_START
        while start <= dt.date.today():  # one request per year keeps each response small
            end = min(dt.date(start.year, 12, 31), dt.date.today())
            parts.append(hourly(FORECAST.format(start=start, end=end, model=model)))
            start = end + dt.timedelta(days=1)
        models.append(pd.concat(parts).set_index("hour").add_suffix(suffix))
    f = pd.concat(models, axis=1).dropna(how="all").reset_index()
    f.to_parquet(data / "weather_forecast.parquet", index=False)
    print(f"weather_forecast: {len(f):,} hours, {f.hour.min()} -> {f.hour.max()}")


data, _ = mlkit.paths("grid_load")
STEPS = {"elia_load": elia_load, "weather": weather, "weather_forecast": weather_forecast}
for name in sys.argv[1:] or STEPS:
    STEPS[name]()
