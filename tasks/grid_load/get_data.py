"""Download the inputs of the grid-load task into data/grid_load/ (open data, no keys).

- elia_load.parquet: Elia ods001, Belgian grid load every 15 min since 2015, with Elia's own day-ahead forecast.
- weather.parquet:   Open-Meteo archive, hourly weather for central Belgium (Brussels) since 2015.

Re-running refreshes both (about 30 MB in total).
"""
import datetime as dt
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import mlkit  # noqa: E402

import pandas as pd  # noqa: E402

ELIA = "https://opendata.elia.be/api/explore/v2.1/catalog/datasets/ods001/exports/parquet"
METEO = ("https://archive-api.open-meteo.com/v1/archive?latitude=50.85&longitude=4.35"
         "&start_date=2015-01-01&end_date={end}&timezone=UTC"
         "&hourly=temperature_2m,shortwave_radiation,cloud_cover,wind_speed_10m")


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "brussels-transit-ml"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


data, _ = mlkit.paths("grid_load")

(data / "elia_load.parquet").write_bytes(fetch(ELIA))
elia = pd.read_parquet(data / "elia_load.parquet")
print(f"elia_load: {len(elia):,} rows, {elia['datetime'].min()} -> {elia['datetime'].max()}")

# the archive lags a few days behind today
end = (dt.date.today() - dt.timedelta(days=6)).isoformat()
h = json.loads(fetch(METEO.format(end=end)))["hourly"]
weather = pd.DataFrame(h).rename(columns={"time": "hour"})
weather["hour"] = pd.to_datetime(weather["hour"], utc=True)
weather.to_parquet(data / "weather.parquet", index=False)
print(f"weather: {len(weather):,} hours, {weather.hour.min()} -> {weather.hour.max()}")
