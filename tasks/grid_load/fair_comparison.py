"""Fair comparison with Elia's day-ahead forecast: weather FORECASTS as input instead of measured weather.

The notebook feeds the models the measured weather of day D, which nobody has at the end of D-1, so its weather scores
are an upper bound. Here the same models (same features, same train/val/test split, same LightGBM settings, trained
on measured weather 2015-2023 as before) are scored on test hours with the weather as it was *forecast* by ECMWF IFS
(Open-Meteo Previous Runs API):

- issued D-1 (previous_day1): the run about 24 h before each hour. Close to Elia's timing; slightly optimistic for
  afternoon/evening hours, whose run comes after Elia publishes on the morning of D-1;
- issued D-2 (previous_day2): the run about 48 h before each hour, all issued before Elia publishes: strictly knowable.

A check with Open-Meteo's default model mix (best_match) is added at the bottom of the table.
Test window = the notebook's test period (2024 onwards) cut to the hours where every forecast column, the measured
weather and Elia's forecast exist. Run: uv run python tasks/grid_load/fair_comparison.py  (needs get_data.py weather_forecast first)
Writes outputs/grid_load/fair-comparison.csv.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import mlkit  # noqa: E402

import holidays  # noqa: E402
import lightgbm as lgb  # noqa: E402
import pandas as pd  # noqa: E402

DATA, OUT = mlkit.paths("grid_load")
TZ = "Europe/Brussels"

# --- same preparation as notebook.py sections 1 and 3 --------------------------------------------------------------
raw = pd.read_parquet(DATA / "elia_load.parquet")
raw["hour"] = raw["datetime"].dt.floor("h")
h = (raw.groupby("hour")[["totalload", "dayaheadforecast"]].mean()
     .rename(columns={"totalload": "load", "dayaheadforecast": "elia_da"}))
h = h[h["load"].notna()]
h = h.join(pd.read_parquet(DATA / "weather.parquet").set_index("hour"), how="left")
h = h.join(pd.read_parquet(DATA / "weather_forecast.parquet").set_index("hour"), how="left")
local = h.index.tz_convert(TZ)
h["hod"], h["dow"], h["month"], h["year"] = local.hour, local.dayofweek, local.month, local.year
h["doy"] = local.dayofyear
be = holidays.Belgium(years=range(2015, 2028))
h["holiday"] = pd.Series(local.date, index=h.index).map(lambda d: d in be).astype(int)
for lag in [48, 72, 168]:
    h[f"load_lag{lag}"] = h["load"].shift(lag, freq="h").reindex(h.index)
h["load_mean_2_7d"] = sum(h["load"].shift(24 * d, freq="h").reindex(h.index) for d in range(2, 8)) / 6

CAL = ["hod", "dow", "month", "doy", "holiday"]
LAGS = ["load_lag48", "load_lag72", "load_lag168", "load_mean_2_7d"]
WEATHER = ["temperature_2m", "shortwave_radiation", "cloud_cover", "wind_speed_10m"]
FEATURES = CAL + LAGS + WEATHER
NO_W = CAL + LAGS
WITH_ELIA = FEATURES + ["elia_da"]

d = h.dropna(subset=["load_lag168"])
train, val, test = mlkit.time_split(d, d.index.to_series(), pd.Timestamp("2023-01-01", tz="UTC"),
                                    pd.Timestamp("2024-01-01", tz="UTC"))
fc = {(k, src): [f"{w}_previous_day{k}{src}" for w in WEATHER] for k in (1, 2) for src in ("", "_best_match")}
test = test.dropna(subset=WEATHER + sum(fc.values(), []) + ["elia_da"])
print(f"fair test window: {test.index.min():%Y-%m-%d %H:%M} -> {test.index.max():%Y-%m-%d %H:%M} UTC, "
      f"{len(test):,} hours ({test['holiday'].sum():,} holiday hours)")


def fit_lgb(features):
    m = lgb.LGBMRegressor(objective="l1", n_estimators=5000, learning_rate=0.03, num_leaves=63,
                          min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
    m.fit(train[features], train["load"], eval_set=[(val[features], val["load"])],
          callbacks=[lgb.early_stopping(200, verbose=False)])
    return m


def as_forecast(rows, k, src=""):
    """The test rows with measured weather swapped for the forecast issued k days earlier."""
    return rows.drop(columns=WEATHER).rename(columns=dict(zip(fc[k, src], WEATHER)))


gbm, gbm_nw, gbm_elia = fit_lgb(FEATURES), fit_lgb(NO_W), fit_lgb(WITH_ELIA)
preds = {
    "Elia day-ahead forecast": test["elia_da"],
    "LightGBM, no weather": gbm_nw.predict(test[NO_W]),
    "LightGBM, measured weather (upper bound)": gbm.predict(test[FEATURES]),
    "LightGBM, forecast weather issued D-1": gbm.predict(as_forecast(test, 1)[FEATURES]),
    "LightGBM, forecast weather issued D-2": gbm.predict(as_forecast(test, 2)[FEATURES]),
    "LightGBM + Elia as input, measured weather (upper bound)": gbm_elia.predict(test[WITH_ELIA]),
    "LightGBM + Elia as input, forecast weather issued D-1": gbm_elia.predict(as_forecast(test, 1)[WITH_ELIA]),
    "LightGBM + Elia as input, forecast weather issued D-2": gbm_elia.predict(as_forecast(test, 2)[WITH_ELIA]),
    "check: LightGBM, best_match forecast D-1": gbm.predict(as_forecast(test, 1, "_best_match")[FEATURES]),
    "check: LightGBM, best_match forecast D-2": gbm.predict(as_forecast(test, 2, "_best_match")[FEATURES]),
    "check: LightGBM + Elia, best_match forecast D-1": gbm_elia.predict(as_forecast(test, 1, "_best_match")[WITH_ELIA]),
    "check: LightGBM + Elia, best_match forecast D-2": gbm_elia.predict(as_forecast(test, 2, "_best_match")[WITH_ELIA]),
}
hol = test["holiday"].eq(1).to_numpy()
elia_all, elia_hol = mlkit.mae(test["load"], test["elia_da"]), mlkit.mae(test["load"][hol], test["elia_da"][hol])
rows = []
for name, p in preds.items():
    p = pd.Series(p, index=test.index)
    mae, mae_h = mlkit.mae(test["load"], p), mlkit.mae(test["load"][hol], p[hol])
    rows.append({"model": name, "mae_mw": mae, "vs_elia_pct": 100 * (mae / elia_all - 1),
                 "mape_pct": mlkit.mape(test["load"], p), "holiday_mae_mw": mae_h,
                 "holiday_vs_elia_pct": 100 * (mae_h / elia_hol - 1)})
res = pd.DataFrame(rows).set_index("model")
res.attrs["window"] = f"{test.index.min():%Y-%m-%d} to {test.index.max():%Y-%m-%d}"
res.round(2).to_csv(OUT / "fair-comparison.csv")
pd.set_option("display.width", 200)
print(res.round(1).to_string())
print(f"N hours {len(test)}, holiday hours {int(hol.sum())}, window {res.attrs['window']}")
