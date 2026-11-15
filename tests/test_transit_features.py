"""transit_delay features: the SQL features match the original pandas notebook and never look ahead."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tasks" / "transit_delay"))
import mlkit  # noqa: E402

SNAPSHOT = ROOT / "data" / "transit_delay" / "snapshot-2026-10-27"
FLOATS = ["scheduled_wait_min", "excess_wait_min", "prev_hour_ewt", "prev_days_same_hod_ewt",
          "city_prev_ewt", "no2_prev", "load_prev"]
KEY = ["stop_id", "line_id", "hour"]


def sql_features(slh_path, city_path):
    features = pytest.importorskip("features")  # needs duckdb (not installed in CI)
    con = features.build(features.connect(), slh_path, city_path)
    return con, con.execute("select * from features").df()


def notebook_features(slh_path, city_path):
    """The preprocessing and features of the original notebook (tasks/transit_delay/notebook.py before the
    pipeline), copied as it was."""
    slh, city = pd.read_parquet(slh_path), pd.read_parquet(city_path)
    for df in (slh, city):
        df["hour"] = pd.to_datetime(df["hour"], utc=True)
        local = df["hour"].dt.tz_convert("Europe/Brussels")
        df["date"], df["hod"], df["dow"] = local.dt.date, local.dt.hour, local.dt.day_of_week
    per_day = slh.groupby("date").agg(hours=("hour", "nunique"), median_excess=("excess_wait_min", "median"))
    full_days = per_day.index[per_day["hours"] >= 20]
    normal = per_day.loc[full_days, "median_excess"].median()
    disrupted = [d for d in full_days if per_day.loc[d, "median_excess"] > 3 * max(normal, 0.1)]
    slh = slh[slh["date"].isin(full_days)].copy()
    slh["excess_wait_min"] = slh["excess_wait_min"].clip(-30, 30)
    slh["disrupted"] = slh["date"].isin(disrupted)
    df = slh.sort_values("hour").copy()
    key = ["stop_id", "line_id"]
    prev = df[key + ["hour", "excess_wait_min"]].copy()
    prev["hour"] += pd.Timedelta(hours=1)
    df = df.merge(prev.rename(columns={"excess_wait_min": "prev_hour_ewt"}), on=key + ["hour"], how="left")
    df = df.sort_values("hour").reset_index(drop=True)
    hist = df["excess_wait_min"].where(~df["disrupted"])
    df["prev_days_same_hod_ewt"] = mlkit.past_mean(hist, [df["stop_id"], df["line_id"], df["hod"]])
    cprev = city[["hour", "excess_wait_min", "no2", "load_mw"]].copy()
    cprev["hour"] += pd.Timedelta(hours=1)
    df = df.merge(cprev.rename(columns={"excess_wait_min": "city_prev_ewt", "no2": "no2_prev", "load_mw": "load_prev"}),
                  on="hour", how="left")
    df["hour"] = df["hour"].dt.tz_convert(None)
    return df, sorted(disrupted)


def aligned(a, b):
    a = a.sort_values(KEY).reset_index(drop=True)
    b = b.sort_values(KEY).reset_index(drop=True)
    return a, b


@pytest.mark.skipif(not (SNAPSHOT / "stop_line_hourly.parquet").exists(), reason="18-day snapshot not on this machine")
def test_sql_features_equal_notebook_features_on_snapshot():
    slh, city = SNAPSHOT / "stop_line_hourly.parquet", SNAPSHOT / "city_hourly.parquet"
    con, got = sql_features(slh, city)
    ref, disrupted = notebook_features(slh, city)
    assert len(got) == len(ref)
    got, ref = aligned(got, ref)
    assert (got[KEY].astype(str).values == ref[KEY].astype(str).values).all()  # same row set
    assert (pd.to_datetime(got["date"]).dt.date.values == ref["date"].values).all()
    for c in ["hod", "dow"]:
        assert (got[c].astype(int).values == ref[c].astype(int).values).all(), c
    assert (got["disrupted"].values == ref["disrupted"].values).all()
    for c in FLOATS:
        g, r = got[c].astype(float).values, ref[c].astype(float).values
        assert (np.isnan(g) == np.isnan(r)).all(), f"{c}: missing values differ"
        diff = np.nanmax(np.abs(g - r)) if (~np.isnan(g)).any() else 0.0
        assert diff < 1e-6, f"{c}: max abs diff {diff}"
    # the compact float32 file only rounds (relative error of float32)
    import features
    small = con.execute(f"select {features.compact_select()} from features").df()
    small, _ = aligned(small, ref)
    for c in FLOATS:
        np.testing.assert_allclose(small[c].astype(float), got[c].astype(float), rtol=1e-6, atol=1e-6, equal_nan=True)
    per_day = con.execute("select date from per_day where disrupted order by date").fetchall()
    assert [d for (d,) in per_day] == disrupted


def synthetic(tmp_path, values=None):
    """10 local days x 24 h, one stop, two lines; day 3 disrupted (+10 min); one missing hour on L1."""
    rng = np.random.default_rng(0)
    hours = pd.date_range("2026-03-01 23:00", periods=240, freq="h")  # = 2 Mar 00:00 Brussels (CET)
    df = pd.concat([pd.DataFrame({"stop_id": "S1", "line_id": line, "mode": "bus", "hour": hours,
                                  "snapshots": 12, "actual_wait_min": 5.0, "scheduled_wait_min": 4.0,
                                  "excess_wait_min": rng.normal(0.5, 0.3, len(hours))}) for line in ["L1", "L2"]])
    df.loc[(df.hour >= hours[48]) & (df.hour < hours[72]), "excess_wait_min"] += 10
    df = df[~((df.line_id == "L1") & (df.hour == hours[100]))].reset_index(drop=True)
    if values is not None:
        df["excess_wait_min"] = values(df)
    city = pd.DataFrame({"hour": hours, "excess_wait_min": rng.normal(0.4, 0.1, 240), "no2": 20.0, "load_mw": 9000.0,
                         "median_wait_min": 5.0, "stops_served": 100, "vehicles_avg": 50.0, "pm10": 1.0, "pm25": 1.0})
    tmp_path.mkdir(exist_ok=True)
    df.to_parquet(tmp_path / "slh.parquet"); city.to_parquet(tmp_path / "city.parquet")
    return sql_features(tmp_path / "slh.parquet", tmp_path / "city.parquet")[1], df


def test_past_mean_uses_only_earlier_normal_days(tmp_path):
    got, raw = synthetic(tmp_path)
    got = got.sort_values(KEY).reset_index(drop=True)
    assert sorted(got.loc[got.disrupted, "date"].astype(str).unique()) == ["2026-03-04"]  # day 3 found by the median rule
    for _, r in got.sample(150, random_state=0).iterrows():
        same = got[(got.stop_id == r.stop_id) & (got.line_id == r.line_id) & (got.hod == r.hod)
                   & (got.hour < r.hour) & ~got.disrupted]
        want = same["excess_wait_min"].mean() if len(same) else np.nan
        assert (np.isnan(want) and np.isnan(r.prev_days_same_hod_ewt)) or abs(want - r.prev_days_same_hod_ewt) < 1e-9
    # the hour after the missing one has no previous-hour value
    gap = pd.Timestamp("2026-03-01 23:00") + pd.Timedelta(hours=101)
    assert np.isnan(got.loc[(got.line_id == "L1") & (got.hour == gap), "prev_hour_ewt"]).all()
    assert got.loc[(got.line_id == "L2") & (got.hour == gap), "prev_hour_ewt"].notna().all()


def test_changing_the_future_changes_no_feature_of_the_past(tmp_path):
    before, raw = synthetic(tmp_path / "a")
    last_day = pd.Timestamp("2026-03-01 23:00") + pd.Timedelta(days=9)

    def bigger_future(df):  # raise the top quarter of the last day's values: same medians, so the same disrupted days
        v = df["excess_wait_min"].copy()
        last = df.hour >= last_day
        v[last & (v > v[last].quantile(0.75))] *= 5
        return v
    after, _ = synthetic(tmp_path / "b", bigger_future)
    before, after = aligned(before, after)
    assert (before.disrupted.values == after.disrupted.values).all()
    past = before.hour < last_day
    for c in ["prev_hour_ewt", "prev_days_same_hod_ewt", "city_prev_ewt"]:
        np.testing.assert_array_equal(before.loc[past, c].values, after.loc[past, c].values)
    # a row's own target is never inside its own history either
    np.testing.assert_array_equal(before["prev_days_same_hod_ewt"].values, after["prev_days_same_hod_ewt"].values)
