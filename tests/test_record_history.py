import importlib.util
from pathlib import Path

import pandas as pd

spec = importlib.util.spec_from_file_location(
    "record_history", Path(__file__).resolve().parents[1] / "tasks" / "transit_delay" / "record_history.py")
rh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rh)

META = {"snapshot_id": "x", "first_hour_utc": "2026-10-09 22:00:00", "last_hour_utc": "2026-11-15 15:00:00", "input_rows": 10}


def frames(lgbm_week1):
    metrics = pd.Series({rh.MODEL: 0.9, rh.RULES[0]: 1.0, rh.RULES[1]: 1.2, rh.RULES[2]: 1.3})
    rows = []
    for fold, lg in [(1, lgbm_week1), (2, 0.88)]:
        rows += [{"fold": fold, "model": rh.MODEL, "mae": lg}, {"fold": fold, "model": rh.RULES[0], "mae": 1.0},
                 {"fold": fold, "model": rh.RULES[1], "mae": 1.2}, {"fold": fold, "model": rh.RULES[2], "mae": 1.3}]
    return metrics, pd.DataFrame(rows)


def test_gain_against_best_rule():
    r = rh.summarise(*frames(0.9), META)
    assert r["gain_pct"] == 10.0 and r["newest_week_gain_pct"] == 10.0
    assert r["backtest_folds"] == 2 and r["backtest_lgbm_mean"] == 0.89


def test_newest_week_is_fold_1():
    r = rh.summarise(*frames(0.99), META)
    assert r["newest_week_gain_pct"] == 1.0  # fold 1 = newest week; this is what the gate checks
